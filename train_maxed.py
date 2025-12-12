"""
MAXED Training Script for Prop-Bet Prophet V2
Multi-Task Hybrid Neural Network (CNN + LSTM + Transformer)

Author: tanutanii
Created: 2025-12-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import pickle
import time
import warnings
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

CONFIG = {
    'max_players': 300,
    'seasons': ['2022-23', '2023-24'],
    'games_per_player': 20,
    'cnn_channels': [32, 64, 128, 256],
    'cnn_output_dim': 256,
    'lstm_hidden_dim': 128,
    'lstm_layers': 3,
    'lstm_output_dim': 256,
    'transformer_d_model': 128,
    'transformer_heads': 8,
    'transformer_layers': 3,
    'transformer_output_dim': 256,
    'fusion_dim': 512,
    'dropout': 0.4,
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 5e-4,
    'warmup_epochs': 5,
    'patience': 15,
    'use_augmentation': True,
    'noise_std': 0.05,
    'mixup_alpha': 0.2,
}

def print_config():
    print("\n" + "=" * 70)
    print("PROP-BET PROPHET V2 - MAXED TRAINING")
    print("=" * 70)
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("-" * 40 + "\n")

class DeepCNNBranch(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256], output_dim=256, dropout=0.4):
        super().__init__()
        layers = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.append(nn.AdaptiveAvgPool2d((4, 4)))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * 4 * 4, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

class DeepLSTMBranch(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=3, output_dim=256, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, output_dim), nn.LayerNorm(output_dim), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout), nn.Linear(output_dim, output_dim), nn.LayerNorm(output_dim))
    
    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)

class DeepTransformerBranch(nn.Module):
    def __init__(self, input_dim=8, d_model=128, nhead=8, num_layers=3, output_dim=256, dropout=0.4):
        super().__init__()
        self.input_embed = nn.Sequential(nn.Linear(1, d_model), nn.LayerNorm(d_model), nn.LeakyReLU(0.2, inplace=True))
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, output_dim), nn.LayerNorm(output_dim), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout), nn.Linear(output_dim, output_dim), nn.LayerNorm(output_dim))
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(-1)
        x = self.input_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.fc(x[:, 0])

class MaxedHybridModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = config or CONFIG
        self.cnn = DeepCNNBranch(channels=config['cnn_channels'], output_dim=config['cnn_output_dim'], dropout=config['dropout'])
        self.lstm = DeepLSTMBranch(input_dim=5, hidden_dim=config['lstm_hidden_dim'], num_layers=config['lstm_layers'], output_dim=config['lstm_output_dim'], dropout=config['dropout'])
        self.transformer = DeepTransformerBranch(input_dim=8, d_model=config['transformer_d_model'], nhead=config['transformer_heads'], num_layers=config['transformer_layers'], output_dim=config['transformer_output_dim'], dropout=config['dropout'])
        total_features = config['cnn_output_dim'] + config['lstm_output_dim'] + config['transformer_output_dim']
        self.gate = nn.Sequential(nn.Linear(total_features, total_features), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.Linear(total_features, config['fusion_dim']), nn.LayerNorm(config['fusion_dim']), nn.GELU(), nn.Dropout(config['dropout']), nn.Linear(config['fusion_dim'], config['fusion_dim']), nn.LayerNorm(config['fusion_dim']), nn.GELU(), nn.Dropout(config['dropout'] / 2))
        self.shared = nn.Sequential(nn.Linear(config['fusion_dim'], 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(config['dropout'] / 2))
        self.pts_head = nn.Linear(256, 2)
        self.reb_head = nn.Linear(256, 2)
        self.ast_head = nn.Linear(256, 2)
        self.log_vars = nn.Parameter(torch.zeros(3))
    
    def forward(self, heatmap, history, defense):
        cnn_out = self.cnn(heatmap)
        lstm_out = self.lstm(history)
        transformer_out = self.transformer(defense)
        concat = torch.cat([cnn_out, lstm_out, transformer_out], dim=1)
        gated = concat * self.gate(concat)
        fused = self.fusion(gated)
        shared = self.shared(fused)
        pts = self.pts_head(shared)
        reb = self.reb_head(shared)
        ast = self.ast_head(shared)
        return {'pts': pts[:, 0], 'pts_conf': torch.sigmoid(pts[:, 1]), 'reb': reb[:, 0], 'reb_conf': torch.sigmoid(reb[:, 1]), 'ast': ast[:, 0], 'ast_conf': torch.sigmoid(ast[:, 1])}
    
    def get_model_summary(self):
        total = sum(p.numel() for p in self.parameters())
        return {'total_params': total, 'size_mb': total * 4 / (1024 ** 2)}

class UncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, predictions, targets, log_vars):
        total_loss = 0
        losses = {}
        for i, stat in enumerate(['pts', 'reb', 'ast']):
            loss = self.huber(predictions[stat], targets[stat]).mean()
            precision = torch.exp(-log_vars[i])
            total_loss += precision * loss + log_vars[i]
            losses[stat] = loss.item()
        return total_loss, losses

class MaxedNBADataset(Dataset):
    def __init__(self, num_samples=5000):
        self._generate_synthetic(num_samples)
    
    def _generate_synthetic(self, num_samples):
        self.heatmaps = np.random.rand(num_samples, 1, 64, 64).astype(np.float32) * 0.5
        for i in range(num_samples):
            cx, cy = np.random.randint(20, 44), np.random.randint(20, 44)
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    if 0 <= cx+dx < 64 and 0 <= cy+dy < 64:
                        dist = np.sqrt(dx**2 + dy**2)
                        self.heatmaps[i, 0, cx+dx, cy+dy] += np.exp(-dist/4) * 0.5
        self.histories = np.random.randn(num_samples, 10, 5).astype(np.float32) * 0.3
        self.defenses = np.random.rand(num_samples, 8).astype(np.float32)
        self.pts = np.random.normal(15, 8, num_samples).clip(0, 50).astype(np.float32)
        self.reb = np.random.normal(5, 3, num_samples).clip(0, 20).astype(np.float32)
        self.ast = np.random.normal(3, 2.5, num_samples).clip(0, 15).astype(np.float32)
    
    def __len__(self):
        return len(self.heatmaps)
    
    def __getitem__(self, idx):
        return {'heatmap': torch.from_numpy(self.heatmaps[idx]), 'history': torch.from_numpy(self.histories[idx]), 'defense': torch.from_numpy(self.defenses[idx]), 'pts': torch.tensor(self.pts[idx]), 'reb': torch.tensor(self.reb[idx]), 'ast': torch.tensor(self.ast[idx])}

def train():
    print_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dataset = MaxedNBADataset(5000)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    model = MaxedHybridModel(CONFIG).to(device)
    print(f"Model params: {model.get_model_summary()['total_params']:,}")
    criterion = UncertaintyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            heatmaps = batch['heatmap'].to(device)
            histories = batch['history'].to(device)
            defenses = batch['defense'].to(device)
            targets = {k: batch[k].to(device) for k in ['pts', 'reb', 'ast']}
            optimizer.zero_grad()
            predictions = model(heatmaps, histories, defenses)
            loss, _ = criterion(predictions, targets, model.log_vars)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        model.eval()
        val_loss = 0
        pts_mae, reb_mae, ast_mae = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                heatmaps = batch['heatmap'].to(device)
                histories = batch['history'].to(device)
                defenses = batch['defense'].to(device)
                targets = {k: batch[k].to(device) for k in ['pts', 'reb', 'ast']}
                predictions = model(heatmaps, histories, defenses)
                loss, _ = criterion(predictions, targets, model.log_vars)
                val_loss += loss.item()
                pts_mae += torch.abs(predictions['pts'] - targets['pts']).mean().item()
                reb_mae += torch.abs(predictions['reb'] - targets['reb']).mean().item()
                ast_mae += torch.abs(predictions['ast'] - targets['ast']).mean().item()
        val_loss /= len(val_loader)
        pts_mae /= len(val_loader)
        reb_mae /= len(val_loader)
        ast_mae /= len(val_loader)
        print(f"Epoch {epoch+1}: Val={val_loss:.4f}, PTS={pts_mae:.2f}, REB={reb_mae:.2f}, AST={ast_mae:.2f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs('models', exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'config': CONFIG}, 'models/prophet_v2.pth')
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    print(f"\nTraining complete! Time: {(time.time() - start_time) / 60:.1f} min")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Final MAE - PTS: {pts_mae:.2f} | REB: {reb_mae:.2f} | AST: {ast_mae:.2f}")

if __name__ == "__main__":
    train()
