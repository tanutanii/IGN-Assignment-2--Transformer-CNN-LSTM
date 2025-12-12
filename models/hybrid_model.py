"""
MAXED Hybrid Model for NBA Prediction System
Combines CNN, LSTM, and Transformer architectures for robust time-series prediction.

Author: tanutanii
Created: 2025-12-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Data Configuration
    'input_features': 64,           # Number of input features
    'sequence_length': 10,          # Length of input sequences
    'num_classes': 2,               # Binary classification (Win/Loss)
    
    # CNN Branch Configuration
    'cnn': {
        'channels': [64, 128, 256, 512],
        'kernel_sizes': [3, 3, 3, 3],
        'pool_size': 2,
        'dropout': 0.3,
        'use_batch_norm': True,
    },
    
    # LSTM Branch Configuration
    'lstm': {
        'hidden_size': 256,
        'num_layers': 3,
        'dropout': 0.3,
        'bidirectional': True,
        'use_attention': True,
    },
    
    # Transformer Branch Configuration
    'transformer': {
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 4,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'max_seq_length': 512,
    },
    
    # Fusion Configuration
    'fusion': {
        'method': 'attention',       # 'concat', 'attention', 'gated'
        'hidden_size': 512,
        'dropout': 0.4,
    },
    
    # Training Configuration
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 100,
        'early_stopping_patience': 15,
    }
}


# =============================================================================
# POSITIONAL ENCODING (for Transformer)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer architecture."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# =============================================================================
# DEEP CNN BRANCH
# =============================================================================

class DeepCNNBranch(nn.Module):
    """
    Deep Convolutional Neural Network branch for spatial feature extraction.
    Uses 1D convolutions to capture local patterns in sequential data.
    """
    
    def __init__(
        self,
        input_features: int,
        channels: list = [64, 128, 256, 512],
        kernel_sizes: list = [3, 3, 3, 3],
        pool_size: int = 2,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super(DeepCNNBranch, self).__init__()
        
        self.input_features = input_features
        self.channels = channels
        self.use_batch_norm = use_batch_norm
        
        # Build convolutional layers
        layers = []
        in_channels = input_features
        
        for i, (out_channels, kernel_size) in enumerate(zip(channels, kernel_sizes)):
            # Convolutional layer
            layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            
            # Activation
            layers.append(nn.ReLU(inplace=True))
            
            # Dropout
            layers.append(nn.Dropout(dropout))
            
            # Pooling (except for last layer)
            if i < len(channels) - 1:
                layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=1, padding=pool_size // 2))
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Attention mechanism for CNN
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], channels[-1] // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels[-1] // 4, channels[-1]),
            nn.Sigmoid()
        )
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Output projection
        self.output_proj = nn.Linear(channels[-1] * 2, channels[-1])
        
        self.output_size = channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, input_features]
        Returns:
            Tensor of shape [batch_size, output_size]
        """
        # Transpose for Conv1d: [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Channel attention
        att_weights = self.channel_attention(x).unsqueeze(-1)
        x = x * att_weights
        
        # Global pooling (combine avg and max)
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        
        # Concatenate and project
        x = torch.cat([avg_pool, max_pool], dim=-1)
        x = self.output_proj(x)
        
        return x


# =============================================================================
# DEEP LSTM BRANCH
# =============================================================================

class DeepLSTMBranch(nn.Module):
    """
    Deep LSTM branch for sequential pattern recognition.
    Supports bidirectional processing and attention mechanism.
    """
    
    def __init__(
        self,
        input_features: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True
    ):
        super(DeepLSTMBranch, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_proj = nn.Linear(input_features, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Layer normalization after LSTM
        lstm_output_size = hidden_size * self.num_directions
        self.lstm_norm = nn.LayerNorm(lstm_output_size)
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(lstm_output_size)
        
        # Output layers
        self.output_fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.output_size = hidden_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, input_features]
        Returns:
            Tensor of shape [batch_size, output_size]
        """
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        
        # Apply attention if enabled
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = self.attention_norm(lstm_out + attn_out)
        
        # Combine final hidden states from both directions
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            combined = torch.cat([h_forward, h_backward], dim=-1)
        else:
            combined = h_n[-1, :, :]
        
        # Also use attention-weighted sum of outputs
        attention_weights = F.softmax(torch.mean(lstm_out, dim=-1), dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        # Combine hidden state and context
        output = combined + context
        
        # Final projection
        output = self.output_fc(output)
        
        return output


# =============================================================================
# DEEP TRANSFORMER BRANCH
# =============================================================================

class DeepTransformerBranch(nn.Module):
    """
    Deep Transformer branch for capturing long-range dependencies.
    Uses self-attention mechanism for global context understanding.
    """
    
    def __init__(
        self,
        input_features: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 512
    ):
        super(DeepTransformerBranch, self).__init__()
        
        self.d_model = d_model
        
        # Input embedding
        self.input_embedding = nn.Linear(input_features, d_model)
        self.embedding_norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Output projection
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        self.output_size = d_model
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, input_features]
            mask: Optional attention mask
        Returns:
            Tensor of shape [batch_size, output_size]
        """
        batch_size = x.size(0)
        
        # Input embedding
        x = self.input_embedding(x)
        x = self.embedding_norm(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding (transpose for positional encoding format)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Extract CLS token output
        cls_output = x[:, 0, :]
        
        # Also compute mean pooling for additional context
        if mask is not None:
            # Mask out padded positions
            mask_expanded = (~mask).unsqueeze(-1).float()
            mean_output = (x[:, 1:, :] * mask_expanded[:, 1:, :]).sum(dim=1) / mask_expanded[:, 1:, :].sum(dim=1).clamp(min=1)
        else:
            mean_output = x[:, 1:, :].mean(dim=1)
        
        # Combine CLS and mean pooling
        combined = cls_output + mean_output
        
        # Output projection
        output = self.output_fc(combined)
        
        return output


# =============================================================================
# MAXED HYBRID MODEL
# =============================================================================

class MaxedHybridModel(nn.Module):
    """
    MAXED Hybrid Model combining CNN, LSTM, and Transformer branches.
    Uses attention-based fusion to combine features from all branches.
    """
    
    def __init__(
        self,
        config: Dict = None,
        input_features: int = None,
        sequence_length: int = None,
        num_classes: int = None
    ):
        super(MaxedHybridModel, self).__init__()
        
        # Use provided config or default CONFIG
        self.config = config or CONFIG
        
        # Override with explicit parameters if provided
        self.input_features = input_features or self.config['input_features']
        self.sequence_length = sequence_length or self.config['sequence_length']
        self.num_classes = num_classes or self.config['num_classes']
        
        # Initialize branches
        self._build_cnn_branch()
        self._build_lstm_branch()
        self._build_transformer_branch()
        
        # Calculate total feature size from all branches
        self.total_features = (
            self.cnn_branch.output_size +
            self.lstm_branch.output_size +
            self.transformer_branch.output_size
        )
        
        # Build fusion layer
        self._build_fusion_layer()
        
        # Build classifier head
        self._build_classifier()
        
        # Initialize weights
        self._init_weights()
    
    def _build_cnn_branch(self):
        """Build the CNN branch."""
        cnn_config = self.config['cnn']
        self.cnn_branch = DeepCNNBranch(
            input_features=self.input_features,
            channels=cnn_config['channels'],
            kernel_sizes=cnn_config['kernel_sizes'],
            pool_size=cnn_config['pool_size'],
            dropout=cnn_config['dropout'],
            use_batch_norm=cnn_config['use_batch_norm']
        )
    
    def _build_lstm_branch(self):
        """Build the LSTM branch."""
        lstm_config = self.config['lstm']
        self.lstm_branch = DeepLSTMBranch(
            input_features=self.input_features,
            hidden_size=lstm_config['hidden_size'],
            num_layers=lstm_config['num_layers'],
            dropout=lstm_config['dropout'],
            bidirectional=lstm_config['bidirectional'],
            use_attention=lstm_config['use_attention']
        )
    
    def _build_transformer_branch(self):
        """Build the Transformer branch."""
        transformer_config = self.config['transformer']
        self.transformer_branch = DeepTransformerBranch(
            input_features=self.input_features,
            d_model=transformer_config['d_model'],
            nhead=transformer_config['nhead'],
            num_encoder_layers=transformer_config['num_encoder_layers'],
            dim_feedforward=transformer_config['dim_feedforward'],
            dropout=transformer_config['dropout'],
            max_seq_length=transformer_config['max_seq_length']
        )
    
    def _build_fusion_layer(self):
        """Build the fusion layer to combine branch outputs."""
        fusion_config = self.config['fusion']
        fusion_method = fusion_config['method']
        hidden_size = fusion_config['hidden_size']
        dropout = fusion_config['dropout']
        
        if fusion_method == 'attention':
            # Attention-based fusion
            self.branch_projections = nn.ModuleList([
                nn.Linear(self.cnn_branch.output_size, hidden_size),
                nn.Linear(self.lstm_branch.output_size, hidden_size),
                nn.Linear(self.transformer_branch.output_size, hidden_size)
            ])
            
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            
            self.fusion_norm = nn.LayerNorm(hidden_size)
            self.fusion_output_size = hidden_size
            
        elif fusion_method == 'gated':
            # Gated fusion
            self.gate_fc = nn.Linear(self.total_features, 3)
            self.branch_projections = nn.ModuleList([
                nn.Linear(self.cnn_branch.output_size, hidden_size),
                nn.Linear(self.lstm_branch.output_size, hidden_size),
                nn.Linear(self.transformer_branch.output_size, hidden_size)
            ])
            self.fusion_output_size = hidden_size
            
        else:  # concat
            # Simple concatenation
            self.fusion_fc = nn.Sequential(
                nn.Linear(self.total_features, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            )
            self.fusion_output_size = hidden_size
        
        self.fusion_method = fusion_method
    
    def _build_classifier(self):
        """Build the classification head."""
        fusion_config = self.config['fusion']
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_output_size, self.fusion_output_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_config['dropout']),
            nn.Linear(self.fusion_output_size // 2, self.fusion_output_size // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_config['dropout'] / 2),
            nn.Linear(self.fusion_output_size // 4, self.num_classes)
        )
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _fuse_features(
        self,
        cnn_features: torch.Tensor,
        lstm_features: torch.Tensor,
        transformer_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse features from all branches.
        
        Args:
            cnn_features: Features from CNN branch
            lstm_features: Features from LSTM branch
            transformer_features: Features from Transformer branch
            
        Returns:
            Fused feature tensor
        """
        if self.fusion_method == 'attention':
            # Project all features to same dimension
            proj_cnn = self.branch_projections[0](cnn_features)
            proj_lstm = self.branch_projections[1](lstm_features)
            proj_transformer = self.branch_projections[2](transformer_features)
            
            # Stack as sequence for attention
            stacked = torch.stack([proj_cnn, proj_lstm, proj_transformer], dim=1)
            
            # Self-attention over branches
            attn_out, _ = self.fusion_attention(stacked, stacked, stacked)
            attn_out = self.fusion_norm(stacked + attn_out)
            
            # Mean pooling over branches
            fused = attn_out.mean(dim=1)
            
        elif self.fusion_method == 'gated':
            # Concatenate for gate computation
            concat_features = torch.cat([cnn_features, lstm_features, transformer_features], dim=-1)
            
            # Compute gates
            gates = F.softmax(self.gate_fc(concat_features), dim=-1)
            
            # Project and weight by gates
            proj_cnn = self.branch_projections[0](cnn_features)
            proj_lstm = self.branch_projections[1](lstm_features)
            proj_transformer = self.branch_projections[2](transformer_features)
            
            fused = (
                gates[:, 0:1] * proj_cnn +
                gates[:, 1:2] * proj_lstm +
                gates[:, 2:3] * proj_transformer
            )
            
        else:  # concat
            concat_features = torch.cat([cnn_features, lstm_features, transformer_features], dim=-1)
            fused = self.fusion_fc(concat_features)
        
        return fused
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_features]
            mask: Optional attention mask for transformer
            return_features: If True, return intermediate features
            
        Returns:
            Classification logits of shape [batch_size, num_classes]
            If return_features=True, also returns dict of branch features
        """
        # Extract features from each branch
        cnn_features = self.cnn_branch(x)
        lstm_features = self.lstm_branch(x)
        transformer_features = self.transformer_branch(x, mask)
        
        # Fuse features
        fused_features = self._fuse_features(cnn_features, lstm_features, transformer_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        if return_features:
            features = {
                'cnn': cnn_features,
                'lstm': lstm_features,
                'transformer': transformer_features,
                'fused': fused_features
            }
            return logits, features
        
        return logits
    
    def predict(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Make predictions (probability distribution).
        
        Args:
            x: Input tensor
            mask: Optional attention mask
            
        Returns:
            Probability distribution over classes
        """
        logits = self.forward(x, mask)
        return F.softmax(logits, dim=-1)
    
    def predict_class(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Make class predictions.
        
        Args:
            x: Input tensor
            mask: Optional attention mask
            
        Returns:
            Predicted class indices
        """
        logits = self.forward(x, mask)
        return torch.argmax(logits, dim=-1)
    
    def get_branch_contributions(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Analyze the contribution of each branch to the final prediction.
        Only available for gated fusion method.
        
        Args:
            x: Input tensor
            mask: Optional attention mask
            
        Returns:
            Dictionary with branch contribution percentages
        """
        if self.fusion_method != 'gated':
            raise ValueError("Branch contributions only available for gated fusion")
        
        with torch.no_grad():
            cnn_features = self.cnn_branch(x)
            lstm_features = self.lstm_branch(x)
            transformer_features = self.transformer_branch(x, mask)
            
            concat_features = torch.cat([cnn_features, lstm_features, transformer_features], dim=-1)
            gates = F.softmax(self.gate_fc(concat_features), dim=-1)
            
            # Average over batch
            avg_gates = gates.mean(dim=0)
            
            return {
                'cnn': avg_gates[0].item(),
                'lstm': avg_gates[1].item(),
                'transformer': avg_gates[2].item()
            }
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable parameters in each component.
        
        Returns:
            Dictionary with parameter counts
        """
        cnn_params = sum(p.numel() for p in self.cnn_branch.parameters() if p.requires_grad)
        lstm_params = sum(p.numel() for p in self.lstm_branch.parameters() if p.requires_grad)
        transformer_params = sum(p.numel() for p in self.transformer_branch.parameters() if p.requires_grad)
        
        # Fusion and classifier params
        other_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        other_params -= (cnn_params + lstm_params + transformer_params)
        
        total_params = cnn_params + lstm_params + transformer_params + other_params
        
        return {
            'cnn_branch': cnn_params,
            'lstm_branch': lstm_params,
            'transformer_branch': transformer_params,
            'fusion_classifier': other_params,
            'total': total_params
        }


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model(
    input_features: int = 64,
    sequence_length: int = 10,
    num_classes: int = 2,
    config: Dict = None
) -> MaxedHybridModel:
    """
    Factory function to create a MaxedHybridModel.
    
    Args:
        input_features: Number of input features
        sequence_length: Length of input sequences
        num_classes: Number of output classes
        config: Optional configuration dictionary
        
    Returns:
        Initialized MaxedHybridModel
    """
    model = MaxedHybridModel(
        config=config,
        input_features=input_features,
        sequence_length=sequence_length,
        num_classes=num_classes
    )
    return model


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test the model
    print("Testing MAXED Hybrid Model...")
    print("=" * 60)
    
    # Create model
    model = create_model(
        input_features=64,
        sequence_length=10,
        num_classes=2
    )
    
    # Print parameter counts
    param_counts = model.count_parameters()
    print("\nParameter Counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    
    # Test forward pass
    batch_size = 8
    seq_len = 10
    features = 64
    
    x = torch.randn(batch_size, seq_len, features)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    logits = model(x)
    print(f"Output logits shape: {logits.shape}")
    
    # Forward pass with features
    logits, feature_dict = model(x, return_features=True)
    print("\nFeature shapes:")
    for name, feat in feature_dict.items():
        print(f"  {name}: {feat.shape}")
    
    # Predictions
    probs = model.predict(x)
    classes = model.predict_class(x)
    print(f"\nPrediction probabilities shape: {probs.shape}")
    print(f"Predicted classes shape: {classes.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
