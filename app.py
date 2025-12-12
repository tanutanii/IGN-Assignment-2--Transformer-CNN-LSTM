"""
Prop-Bet Prophet V2 - Streamlit Dashboard
NBA Player Performance Prediction System
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.patches import Circle, Rectangle, Arc
import warnings
import sys
import os

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Prop-Bet Prophet V2", page_icon="🏀", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: bold; text-align: center; background: linear-gradient(90deg, #39FF14, #00D4FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .player-card { background: #1a1a2e; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; border-left: 4px solid #39FF14; }
    .over-badge { background: #00C853; color: white; padding: 0.25rem 0.75rem; border-radius: 15px; }
    .under-badge { background: #FF5252; color: white; padding: 0.25rem 0.75rem; border-radius: 15px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        from train_maxed import MaxedHybridModel, CONFIG
        model = MaxedHybridModel(CONFIG)
        if os.path.exists('models/prophet_v2.pth'):
            checkpoint = torch.load('models/prophet_v2.pth', map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model, True, CONFIG
        return model, False, CONFIG
    except Exception as e:
        st.error(f"Error: {e}")
        return None, False, {}

@st.cache_resource
def load_data_loader():
    try:
        from utils.nba_loader import NBADataLoader
        return NBADataLoader()
    except:
        return None

def draw_court(ax, color='white', lw=2):
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)
    ax.add_patch(hoop)
    ax.plot([-30, 30], [-7.5, -7.5], color=color, linewidth=lw)
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False)
    ax.add_patch(outer_box)
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)
    ax.add_patch(three_arc)
    ax.set_xlim(-250, 250)
    ax.set_ylim(-50, 425)
    ax.set_aspect('equal')
    ax.axis('off')
    return ax

def plot_radar_chart(predictions, player_name):
    categories = ['Points', 'Rebounds', 'Assists']
    values = [predictions.get('pts', 0), predictions.get('reb', 0), predictions.get('ast', 0)]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself', name=player_name, fillcolor='rgba(57, 255, 20, 0.3)', line=dict(color='#39FF14', width=3)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max(values) + 10])), paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    return fig

def run_inference(model, heatmap, history, defense):
    if model is None:
        return {'pts': np.random.uniform(15, 30), 'reb': np.random.uniform(3, 10), 'ast': np.random.uniform(2, 8)}, {'pts_conf': 0.75, 'reb_conf': 0.75, 'ast_conf': 0.75}
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(heatmap).unsqueeze(0).float(), torch.from_numpy(history).unsqueeze(0).float(), torch.from_numpy(defense).unsqueeze(0).float())
        return {'pts': output['pts'].item(), 'reb': output['reb'].item(), 'ast': output['ast'].item()}, {'pts_conf': output['pts_conf'].item(), 'reb_conf': output['reb_conf'].item(), 'ast_conf': output['ast_conf'].item()}

def main():
    st.markdown('<h1 class="main-header">🏀 Prop-Bet Prophet V2</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#888;">Multi-Task Hybrid Neural Network (CNN + LSTM + Transformer)</p>', unsafe_allow_html=True)
    
    model, is_trained, config = load_model()
    data_loader = load_data_loader()
    
    with st.sidebar:
        st.markdown("### 🔧 Model Status")
        st.success("✅ Trained" if is_trained else "⚠️ Untrained")
        if model:
            st.metric("Parameters", f"{model.get_model_summary()['total_params']:,}")
        st.markdown("---")
        st.markdown("### 🎮 Select Game")
        games = data_loader.get_todays_scoreboard() if data_loader else [{'game_id': 'DEMO', 'away_team': 'Warriors', 'home_team': 'Lakers', 'status': 'Demo'}]
        game_opts = [f"{g['away_team']} @ {g['home_team']}" for g in games]
        selected_idx = st.selectbox("Game", range(len(game_opts)), format_func=lambda x: game_opts[x])
        selected_game = games[selected_idx]
    
    st.markdown(f"### 🏟️ {selected_game['away_team']} @ {selected_game['home_team']}")
    
    roster = data_loader.get_game_roster(selected_game['game_id']) if data_loader else {'home': [{'player_id': 2544, 'player_name': 'LeBron James', 'team': 'LAL'}], 'away': [{'player_id': 201939, 'player_name': 'Stephen Curry', 'team': 'GSW'}]}
    
    if st.button("🔮 Analyze Players", type="primary", use_container_width=True):
        results = []
        all_players = roster.get('home', []) + roster.get('away', [])
        progress = st.progress(0)
        
        for i, player in enumerate(all_players):
            if data_loader:
                history = data_loader.fetch_player_history(player['player_id'])
                x, y = data_loader.fetch_shot_chart(player['player_id'])
                heatmap = data_loader.generate_heatmap(x, y)
                lstm_input = data_loader.prepare_lstm_input(history)
                defense = data_loader.fetch_opponent_defense_stats(0)
            else:
                heatmap, lstm_input, defense = np.random.rand(1, 64, 64).astype(np.float32), np.random.randn(10, 5).astype(np.float32), np.random.rand(8).astype(np.float32)
            
            preds, confs = run_inference(model, heatmap, lstm_input, defense)
            odds = data_loader.generate_mock_odds(player['player_name'], preds) if data_loader else preds
            
            results.append({'Player': player['player_name'], 'Team': player.get('team', 'N/A'), 'PTS Pred': round(preds['pts'], 1), 'PTS Line': round(odds.get('pts', 0), 1), 'PTS Edge': round(preds['pts'] - odds.get('pts', preds['pts']), 1), 'REB Pred': round(preds['reb'], 1), 'AST Pred': round(preds['ast'], 1), 'Confidence': round(np.mean([confs['pts_conf'], confs['reb_conf'], confs['ast_conf']]) * 100, 1)})
            progress.progress((i + 1) / len(all_players))
        
        st.session_state['results'] = pd.DataFrame(results)
    
    if 'results' in st.session_state:
        df = st.session_state['results']
        tab1, tab2 = st.tabs(["🔥 Hot Picks", "📋 Full Board"])
        
        with tab1:
            for _, row in df.nlargest(3, 'PTS Edge').iterrows():
                badge = "over-badge" if row['PTS Edge'] > 0 else "under-badge"
                st.markdown(f'<div class="player-card"><strong>{row["Player"]}</strong> | Pred: {row["PTS Pred"]} pts | <span class="{badge}">{"OVER" if row["PTS Edge"] > 0 else "UNDER"} {abs(row["PTS Edge"]):+.1f}</span></div>', unsafe_allow_html=True)
        
        with tab2:
            st.dataframe(df, use_container_width=True)
            st.download_button("📥 Download", df.to_csv(index=False), "predictions.csv")
    
    st.markdown("---")
    st.caption("🏀 Prop-Bet Prophet V2 | For educational purposes only")

if __name__ == "__main__":
    main()
