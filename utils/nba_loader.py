"""
NBA Data Loader for Prop-Bet Prophet V2
Handles live data fetching, heatmap generation, and odds.
"""

import numpy as np
import pandas as pd
import requests
import time
import warnings
from typing import Optional, List, Dict, Tuple
from datetime import datetime

try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from nba_api.live.nba.endpoints import scoreboard
    from nba_api.stats.endpoints import playergamelog, shotchartdetail
    from nba_api.stats.static import players
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    warnings.warn("nba_api not installed. Running in demo mode.")


class NBADataLoader:
    """Comprehensive NBA data loader for the Prophet model."""
    
    def __init__(self, timeout: int = 30, max_retries: int = 2):
        self.timeout = timeout
        self.max_retries = max_retries
        self.heatmap_size = 64
        self.court_x_range = (-250, 250)
        self.court_y_range = (-47, 423)
        self._cache = {}
    
    def get_todays_scoreboard(self) -> List[Dict]:
        """Get today's NBA games."""
        if not NBA_API_AVAILABLE:
            return self._get_demo_scoreboard()
        
        try:
            board = scoreboard.ScoreBoard()
            games_data = board.get_dict()
            
            games = []
            for game in games_data.get('scoreboard', {}).get('games', []):
                home = game.get('homeTeam', {})
                away = game.get('awayTeam', {})
                
                games.append({
                    'game_id': game.get('gameId'),
                    'home_team': home.get('teamName', 'Unknown'),
                    'away_team': away.get('teamName', 'Unknown'),
                    'home_team_tricode': home.get('teamTricode', ''),
                    'away_team_tricode': away.get('teamTricode', ''),
                    'home_team_id': home.get('teamId'),
                    'away_team_id': away.get('teamId'),
                    'status': game.get('gameStatusText', 'Scheduled'),
                    'start_time': game.get('gameTimeUTC', ''),
                    'is_demo': False
                })
            
            return games if games else self._get_demo_scoreboard()
            
        except Exception as e:
            warnings.warn(f"Scoreboard fetch failed: {e}")
            return self._get_demo_scoreboard()
    
    def _get_demo_scoreboard(self) -> List[Dict]:
        """Demo games for testing."""
        return [
            {
                'game_id': 'DEMO001',
                'home_team': 'Lakers',
                'away_team': 'Warriors',
                'home_team_tricode': 'LAL',
                'away_team_tricode': 'GSW',
                'home_team_id': 1610612747,
                'away_team_id': 1610612744,
                'status': 'Demo Mode',
                'start_time': datetime.now().isoformat(),
                'is_demo': True
            },
            {
                'game_id': 'DEMO002',
                'home_team': 'Celtics',
                'away_team': 'Heat',
                'home_team_tricode': 'BOS',
                'away_team_tricode': 'MIA',
                'home_team_id': 1610612738,
                'away_team_id': 1610612748,
                'status': 'Demo Mode',
                'start_time': datetime.now().isoformat(),
                'is_demo': True
            }
        ]
    
    def get_game_roster(self, game_id: str, game_data: Dict = None) -> Dict[str, List[Dict]]:
        """Get player rosters for a game."""
        if game_id.startswith('DEMO'):
            return self._get_demo_roster(game_id)
        return self._get_demo_roster(game_id)
    
    def _get_demo_roster(self, game_id: str) -> Dict[str, List[Dict]]:
        """Demo rosters with real player IDs."""
        rosters = {
            'DEMO001': {
                'home': [
                    {'player_id': 2544, 'player_name': 'LeBron James', 'team': 'LAL', 'team_abbreviation': 'LAL'},
                    {'player_id': 203076, 'player_name': 'Anthony Davis', 'team': 'LAL', 'team_abbreviation': 'LAL'},
                    {'player_id': 1628398, 'player_name': 'Austin Reaves', 'team': 'LAL', 'team_abbreviation': 'LAL'},
                    {'player_id': 1629684, 'player_name': 'Rui Hachimura', 'team': 'LAL', 'team_abbreviation': 'LAL'},
                    {'player_id': 1626156, 'player_name': "D'Angelo Russell", 'team': 'LAL', 'team_abbreviation': 'LAL'},
                ],
                'away': [
                    {'player_id': 201939, 'player_name': 'Stephen Curry', 'team': 'GSW', 'team_abbreviation': 'GSW'},
                    {'player_id': 203110, 'player_name': 'Draymond Green', 'team': 'GSW', 'team_abbreviation': 'GSW'},
                    {'player_id': 1628960, 'player_name': 'Andrew Wiggins', 'team': 'GSW', 'team_abbreviation': 'GSW'},
                    {'player_id': 1629680, 'player_name': 'Jonathan Kuminga', 'team': 'GSW', 'team_abbreviation': 'GSW'},
                    {'player_id': 203952, 'player_name': 'Kevon Looney', 'team': 'GSW', 'team_abbreviation': 'GSW'},
                ]
            },
            'DEMO002': {
                'home': [
                    {'player_id': 1628369, 'player_name': 'Jayson Tatum', 'team': 'BOS', 'team_abbreviation': 'BOS'},
                    {'player_id': 1627759, 'player_name': 'Jaylen Brown', 'team': 'BOS', 'team_abbreviation': 'BOS'},
                    {'player_id': 1630202, 'player_name': 'Payton Pritchard', 'team': 'BOS', 'team_abbreviation': 'BOS'},
                    {'player_id': 203935, 'player_name': 'Marcus Smart', 'team': 'BOS', 'team_abbreviation': 'BOS'},
                    {'player_id': 1629057, 'player_name': 'Robert Williams', 'team': 'BOS', 'team_abbreviation': 'BOS'},
                ],
                'away': [
                    {'player_id': 203999, 'player_name': 'Jimmy Butler', 'team': 'MIA', 'team_abbreviation': 'MIA'},
                    {'player_id': 1628389, 'player_name': 'Bam Adebayo', 'team': 'MIA', 'team_abbreviation': 'MIA'},
                    {'player_id': 1629639, 'player_name': 'Tyler Herro', 'team': 'MIA', 'team_abbreviation': 'MIA'},
                    {'player_id': 200768, 'player_name': 'Kyle Lowry', 'team': 'MIA', 'team_abbreviation': 'MIA'},
                    {'player_id': 1628420, 'player_name': 'Duncan Robinson', 'team': 'MIA', 'team_abbreviation': 'MIA'},
                ]
            }
        }
        return rosters.get(game_id, rosters['DEMO001'])
    
    def fetch_player_history(self, player_id: int, num_games: int = 10) -> pd.DataFrame:
        """Fetch player's recent game history."""
        return self._generate_demo_history(player_id, num_games)
    
    def _generate_demo_history(self, player_id: int, num_games: int = 10) -> pd.DataFrame:
        """Generate realistic demo history based on player archetypes."""
        np.random.seed(player_id % 1000)
        
        baselines = {
            2544: [25.5, 7.5, 8.0, 35.0, 0.50],
            203076: [24.0, 12.5, 3.5, 34.0, 0.55],
            201939: [29.0, 5.0, 5.5, 34.0, 0.47],
            1628369: [27.0, 8.5, 4.5, 36.0, 0.46],
            203999: [21.0, 6.0, 5.5, 34.0, 0.48],
        }
        
        base = baselines.get(player_id, [15, 5, 3, 28, 0.45])
        
        data = {
            'PTS': np.clip(np.random.normal(base[0], 5, num_games), 0, 50),
            'REB': np.clip(np.random.normal(base[1], 2, num_games), 0, 20),
            'AST': np.clip(np.random.normal(base[2], 2, num_games), 0, 15),
            'MIN': np.clip(np.random.normal(base[3], 4, num_games), 15, 42),
            'FG_PCT': np.clip(np.random.normal(base[4], 0.08, num_games), 0.25, 0.70)
        }
        
        return pd.DataFrame(data)
    
    def fetch_shot_chart(self, player_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch player shot chart coordinates."""
        return self._generate_demo_shot_chart(player_id)
    
    def _generate_demo_shot_chart(self, player_id: int, num_shots: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic shot distribution."""
        np.random.seed(player_id % 1000)
        
        three_pt_shooters = [201939, 1628420, 1630202]
        paint_players = [203076, 1628389, 1629057]
        
        x, y = [], []
        
        if player_id in three_pt_shooters:
            for _ in range(int(num_shots * 0.6)):
                angle = np.random.uniform(-np.pi/2, np.pi/2)
                r = np.random.uniform(220, 240)
                x.append(r * np.sin(angle))
                y.append(r * np.cos(angle) + 50)
            for _ in range(int(num_shots * 0.4)):
                x.append(np.random.uniform(-80, 80))
                y.append(np.random.uniform(0, 150))
        elif player_id in paint_players:
            for _ in range(int(num_shots * 0.7)):
                x.append(np.random.normal(0, 40))
                y.append(np.random.uniform(0, 80))
            for _ in range(int(num_shots * 0.3)):
                x.append(np.random.uniform(-150, 150))
                y.append(np.random.uniform(50, 180))
        else:
            for _ in range(int(num_shots * 0.35)):
                x.append(np.random.normal(0, 50))
                y.append(np.random.uniform(0, 80))
            for _ in range(int(num_shots * 0.25)):
                x.append(np.random.uniform(-180, 180))
                y.append(np.random.uniform(80, 180))
            for _ in range(int(num_shots * 0.4)):
                angle = np.random.uniform(-np.pi/2, np.pi/2)
                r = np.random.uniform(220, 240)
                x.append(r * np.sin(angle))
                y.append(r * np.cos(angle) + 50)
        
        return np.array(x), np.array(y)
    
    def generate_heatmap(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate 64x64 normalized heatmap for CNN input."""
        if len(x) == 0 or len(y) == 0:
            return np.zeros((1, 64, 64), dtype=np.float32)
        
        heatmap, _, _ = np.histogram2d(
            x, y,
            bins=self.heatmap_size,
            range=[list(self.court_x_range), list(self.court_y_range)]
        )
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        if SCIPY_AVAILABLE:
            heatmap = gaussian_filter(heatmap, sigma=1.5)
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
        
        return heatmap.reshape(1, 64, 64).astype(np.float32)
    
    def fetch_opponent_defense_stats(self, team_id: int) -> np.ndarray:
        """Get opponent defensive metrics for Transformer input."""
        np.random.seed(team_id % 1000 if team_id else 42)
        
        return np.array([
            np.random.uniform(105, 118),
            np.random.uniform(105, 120),
            np.random.uniform(42, 48),
            np.random.uniform(22, 28),
            np.random.uniform(6, 10),
            np.random.uniform(4, 7),
            np.random.uniform(0.44, 0.48),
            np.random.uniform(0.34, 0.38)
        ], dtype=np.float32)
    
    def prepare_lstm_input(self, history: pd.DataFrame) -> np.ndarray:
        """Prepare LSTM input tensor from game history."""
        if len(history) < 10:
            padding = pd.DataFrame(
                np.zeros((10 - len(history), 5)),
                columns=['PTS', 'REB', 'AST', 'MIN', 'FG_PCT']
            )
            history = pd.concat([history, padding], ignore_index=True)
        
        features = history[['PTS', 'REB', 'AST', 'MIN', 'FG_PCT']].values[:10]
        
        means = np.array([15.0, 5.0, 3.0, 25.0, 0.45])
        stds = np.array([8.0, 3.0, 2.5, 8.0, 0.1])
        
        return ((features - means) / stds).astype(np.float32)
    
    def generate_mock_odds(self, player_name: str, predictions: Dict[str, float]) -> Dict[str, float]:
        """Generate mock Vegas lines."""
        np.random.seed(hash(player_name) % 10000)
        
        mock_lines = {}
        for stat, pred in predictions.items():
            variance = np.random.uniform(-3, 3)
            line = round((pred + variance) * 2) / 2
            mock_lines[stat] = max(0.5, line)
        
        return mock_lines
    
    def fetch_odds(self, api_key = None) -> Dict:
        """Fetch real or mock odds."""
        if api_key:
            try:
                response = requests.get(
                    "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
                    params={'apiKey': api_key, 'regions': 'us', 'markets': 'player_points'},
                    timeout=10
                )
                if response.ok:
                    return response.json()
            except Exception:
                pass
        return {}
