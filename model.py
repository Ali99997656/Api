import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class SoccerMatchmakerAI:
    def __init__(self):
        self.ai_model = KMeans(n_clusters=3, random_state=42, n_init='auto')
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self.mean_stats = {}
        self.rating_max = 100.0
        self.rating_min = 0.0
        
    def _calculate_base_rating(self, df):
        w_goals, w_assists, w_shots, w_saves, w_behavior = 10.0, 8.0, 3.0, 12.0, 2.0
        return ((df['goals'] * w_goals) + (df['assists'] * w_assists) + 
                (df['shots'] * w_shots) + (df['saves'] * w_saves) + 
                (df['behavior_rank'] * w_behavior))

    def _preprocess_backend_payload(self, raw_json_data):
    
        df = pd.DataFrame(raw_json_data)
        
        required_cols = ['goals', 'assists', 'shots', 'saves']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].astype(float)
                
        if 'behavior_rank' not in df.columns:
            behavior_cols = [col for col in df.columns if 'الالتزام' in col or 'behavior' in col.lower()]
            if len(behavior_cols) > 0:
                df['behavior_sum'] = df[behavior_cols].sum(axis=1)
                df['behavior_rank'] = (df['behavior_sum'] / len(behavior_cols)) * 10.0
            else:
                df['behavior_rank'] = 5.0
                
        df['behavior_rank'] = df['behavior_rank'].astype(float)
                
        return df

    def train_model(self, historical_raw_data):
      
        print("training model with historical data...")
        df = self._preprocess_backend_payload(historical_raw_data).fillna(0)
        df['raw_rating'] = self._calculate_base_rating(df)
        
        self.rating_max = df['raw_rating'].max()
        self.rating_min = df['raw_rating'].min()
        
        if self.rating_max > self.rating_min:
            df['calculated_overall_rating'] = ((df['raw_rating'] - self.rating_min) / (self.rating_max - self.rating_min)) * 100
        else:
            df['calculated_overall_rating'] = 50.0
            
        self.mean_stats = {
            'goals': df['goals'].mean(), 'assists': df['assists'].mean(),
            'shots': df['shots'].mean(), 'saves': df['saves'].mean(),
            'calculated_overall_rating': df['calculated_overall_rating'].mean()
        }
        
        features = df[['goals', 'assists', 'saves', 'shots', 'calculated_overall_rating']]
        scaled_features = self.scaler.fit_transform(features)
        self.ai_model.fit(scaled_features)
        self.is_trained = True
        print("model training completed.")

    def generate_match(self, raw_json_payload, team_size=5):
      
        if not self.is_trained:
            raise Exception("Model is not trained yet!")
            
        df = self._preprocess_backend_payload(raw_json_payload).fillna(0)
        total_players_received = len(df)
        starters_needed = team_size * 2
        
        if total_players_received < starters_needed:
            return {"error": f"game needs {starters_needed} players minimum, input received {total_players_received}."}

        df['raw_rating'] = self._calculate_base_rating(df)
        if self.rating_max > self.rating_min:
            df['calculated_overall_rating'] = ((df['raw_rating'] - self.rating_min) / (self.rating_max - self.rating_min)) * 100
        else:
            df['calculated_overall_rating'] = 50.0
        df['calculated_overall_rating'] = df['calculated_overall_rating'].clip(0, 100)
        
        is_new = (df['goals'] == 0) & (df['assists'] == 0) & (df['shots'] == 0) & (df['saves'] == 0)
        if is_new.any():
            for col in ['goals', 'assists', 'shots', 'saves', 'calculated_overall_rating']:
                df.loc[is_new, col] = self.mean_stats[col]
        
        df = df.round(1)
        
        features = df[['goals', 'assists', 'saves', 'shots', 'calculated_overall_rating']]
        scaled_features = self.scaler.transform(features)
        df['ai_cluster'] = self.ai_model.predict(scaled_features) 
        
     
        gks = df[df['saves'] > 0].sort_values(by=['saves', 'calculated_overall_rating'], ascending=[False, False])
        non_gks = df[df['saves'] == 0].sort_values(by='calculated_overall_rating', ascending=True)
        
        if len(gks) >= 2:
            gk_a, gk_b = gks.iloc[0], gks.iloc[1]
        elif len(gks) == 1:
            gk_a, gk_b = gks.iloc[0], non_gks.iloc[0]
        else:
            gk_a, gk_b = non_gks.iloc[0], non_gks.iloc[1]
            
        remaining_pool = df[~df['player_id'].isin([gk_a['player_id'], gk_b['player_id']])].copy()
        
  
        field_starters_needed = (team_size - 1) * 2
        remaining_pool = remaining_pool.sort_values(by='calculated_overall_rating', ascending=False) 
        
        starting_field_players = remaining_pool.head(field_starters_needed)
        substitutes = remaining_pool.iloc[field_starters_needed:]
        
        team_a, team_b = [], []
        team_a_score, team_b_score = gk_a['calculated_overall_rating'], gk_b['calculated_overall_rating']
        
        team_a.append({'player_id': int(gk_a['player_id']), 'Role': 'GK', 'Status': 'Starter', 'AI_Class': int(gk_a['ai_cluster']), 'Rating': gk_a['calculated_overall_rating']})
        team_b.append({'player_id': int(gk_b['player_id']), 'Role': 'GK', 'Status': 'Starter', 'AI_Class': int(gk_b['ai_cluster']), 'Rating': gk_b['calculated_overall_rating']})
        
        for _, player in starting_field_players.iterrows():
            if team_a_score <= team_b_score:
                team_a.append({'player_id': int(player['player_id']), 'Role': 'Player', 'Status': 'Starter', 'AI_Class': int(player['ai_cluster']), 'Rating': player['calculated_overall_rating']})
                team_a_score += player['calculated_overall_rating']
            else:
                team_b.append({'player_id': int(player['player_id']), 'Role': 'Player', 'Status': 'Starter', 'AI_Class': int(player['ai_cluster']), 'Rating': player['calculated_overall_rating']})
                team_b_score += player['calculated_overall_rating']
                
        for _, sub in substitutes.iterrows():
            if team_a_score <= team_b_score:
                team_a.append({'player_id': int(sub['player_id']), 'Role': 'Player', 'Status': 'Bench', 'AI_Class': int(sub['ai_cluster']), 'Rating': sub['calculated_overall_rating']})
                team_a_score += sub['calculated_overall_rating']
            else:
                team_b.append({'player_id': int(sub['player_id']), 'Role': 'Player', 'Status': 'Bench', 'AI_Class': int(sub['ai_cluster']), 'Rating': sub['calculated_overall_rating']})
                team_b_score += sub['calculated_overall_rating']
                
        return {
            'Team_A': team_a,
            'Team_B': team_b,
            'Team_A_Total': round(team_a_score, 1),
            'Team_B_Total': round(team_b_score, 1),
            'Difference': round(abs(team_a_score - team_b_score), 1)
        }
