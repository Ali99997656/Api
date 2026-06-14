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

    def train_model(self, historical_raw_data):
        print("جاري تدريب الموديل وحساب المتوسطات العامة للنظام...")
        df = historical_raw_data.copy().fillna(0)
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
        print("تم التدريب بنجاح!")

    def generate_match(self, match_raw_data, team_size=5):
        if not self.is_trained:
            raise Exception("يجب تدريب الموديل أولاً!")
            
        df = match_raw_data.copy().fillna(0)
        total_players_received = len(df)
        starters_needed = team_size * 2
        
        if total_players_received < starters_needed:
            return f"خطأ: المباراة {team_size}ضد{team_size} تحتاج على الأقل {starters_needed} لاعبين، لكنك أدخلت {total_players_received} فقط."

        df['raw_rating'] = self._calculate_base_rating(df)
        if self.rating_max > self.rating_min:
            df['calculated_overall_rating'] = ((df['raw_rating'] - self.rating_min) / (self.rating_max - self.rating_min)) * 100
        else:
            df['calculated_overall_rating'] = 50.0
        df['calculated_overall_rating'] = df['calculated_overall_rating'].clip(0, 100)
        
        # معالجة اللاعبين الجدد
        is_new = (df['goals'] == 0) & (df['assists'] == 0) & (df['shots'] == 0) & (df['saves'] == 0)
        if is_new.any():
            for col in ['goals', 'assists', 'shots', 'saves', 'calculated_overall_rating']:
                df.loc[is_new, col] = self.mean_stats[col]
        
        df = df.round(1)
        features = df[['goals', 'assists', 'saves', 'shots', 'calculated_overall_rating']]
        scaled_features = self.scaler.transform(features)
        df['ai_cluster'] = self.ai_model.predict(scaled_features) 
        
        # ==========================================
        # 1. منطق حراس المرمى (أولوية للحارس الحقيقي، ثم الأضعف)
        # ==========================================
        gks = df[df['is_gk'] == 1].sort_values(by='calculated_overall_rating', ascending=False)
        non_gks = df[df['is_gk'] == 0].sort_values(by='calculated_overall_rating', ascending=True) # تصاعدي لنجد الأضعف
        
        if len(gks) >= 2:
            gk_a, gk_b = gks.iloc[0], gks.iloc[1]
        elif len(gks) == 1:
            gk_a, gk_b = gks.iloc[0], non_gks.iloc[0]
        else:
            gk_a, gk_b = non_gks.iloc[0], non_gks.iloc[1]
            
        remaining_pool = df[~df['player_id'].isin([gk_a['player_id'], gk_b['player_id']])].copy()
        
        # ==========================================
        # 2. فرز الأساسيين والاحتياط (الأقوى يلعب أساسي)
        # ==========================================
        field_starters_needed = (team_size - 1) * 2
        remaining_pool = remaining_pool.sort_values(by='calculated_overall_rating', ascending=False) # من الأقوى للأضعف
        
        starting_field_players = remaining_pool.head(field_starters_needed)
        substitutes = remaining_pool.iloc[field_starters_needed:]
        
        # ==========================================
        # 3. توزيع التشكيلة بالعدل
        # ==========================================
        team_a, team_b = [], []
        team_a_score, team_b_score = gk_a['calculated_overall_rating'], gk_b['calculated_overall_rating']
        
        # إضافة الحراس للأساسيين
        team_a.append({'player_id': gk_a['player_id'], 'Role': 'GK', 'Status': 'Starter', 'Rating': gk_a['calculated_overall_rating']})
        team_b.append({'player_id': gk_b['player_id'], 'Role': 'GK', 'Status': 'Starter', 'Rating': gk_b['calculated_overall_rating']})
        
        # توزيع اللاعبين الأساسيين
        for _, player in starting_field_players.iterrows():
            if team_a_score <= team_b_score:
                team_a.append({'player_id': player['player_id'], 'Role': 'Player', 'Status': 'Starter', 'Rating': player['calculated_overall_rating']})
                team_a_score += player['calculated_overall_rating']
            else:
                team_b.append({'player_id': player['player_id'], 'Role': 'Player', 'Status': 'Starter', 'Rating': player['calculated_overall_rating']})
                team_b_score += player['calculated_overall_rating']
                
        # توزيع الاحتياط (إن وجدوا)
        for _, sub in substitutes.iterrows():
            if team_a_score <= team_b_score:
                team_a.append({'player_id': sub['player_id'], 'Role': 'Player', 'Status': 'Bench', 'Rating': sub['calculated_overall_rating']})
                team_a_score += sub['calculated_overall_rating']
            else:
                team_b.append({'player_id': sub['player_id'], 'Role': 'Player', 'Status': 'Bench', 'Rating': sub['calculated_overall_rating']})
                team_b_score += sub['calculated_overall_rating']
                
        return {
            'Team_A': pd.DataFrame(team_a),
            'Team_B': pd.DataFrame(team_b),
            'Team_A_Total': round(team_a_score, 1),
            'Team_B_Total': round(team_b_score, 1),
            'Difference': round(abs(team_a_score - team_b_score), 1)
        }