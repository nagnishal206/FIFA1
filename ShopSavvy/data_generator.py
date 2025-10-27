import pandas as pd
import numpy as np
from datetime import datetime

def generate_historical_fifa_data(num_records: int = 200) -> pd.DataFrame:
    """
    Generate realistic historical FIFA World Cup data for model training.
    
    This creates a synthetic dataset with realistic features based on
    historical World Cup patterns and team performance statistics.
    
    Features:
    - Team demographics (name, confederation)
    - Performance metrics (goals, wins, ranking)
    - Player statistics (average age, experience)
    - Tournament outcomes (finalist status)
    """
    
    np.random.seed(42)
    
    teams = [
        'Brazil', 'Germany', 'Argentina', 'France', 'Italy', 'Spain', 'England',
        'Netherlands', 'Uruguay', 'Portugal', 'Belgium', 'Croatia', 'Mexico',
        'Colombia', 'Chile', 'USA', 'Japan', 'South Korea', 'Senegal', 'Ghana',
        'Nigeria', 'Morocco', 'Egypt', 'Cameroon', 'Australia', 'Denmark',
        'Sweden', 'Switzerland', 'Poland', 'Austria', 'Serbia', 'Ecuador'
    ]
    
    confederations = {
        'Brazil': 'CONMEBOL', 'Argentina': 'CONMEBOL', 'Uruguay': 'CONMEBOL',
        'Colombia': 'CONMEBOL', 'Chile': 'CONMEBOL', 'Ecuador': 'CONMEBOL',
        'Germany': 'UEFA', 'France': 'UEFA', 'Italy': 'UEFA', 'Spain': 'UEFA',
        'England': 'UEFA', 'Netherlands': 'UEFA', 'Portugal': 'UEFA',
        'Belgium': 'UEFA', 'Croatia': 'UEFA', 'Denmark': 'UEFA', 'Sweden': 'UEFA',
        'Switzerland': 'UEFA', 'Poland': 'UEFA', 'Austria': 'UEFA', 'Serbia': 'UEFA',
        'Mexico': 'CONCACAF', 'USA': 'CONCACAF',
        'Japan': 'AFC', 'South Korea': 'AFC', 'Australia': 'AFC',
        'Senegal': 'CAF', 'Ghana': 'CAF', 'Nigeria': 'CAF', 'Morocco': 'CAF',
        'Egypt': 'CAF', 'Cameroon': 'CAF'
    }
    
    years = [2006, 2010, 2014, 2018, 2022]
    
    data = []
    
    for year in years:
        num_teams_this_year = 32
        selected_teams = np.random.choice(teams, size=min(num_teams_this_year, len(teams)), replace=False)
        
        for i, team in enumerate(selected_teams):
            is_strong_team = team in ['Brazil', 'Germany', 'Argentina', 'France', 'Italy', 'Spain', 'England']
            
            fifa_ranking = np.random.randint(1, 50) if is_strong_team else np.random.randint(10, 100)
            
            avg_age = np.random.uniform(24.5, 29.5)
            
            matches_played = np.random.randint(4, 8)
            wins = np.random.randint(0, matches_played + 1)
            draws = np.random.randint(0, matches_played - wins + 1)
            losses = matches_played - wins - draws
            
            goals_scored = np.random.randint(2, 20)
            goals_conceded = np.random.randint(1, 15)
            goal_difference = goals_scored - goals_conceded
            
            win_rate = wins / matches_played if matches_played > 0 else 0
            
            avg_player_caps = np.random.randint(20, 80)
            
            previous_wc_appearances = np.random.randint(0, 15) if is_strong_team else np.random.randint(0, 8)
            
            qualification_status = np.random.choice(['Direct', 'Playoff', 'Host'])
            
            finalist_probability = (
                0.3 if is_strong_team else 0.05
            ) * (1 + win_rate) * (1 if goal_difference > 0 else 0.3)
            
            is_finalist = 1 if np.random.random() < finalist_probability else 0
            
            data.append({
                'Year': year,
                'Team': team,
                'Confederation': confederations.get(team, 'UEFA'),
                'FIFA_Ranking': fifa_ranking,
                'Avg_Age': round(avg_age, 1),
                'Matches_Played': matches_played,
                'Wins': wins,
                'Draws': draws,
                'Losses': losses,
                'Goals_Scored': goals_scored,
                'Goals_Conceded': goals_conceded,
                'Goal_Difference': goal_difference,
                'Win_Rate': round(win_rate, 3),
                'Avg_Player_Caps': avg_player_caps,
                'Previous_WC_Appearances': previous_wc_appearances,
                'Qualification_Status': qualification_status,
                'Finalist': is_finalist
            })
    
    df = pd.DataFrame(data)
    
    mask = np.random.random(len(df)) < 0.05
    df.loc[mask, 'Avg_Age'] = np.nan
    
    mask = np.random.random(len(df)) < 0.03
    df.loc[mask, 'Avg_Player_Caps'] = np.nan
    
    return df


def generate_2026_prediction_data() -> pd.DataFrame:
    """
    Generate sample data for 2026 World Cup teams for prediction.
    """
    np.random.seed(2026)
    
    teams_2026 = [
        'Brazil', 'Argentina', 'France', 'England', 'Spain', 'Germany',
        'Netherlands', 'Portugal', 'Belgium', 'Croatia', 'Italy', 'Uruguay',
        'Mexico', 'USA', 'Colombia', 'Japan', 'South Korea', 'Senegal',
        'Morocco', 'Switzerland', 'Denmark', 'Poland', 'Australia', 'Ecuador'
    ]
    
    confederations = {
        'Brazil': 'CONMEBOL', 'Argentina': 'CONMEBOL', 'Uruguay': 'CONMEBOL',
        'Colombia': 'CONMEBOL', 'Ecuador': 'CONMEBOL',
        'Germany': 'UEFA', 'France': 'UEFA', 'Italy': 'UEFA', 'Spain': 'UEFA',
        'England': 'UEFA', 'Netherlands': 'UEFA', 'Portugal': 'UEFA',
        'Belgium': 'UEFA', 'Croatia': 'UEFA', 'Denmark': 'UEFA',
        'Switzerland': 'UEFA', 'Poland': 'UEFA',
        'Mexico': 'CONCACAF', 'USA': 'CONCACAF',
        'Japan': 'AFC', 'South Korea': 'AFC', 'Australia': 'AFC',
        'Senegal': 'CAF', 'Morocco': 'CAF'
    }
    
    data = []
    
    for team in teams_2026:
        is_strong_team = team in ['Brazil', 'Argentina', 'France', 'England', 'Spain', 'Germany']
        
        fifa_ranking = np.random.randint(1, 30) if is_strong_team else np.random.randint(15, 60)
        avg_age = np.random.uniform(25.0, 28.5)
        matches_played = np.random.randint(8, 15)
        wins = np.random.randint(4, matches_played)
        draws = np.random.randint(0, matches_played - wins)
        losses = matches_played - wins - draws
        
        goals_scored = np.random.randint(10, 35)
        goals_conceded = np.random.randint(5, 25)
        goal_difference = goals_scored - goals_conceded
        
        win_rate = wins / matches_played
        avg_player_caps = np.random.randint(25, 75)
        previous_wc_appearances = np.random.randint(3, 16) if is_strong_team else np.random.randint(1, 10)
        qualification_status = np.random.choice(['Direct', 'Playoff', 'Host'])
        
        data.append({
            'Team': team,
            'Confederation': confederations.get(team, 'UEFA'),
            'FIFA_Ranking': fifa_ranking,
            'Avg_Age': round(avg_age, 1),
            'Matches_Played': matches_played,
            'Wins': wins,
            'Draws': draws,
            'Losses': losses,
            'Goals_Scored': goals_scored,
            'Goals_Conceded': goals_conceded,
            'Goal_Difference': goal_difference,
            'Win_Rate': round(win_rate, 3),
            'Avg_Player_Caps': avg_player_caps,
            'Previous_WC_Appearances': previous_wc_appearances,
            'Qualification_Status': qualification_status
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("Generating historical FIFA World Cup dataset...")
    df_historical = generate_historical_fifa_data()
    df_historical.to_csv('historical_fifa_data.csv', index=False)
    print(f"Generated {len(df_historical)} records")
    print(f"\nDataset shape: {df_historical.shape}")
    print(f"\nColumns: {list(df_historical.columns)}")
    print(f"\nFinalists distribution:\n{df_historical['Finalist'].value_counts()}")
    
    print("\n" + "="*50)
    print("Generating 2026 prediction dataset...")
    df_2026 = generate_2026_prediction_data()
    df_2026.to_csv('teams_2026.csv', index=False)
    print(f"Generated {len(df_2026)} teams for 2026 prediction")
