import pandas as pd
import numpy as np
import requests
import io
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType

# --- CONFIGURATION ---
LEAGUE_CONFIG = {
    'E0': ["https://www.football-data.co.uk/mmz4281/2425/E0.csv", "https://www.football-data.co.uk/mmz4281/2526/E0.csv"],
    'E1': ["https://www.football-data.co.uk/mmz4281/2425/E1.csv", "https://www.football-data.co.uk/mmz4281/2526/E1.csv"],
    'E2': ["https://www.football-data.co.uk/mmz4281/2425/E2.csv", "https://www.football-data.co.uk/mmz4281/2526/E2.csv"],
    'E3': ["https://www.football-data.co.uk/mmz4281/2425/E3.csv", "https://www.football-data.co.uk/mmz4281/2526/E3.csv"],
    'SP1': ["https://www.football-data.co.uk/mmz4281/2425/SP1.csv", "https://www.football-data.co.uk/mmz4281/2526/SP1.csv"],
    'D1': ["https://www.football-data.co.uk/mmz4281/2425/D1.csv", "https://www.football-data.co.uk/mmz4281/2526/D1.csv"],
    'D2': ["https://www.football-data.co.uk/mmz4281/2425/D2.csv", "https://www.football-data.co.uk/mmz4281/2526/D2.csv"],
    'I1': ["https://www.football-data.co.uk/mmz4281/2425/I1.csv", "https://www.football-data.co.uk/mmz4281/2526/I1.csv"],
    'F1': ["https://www.football-data.co.uk/mmz4281/2425/F1.csv", "https://www.football-data.co.uk/mmz4281/2526/F1.csv"],
    'SC0': ["https://www.football-data.co.uk/mmz4281/2425/SC0.csv", "https://www.football-data.co.uk/mmz4281/2526/SC0.csv"],
}

def fetch_and_combine_data(urls):
    combined_data = pd.DataFrame()
    for url in urls:
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                csv_file = io.StringIO(response.content.decode('ISO-8859-1'))
                if csv_file.getvalue().strip():
                    data = pd.read_csv(csv_file)
                    required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
                    if all(col in data.columns for col in required_cols):
                        data = data[required_cols]
                        data.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals', 'FTR': 'Result'}, inplace=True)
                        combined_data = pd.concat([combined_data, data], ignore_index=True)
                    else:
                        print(f"Warning: Skipping {url} due to missing required columns.")
                else:
                    print(f"Warning: Skipping empty file at {url}.")
            else:
                print(f"Warning: Could not fetch {url}. Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Network error processing {url}. Error: {e}")
        except Exception as e:
            print(f"Warning: General error processing {url}. Error: {e}")
            
    combined_data.dropna(inplace=True)
    return combined_data

def calculate_team_strengths(data):
    if data.empty:
        return pd.DataFrame()
    avg_home_goals = data['HomeGoals'].mean()
    avg_away_goals = data['AwayGoals'].mean()
    all_teams = pd.unique(data[['HomeTeam', 'AwayTeam']].values.ravel('K'))
    team_strengths = pd.DataFrame(index=all_teams, columns=['AttackStrength', 'DefenceStrength'])
    for team in all_teams:
        home_games = data[data['HomeTeam'] == team]
        away_games = data[data['AwayTeam'] == team]
        avg_home_scored = home_games['HomeGoals'].mean() if not home_games.empty else 0
        avg_away_scored = away_games['AwayGoals'].mean() if not away_games.empty else 0
        avg_home_conceded = home_games['AwayGoals'].mean() if not home_games.empty else 0
        avg_away_conceded = away_games['HomeGoals'].mean() if not away_games.empty else 0
        
        attack_strength = ((avg_home_scored / avg_home_goals) + (avg_away_scored / avg_away_goals)) / 2 if avg_home_goals and avg_away_goals else 1.0
        defence_strength = ((avg_home_conceded / avg_away_goals) + (avg_away_conceded / avg_home_goals)) / 2 if avg_away_goals and avg_home_goals else 1.0
        
        team_strengths.loc[team] = [attack_strength, defence_strength]
    return team_strengths.fillna(1.0)

def train_and_export_model(league_code, urls):
    print(f"\n--- Processing League: {league_code} ---")
    data = fetch_and_combine_data(urls)
    
    if data.empty:
        print(f"No usable data for {league_code}, skipping model training.")
        return

    strengths = calculate_team_strengths(data)
    data = data.merge(strengths, left_on='HomeTeam', right_index=True).rename(columns={'AttackStrength': 'HomeAttack', 'DefenceStrength': 'HomeDefence'})
    data = data.merge(strengths, left_on='AwayTeam', right_index=True).rename(columns={'AttackStrength': 'AwayAttack', 'DefenceStrength': 'AwayDefence'})
    
    X = data[['HomeAttack', 'HomeDefence', 'AwayAttack', 'AwayDefence']].astype(np.float32)
    y = data['Result']
    
    if len(pd.unique(y)) < 3:
        print(f"Not enough outcome variety for {league_code} to train a model (found {len(pd.unique(y))} outcomes). Skipping.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=5)
    rf_model.fit(X_train, y_train)
    
    accuracy = rf_model.score(X_test, y_test)
    print(f"Model for {league_code} trained with accuracy: {accuracy:.2%}")

    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"model_{league_code}.onnx")

    initial_types = [('float_input', FloatTensorType([None, 4]))]
    onnx_model = skl2onnx.convert_sklearn(rf_model, initial_types=initial_types, target_opset=12)
    
    with open(output_filename, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"✅ Model for {league_code} exported to '{output_filename}'")


if __name__ == "__main__":
    for code, url_list in LEAGUE_CONFIG.items():
        train_and_export_model(code, url_list)
    print("\nAll models have been trained and exported.")

