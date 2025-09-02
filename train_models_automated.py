    """Trains a model for a specific league and exports it."""
    print(f"\n--- Processing League: {league_code} ---")
    data = fetch_and_combine_data(urls)
    if data.empty:
        print(f"No data for {league_code}, skipping.")
        return

    strengths = calculate_team_strengths(data)
    data = data.merge(strengths, left_on='HomeTeam', right_index=True).rename(columns={'AttackStrength': 'HomeAttack', 'DefenceStrength': 'HomeDefence'})
    data = data.merge(strengths, left_on='AwayTeam', right_index=True).rename(columns={'AttackStrength': 'AwayAttack', 'DefenceStrength': 'AwayDefence'})
    
    X = data[['HomeAttack', 'HomeDefence', 'AwayAttack', 'AwayDefence']].astype(np.float32)
    y = data['Result']
    
    if len(pd.unique(y)) < 3:
        print(f"Not enough outcome variety for {league_code} to train a model. Skipping.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=5)
    rf_model.fit(X_train, y_train)
    
    accuracy = rf_model.score(X_test, y_test)
    print(f"Model for {league_code} trained with accuracy: {accuracy:.2%}")

    # NEW: Create a directory for the models if it doesn't exist
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"model_{league_code}.onnx")

    initial_types = [('float_input', FloatTensorType([None, 4]))]
    onnx_model = skl2onnx.convert_sklearn(rf_model, initial_types=initial_types, target_opset=12)
    
    with open(output_filename, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"✅ Model for {league_code} exported to '{output_filename}'")


if __name__ == "__main__":
    # NEW: Loop through all configured leagues and train a model for each
    for code, url_list in LEAGUE_CONFIG.items():
        train_and_export_model(code, url_list)
    print("\nAll models have been trained and exported.")

