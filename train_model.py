import pandas as pd
import numpy as np
import pickle
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    # Drop timestamp for modeling
    df = df.drop(columns=['timestamp'])
    return df

def train_and_evaluate(df, target_col='carbon_capture_cci'):
    print("Preparing train/test split...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    }
    
    results = {}
    best_model = None
    best_r2 = -float('inf')
    best_name = ""
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        print(f"{name} Results: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name
            
    # Feature Importance for best model
    print(f"Extracting Feature Importance from {best_name}...")
    if best_name == 'RandomForest':
        importances = best_model.feature_importances_
    else:
        importances = best_model.feature_importances_
        
    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
    
    # Save best model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/best_{best_name.lower()}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save scaler/columns info for inference
    meta_path = "models/model_metadata.pkl"
    with open(meta_path, 'wb') as f:
        pickle.dump({'features': list(X.columns)}, f)
        
    print(f"Saved best model to {model_path}")
    
    # Save evaluation metrics to artifact text file
    with open("models/metrics.txt", "w") as f:
        f.write("=== Model Evaluation ===\n")
        for k, v in results.items():
            f.write(f"\n{k}:\n")
            f.write(f"  RMSE: {v['RMSE']:.4f}\n")
            f.write(f"  MAE:  {v['MAE']:.4f}\n")
            f.write(f"  R2:   {v['R2']:.4f}\n")
            
        f.write("\n=== Feature Importance (Best Model) ===\n")
        for idx, row in feat_imp.iterrows():
            f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/training_data.csv')
    args = parser.parse_args()
    
    df = load_data(args.data)
    train_and_evaluate(df)
