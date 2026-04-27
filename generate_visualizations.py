import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde

# ---------------------------------------------------------
# Set High-Quality "Journal-Level" Style
# ---------------------------------------------------------
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.linewidth': 1.5,
    'figure.dpi': 300,  # High resolution for journals
    'savefig.dpi': 300,
    'axes.edgecolor': '#333333',
    'text.color': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
})
sns.set_theme(style="ticks", context="paper", font_scale=1.2)

def make_dirs():
    os.makedirs('visualisasi', exist_ok=True)

def plot_correlation_matrix(df):
    print("Generating Correlation Matrix...")
    # Select important numeric features for the correlation matrix (exclude derived time/lags to reduce clutter)
    cols_of_interest = [
        'air_temperature_c', 'air_humidity_percent', 'solar_radiation_w_m2',
        'air_co2_ppm', # added CO2
        'soil_moisture_percent', 'soil_temperature_c', 'soil_ph', 
        'soil_n_mg_kg', 'soil_ec_ms_cm', 'carbon_capture_cci' # changed target
    ]
    
    # Filter columns that exist
    cols = [c for c in cols_of_interest if c in df.columns]
    corr = df[cols].corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Custom diverging colormap standard in publications
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .7, "label": "Pearson Correlation"},
                annot=True, fmt=".2f", annot_kws={"size": 10}, ax=ax)
                
    ax.set_title('Eco-Physiological Sensor Variables Correlation Matrix', pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualisasi/1_correlation_matrix.png', bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names):
    print("Generating Feature Importance...")
    importances = model.feature_importances_
    
    # Create DataFrame
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(15) # Top 15
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Premium barplot with palette
    sns.barplot(x='Importance', y='Feature', data=df_imp, 
                palette='viridis', edgecolor='black', ax=ax)
                
    ax.set_title('XGBoost Model: Top 15 Drivers of CCI (Carbon Capture Index)', pad=20, fontweight='bold')
    ax.set_xlabel('Relative Feature Importance (Gain)')
    ax.set_ylabel('Sensor Feature')
    sns.despine(left=True, bottom=True)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('visualisasi/2_feature_importance.png', bbox_inches='tight')
    plt.close()

def plot_model_evaluation(model, df):
    print("Generating Model Evaluation Plots...")
    target = 'carbon_capture_cci'
    X = df.drop(columns=[target, 'timestamp'], errors='ignore')
    y = df[target]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    
    fig = plt.figure(figsize=(16, 6))
    
    # --- Plot 1: Predicted vs Actual with Density ---
    ax1 = plt.subplot(1, 2, 1)
    
    # Calculate density for scatter plot coloring (looks highly professional)
    xy = np.vstack([y_test, y_pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x_s, y_s, z_s = y_test.iloc[idx], y_pred[idx], z[idx]
    
    scatter = ax1.scatter(x_s, y_s, c=z_s, s=30, cmap='Spectral_r', edgecolor='none', alpha=0.8)
    
    # 1:1 Line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, zorder=3, label='1:1 Perfect Fit')
    
    ax1.set_xlabel('Actual CCI Observed (0-100)')
    ax1.set_ylabel('Model Predicted CCI (0-100)')
    ax1.set_title('Model Performance: Predicted vs Actual', fontweight='bold', pad=15)
    ax1.legend(loc='upper left')
    plt.colorbar(scatter, ax=ax1, label='Point Density')
    
    # --- Plot 2: Residual Distribution ---
    ax2 = plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    
    sns.histplot(residuals, kde=True, color='#2ca02c', edgecolor='black', bins=40, ax=ax2, alpha=0.6)
    ax2.axvline(0, color='black', linestyle='--', lw=2)
    
    ax2.set_xlabel('Prediction Error / Residual (Index Points)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residual Error Distribution', fontweight='bold', pad=15)
    
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig('visualisasi/3_model_evaluation.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    make_dirs()
    
    # Load Data
    data_path = 'data/training_data.csv'
    model_path = 'models/best_xgboost_model.pkl'
    
    if os.path.exists(data_path) and os.path.exists(model_path):
        df = pd.read_csv(data_path)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        with open('models/model_metadata.pkl', 'rb') as f:
            meta = pickle.load(f)
            features = meta['features']
            
        plot_correlation_matrix(df)
        plot_feature_importance(model, features)
        plot_model_evaluation(model, df)
        print("All visualizations generated successfully in the '/visualisasi' folder.")
    else:
        print("Data or Model missing. Please ensure the pipeline and training script have been run.")
