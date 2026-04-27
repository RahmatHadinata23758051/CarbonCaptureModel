import pandas as pd
import numpy as np
import os
import argparse

def load_fluxnet(filepath):
    print(f"Loading FLUXNET data from {filepath}...")
    df = pd.read_csv(filepath)
    df.replace(-9999, np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

def map_features_to_iot_schema(df):
    print("Mapping features to IoT Schema...")
    mapping = {
        'TA_F': 'air_temperature_c',
        'RH': 'air_humidity_percent',
        'PA_F': 'air_pressure_hpa',
        'P_F': 'rainfall_mm',
        'WS_F': 'wind_speed_m_s',
        'SW_IN_F': 'solar_radiation_w_m2',
        'TS_F_MDS_1': 'soil_temperature_c',
        'SWC_F_MDS_1': 'soil_moisture_percent',
        'CO2_PPM': 'air_co2_ppm',
        'CCI_INDEX': 'carbon_capture_cci'
    }
    
    keep_cols = ['TIMESTAMP'] + [col for col in mapping.keys() if col in df.columns]
    df = df[keep_cols].copy()
    df.rename(columns=mapping, inplace=True)
    
    df['latitude'] = -6.2
    df['longitude'] = 106.8
    if 'solar_radiation_w_m2' in df.columns:
        df['light_lux'] = df['solar_radiation_w_m2'] * 120
    
    return df

def feature_engineering(df):
    print("Performing feature engineering...")
    df['timestamp'] = pd.to_datetime(df['TIMESTAMP'].astype(str), format='%Y%m%d%H%M')
    
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    if 'air_temperature_c' in df.columns:
        df['air_temp_lag_1h'] = df['air_temperature_c'].shift(1)
        df['air_temp_rolling_6h'] = df['air_temperature_c'].rolling(6, min_periods=1).mean()
    
    if 'solar_radiation_w_m2' in df.columns:
        df['solar_rad_lag_1h'] = df['solar_radiation_w_m2'].shift(1)
        
    if 'air_co2_ppm' in df.columns:
        df['co2_lag_1h'] = df['air_co2_ppm'].shift(1)
    
    if 'air_temperature_c' in df.columns and 'solar_radiation_w_m2' in df.columns:
        df['temp_x_radiation'] = df['air_temperature_c'] * df['solar_radiation_w_m2']
        
    if 'soil_moisture_percent' in df.columns and 'air_temperature_c' in df.columns:
        df['moisture_x_temp'] = df['soil_moisture_percent'] * df['air_temperature_c']
        
    df.fillna(method='bfill', inplace=True)
    return df

def integrate_wosis_soil(df, wosis_filepath):
    print(f"Integrating WoSIS soil baseline from {wosis_filepath}...")
    wosis = pd.read_csv(wosis_filepath)
    
    soil_stats = {}
    for msr in ['pH', 'Nitrogen', 'Phosphorus', 'EC']:
        sub = wosis[wosis['measurement'] == msr]
        if not sub.empty:
            mean_val = sub['value_avg'].mean()
            std_val = sub['value_avg'].std()
            if pd.isna(std_val) or std_val == 0:
                std_val = mean_val * 0.05
            soil_stats[msr] = {'mean': mean_val, 'std': std_val}
        else:
            fallbacks = {'pH': (6.5, 0.3), 'Nitrogen': (1200, 100), 'Phosphorus': (45, 5), 'EC': (0.5, 0.05)}
            soil_stats[msr] = {'mean': fallbacks[msr][0], 'std': fallbacks[msr][1]}
            
    n_rows = len(df)
    
    def generate_soil_series(mean, std):
        noise = np.random.normal(0, std * 0.1, n_rows)
        series = np.cumsum(noise)
        series = series - np.mean(series)
        series = series * (std / (np.std(series) + 1e-6)) + mean
        return series
    
    df['soil_ph'] = generate_soil_series(soil_stats['pH']['mean'], soil_stats['pH']['std'])
    df['soil_n_mg_kg'] = generate_soil_series(soil_stats['Nitrogen']['mean'], soil_stats['Nitrogen']['std'])
    df['soil_p_mg_kg'] = generate_soil_series(soil_stats['Phosphorus']['mean'], soil_stats['Phosphorus']['std'])
    df['soil_ec_ms_cm'] = np.clip(generate_soil_series(soil_stats['EC']['mean'], soil_stats['EC']['std']), 0, None)
    
    return df

def run_pipeline(fluxnet_in, wosis_in, out_filepath):
    df = load_fluxnet(fluxnet_in)
    df = map_features_to_iot_schema(df)
    df = feature_engineering(df)
    df = integrate_wosis_soil(df, wosis_in)
    
    if 'soil_k_mg_kg' not in df.columns:
        df['soil_k_mg_kg'] = np.random.normal(250, 20, len(df))
        
    # Re-adjust CCI target slightly based on generated soil NPK (simulate Boss's logic that NPK limits/boosts CCI)
    n_factor = np.clip(df['soil_n_mg_kg'] / 1200.0, 0.5, 1.5)
    p_factor = np.clip(df['soil_p_mg_kg'] / 45.0, 0.5, 1.5)
    df['carbon_capture_cci'] = np.clip(df['carbon_capture_cci'] * n_factor * p_factor, 0, 100)
    
    target = 'carbon_capture_cci'
    features = [
        'timestamp', 'latitude', 'longitude', 
        'air_temperature_c', 'air_humidity_percent', 'air_pressure_hpa',
        'air_co2_ppm', # New field
        'rainfall_mm', 'wind_speed_m_s', 'solar_radiation_w_m2', 'light_lux',
        'soil_moisture_percent', 'soil_temperature_c', 'soil_ec_ms_cm',
        'soil_ph', 'soil_n_mg_kg', 'soil_p_mg_kg', 'soil_k_mg_kg',
        'hour_of_day', 'day_of_year', 'month',
        'air_temp_lag_1h', 'air_temp_rolling_6h',
        'solar_rad_lag_1h', 'co2_lag_1h', 'temp_x_radiation', 'moisture_x_temp'
    ]
    
    final_cols = [c for c in features if c in df.columns] + [target]
    df = df[final_cols]
    
    df.to_csv(out_filepath, index=False)
    print(f"Pipeline complete. Saved training data to {out_filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fluxnet', default='data/FLUXNET_sample.csv')
    parser.add_argument('--wosis', default='data/wosis_0_30cm.csv')
    parser.add_argument('--out', default='data/training_data.csv')
    args = parser.parse_args()
    
    run_pipeline(args.fluxnet, args.wosis, args.out)
