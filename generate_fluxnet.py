import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_realistic_fluxnet(filename, num_days=365):
    print(f"Generating realistic FLUXNET sample for CCI: {filename}")
    
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(num_days * 24)]
    df = pd.DataFrame({'TIMESTAMP': timestamps})
    
    hours = df['TIMESTAMP'].dt.hour
    day_of_year = df['TIMESTAMP'].dt.dayofyear
    
    # 1. Temperature
    yearly_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
    daily_temp = 5 * np.sin(2 * np.pi * (hours - 8) / 24)
    df['TA_F'] = yearly_temp + daily_temp + np.random.normal(0, 2, len(df))
    
    # 2. Humidity
    df['RH'] = np.clip(80 - (df['TA_F'] - 10) * 1.5 + np.random.normal(0, 5, len(df)), 20, 100)
    df['PA_F'] = 1013 + np.random.normal(0, 5, len(df))
    df['WS_F'] = np.random.lognormal(mean=0.5, sigma=0.5, size=len(df))
    
    # 3. Precip
    rain_chance = np.random.uniform(0, 1, len(df))
    df['P_F'] = np.where(rain_chance > 0.95, np.random.exponential(scale=2.0, size=len(df)), 0)
    
    # 4. Solar Radiation
    solar_peak = 800 + 200 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
    daily_solar = np.where((hours > 6) & (hours < 18), 
                           np.sin(np.pi * (hours - 6) / 12) * solar_peak, 0)
    df['SW_IN_F'] = np.clip(daily_solar + np.where(daily_solar > 0, np.random.normal(0, 50, len(df)), 0), 0, None)
    
    # 5. Soil Temp & Moisture
    df['TS_F_MDS_1'] = df['TA_F'].rolling(window=12, min_periods=1).mean() + np.random.normal(0, 0.5, len(df))
    swc = np.zeros(len(df))
    current_swc = 20.0
    for i in range(len(df)):
        if df['P_F'].iloc[i] > 0: current_swc += df['P_F'].iloc[i] * 2
        current_swc = np.clip(current_swc * 0.995, 10, 45)
        swc[i] = current_swc
    df['SWC_F_MDS_1'] = swc
    
    # 6. CO2 PPM - peaks at night (respiration), dips during day (photosynthesis)
    base_co2 = 420
    diurnal_co2 = -15 * np.sin(2 * np.pi * (hours - 8) / 24) # dips around 2 PM
    seasonal_co2 = 5 * np.sin(2 * np.pi * (day_of_year - 150) / 365) # minor seasonal variation
    df['CO2_PPM'] = base_co2 + diurnal_co2 + seasonal_co2 + np.random.normal(0, 2, len(df))
    
    # 7. Carbon Capture Index (CCI) Target (0-100)
    # Normalized drivers:
    # Light (0-1000 W/m2 -> 0-1)
    light_norm = np.clip(df['SW_IN_F'] / 800.0, 0, 1)
    # Temp (optimal at 25C -> 0-1 bell curve)
    temp_norm = np.exp(-0.5 * ((df['TA_F'] - 25) / 7)**2)
    # Moisture (optimal at >25% -> 0-1)
    moist_norm = np.clip((df['SWC_F_MDS_1'] - 10) / 20.0, 0, 1)
    # CO2 (more CO2 = slight boost up to a point)
    co2_norm = np.clip((df['CO2_PPM'] - 380) / 100.0, 0.5, 1.2)
    
    # Base index
    cci_base = 100 * (light_norm * 0.5 + light_norm * temp_norm * 0.3 + light_norm * moist_norm * 0.2)
    # Multiply by CO2 factor
    cci = cci_base * co2_norm + np.random.normal(0, 2, len(df))
    
    df['CCI_INDEX'] = np.clip(cci, 0, 100)
    
    df['TIMESTAMP'] = df['TIMESTAMP'].dt.strftime('%Y%m%d%H%M')
    df.to_csv(filename, index=False)
    print(f"Generated {len(df)} rows.")

if __name__ == "__main__":
    generate_realistic_fluxnet("data/FLUXNET_sample.csv", num_days=365)
