import json
import pickle
import pandas as pd
import numpy as np
import datetime
import argparse

class CarbonCapturePredictor:
    def __init__(self, model_path="models/best_xgboost_model.pkl", meta_path="models/model_metadata.pkl"):
        # Load model and features
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(meta_path, 'rb') as f:
            self.meta = pickle.load(f)
            self.features = self.meta['features']
            
        # Store state for rolling/lag features in a real scenario (mocked here for stateless API)
        self.history = []

    def _parse_payload(self, payload):
        """
        Parses Agrisense IoT JSON payload into a flat dictionary matching pipeline schema.
        """
        data = {
            'timestamp': payload.get('timestamp', datetime.datetime.now().isoformat()),
            'latitude': payload['location']['latitude'],
            'longitude': payload['location']['longitude'],
            
            'air_temperature_c': payload['environment']['air_temperature_c'] if payload['environment']['air_temperature_c'] is not None else 25.0,
            'air_humidity_percent': payload['environment']['air_humidity_percent'] if payload['environment']['air_humidity_percent'] is not None else 70.0,
            'air_pressure_hpa': payload['environment']['air_pressure_hpa'] if payload['environment']['air_pressure_hpa'] is not None else 1013.0,
            'air_co2_ppm': payload.get('carbon_data', {}).get('co2_ppm', 420.0), # Updated location
            'light_lux': payload['environment']['light_lux'] if payload['environment']['light_lux'] is not None else 0.0,
            # Derive solar radiation approximately from lux
            'solar_radiation_w_m2': payload['environment']['light_lux'] / 120.0,
            
            # Assume 0 rainfall and wind if not provided by sensor
            'rainfall_mm': 0.0,
            'wind_speed_m_s': 0.5,
            
            'soil_moisture_percent': payload.get('soil_7in1', {}).get('soil_moisture_percent', 35.0),
            'soil_temperature_c': payload.get('soil_7in1', {}).get('soil_temperature_c', 25.0),
            'soil_ec_ms_cm': payload.get('soil_7in1', {}).get('soil_ec_ms_cm', 0.5),
            'soil_ph': payload.get('soil_7in1', {}).get('soil_ph', 6.5),
            'soil_n_mg_kg': payload.get('soil_7in1', {}).get('soil_n_mg_kg', 1200.0),
            'soil_p_mg_kg': payload.get('soil_7in1', {}).get('soil_p_mg_kg', 45.0),
            'soil_k_mg_kg': payload.get('soil_7in1', {}).get('soil_k_mg_kg', 250.0),
        }
        return data

    def _feature_engineering(self, data):
        """
        Adds engineered features required by the model.
        """
        dt = pd.to_datetime(data['timestamp'])
        
        data['hour_of_day'] = dt.hour
        data['day_of_year'] = dt.dayofyear
        data['month'] = dt.month
        
        # In a stateless API, we approximate lag/rolling if history is missing
        # For a true production system, you'd fetch the last 24h of data from a DB or Redis
        data['air_temp_lag_1h'] = data['air_temperature_c'] # approximate
        data['air_temp_rolling_6h'] = data['air_temperature_c'] # approximate
        data['solar_rad_lag_1h'] = data['solar_radiation_w_m2']
        data['co2_lag_1h'] = data['air_co2_ppm'] # approximate
        
        data['temp_x_radiation'] = data['air_temperature_c'] * data['solar_radiation_w_m2']
        data['moisture_x_temp'] = data['soil_moisture_percent'] * data['air_temperature_c']
        
        return data

    def predict(self, payload_json_str):
        # 1. Parse JSON
        payload = json.loads(payload_json_str)
        
        # 2. Extract Data
        raw_data = self._parse_payload(payload)
        
        # 3. Engineer Features
        processed_data = self._feature_engineering(raw_data)
        
        # 4. Format for Model (ensure columns match training)
        df = pd.DataFrame([processed_data])
        
        # Add missing columns with 0 if any (failsafe)
        for col in self.features:
            if col not in df.columns:
                df[col] = 0.0
                
        # Reorder to match model
        X = df[self.features]
        
        # 5. Predict
        prediction = float(self.model.predict(X)[0])
        
        return {
            "status": "success",
            "timestamp": payload['timestamp'],
            "predicted_cci": round(prediction, 2),
            "unit": "Index (0-100)"
        }

if __name__ == "__main__":
    # Test Payload mimicking Agrisense Schema
    sample_payload = """
    {
      "device_id": "AGRISENSE-CC-001",
      "message_id": "MSG-20260414-0001",
      "timestamp": "2026-04-14T10:15:30Z",
      "location": {
        "latitude": -6.914744,
        "longitude": 107.60981,
        "altitude_m": 768.5
      },
      "carbon_data": {
        "co2_ppm": 421.6,
        "tvoc_ppb": 112
      },
      "environment": {
        "air_temperature_c": null,
        "air_humidity_percent": null,
        "air_pressure_hpa": null,
        "light_lux": 0
      },
      "soil_7in1": {
        "soil_moisture_percent": 42.5,
        "soil_temperature_c": 26.8,
        "soil_ec_ms_cm": 1.45,
        "soil_ph": 6.7,
        "soil_n_mg_kg": 78,
        "soil_p_mg_kg": 32,
        "soil_k_mg_kg": 115
      },
      "power": {
        "battery_voltage": 3.92,
        "battery_percent": 81
      },
      "communication": {
        "network_type": "WiFi",
        "rssi_dbm": -65
      },
      "status": {
        "node_status": "online",
        "sensor_status": "normal",
        "firmware_version": "1.0.1"
      }
    }
    """
    
    predictor = CarbonCapturePredictor()
    result = predictor.predict(sample_payload)
    print("Inference Result:")
    print(json.dumps(result, indent=2))
