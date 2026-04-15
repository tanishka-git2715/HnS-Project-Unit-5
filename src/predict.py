import joblib
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class ChurnPredictor:
    def __init__(self, model_dir='models'):
        self.model_path = os.path.join(model_dir, 'model.pkl')
        self.scaler_path = os.path.join(model_dir, 'scaler.pkl')
        self.features_path = os.path.join(model_dir, 'features.joblib')
        
        if not all(os.path.exists(p) for p in [self.model_path, self.scaler_path, self.features_path]):
            raise FileNotFoundError("Model artifacts not found. Please run training first.")
            
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.features = joblib.load(self.features_path)
        
    def predict(self, input_df: pd.DataFrame):
        """
        Predict churn for the given input data.
        input_df should be preprocessed to match the training feature set.
        """
        # Ensure correct column order and missing columns
        for col in self.features:
            if col not in input_df.columns:
                input_df[col] = 0
                
        # Reorder columns to match training
        input_df = input_df[self.features]
        
        # Scale numeric columns
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_df[numeric_cols] = self.scaler.transform(input_df[numeric_cols])
        
        # Prediction
        prediction = self.model.predict(input_df)[0]
        probability = self.model.predict_proba(input_df)[0][1]
        
        return {
            "prediction": "Yes" if prediction == 1 else "No",
            "probability": float(probability)
        }

def get_predictor():
    return ChurnPredictor()
