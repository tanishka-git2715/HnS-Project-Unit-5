from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from app.schemas import ChurnInput, PredictionResponse, HealthResponse
from app.middleware import PrometheusMiddleware, PREDICTION_COUNT
from src.predict import ChurnPredictor, get_predictor
from src.retrain import trigger_retraining
import pandas as pd
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API to predict customer churn using a machine learning model.",
    version="1.0.0"
)

# Add Middleware
app.add_middleware(PrometheusMiddleware)

# Persistence for model
predictor = None

@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        predictor = ChurnPredictor()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to the Churn Prediction API. Go to /docs for API documentation."}

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    return {
        "status": "healthy" if predictor else "unhealthy",
        "model_loaded": predictor is not None
    }

@app.get("/metrics", tags=["Monitoring"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(input_data: ChurnInput):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        # Convert input to DataFrame (flattened for preprocessing)
        input_dict = input_data.dict()
        
        # Add the Tenure_Group logic back in as it's part of preprocessing in src
        # But wait, src/predict handles preprocessing? 
        # Actually src/predict.pd handles scaling and column reordering.
        # We need to apply the same feature engineering as in src/data_processing.
        
        # In a real production app, preprocessing should be a shared utility or a pipeline.
        # For this demo, we'll manually apply the categorical expansion.
        
        df = pd.DataFrame([input_dict])
        
        # Feature engineering (Sync with src/data_processing.py)
        def tenure_group(tenure):
            if tenure <= 12: return '0-12 Month'
            elif tenure <= 24: return '12-24 Month'
            elif tenure <= 48: return '24-48 Month'
            elif tenure <= 60: return '48-60 Month'
            else: return '> 60 Month'
        df['Tenure_Group'] = df['tenure'].apply(tenure_group)
        
        # Encoding for specific fields (sync with src/data_processing.py)
        # Note: Usually you'd save the encoder or use pd.get_dummies if you have a fixed set of columns.
        # Here we rely on src/predict.py to fill in zeros for missing one-hot columns.
        df['gender'] = 1 if input_dict['gender'] == 'Male' else 0
        df['Partner'] = 1 if input_dict['Partner'] == 'Yes' else 0
        df['Dependents'] = 1 if input_dict['Dependents'] == 'Yes' else 0
        df['PhoneService'] = 1 if input_dict['PhoneService'] == 'Yes' else 0
        df['PaperlessBilling'] = 1 if input_dict['PaperlessBilling'] == 'Yes' else 0
        
        # Build one-hot columns prefix_value
        # This is simple manual encoding for the demo
        df[f"MultipleLines_{input_dict['MultipleLines']}"] = 1
        df[f"InternetService_{input_dict['InternetService']}"] = 1
        df[f"OnlineSecurity_{input_dict['OnlineSecurity']}"] = 1
        df[f"OnlineBackup_{input_dict['OnlineBackup']}"] = 1
        df[f"DeviceProtection_{input_dict['DeviceProtection']}"] = 1
        df[f"TechSupport_{input_dict['TechSupport']}"] = 1
        df[f"StreamingTV_{input_dict['StreamingTV']}"] = 1
        df[f"StreamingMovies_{input_dict['StreamingMovies']}"] = 1
        df[f"Contract_{input_dict['Contract']}"] = 1
        df[f"PaymentMethod_{input_dict['PaymentMethod']}"] = 1
        df[f"Tenure_Group_{df['Tenure_Group'][0]}"] = 1

        result = predictor.predict(df)
        
        PREDICTION_COUNT.labels(outcome=result["prediction"]).inc()
        
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain", tags=["Maintenance"])
def retrain_model():
    success = trigger_retraining()
    if success:
        global predictor
        predictor = ChurnPredictor() # Reload model
        return {"message": "Retraining triggered and model reloaded successfully."}
    else:
        raise HTTPException(status_code=500, detail="Retraining failed.")
