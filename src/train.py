import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_processing import generate_synthetic_data, load_data, preprocess_data, split_and_scale

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train():
    # 1. Ensure data exists
    dataset_path = 'data/raw/telco_churn.csv'
    if not os.path.exists(dataset_path):
        generate_synthetic_data(dataset_path)
    
    # 2. Load and preprocess
    df = load_data(dataset_path)
    processed_df = preprocess_data(df)
    X_train, X_test, y_train, y_test, scaler, features = split_and_scale(processed_df)
    
    # 3. Model setup
    n_estimators = 100
    max_depth = 10
    random_state = 42
    
    # Start MLflow run
    mlflow.set_experiment("Customer_Churn_Prediction")
    
    with mlflow.start_run():
        logger.info("Training model...")
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # 4. Predictions and Metrics
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_prob)
        }
        
        # Log params and metrics
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state
        })
        mlflow.log_metrics(metrics)
        
        logger.info(f"Metrics: {metrics}")
        
        # 5. Confusion Matrix Visualization
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        os.remove('confusion_matrix.png')
        
        # 6. Save Artifacts locally
        os.makedirs('models', exist_ok=True)
        model_path = 'models/model.pkl'
        scaler_path = 'models/scaler.pkl'
        features_path = 'models/features.joblib'
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(features, features_path)
        
        # 7. Log model to MLflow
        mlflow.sklearn.log_model(model, "churn_model_rf")
        
        logger.info(f"Model trained and saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info("MLflow run complete.")

if __name__ == "__main__":
    train()
 Mount folder monitoring/grafana/dashboards to avoid errors with empty dirs
