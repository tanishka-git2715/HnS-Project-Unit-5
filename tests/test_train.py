import pytest
import os
import joblib
from src.train import train

def test_train_execution():
    # Test if training runs without error and produces artifacts
    # We'll use a smaller dataset if possible, but our train() function uses the full path.
    # For simplicity, we just run it.
    train()
    
    assert os.path.exists('models/model.pkl')
    assert os.path.exists('models/scaler.pkl')
    assert os.path.exists('models/features.joblib')
    
    model = joblib.load('models/model.pkl')
    assert hasattr(model, 'predict')
