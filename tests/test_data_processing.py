import pytest
import pandas as pd
import numpy as np
import os
from src.data_processing import generate_synthetic_data, load_data, preprocess_data, split_and_scale

def test_generate_synthetic_data():
    path = 'data/raw/test_churn.csv'
    if os.path.exists(path):
        os.remove(path)
    generate_synthetic_data(path, n_samples=100)
    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert len(df) == 100
    os.remove(path)

def test_preprocess_data():
    # Create a small dummy dataframe
    data = {
        'customerID': ['1', '2'],
        'gender': ['Male', 'Female'],
        'SeniorCitizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'Dependents': ['No', 'Yes'],
        'tenure': [10, 20],
        'PhoneService': ['Yes', 'No'],
        'MultipleLines': ['Yes', 'No phone service'],
        'InternetService': ['Fiber optic', 'No'],
        'OnlineSecurity': ['Yes', 'No internet service'],
        'OnlineBackup': ['No', 'No internet service'],
        'DeviceProtection': ['Yes', 'No internet service'],
        'TechSupport': ['No', 'No internet service'],
        'StreamingTV': ['Yes', 'No internet service'],
        'StreamingMovies': ['No', 'No internet service'],
        'Contract': ['One year', 'Month-to-month'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Bank transfer (automatic)', 'Electronic check'],
        'MonthlyCharges': [70.0, 20.0],
        'TotalCharges': [700.0, 400.0],
        'Churn': ['No', 'Yes']
    }
    df = pd.DataFrame(data)
    processed = preprocess_data(df)
    assert 'customerID' not in processed.columns
    assert 'Tenure_Group' not in processed.columns # it's one-hot encoded
    assert any(col.startswith('Tenure_Group_') for col in processed.columns)
    assert processed['gender'].iloc[0] in [0, 1]

def test_split_and_scale():
    # Need a slightly larger dataset for stratify
    generate_synthetic_data('data/raw/test_split.csv', n_samples=20)
    df = load_data('data/raw/test_split.csv')
    processed = preprocess_data(df)
    X_train, X_test, y_train, y_test, scaler, features = split_and_scale(processed)
    
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) == 20
    assert X_train.shape[1] == len(features)
    
    os.remove('data/raw/test_split.csv')
