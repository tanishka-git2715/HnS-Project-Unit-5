import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_data(output_path='data/raw/telco_churn.csv', n_samples=7043):
    """
    Generate synthetic Telco Customer Churn data.
    Based on the popular IBM Telco Churn dataset structure.
    """
    if os.path.exists(output_path):
        logger.info(f"Dataset already exists at {output_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    
    data = {
        'customerID': [f'{i:04d}-ID' for i in range(n_samples)],
        'gender': np.random.choice(['Female', 'Male'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'tenure': np.random.randint(0, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['No phone service', 'No', 'Yes'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['No', 'Yes', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['No', 'Yes', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['No', 'Yes', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['No', 'Yes', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['No', 'Yes', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['No', 'Yes', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
        'MonthlyCharges': np.random.uniform(18.25, 118.75, n_samples),
        'TotalCharges': np.random.uniform(18.25, 8000.0, n_samples),
        'Churn': np.random.choice(['No', 'Yes'], n_samples, p=[0.73, 0.27])
    }
    
    df = pd.DataFrame(data)
    
    # Logic adjustment: TotalCharges should roughly be tenure * MonthlyCharges
    df['TotalCharges'] = df['tenure'] * df['MonthlyCharges']
    # Add some noise
    df['TotalCharges'] += np.random.normal(0, df['MonthlyCharges'], n_samples)
    df['TotalCharges'] = df['TotalCharges'].clip(lower=df['MonthlyCharges'])
    
    # Handle consistency (e.g., No internet service)
    internet_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in internet_cols:
        df.loc[df['InternetService'] == 'No', col] = 'No internet service'
        
    df.loc[df['PhoneService'] == 'No', 'MultipleLines'] = 'No phone service'

    df.to_csv(output_path, index=False)
    logger.info(f"Synthetic dataset generated and saved to {output_path}")

def load_data(path):
    """Load data from CSV."""
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path)

def preprocess_data(df):
    """Clean and preprocess the data."""
    logger.info("Preprocessing data...")
    
    # 1. Drop customerID (not useful for prediction)
    df = df.drop('customerID', axis=1)
    
    # 2. Convert TotalCharges to numeric (handle empty strings if any)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # 3. Feature Engineering: Tenure Groups
    def tenure_group(tenure):
        if tenure <= 12: return '0-12 Month'
        elif tenure <= 24: return '12-24 Month'
        elif tenure <= 48: return '24-48 Month'
        elif tenure <= 60: return '48-60 Month'
        else: return '> 60 Month'
    
    df['Tenure_Group'] = df['tenure'].apply(tenure_group)
    
    # 4. Encoding
    # Binary encoding for categorical with 2 options
    le = LabelEncoder()
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
        
    # One-hot encoding for the rest
    cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                'Contract', 'PaymentMethod', 'Tenure_Group']
    df = pd.get_dummies(df, columns=cat_cols)
    
    return df

def split_and_scale(df, target='Churn'):
    """Split data into train/test and scale numeric features."""
    logger.info("Splitting and scaling data...")
    
    X = df.drop(target, axis=1)
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()

if __name__ == "__main__":
    generate_synthetic_data()
    data = load_data('data/raw/telco_churn.csv')
    processed_data = preprocess_data(data)
    X_train, X_test, y_train, y_test, scaler, features = split_and_scale(processed_data)
    logger.info(f"Preprocessing complete. Training shapes: {X_train.shape}, {y_train.shape}")
    logger.info(f"Features: {features}")
