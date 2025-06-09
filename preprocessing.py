import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, SCALER_PATH, ENCODERS_PATH

def preprocess_data(df: pd.DataFrame, fit=False):
    """
    Cleans, encodes, and scales the input DataFrame.
    If fit=True, it fits and saves the transformers.
    """
    df = df.copy()

    # Handle missing values (basic strategy - mean for numerical, mode for categorical)
    for col in NUMERICAL_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Initialize containers
    encoders = {}
    
    # Encode categorical features
    for col in CATEGORICAL_FEATURES:
        if fit:
            enc = LabelEncoder()
            df[col] = enc.fit_transform(df[col])
            encoders[col] = enc
        else:
            enc = load_label_encoders().get(col)
            df[col] = enc.transform(df[col]) if enc else df[col]

    # Scale numerical features
    if fit:
        scaler = StandardScaler()
        df[NUMERICAL_FEATURES] = scaler.fit_transform(df[NUMERICAL_FEATURES])
        save_scaler(scaler)
        save_label_encoders(encoders)
    else:
        scaler = load_scaler()
        df[NUMERICAL_FEATURES] = scaler.transform(df[NUMERICAL_FEATURES])

    return df

# -------------------------------
# Save/Load Utilities
# -------------------------------

def save_scaler(scaler):
    joblib.dump(scaler, SCALER_PATH)

def load_scaler():
    return joblib.load(SCALER_PATH)

def save_label_encoders(encoders: dict):
    joblib.dump(encoders, ENCODERS_PATH)

def load_label_encoders():
    return joblib.load(ENCODERS_PATH)
