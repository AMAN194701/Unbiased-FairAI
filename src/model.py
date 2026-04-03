"""
model.py — Handles data encoding, model training, and predictions
Part of Unbiased-FairAI | Solution Challenge 2026
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ─────────────────────────────────────────
# STEP 1: ENCODE DATA
# ─────────────────────────────────────────

def encode_data(df):
    """
    Converts text columns to numbers so ML model can understand them.
    
    Example:
        Gender: Male → 1, Female → 0
        Income: <=50K → 0, >50K → 1
    
    Args:
        df: cleaned pandas DataFrame (from clean_data.csv)
    
    Returns:
        X            → all input features (numbers only)
        y            → target column (Income as 0 or 1)
        X_train      → 80% of X for training
        X_test       → 20% of X for testing
        y_train      → 80% of y for training
        y_test       → 20% of y for testing
        encoders     → dictionary of LabelEncoders (needed for What-If simulator)
        sensitive_features → Gender column from test set (for fairness metrics)
    """

    df_model = df.copy()

    # Save encoders so What-If simulator can reverse them later
    encoders = {}

    # Find all text columns
    categorical_cols = df_model.select_dtypes(include='object').columns

    for col in categorical_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        encoders[col] = le  # save encoder for this column

    # Save encoders to disk for later use in app.py
    encoder_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'encoders.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoders, f)
    print("✅ Encoders saved to data/encoders.pkl")

    # X = everything except Income
    # y = Income column (target we want to predict)
    X = df_model.drop('Income', axis=1)
    y = df_model['Income']

    # Split into 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Get Gender from original df (not encoded) — needed for fairness metrics
    sensitive_features = df.loc[X_test.index, 'Gender']

    print(f"✅ Data encoded successfully")
    print(f"   Training samples : {X_train.shape[0]}")
    print(f"   Testing samples  : {X_test.shape[0]}")
    print(f"   Features         : {X_train.shape[1]}")

    return X, y, X_train, X_test, y_train, y_test, encoders, sensitive_features


# ─────────────────────────────────────────
# STEP 2: TRAIN MODEL
# ─────────────────────────────────────────

def train_model(X_train, y_train):
    """
    Trains a Random Forest classifier on the training data.
    
    What is Random Forest?
        It builds 100 decision trees and combines their answers.
        Think of it like asking 100 experts and taking majority vote.
    
    Args:
        X_train: training features
        y_train: training labels (Income)
    
    Returns:
        model: trained RandomForestClassifier
    """

    print("🔄 Training model...")

    model = RandomForestClassifier(
        n_estimators=100,   # 100 decision trees
        random_state=42,    # same result every run
        n_jobs=-1           # use all CPU cores → faster
    )

    model.fit(X_train, y_train)

    # Save model to disk
    model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved to data/model.pkl")
    return model


# ─────────────────────────────────────────
# STEP 3: PREDICT
# ─────────────────────────────────────────

def predict(model, X_test):
    """
    Uses trained model to make predictions on test data.
    
    Args:
        model  : trained RandomForestClassifier
        X_test : test features
    
    Returns:
        y_pred : array of predictions (0 = <=50K, 1 = >50K)
    """

    y_pred = model.predict(X_test)
    print(f"✅ Predictions made for {len(y_pred)} samples")
    return y_pred


# ─────────────────────────────────────────
# STEP 4: LOAD SAVED MODEL (for app.py)
# ─────────────────────────────────────────

def load_model():
    """
    Loads previously saved model from disk.
    Used in app.py so we don't retrain every time.
    
    Returns:
        model: trained RandomForestClassifier
    """

    model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'model.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Train the model first.")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print("✅ Model loaded from data/model.pkl")
    return model


def load_encoders():
    """
    Loads previously saved encoders from disk.
    Used in What-If simulator to encode user input.
    
    Returns:
        encoders: dictionary of LabelEncoders
    """

    encoder_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'encoders.pkl')

    if not os.path.exists(encoder_path):
        raise FileNotFoundError("Encoders not found. Run encode_data() first.")

    with open(encoder_path, 'rb') as f:
        encoders = pickle.load(f)

    print("✅ Encoders loaded")
    return encoders


# ─────────────────────────────────────────
# TEST — Run this file directly to verify
# ─────────────────────────────────────────

if __name__ == "__main__":
    
    # Load cleaned data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean_data.csv')
    df = pd.read_csv(data_path)
    print(f"📂 Loaded data: {df.shape}")

    # Encode
    X, y, X_train, X_test, y_train, y_test, encoders, sensitive_features = encode_data(df)

    # Train
    model = train_model(X_train, y_train)

    # Predict
    y_pred = predict(model, X_test)

    # Quick accuracy check
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Model Accuracy: {acc * 100:.2f}%")
    print(f"   (Expected: ~85-87%)")
    