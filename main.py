
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
from schemas import TransactionInput, PredictionOutput


# Initialize FastAPI app
app = FastAPI(
    title="Financial Fraud Detector API",
    description="Real-time credit card fraud detection using Random Forest ML model. Built with FastAPI and Scikit-learn.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model and scaler at startup
MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"
FRAUD_THRESHOLD = 0.3

print("Loading model and scaler...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("Model and scaler loaded successfully!")


@app.get("/")
def root():
    return {
        "message": "Financial Fraud Detector API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_fraud(transaction: TransactionInput):
    try:
        # Convert input to dataframe
        input_data = pd.DataFrame([{
            'V1': transaction.V1, 'V2': transaction.V2,
            'V3': transaction.V3, 'V4': transaction.V4,
            'V5': transaction.V5, 'V6': transaction.V6,
            'V7': transaction.V7, 'V8': transaction.V8,
            'V9': transaction.V9, 'V10': transaction.V10,
            'V11': transaction.V11, 'V12': transaction.V12,
            'V13': transaction.V13, 'V14': transaction.V14,
            'V15': transaction.V15, 'V16': transaction.V16,
            'V17': transaction.V17, 'V18': transaction.V18,
            'V19': transaction.V19, 'V20': transaction.V20,
            'V21': transaction.V21, 'V22': transaction.V22,
            'V23': transaction.V23, 'V24': transaction.V24,
            'V25': transaction.V25, 'V26': transaction.V26,
            'V27': transaction.V27, 'V28': transaction.V28,
            'Amount': transaction.Amount
        }])

        # Scale Amount column
        input_data['Amount'] = scaler.transform(
            input_data[['Amount']]
        )

        # Get fraud probability
        fraud_probability = model.predict_proba(input_data)[0][1]

        # Apply threshold
        is_fraud = fraud_probability >= FRAUD_THRESHOLD

        # Determine confidence level
        if fraud_probability >= 0.8:
            confidence = "HIGH"
        elif fraud_probability >= 0.5:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Build response
        prediction = "FRAUD" if is_fraud else "LEGITIMATE"
        message = (
            f"Transaction flagged as {prediction}. "
            f"Fraud probability: {fraud_probability:.2%}"
        )

        return PredictionOutput(
            prediction=prediction,
            fraud_probability=round(fraud_probability, 4),
            confidence=confidence,
            amount=transaction.Amount,
            message=message
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    return {
        "model_type": "Random Forest Classifier",
        "n_estimators": model.n_estimators,
        "fraud_threshold": FRAUD_THRESHOLD,
        "features": 29,
        "feature_names": [
            "V1", "V2", "V3", "V4", "V5", "V6", "V7",
            "V8", "V9", "V10", "V11", "V12", "V13", "V14",
            "V15", "V16", "V17", "V18", "V19", "V20", "V21",
            "V22", "V23", "V24", "V25", "V26", "V27", "V28",
            "Amount"
        ],
        "training_dataset": "Kaggle Credit Card Fraud Detection",
        "total_transactions": 284807,
        "fraud_cases": 492,
        "fraud_rate": "0.17%",
        "target_auc_roc": 0.95,
        "achieved_auc_roc": 0.9580,
        "compliance": [
            "PCI DSS",
            "FINTRAC",
            "OSFI",
            "PIPEDA"
        ],
        "version": "1.0.0",
        "author": "Aamir Qadeer — AI Engineer"
    }



