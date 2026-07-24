from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts whether a customer will default on a loan.",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE, 'models', 'best_model.pkl'))
scaler = joblib.load(os.path.join(BASE, 'models', 'pipeline.pkl'))
feature_names = joblib.load(os.path.join(BASE, 'models', 'feature_names.pkl'))

class LoanApplication(BaseModel):
    customer_id: int
    tbl_loan_id: int
    lender_id: int
    Total_Amount: float
    duration: int
    Amount_Funded_By_Lender: float
    Lender_portion_Funded: float
    loan_year: int
    interest_amount: float
    repayment_ratio: float
    lender_profit: float
    loan_month: int
    loan_dayofweek: int
    loan_quarter: int
    customer_loan_count: int
    loan_type: str
    New_versus_Repeat: str

@app.get("/")
def root():
    return {"message": "Credit Risk Prediction API is running"}

@app.post("/predict")
def predict(data: LoanApplication):
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df, columns=['loan_type', 'New_versus_Repeat'])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "result": "DEFAULT RISK" if prediction == 1 else "LOW RISK",
        "default_probability": round(float(probability), 4),
        "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost", "version": "1.0.0"}