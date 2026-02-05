"""
Telco Customer Churn Prediction API
FastAPI backend with Jinja2 template rendering.

Usage:
    uvicorn main:app --reload
"""

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# ── Initialize App ────────────────────────────────────────────
app = FastAPI(title="Churn Predictor", version="1.0.0")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Load Model Artifacts ──────────────────────────────────────
with open("model/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


# ── Request Schema ────────────────────────────────────────────
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# ── Routes ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(customer: CustomerData):
    """Predict churn probability for a customer."""
    try:
        # Convert input to DataFrame
        input_dict = customer.dict()
        df = pd.DataFrame([input_dict])

        # Categorical columns to encode (same as training)
        categorical_cols = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]

        # One-hot encode
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Align columns with training features (add missing, remove extra)
        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[feature_columns]

        # Scale numerical features
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df_encoded[num_cols] = scaler.transform(df_encoded[num_cols])

        # Predict
        prediction = model.predict(df_encoded)[0]
        probability = model.predict_proba(df_encoded)[0]

        churn_prob = round(float(probability[1]) * 100, 1)
        stay_prob = round(float(probability[0]) * 100, 1)

        # Risk level
        if churn_prob >= 70:
            risk_level = "Critical"
            risk_color = "#ef4444"
        elif churn_prob >= 40:
            risk_level = "High"
            risk_color = "#f97316"
        elif churn_prob >= 20:
            risk_level = "Medium"
            risk_color = "#eab308"
        else:
            risk_level = "Low"
            risk_color = "#22c55e"

        return {
            "success": True,
            "prediction": int(prediction),
            "churn_probability": churn_prob,
            "stay_probability": stay_prob,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "message": f"This customer has a {churn_prob}% chance of churning."
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_features": len(feature_columns)}
