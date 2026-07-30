from fastapi import FastAPI
import joblib
import pandas as pd

from src.api.pydantic_models import (
    CustomerData,
    PredictionResponse
)

app = FastAPI()

model = joblib.load("best_model.pkl")


@app.get("/")
def home():
    return {
        "message": "Credit Risk API Running"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: CustomerData):

    df = pd.DataFrame(
        [data.model_dump()]
    )

    probability = (
        model.predict_proba(df)[0][1]
    )

    return PredictionResponse(
        risk_probability=float(probability)
    )