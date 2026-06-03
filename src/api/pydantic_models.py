from pydantic import BaseModel


class CustomerData(BaseModel):
    Amount: float
    Value: float
    CountryCode: int
    PricingStrategy: int


class PredictionResponse(BaseModel):
    risk_probability: float