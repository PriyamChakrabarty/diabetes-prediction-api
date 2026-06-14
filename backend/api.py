from fastapi import FastAPI
from pydantic import BaseModel

from backend.predict import predict_diabetes

app = FastAPI(
    title="Diabetes Prediction API",
    version="1.0.0"
)


class DiabetesInput(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float


@app.get("/")
def health():
    return {"status": "running"}


@app.post("/predict")
def predict(payload: DiabetesInput):

    result = predict_diabetes(
        payload.model_dump()
    )

    return result