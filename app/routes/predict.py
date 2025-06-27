from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import logging

router = APIRouter()

# Load the trained model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Input data structure
class NewsInput(BaseModel):
    text: str

logging.basicConfig(level=logging.INFO)

@router.post("/predict")
def predict(input: NewsInput):
    if not input.text or input.text.strip() == "":
        return {"error": "Input text is empty or invalid."}

    # Vectorize the input text
    X = vectorizer.transform([input.text])
    # prediction = model.predict(X)[0]
    # return {"prediction": int(prediction)}

    prediction = model.predict(X)[0]
    probablity = model.predict_proba(X)[0][1]

    logging.into(f"Input: {input.text}")
    logging.info(f"Prediction: {prediction}, Probablity: {probablity}")

    return {
        "prediction": int(prediction),
        "probablity": float(probablity)
    }