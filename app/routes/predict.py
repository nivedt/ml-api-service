from fastapi import APIRouter
from pydantic import BaseModel
import joblib

# Load the trained model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

router = APIRouter()

# Input data structure
class NewsInput(BaseModel):
    text: str

@router.post("/predict")
def predict(input: NewsInput):
    if not input.text.strip():
        return {"error": "Input text is empty"}

    # Vectorize the input text
    X = vectorizer.transform([input.text])
    prediction = model.predict(X)[0]
    return {"prediction": int(prediction)}