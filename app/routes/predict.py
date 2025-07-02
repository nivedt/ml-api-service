# app/routes/predict.py
from fastapi import APIRouter, HTTPException
from fastapi.params import Depends
from pydantic import BaseModel
import joblib
import os
import pickle

from app.db.database import SessionLocal
from app.db.models import PredictionLog
from sqlalchemy.orm import Session

router = APIRouter()

# Global variables for model and vectorizer
model = None
vectorizer = None


def load_models():
    """Load the trained model and vectorizer"""
    global model, vectorizer

    model_path = "model/model.pkl"
    vectorizer_path = "model/vectorizer.pkl"

    # Check if model files exist
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model files not found. Please train the model first.")

    try:
        # Try loading with joblib first
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("Models loaded successfully with joblib!")
    except Exception as joblib_error:
        print(f"Joblib loading failed: {joblib_error}")
        try:
            # Fallback to pickle
            with open("model/model_backup.pkl", 'rb') as f:
                model = pickle.load(f)
            with open("model/vectorizer_backup.pkl", 'rb') as f:
                vectorizer = pickle.load(f)
            print("Models loaded successfully with pickle backup!")
        except Exception as pickle_error:
            raise Exception(f"Failed to load models with both methods. Joblib: {joblib_error}, Pickle: {pickle_error}")


# Try to load models when module is imported
try:
    load_models()
except Exception as e:
    print(f"Warning: Could not load models at startup: {e}")
    print("Models will need to be loaded manually or retrained.")


class TextInput(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    label: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/predict", response_model=PredictionResponse)
async def predict_news(input_data: TextInput, db: Session = Depends(get_db)):
    """Predict whether news is fake or real"""
    global model, vectorizer

    # Check if models are loaded
    if model is None or vectorizer is None:
        try:
            load_models()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Models not available: {str(e)}")

    try:
        # Preprocess and vectorize the input text
        text_vector = vectorizer.transform([input_data.text])

        # Make prediction
        prediction = model.predict(text_vector)[0]
        confidence = max(model.predict_proba(text_vector)[0])

        # Convert prediction to label
        label = "Real" if prediction == 1 else "Fake"

        # Log to DB
        log = PredictionLog(
            input_text=input_data.text,
            prediction=int(prediction)
        )
        db.add(log)
        db.commit()
        db.refresh(log)

        return PredictionResponse(
            prediction=int(prediction),
            confidence=float(confidence),
            label=label
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/model-status")
async def model_status():
    """Check if models are loaded"""
    return {
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "status": "ready" if (model is not None and vectorizer is not None) else "not ready"
    }

@router.get("/logs")
async def get_logs(db: Session = Depends(get_db)):
    logs = db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(10).all()
    return [
        {
            "input": log.input_text,
            "prediction": log.prediction,
            "timestamp": log.timestamp.isoformat()
        } for log in logs
    ]



# from fastapi import APIRouter
# from pydantic import BaseModel
# import joblib
# import logging
#
# router = APIRouter()
#
# # Load the trained model and vectorizer
# model = joblib.load("model/model.pkl")
# vectorizer = joblib.load("model/vectorizer.pkl")
#
# # Input data structure
# class NewsInput(BaseModel):
#     text: str
#
# logging.basicConfig(level=logging.INFO)
#
# @router.post("/predict")
# def predict(input: NewsInput):
#     if not input.text or input.text.strip() == "":
#         return {"error": "Input text is empty or invalid."}
#
#     # Vectorize the input text
#     X = vectorizer.transform([input.text])
#     # prediction = model.predict(X)[0]
#     # return {"prediction": int(prediction)}
#
#     prediction = model.predict(X)[0]
#     probablity = model.predict_proba(X)[0][1]
#
#     logging.into(f"Input: {input.text}")
#     logging.info(f"Prediction: {prediction}, Probablity: {probablity}")
#
#     return {
#         "prediction": int(prediction),
#         "probablity": float(probablity)
#     }