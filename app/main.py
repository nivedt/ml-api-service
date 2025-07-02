from fastapi import FastAPI
from app.routes import predict

app = FastAPI(
    title = "Fake New Detection API",
    description = "An ML-powered API to classify whether a news article is fake or real.",
    version = "1.0.0"
)

app.include_router(predict.router)

@app.get("/")
def root():
    return {"message": "ML API is live"}