from fastapi import FastAPI
from app.routes import predict
from app.utils import analytics

app = FastAPI(
    title = "Fake New Detection API",
    description = "An ML-powered API to classify whether a news article is fake or real.",
    version = "1.0.0"
)

app.include_router(predict.router)
app.include_router(analytics.router)

@app.get("/")
def root():
    return {"message": "ML API is live"}