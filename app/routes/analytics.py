# app/routes/analytics.py

from fastapi import APIRouter
from app.utils.analytics import tracker

router = APIRouter()

@router.get("/analytics")
def get_analytics():
    return tracker.get_summary()
