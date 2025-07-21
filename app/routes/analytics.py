# app/routes/analytics.py

from fastapi import APIRouter
from fastapi.params import Depends
from app.auth.security import verify_api_key

from app.auth.security import verify_api_key
from app.utils.analytics import tracker

router = APIRouter()

@router.get("/analytics", dependencies=[Depends(verify_api_key)])
def get_analytics():
    return tracker.get_summary()
