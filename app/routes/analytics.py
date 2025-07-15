from fastapi import APIRouter
import os
import json

router = APIRouter()

LOG_FILE = "logs.json"

@router.get("/analytics")
def get_analytics():
    if not os.path.exists(LOG_FILE):
        return {
            "total_predictions": 0,
            "class_distribution": {"real": 0, "fake": 0},
            "recent_timestamps": []
        }

    with open(LOG_FILE, "r") as f:
        logs = json.load(f)

    total = len(logs)
    real = sum(1 for entry in logs if entry["prediction"] == 0)
    fake = sum(1 for entry in logs if entry["prediction"] == 1)
    recent_timestamps = [entry["timestamp"] for entry in logs[-5:]]

    return {
        "total_predictions": total,
        "class_distribution": {"real": real, "fake": fake},
        "recent_timestamps": recent_timestamps
    }
