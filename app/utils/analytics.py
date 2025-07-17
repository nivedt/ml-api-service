# app/utils/analytics.py

from collections import Counter
from datetime import datetime

class AnalyticsTracker:
    def __init__(self):
        self.predictions = []

    def log_prediction(self, label: int):
        self.predictions.append((label, datetime.utcnow()))

    def get_summary(self):
        total = len(self.predictions)
        class_counts = Counter(label for label, _ in self.predictions)
        timestamps = [timestamp.isoformat() for _, timestamp in self.predictions][-5:]  # last 5
        return {
            "total_predictions": total,
            "class_distribution": {
                "real": class_counts.get(1, 0),
                "fake": class_counts.get(0, 0)
            },
            "recent_timestamps": timestamps
        }

# ✅ Declare one global instance here
tracker = AnalyticsTracker()



# from fastapi import APIRouter
# import os
# import json
#
# router = APIRouter()
#
# LOG_FILE = "logs.json"
#
# @router.get("/analytics")
# def get_analytics():
#     if not os.path.exists(LOG_FILE):
#         return {
#             "total_predictions": 0,
#             "class_distribution": {"real": 0, "fake": 0},
#             "recent_timestamps": []
#         }
#
#     with open(LOG_FILE, "r") as f:
#         logs = json.load(f)
#
#     total = len(logs)
#     real = sum(1 for entry in logs if entry["prediction"] == 0)
#     fake = sum(1 for entry in logs if entry["prediction"] == 1)
#     recent_timestamps = [entry["timestamp"] for entry in logs[-5:]]
#
#     return {
#         "total_predictions": total,
#         "class_distribution": {"real": real, "fake": fake},
#         "recent_timestamps": recent_timestamps
#     }
