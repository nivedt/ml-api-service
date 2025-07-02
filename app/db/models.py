from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(String, nullable=False)
    prediction = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
