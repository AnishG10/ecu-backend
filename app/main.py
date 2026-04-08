from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from app.ml.driver_behaviour import predict_from_raw_window
from app.ml.lstm_anomaly import detect_anomaly

app = FastAPI(title="ECU Guardian API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ UPDATED request model (removed accel_d & accel_e)
class WindowRequest(BaseModel):
    rpm_values: List[float]
    speed_values: List[float]
    throttle_values: List[float]

class SequenceRequest(BaseModel):
    data: List[dict]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/driver/predict")
def predict(body: WindowRequest):
    try:
        if len(body.rpm_values) < 5:
            raise HTTPException(status_code=400, detail="Not enough data points")

        result = predict_from_raw_window(
            body.rpm_values,
            body.speed_values,
            body.throttle_values
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/driver/anomaly")
def anomaly_detection(body: SequenceRequest):
    try:
        if len(body.data) < 50:
            raise HTTPException(status_code=400, detail="Minimum 50 data points required")

        result = detect_anomaly(body.data)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
