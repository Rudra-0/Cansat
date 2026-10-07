"""Live ML inference endpoints (REST and WebSocket)."""

import json
import time
import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from backend import ml_service as ml
from backend.ml_service import TelemetryPayload, run_ml_inference, live_telemetry_history

logger = logging.getLogger("cansat.server.ml")

router = APIRouter(tags=["ml"])

@router.get("/api/latest_telemetry")
def get_latest_telemetry():
    return {
        "status": "success",
        "count": len(live_telemetry_history),
        "packets": list(live_telemetry_history)
    }

@router.get("/api/health")
def get_health():
    return {
        "status": "online" if ml.models_loaded else "degraded",
        "models_loaded": ml.models_loaded,
        "timestamp": time.time()
    }

@router.get("/api/models/info")
def get_models_info():
    if not ml.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "models": {
            "flight_phase_classifier": {
                "type": "RandomForestClassifier",
                "features": ml.phase_classifier['feature_names'],
                "classes": ml.phase_classifier['classes'],
                "metrics": ml.model_metrics.get("flight_phase_classifier", {})
            },
            "anomaly_detector": {
                "type": "PyOD.IForest",
                "metrics": ml.model_metrics.get("anomaly_detector", {})
            },
            "apogee_regressor": {
                "type": "GradientBoostingRegressor",
                "metrics": ml.model_metrics.get("apogee_regressor", {})
            }
        }
    }

@router.post("/api/predict")
def predict_telemetry(payload: TelemetryPayload):
    return run_ml_inference(payload)

@router.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected to live ML telemetry stream.")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                payload = TelemetryPayload(**msg)
                result = run_ml_inference(payload)
                await websocket.send_json(result)
            except Exception as parse_err:
                await websocket.send_json({"status": "error", "message": str(parse_err)})
    except WebSocketDisconnect:
        logger.info("WebSocket telemetry client disconnected.")
