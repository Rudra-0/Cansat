"""
ML inference service: loads the trained scikit-learn / PyOD models and turns a
telemetry packet into phase, anomaly, apogee and sounding predictions.
"""

import os
import json
import math
import time
import logging
from collections import deque
from typing import Any, Dict, Optional

import joblib
from fastapi import HTTPException
from pydantic import BaseModel, Field

from backend.core import features
from backend.settings import MODELS_DIR, METRICS_PATH

logger = logging.getLogger("cansat.server.ml")

models_loaded = False
phase_classifier = None
anomaly_detector = None
apogee_regressor = None
model_metrics = {}

def load_models():
    global models_loaded, phase_classifier, anomaly_detector, apogee_regressor, model_metrics
    try:
        clf_path = os.path.join(MODELS_DIR, "flight_phase_classifier.joblib")
        anom_path = os.path.join(MODELS_DIR, "anomaly_detector.joblib")
        reg_path = os.path.join(MODELS_DIR, "apogee_regressor.joblib")

        if os.path.exists(clf_path):
            phase_classifier = joblib.load(clf_path)
        if os.path.exists(anom_path):
            anomaly_detector = joblib.load(anom_path)
        if os.path.exists(reg_path):
            apogee_regressor = joblib.load(reg_path)
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                model_metrics = json.load(f)

        models_loaded = (phase_classifier is not None and anomaly_detector is not None)
        logger.info("ML models loaded: %s", models_loaded)
    except Exception:
        logger.exception("Error loading ML models")
        models_loaded = False

load_models()

class TelemetryPayload(BaseModel):
    temp: float = Field(..., description="Ambient temperature in °C")
    pressure: float = Field(..., description="Barometric pressure in hPa")
    altitude: float = Field(..., description="Barometric/GPS Altitude in meters")
    gx: float = Field(0.0, description="Gyro X in °/s")
    gy: float = Field(0.0, description="Gyro Y in °/s")
    gz: float = Field(0.0, description="Gyro Z in °/s")
    ax: float = Field(0.0, description="Accel X in G")
    ay: float = Field(0.0, description="Accel Y in G")
    az: float = Field(1.0, description="Accel Z in G")
    lat: float = Field(22.5727, description="GPS Latitude")
    lon: float = Field(88.3655, description="GPS Longitude")
    humidity: float = Field(50.0, description="Relative Humidity in %")
    batteryVoltage: float = Field(4.10, description="Cell Voltage in Volts")
    vSpd: Optional[float] = Field(None, description="Vertical Velocity in m/s")
    padAlt: Optional[float] = Field(0.0, description="Launch pad baseline altitude")
    padTemp: Optional[float] = Field(25.0, description="Launch pad baseline temperature")
    padPress: Optional[float] = Field(1013.25, description="Launch pad baseline pressure")

def compute_physics_soundings(temp_c: float, press_hpa: float, hum_pct: float, alt_m: float, pad_alt: float, pad_temp: float):
    # Magnus-Tetens Dew Point
    a, b = 17.625, 243.04
    clamped_hum = max(1.0, min(100.0, hum_pct))
    alpha = ((a * temp_c) / (b + temp_c)) + math.log(clamped_hum / 100.0)
    dew_point = (b * alpha) / (a - alpha)

    # Ideal Gas Law Air Density
    p_pa = press_hpa * 100.0
    t_kelvin = temp_c + 273.15
    r_specific = 287.058
    air_density = p_pa / (r_specific * t_kelvin)

    # Environmental Lapse Rate (ELR)
    lapse_rate = 0.65
    if abs(alt_m - pad_alt) > 15.0:
        d_alt = alt_m - pad_alt
        d_temp = temp_c - pad_temp
        lapse_rate = -(d_temp / d_alt) * 100.0

    return dew_point, air_density, lapse_rate

live_telemetry_history = deque(maxlen=20)

def run_ml_inference(payload: TelemetryPayload) -> Dict[str, Any]:
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Machine Learning models not loaded")

    v_spd = payload.vSpd if payload.vSpd is not None else 0.0

    live_telemetry_history.append({
        "time": time.time(),
        "temp": payload.temp,
        "pressure": payload.pressure,
        "altitude": payload.altitude,
        "ax": payload.ax,
        "ay": payload.ay,
        "az": payload.az,
        "gx": payload.gx,
        "gy": payload.gy,
        "gz": payload.gz,
        "lat": payload.lat,
        "lon": payload.lon,
        "humidity": payload.humidity,
        "batteryVoltage": payload.batteryVoltage,
        "vSpd": v_spd
    })
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "LIVE AUDIT: Temp=%sC Press=%shPa Alt=%sm Accel=(%s,%s,%s)g Gyro=(%s,%s,%s)deg/s "
            "GPS=(%s,%s) Hum=%s%% Bat=%sV vSpd=%sm/s",
            payload.temp, payload.pressure, payload.altitude, payload.ax, payload.ay, payload.az,
            payload.gx, payload.gy, payload.gz, payload.lat, payload.lon, payload.humidity,
            payload.batteryVoltage, v_spd,
        )
    pad_alt, pad_temp, _ = features.resolve_pad_baseline(
        payload.altitude, payload.temp, payload.padAlt, payload.padTemp, payload.padPress
    )
    feature_row = features.build_inference_features(
        temp=payload.temp, pressure=payload.pressure, altitude=payload.altitude,
        ax=payload.ax, ay=payload.ay, az=payload.az,
        gx=payload.gx, gy=payload.gy, gz=payload.gz,
        humidity=payload.humidity, battery_voltage=payload.batteryVoltage, v_spd=v_spd,
        pad_alt=payload.padAlt, pad_temp=payload.padTemp, pad_press=payload.padPress,
    )
    accel_mag = feature_row['accelMag']
    gyro_mag = feature_row['gyroMag']
    input_df = features.to_frame(feature_row, phase_classifier['feature_names'])

    # 1. Flight Phase Classifier Prediction (Random Forest)
    pipeline = phase_classifier['pipeline']
    predicted_phase = str(pipeline.predict(input_df)[0])
    probabilities = pipeline.predict_proba(input_df)[0]
    classes = phase_classifier['classes']
    prob_dict = {str(c): float(round(p, 4)) for c, p in zip(classes, probabilities)}

    # 2. Anomaly Detection (PyOD IForest)
    anom_scaler = anomaly_detector['scaler']
    anom_model = anomaly_detector['model']
    input_scaled = anom_scaler.transform(input_df)
    raw_anom_score = float(anom_model.decision_function(input_scaled)[0])
    
    score_min = anomaly_detector['score_min']
    score_max = anomaly_detector['score_max']
    normalized_anom_score = max(0.0, min(1.0, (raw_anom_score - score_min) / (score_max - score_min + 1e-9)))

    # Identify specific outlier heuristic category if elevated
    anomaly_type = "NOMINAL"
    is_anomaly = False

    if normalized_anom_score > 0.65 or raw_anom_score > 0.15:
        is_anomaly = True
        if gyro_mag > 120.0:
            anomaly_type = "GYRO_TUMBLE"
        elif accel_mag > 4.0:
            anomaly_type = "HIGH_G_SHOCK"
        elif payload.batteryVoltage < 3.55:
            anomaly_type = "VOLTAGE_SAG"
        elif predicted_phase == "PARACHUTE_DESCENT" and v_spd < -11.0:
            anomaly_type = "BALLISTIC_DESCENT"
        elif accel_mag < 0.20 and predicted_phase == "PARACHUTE_DESCENT":
            anomaly_type = "FREEFALL_ANOMALY"
        else:
            anomaly_type = "OUT_OF_ENVELOPE"

    # 3. Apogee Regressor
    predicted_apogee = payload.altitude
    if apogee_regressor is not None:
        try:
            reg_pipeline = apogee_regressor['pipeline']
            reg_pred = float(reg_pipeline.predict(input_df)[0])
            predicted_apogee = max(payload.altitude, reg_pred)
        except Exception:
            logger.warning("Apogee regressor failed; falling back to current altitude", exc_info=True)
            predicted_apogee = payload.altitude

    # 4. Thermodynamic Sounding
    dew_point, air_density, lapse_rate = compute_physics_soundings(
        payload.temp, payload.pressure, payload.humidity, payload.altitude, pad_alt, pad_temp
    )

    return {
        "status": "success",
        "predicted_phase": predicted_phase,
        "phase_confidence": float(round(prob_dict.get(predicted_phase, 1.0), 4)),
        "phase_probabilities": prob_dict,
        "anomaly_score": float(round(normalized_anom_score, 4)),
        "raw_anomaly_score": float(round(raw_anom_score, 4)),
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
        "predicted_apogee_m": float(round(predicted_apogee, 2)),
        "dew_point_c": float(round(dew_point, 2)),
        "air_density_kg_m3": float(round(air_density, 4)),
        "lapse_rate_c_100m": float(round(lapse_rate, 2))
    }
