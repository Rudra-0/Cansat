"""
Cognitive CanSat FastAPI Real-Time Machine Learning Backend Server
Serves real-time inference using trained scikit-learn and PyOD models.
"""
import os
import re
import hmac
import math
import json
import time
import logging
import warnings
from typing import Dict, Any, Optional, List
from collections import deque
import numpy as np
import pandas as pd
import joblib

import io
import zipfile

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.core import database, features

# Suppress minor version warnings for clean telemetry logs
warnings.filterwarnings("ignore", category=UserWarning)

# No-op when uvicorn (or another host) has already configured the root logger.
logging.basicConfig(
    level=os.environ.get("CANSAT_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("cansat.server")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "ml", "saved_models")
METRICS_PATH = os.path.join(BASE_DIR, "ml", "model_metrics.json")

# Initialize persistent SQLite telemetry & mission database
try:
    database.init_db()
    logger.info("SQLite telemetry & mission database initialized in WAL mode.")
except Exception as db_err:
    logger.warning("Could not initialize database: %s", db_err)

app = FastAPI(
    title="Cognitive CanSat ML Telemetry Server",
    description="Real-Time Machine Learning Inference API for Sounding Pico-Satellite Telemetry",
    version="2.0.0"
)

# Pages served by this server are same-origin and need no CORS. Cross-origin
# access is allowed for local development (file:// pages send Origin "null",
# dev servers on other localhost ports) plus any origins in CANSAT_CORS_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CANSAT_CORS_ORIGINS", "").split(",") if o.strip()],
    allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|null)$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOCAL_CLIENT_HOSTS = {"127.0.0.1", "::1", "localhost"}


def require_operator(
    request: Request,
    x_api_key: Optional[str] = Header(None),
    api_key: Optional[str] = Query(None, description="Alternative to the X-API-Key header"),
) -> None:
    """Guard for destructive / hardware-control endpoints.

    With CANSAT_API_KEY set, callers must present it (X-API-Key header or
    ?api_key=). Without it, only clients on this machine are allowed, so a
    laptop ground station works out of the box while a deployed server is
    locked down until a key is configured.
    """
    expected = os.environ.get("CANSAT_API_KEY")
    if expected:
        supplied = x_api_key or api_key or ""
        if hmac.compare_digest(supplied.encode(), expected.encode()):
            return
        raise HTTPException(status_code=401, detail="Valid X-API-Key required",
                            headers={"WWW-Authenticate": "ApiKey"})
    client_host = request.client.host if request.client else ""
    if client_host in LOCAL_CLIENT_HOSTS:
        return
    raise HTTPException(status_code=403,
                        detail="Operator endpoints are restricted to localhost unless CANSAT_API_KEY is set")

# Global model state
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

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/index.html")
@app.get("/home.html")
def serve_index_html():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/tinyml.html")
def serve_tinyml_html():
    return FileResponse(os.path.join(FRONTEND_DIR, "tinyml.html"))

@app.get("/dashboard.html")
def serve_dashboard_html():
    return FileResponse(os.path.join(FRONTEND_DIR, "dashboard.html"))

@app.get("/models.html")
@app.get("/models")
def serve_models_html():
    return FileResponse(os.path.join(FRONTEND_DIR, "models.html"))

@app.get("/analysis.html")
@app.get("/analysis")
def serve_analysis_html():
    return FileResponse(os.path.join(FRONTEND_DIR, "analysis.html"))

@app.get("/results.html")
def serve_results_html():
    return FileResponse(os.path.join(FRONTEND_DIR, "results.html"))


css_dir = os.path.join(FRONTEND_DIR, "css")
if os.path.exists(css_dir):
    app.mount("/css", StaticFiles(directory=css_dir), name="css")

js_dir = os.path.join(FRONTEND_DIR, "js")
if os.path.exists(js_dir):
    app.mount("/js", StaticFiles(directory=js_dir), name="js")


test_cases_dir = os.path.join(BASE_DIR, "test_cases")
if os.path.exists(test_cases_dir):
    app.mount("/test_cases", StaticFiles(directory=test_cases_dir), name="test_cases")

reports_dir = os.path.join(BASE_DIR, "reports")
if os.path.exists(reports_dir):
    app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")


# -----------------------------------------------------------------------------
# Post-Flight Telemetry & Sensor Suite Ablation API Endpoints
# -----------------------------------------------------------------------------
class AblationRunRequest(BaseModel):
    scenario: str = Field(default="01_nominal_sounding_flight")
    feature_set: str = Field(default="Full_Suite")
    dataset_mode: str = Field(default="testing")

@app.get("/api/analysis/reports-zip")
def download_reports_zip():
    """Packages all 10 verified PDF flight reports and all_missions_summary.csv into a ZIP."""
    pdf_dir = os.path.join(BASE_DIR, "reports", "pdf")
    summary_csv = os.path.join(BASE_DIR, "reports", "all_missions_summary.csv")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(summary_csv):
            zf.write(summary_csv, arcname="all_missions_summary.csv")
        if os.path.exists(pdf_dir):
            for fname in sorted(os.listdir(pdf_dir)):
                if fname.endswith(".pdf"):
                    zf.write(os.path.join(pdf_dir, fname), arcname=os.path.join("pdf_reports", fname))
    zip_buffer.seek(0)
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=Cognitive_CanSat_Flight_Reports.zip"}
    )

@app.get("/api/analysis/ablation-data")
def get_ablation_benchmarks():
    """Return precomputed sensor suite ablation benchmarks for training and testing data."""
    ablation_file = os.path.join(BASE_DIR, "ml", "ablation_benchmarks.json")
    if os.path.exists(ablation_file):
        try:
            with open(ablation_file, "r") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=404, detail="Ablation data not found")

@app.get("/api/analysis/scenarios")
def list_flight_scenarios():
    """Return list of all 10 flight scenarios and key summary metrics."""
    summary_path = os.path.join(BASE_DIR, "reports", "all_missions_summary.csv")
    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            return {"status": "success", "scenarios": df.to_dict(orient="records")}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"status": "error", "message": "Summary CSV not found"}

TEST_CASES_DIR = os.path.join(BASE_DIR, "test_cases")


def available_scenarios() -> set:
    """Scenario names that may be requested: the CSV files shipped in test_cases/."""
    if not os.path.isdir(TEST_CASES_DIR):
        return set()
    return {f[:-4] for f in os.listdir(TEST_CASES_DIR) if f.endswith(".csv")}


@app.get("/api/analysis/scenario/{scenario_name}")
def get_scenario_telemetry(scenario_name: str):
    """Return time-series telemetry and calculated kinematics for interactive graphing."""
    clean_name = scenario_name.removesuffix(".csv")
    if clean_name not in available_scenarios():
        raise HTTPException(status_code=404, detail=f"Scenario {clean_name} not found")
    csv_path = os.path.join(TEST_CASES_DIR, f"{clean_name}.csv")

    try:
        df = pd.read_csv(csv_path).bfill().ffill()
        n = len(df)
        dt = 0.8
        time_arr = (np.arange(n) * dt).tolist()

        from scipy.signal import savgol_filter
        alt_raw = df['altitude'].values
        if n >= 7:
            w = min(11, n // 2 * 2 + 1)
            p = min(3, w - 2)
            alt_smooth = savgol_filter(alt_raw, w, p)
        else:
            alt_smooth = alt_raw

        vel = np.gradient(alt_smooth, dt)
        accel_mag = (np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)).values if {'ax', 'ay', 'az'}.issubset(df.columns) else np.ones(n)

        p0 = 1013.25
        p = np.clip(df['pressure'].values, 10.0, 1500.0)
        t_k = df['temp'].values + 273.15
        pot_temp = t_k * (p0 / p) ** 0.286 - 273.15

        stride = max(1, n // 200)
        indices = list(range(0, n, stride))
        if (n - 1) not in indices:
            indices.append(n - 1)

        return {
            "status": "success",
            "scenario": clean_name,
            "total_samples": n,
            "time": [round(float(time_arr[i]), 1) for i in indices],
            "altitude_raw": [round(float(alt_raw[i]), 1) for i in indices],
            "altitude_smoothed": [round(float(alt_smooth[i]), 1) for i in indices],
            "velocity": [round(float(vel[i]), 2) for i in indices],
            "accel_mag": [round(float(accel_mag[i]), 2) for i in indices],
            "temp": [round(float(df['temp'].iloc[i]), 1) for i in indices],
            "pressure": [round(float(df['pressure'].iloc[i]), 1) for i in indices],
            "potential_temp": [round(float(pot_temp[i]), 1) for i in indices],
            "kpis": {
                "max_altitude_m": round(float(np.max(alt_smooth)), 1),
                "time_to_apogee_s": round(float(time_arr[int(np.argmax(alt_smooth))]), 1),
                "max_ascent_vel_mps": round(float(np.max(vel)), 1),
                "avg_descent_vel_mps": round(float(np.mean(vel[vel < 0])), 1) if np.any(vel < 0) else 0.0,
                "peak_g_shock": round(float(np.max(accel_mag)), 2),
                "touchdown_impact_g": round(float(accel_mag[-1]), 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analysis/run-ablation")
def run_live_ablation(req: AblationRunRequest):
    """Simulates live model ablation / sensor dropout on selected scenario testing data."""
    ablation_file = os.path.join(BASE_DIR, "ml", "ablation_benchmarks.json")
    if os.path.exists(ablation_file):
        with open(ablation_file, "r") as f:
            data = json.load(f)
        mode = req.dataset_mode if req.dataset_mode in ["training", "testing"] else "testing"
        records = [r for r in data[mode] if r["scenario"] == req.scenario.replace(".csv", "")]
        selected = next((r for r in records if r["feature_set"] == req.feature_set), None)
        return {
            "status": "success",
            "scenario": req.scenario,
            "feature_set": req.feature_set,
            "dataset_mode": mode,
            "result": selected,
            "all_suites_for_scenario": records,
            "module_impact": data.get("module_impact", {})
        }
    raise HTTPException(status_code=404, detail="Ablation data unavailable")

# -----------------------------------------------------------------------------
# Mission Database & Post-Flight Review (PFR) Persistence API
# -----------------------------------------------------------------------------
class MissionStartRequest(BaseModel):
    name: str = Field(default="Live Flight Test", description="Human-readable mission name")
    callsign: str = Field(default="CANSAT-1", description="Vehicle radio callsign")
    operator: str = Field(default="Flight Controller", description="Operator callsign / name")
    notes: Optional[str] = Field(None, description="Pre-flight briefing or weather notes")
    mission_id: Optional[str] = Field(None, description="Custom mission ID")

class TelemetryBatchRequest(BaseModel):
    mission_id: str
    records: List[Dict[str, Any]]

class MissionFinishRequest(BaseModel):
    pad_baseline_alt: Optional[float] = None

class EventLogPayload(BaseModel):
    timestamp_ms: int
    event_type: str
    severity: str = "INFO"
    description: str

@app.post("/api/db/missions/start")
def start_db_mission(req: MissionStartRequest):
    """Arm and create a new mission session in SQLite."""
    try:
        mission = database.create_mission(
            name=req.name,
            callsign=req.callsign,
            operator=req.operator,
            notes=req.notes,
            mission_id=req.mission_id
        )
        return {"status": "success", "mission_id": mission["id"], "mission": mission}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/db/missions/telemetry")
def ingest_mission_telemetry(req: TelemetryBatchRequest):
    """Ingest a batch of telemetry packets into database for the specified mission."""
    try:
        count = database.insert_telemetry_batch(req.mission_id, req.records)
        return {"status": "success", "mission_id": req.mission_id, "inserted": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/db/missions/{mission_id}/events")
def log_mission_event(mission_id: str, payload: EventLogPayload):
    """Log an operational event or anomaly during a mission."""
    try:
        event_id = database.insert_event(
            mission_id=mission_id,
            timestamp_ms=payload.timestamp_ms,
            event_type=payload.event_type,
            severity=payload.severity,
            description=payload.description
        )
        return {"status": "success", "event_id": event_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/db/missions/{mission_id}/finish")
def finish_db_mission(mission_id: str, req: Optional[MissionFinishRequest] = None):
    """Finalize a mission, run SQL aggregations, and generate/store the PFR audit report."""
    try:
        pad_alt = req.pad_baseline_alt if req else None
        report = database.finalize_mission(mission_id)
        return {"status": "success", "mission_id": mission_id, "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/db/missions")
def list_db_missions():
    """List all stored missions with duration, apogee, and report status."""
    try:
        missions = database.list_missions()
        return {"status": "success", "missions": missions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/db/missions/{mission_id}")
def get_db_mission(mission_id: str):
    """Retrieve metadata for a single mission."""
    mission = database.get_mission(mission_id)
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")
    return {"status": "success", "mission": mission}

@app.get("/api/db/missions/{mission_id}/report")
def get_db_mission_report(mission_id: str):
    """Retrieve the stored Post-Flight Review (PFR) audit report for a mission."""
    report = database.get_mission_report(mission_id)
    if not report:
        try:
            report = database.generate_and_save_pfr_report(mission_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Report not found for this mission")
    return {"status": "success", "report": report.get("parsed_json", report), "raw": report}

@app.get("/api/db/missions/{mission_id}/telemetry")
def get_db_mission_telemetry(mission_id: str, limit: Optional[int] = None):
    """Retrieve time-series telemetry records for historic replay and charting."""
    records = database.get_mission_telemetry(mission_id, limit=limit)
    return {"status": "success", "mission_id": mission_id, "count": len(records), "records": records, "telemetry": records}

@app.delete("/api/db/missions/{mission_id}", dependencies=[Depends(require_operator)])
def delete_db_mission(mission_id: str):
    """Delete a mission and all associated records and reports."""
    deleted = database.delete_mission(mission_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Mission not found")
    return {"status": "success", "deleted": True, "deleted_mission_id": mission_id}

@app.delete("/api/db/missions", dependencies=[Depends(require_operator)])
def purge_all_db_missions():
    """Purge all recorded flight missions, resetting telemetry, reports, and events."""
    try:
        deleted_count = database.purge_all_missions()
        return {"status": "success", "deleted_missions": deleted_count, "purged": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/db/stats")
def get_db_stats():
    """Retrieve overall SQLite database storage metrics, row counts, and health."""
    try:
        stats = database.get_database_stats()
        return {"status": "success", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/db/download", dependencies=[Depends(require_operator)])
def download_sqlite_database():
    """Download the raw SQLite database file (cansat_missions.db) for external analysis."""
    db_path = database.DEFAULT_DB_PATH
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database file not found on disk")
    return FileResponse(
        path=db_path,
        filename="cansat_missions.db",
        media_type="application/x-sqlite3"
    )

@app.get("/api/db/missions/{mission_id}/export/csv")
def export_mission_telemetry_csv(mission_id: str):
    """Download standard CanSat CSV formatted telemetry for the specified mission."""
    csv_content = database.export_mission_csv(mission_id)
    if not csv_content:
        raise HTTPException(status_code=404, detail="No telemetry records found for this mission")
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{re.sub(r"[^A-Za-z0-9_.-]", "_", mission_id)}_telemetry.csv"'
        }
    )

@app.get("/api/latest_telemetry")
def get_latest_telemetry():
    return {
        "status": "success",
        "count": len(live_telemetry_history),
        "packets": list(live_telemetry_history)
    }

@app.get("/api/health")
def get_health():
    return {
        "status": "online" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "timestamp": time.time()
    }

@app.get("/api/models/info")
def get_models_info():
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "models": {
            "flight_phase_classifier": {
                "type": "RandomForestClassifier",
                "features": phase_classifier['feature_names'],
                "classes": phase_classifier['classes'],
                "metrics": model_metrics.get("flight_phase_classifier", {})
            },
            "anomaly_detector": {
                "type": "PyOD.IForest",
                "metrics": model_metrics.get("anomaly_detector", {})
            },
            "apogee_regressor": {
                "type": "GradientBoostingRegressor",
                "metrics": model_metrics.get("apogee_regressor", {})
            }
        }
    }

@app.post("/api/predict")
def predict_telemetry(payload: TelemetryPayload):
    return run_ml_inference(payload)

@app.websocket("/ws/telemetry")
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

# --- REAL-TIME DUAL-PORT HARDWARE SERIAL BRIDGE & PYTHON CORE ENGINES ---
import asyncio
import threading
from backend.core.serial_manager import DualSerialManager, list_available_ports, HAS_SERIAL

# Close code sent on /ws/serial when this host has no serial hardware bridge.
WS_CLOSE_NO_HARDWARE = 4404


def hardware_bridge_enabled() -> bool:
    """CANSAT_HARDWARE=off disables the server-side serial bridge (cloud / Docker hosts)."""
    mode = os.environ.get("CANSAT_HARDWARE", "on").strip().lower()
    return HAS_SERIAL and mode not in ("off", "0", "false", "no")
from backend.core.kinematics import KinematicsEngine, KinematicState
from backend.core.atmospheric import AtmosphericEngine, AtmosphericSounding

serial_subscribers: List[WebSocket] = []
subscribers_lock = threading.Lock()

# Core computation engines
kinematics_engine = KinematicsEngine()
atmospheric_engine = AtmosphericEngine()

# Thread-safe cache of latest processed states
latest_kinematic_state: Optional[KinematicState] = None
latest_sounding_state: Optional[AtmosphericSounding] = None
state_lock = threading.Lock()

async def broadcast_serial_line(line: str):
    with subscribers_lock:
        targets = list(serial_subscribers)
    dead = []
    for ws in targets:
        try:
            await ws.send_text(line)
        except Exception as e:
            logger.info("Dropping serial bridge subscriber after failed send: %s", e)
            dead.append(ws)
    if dead:
        with subscribers_lock:
            for ws in dead:
                if ws in serial_subscribers:
                    serial_subscribers.remove(ws)

def on_serial_line_dispatcher(label: str, line: str, loop: asyncio.AbstractEventLoop):
    # Broadcast raw line directly to all active WebSocket clients (index.html HUD)
    asyncio.run_coroutine_threadsafe(broadcast_serial_line(line), loop)

def on_telemetry_csv_processor(line: str):
    """Process incoming 13-field (or 14-field) CSV through Kinematics and Atmospheric engines."""
    global latest_kinematic_state, latest_sounding_state
    try:
        parts = [float(x.strip()) for x in line.split(",")]
        if len(parts) >= 13:
            temp, press, alt, gx, gy, gz, ax, ay, az, lat, lon, hum, volt = parts[:13]
            kin_state = kinematics_engine.process_packet(
                alt, ax, ay, az, gx, gy, gz,
                pressure=press, temp=temp, humidity=hum
            )
            atmo_state = atmospheric_engine.process_sounding(press, temp, hum, alt)
            with state_lock:
                latest_kinematic_state = kin_state
                latest_sounding_state = atmo_state
            live_telemetry_history.append({
                "time": time.time(),
                "temp": temp,
                "pressure": press,
                "altitude": alt,
                "calibrated_altitude": kin_state.calibrated_altitude if kin_state else alt,
                "ax": ax, "ay": ay, "az": az,
                "gx": gx, "gy": gy, "gz": gz,
                "lat": lat, "lon": lon,
                "humidity": hum,
                "batteryVoltage": volt,
                "vSpd": kin_state.vertical_speed if kin_state else 0.0,
                "calibrated_vspd": kin_state.calibrated_vspd if kin_state else 0.0,
            })
        else:
            logger.warning("Dropping telemetry line with %d fields (need 13): %.120r", len(parts), line)
    except ValueError as e:
        logger.warning("Dropping malformed telemetry line %.120r: %s", line, e)
    except Exception:
        logger.exception("Telemetry processing failed for line %.120r", line)

# Global serial manager instance
dual_serial_manager: Optional[DualSerialManager] = None

class HardwareConfigRequest(BaseModel):
    telemetry_port: Optional[str] = None
    telemetry_baud: Optional[int] = None
    video_port: Optional[str] = None
    video_baud: Optional[int] = None

def get_or_create_serial_manager(loop: asyncio.AbstractEventLoop) -> DualSerialManager:
    global dual_serial_manager
    if dual_serial_manager is None:
        telemetry_port = os.environ.get("CANSAT_TELEMETRY_PORT", "COM4")
        telemetry_baud = int(os.environ.get("CANSAT_TELEMETRY_BAUD", "9600"))
        video_port = os.environ.get("CANSAT_VIDEO_PORT", "COM5")
        video_baud = int(os.environ.get("CANSAT_VIDEO_BAUD", "460800"))
        dual_serial_manager = DualSerialManager(
            telemetry_port=telemetry_port,
            telemetry_baud=telemetry_baud,
            video_port=video_port,
            video_baud=video_baud,
            on_line_received=lambda label, line: on_serial_line_dispatcher(label, line, loop),
            on_telemetry_csv=on_telemetry_csv_processor,
        )
    return dual_serial_manager

def _get_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

@app.get("/hardware/ports")
def get_hardware_ports():
    """Enumerate all serial ports on the host system."""
    if not hardware_bridge_enabled():
        return {"ports": [], "hardware_enabled": False}
    return {"ports": list_available_ports(), "hardware_enabled": True}

@app.get("/hardware/config")
def get_hardware_config():
    """Return active serial port and baud rate configuration."""
    if not hardware_bridge_enabled():
        return {"status": "disabled", "hardware_enabled": False, "is_running": False, "available_ports": []}
    loop = _get_loop()
    mgr = get_or_create_serial_manager(loop)
    conn_status = mgr.get_connection_status()
    return {
        "status": "success",
        "telemetry": {
            "port": mgr.telemetry_cfg.port,
            "baud_rate": mgr.telemetry_cfg.baud_rate,
            "connected": conn_status.get(mgr.telemetry_cfg.label, False),
        },
        "video": {
            "port": mgr.video_cfg.port,
            "baud_rate": mgr.video_cfg.baud_rate,
            "connected": conn_status.get(mgr.video_cfg.label, False),
        },
        "is_running": mgr.is_running,
        "available_ports": list_available_ports(),
    }

@app.post("/hardware/config", dependencies=[Depends(require_operator)])
def set_hardware_config(cfg: HardwareConfigRequest):
    """Reconfigure serial ports and baud rates dynamically at runtime."""
    if not hardware_bridge_enabled():
        raise HTTPException(status_code=409, detail="Serial hardware bridge is disabled on this server")
    loop = _get_loop()
    mgr = get_or_create_serial_manager(loop)
    mgr.reconfigure(
        telemetry_port=cfg.telemetry_port,
        telemetry_baud=cfg.telemetry_baud,
        video_port=cfg.video_port,
        video_baud=cfg.video_baud,
    )
    conn_status = mgr.get_connection_status()
    return {
        "status": "reconfigured",
        "telemetry_port": mgr.telemetry_cfg.port,
        "telemetry_baud": mgr.telemetry_cfg.baud_rate,
        "telemetry_connected": conn_status.get(mgr.telemetry_cfg.label, False),
        "video_port": mgr.video_cfg.port,
        "video_baud": mgr.video_cfg.baud_rate,
        "video_connected": conn_status.get(mgr.video_cfg.label, False),
    }

@app.get("/analytics/kinematics")
def get_latest_kinematics():
    """Return latest real-time Kinematics state (Kalman altitude, attitude, alarms)."""
    with state_lock:
        if latest_kinematic_state is None:
            return {"status": "waiting_for_data"}
        return {"status": "active", "data": latest_kinematic_state.__dict__}

@app.get("/analytics/sounding")
def get_latest_sounding():
    """Return latest Atmospheric Thermodynamics state (dew point, air density, ELR)."""
    with state_lock:
        if latest_sounding_state is None:
            return {"status": "waiting_for_data"}
        return {"status": "active", "data": latest_sounding_state.__dict__}

@app.post("/hardware/tare")
def trigger_hardware_tare(samples: int = 15):
    """Trigger stationary pad tare calibration across baro and IMU sensors."""
    kinematics_engine.start_tare(num_samples=samples)
    return {
        "status": "taring_started",
        "samples_target": samples,
        "is_taring": kinematics_engine.is_taring
    }

@app.get("/hardware/tare")
def get_hardware_tare_status():
    """Get current tare calibration state."""
    return {
        "status": "success",
        "is_calibrated": kinematics_engine.tare.is_calibrated,
        "is_taring": kinematics_engine.is_taring,
        "samples_count": kinematics_engine.tare.samples_count,
        "tare": {
            "gyro_bias_x": round(kinematics_engine.tare.gyro_bias_x, 3),
            "gyro_bias_y": round(kinematics_engine.tare.gyro_bias_y, 3),
            "gyro_bias_z": round(kinematics_engine.tare.gyro_bias_z, 3),
            "pad_altitude": round(kinematics_engine.tare.pad_altitude, 2),
            "tare_pitch": round(kinematics_engine.tare.tare_pitch, 2),
            "tare_roll": round(kinematics_engine.tare.tare_roll, 2),
        }
    }

@app.websocket("/ws/serial")
async def websocket_serial_bridge(websocket: WebSocket):
    await websocket.accept()
    if not hardware_bridge_enabled():
        reason = "pyserial not installed" if not HAS_SERIAL else "CANSAT_HARDWARE=off"
        await websocket.send_text(f"# [HARDWARE] Serial bridge disabled on this server ({reason}). Use direct USB Web Serial.")
        await websocket.close(code=WS_CLOSE_NO_HARDWARE)
        return
    with subscribers_lock:
        serial_subscribers.append(websocket)
        count = len(serial_subscribers)
    logger.info("Serial bridge client connected (total subscribers: %d)", count)
    
    loop = asyncio.get_running_loop()
    manager = get_or_create_serial_manager(loop)
    if not manager.is_running:
        manager.start()

    # Send initial hardware link status to HUD
    init_msg = (
        f"# [HARDWARE] Active dual bridge: Telemetry ({manager.telemetry_cfg.port} @ "
        f"{manager.telemetry_cfg.baud_rate} baud) | Video ({manager.video_cfg.port} @ "
        f"{manager.video_cfg.baud_rate} baud)"
    )
    await websocket.send_text(init_msg)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        with subscribers_lock:
            if websocket in serial_subscribers:
                serial_subscribers.remove(websocket)
            remaining = len(serial_subscribers)
        if remaining == 0:
            manager.stop()




