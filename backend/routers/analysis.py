"""Post-flight telemetry analysis and sensor-ablation endpoints."""

import io
import os
import json
import zipfile

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from backend.settings import BASE_DIR, TEST_CASES_DIR

router = APIRouter(tags=["analysis"])

# -----------------------------------------------------------------------------
# Post-Flight Telemetry & Sensor Suite Ablation API Endpoints
# -----------------------------------------------------------------------------
class AblationRunRequest(BaseModel):
    scenario: str = Field(default="01_nominal_sounding_flight")
    feature_set: str = Field(default="Full_Suite")
    dataset_mode: str = Field(default="testing")

@router.get("/api/analysis/reports-zip")
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

@router.get("/api/analysis/ablation-data")
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

@router.get("/api/analysis/scenarios")
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

def available_scenarios() -> set:
    """Scenario names that may be requested: the CSV files shipped in test_cases/."""
    if not os.path.isdir(TEST_CASES_DIR):
        return set()
    return {f[:-4] for f in os.listdir(TEST_CASES_DIR) if f.endswith(".csv")}


@router.get("/api/analysis/scenario/{scenario_name}")
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

@router.post("/api/analysis/run-ablation")
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
