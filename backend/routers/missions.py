"""Mission database & Post-Flight Review (PFR) persistence endpoints."""

import os
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from backend.core import database
from backend.security import require_operator

router = APIRouter(tags=["missions"])

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

@router.post("/api/db/missions/start")
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

@router.post("/api/db/missions/telemetry")
def ingest_mission_telemetry(req: TelemetryBatchRequest):
    """Ingest a batch of telemetry packets into database for the specified mission."""
    try:
        count = database.insert_telemetry_batch(req.mission_id, req.records)
        return {"status": "success", "mission_id": req.mission_id, "inserted": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/db/missions/{mission_id}/events")
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

@router.post("/api/db/missions/{mission_id}/finish")
def finish_db_mission(mission_id: str, req: Optional[MissionFinishRequest] = None):
    """Finalize a mission, run SQL aggregations, and generate/store the PFR audit report."""
    try:
        pad_alt = req.pad_baseline_alt if req else None
        report = database.finalize_mission(mission_id, pad_baseline_alt=pad_alt)
        return {"status": "success", "mission_id": mission_id, "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/db/missions")
def list_db_missions():
    """List all stored missions with duration, apogee, and report status."""
    try:
        missions = database.list_missions()
        return {"status": "success", "missions": missions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/db/missions/{mission_id}")
def get_db_mission(mission_id: str):
    """Retrieve metadata for a single mission."""
    mission = database.get_mission(mission_id)
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")
    return {"status": "success", "mission": mission}

@router.get("/api/db/missions/{mission_id}/report")
def get_db_mission_report(mission_id: str):
    """Retrieve the stored Post-Flight Review (PFR) audit report for a mission."""
    report = database.get_mission_report(mission_id)
    if not report:
        try:
            report = database.generate_and_save_pfr_report(mission_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Report not found for this mission")
    return {"status": "success", "report": report.get("parsed_json", report), "raw": report}

@router.get("/api/db/missions/{mission_id}/telemetry")
def get_db_mission_telemetry(mission_id: str, limit: Optional[int] = None):
    """Retrieve time-series telemetry records for historic replay and charting."""
    records = database.get_mission_telemetry(mission_id, limit=limit)
    return {"status": "success", "mission_id": mission_id, "count": len(records), "records": records, "telemetry": records}

@router.delete("/api/db/missions/{mission_id}", dependencies=[Depends(require_operator)])
def delete_db_mission(mission_id: str):
    """Delete a mission and all associated records and reports."""
    deleted = database.delete_mission(mission_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Mission not found")
    return {"status": "success", "deleted": True, "deleted_mission_id": mission_id}

@router.delete("/api/db/missions", dependencies=[Depends(require_operator)])
def purge_all_db_missions():
    """Purge all recorded flight missions, resetting telemetry, reports, and events."""
    try:
        deleted_count = database.purge_all_missions()
        return {"status": "success", "deleted_missions": deleted_count, "purged": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/db/stats")
def get_db_stats():
    """Retrieve overall SQLite database storage metrics, row counts, and health."""
    try:
        stats = database.get_database_stats()
        return {"status": "success", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/db/download", dependencies=[Depends(require_operator)])
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

@router.get("/api/db/missions/{mission_id}/export/csv")
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
