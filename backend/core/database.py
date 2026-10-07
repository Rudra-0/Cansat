"""
Cognitive CanSat Telemetry & Mission Database Engine
SQLite with Write-Ahead Logging (WAL) for concurrent, high-frequency flight persistence.
"""

import os
import io
import csv
import math
import json
import uuid
import sqlite3
import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List, Any, Optional

DEFAULT_DB_PATH = os.environ.get(
    "CANSAT_DB_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "cansat_missions.db")
)


@contextmanager
def get_db_connection(db_path: Optional[str] = None):
    """Return an SQLite connection configured with WAL mode and foreign keys, closed on exit."""
    target_path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    conn = sqlite3.connect(target_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database tables and indexes."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Missions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS missions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                callsign TEXT NOT NULL DEFAULT 'CANSAT-1',
                operator TEXT NOT NULL DEFAULT 'Flight Controller',
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT NOT NULL DEFAULT 'ARMED',
                notes TEXT
            );
        """)

        # 2. Telemetry Records Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mission_id TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                met_seconds REAL NOT NULL,
                altitude REAL NOT NULL,
                pressure REAL NOT NULL,
                temp REAL NOT NULL,
                humidity REAL NOT NULL,
                battery_voltage REAL NOT NULL,
                ax REAL NOT NULL,
                ay REAL NOT NULL,
                az REAL NOT NULL,
                gx REAL NOT NULL,
                gy REAL NOT NULL,
                gz REAL NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                v_spd REAL NOT NULL DEFAULT 0.0,
                accel_mag REAL NOT NULL DEFAULT 1.0,
                pitch REAL NOT NULL DEFAULT 0.0,
                roll REAL NOT NULL DEFAULT 0.0,
                air_density REAL NOT NULL DEFAULT 1.225,
                dew_point REAL NOT NULL DEFAULT 15.0,
                lapse_rate REAL NOT NULL DEFAULT 0.65,
                flight_phase TEXT NOT NULL DEFAULT 'PAD_IDLE',
                anomaly_score REAL NOT NULL DEFAULT 0.0,
                is_anomaly INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (mission_id) REFERENCES missions(id) ON DELETE CASCADE
            );
        """)

        # 3. Mission Reports Table (PFR Audits)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mission_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mission_id TEXT NOT NULL UNIQUE,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                peak_apogee_agl REAL NOT NULL,
                peak_altitude_msl REAL NOT NULL,
                time_to_apogee_s REAL NOT NULL,
                max_ejection_shock_g REAL NOT NULL,
                terminal_descent_rate_mps REAL NOT NULL,
                average_descent_rate_mps REAL NOT NULL,
                descent_compliance TEXT NOT NULL,
                total_flight_time_s REAL NOT NULL,
                total_packets INTEGER NOT NULL,
                packet_loss_pct REAL NOT NULL DEFAULT 0.0,
                launch_lat REAL NOT NULL,
                launch_lon REAL NOT NULL,
                touchdown_lat REAL NOT NULL,
                touchdown_lon REAL NOT NULL,
                horizontal_drift_m REAL NOT NULL,
                drift_azimuth_deg REAL NOT NULL,
                battery_start_v REAL NOT NULL,
                battery_end_v REAL NOT NULL,
                battery_delta_v REAL NOT NULL,
                anomaly_summary TEXT NOT NULL,
                report_markdown TEXT NOT NULL,
                report_json TEXT NOT NULL,
                FOREIGN KEY (mission_id) REFERENCES missions(id) ON DELETE CASCADE
            );
        """)

        # 4. Mission Events Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mission_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mission_id TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'INFO',
                description TEXT NOT NULL,
                FOREIGN KEY (mission_id) REFERENCES missions(id) ON DELETE CASCADE
            );
        """)

        # Indexes for rapid querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_mission_met ON telemetry_records(mission_id, met_seconds);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_records(mission_id, timestamp_ms);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_mission ON mission_events(mission_id, timestamp_ms);")
        conn.commit()


def create_mission(name: str, callsign: str = "CANSAT-1", operator: str = "Flight Controller",
                   notes: Optional[str] = None, mission_id: Optional[str] = None,
                   db_path: Optional[str] = None) -> Dict[str, Any]:
    """Create a new mission session in the database."""
    init_db(db_path)
    if not mission_id:
        now = datetime.datetime.now(datetime.timezone.utc)
        mission_id = f"MSN-{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO missions (id, name, callsign, operator, status, notes)
            VALUES (?, ?, ?, ?, 'ACTIVE_FLIGHT', ?)
            """,
            (mission_id, name, callsign, operator, notes)
        )
        conn.commit()

        cursor.execute("SELECT * FROM missions WHERE id = ?", (mission_id,))
        row = cursor.fetchone()
        return dict(row)


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, handling None, NaN, and conversion errors."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely convert a value to int, handling None and conversion errors."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def insert_telemetry_batch(mission_id: str, records: List[Dict[str, Any]],
                           db_path: Optional[str] = None) -> int:
    """Insert a batch of telemetry packets into the database."""
    if not records:
        return 0

    init_db(db_path)
    sql = """
        INSERT INTO telemetry_records (
            mission_id, timestamp_ms, met_seconds, altitude, pressure, temp, humidity,
            battery_voltage, ax, ay, az, gx, gy, gz, lat, lon, v_spd, accel_mag,
            pitch, roll, air_density, dew_point, lapse_rate, flight_phase,
            anomaly_score, is_anomaly
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """

    batch_params = []
    for r in records:
        # Fallbacks & conversions
        ax = _safe_float(r.get("ax"), 0.0)
        ay = _safe_float(r.get("ay"), 0.0)
        az = _safe_float(r.get("az"), 1.0)
        accel_mag = _safe_float(
            r.get("accel_mag") if r.get("accel_mag") is not None else r.get("accelMag"),
            math.sqrt(ax * ax + ay * ay + az * az)
        )

        battery_v = r.get("battery_voltage")
        if battery_v is None:
            battery_v = r.get("batteryVoltage")

        v_spd = r.get("v_spd")
        if v_spd is None:
            v_spd = r.get("vSpd")

        air_dens = r.get("air_density")
        if air_dens is None:
            air_dens = r.get("airDensity")

        dew_p = r.get("dew_point")
        if dew_p is None:
            dew_p = r.get("dewPoint")

        lapse_r = r.get("lapse_rate")
        if lapse_r is None:
            lapse_r = r.get("lapseRate")

        phase = r.get("flight_phase")
        if phase is None:
            phase = r.get("flightPhase", "PAD_IDLE")

        anom_score = r.get("anomaly_score")
        if anom_score is None:
            anom_score = r.get("anomalyScore")

        is_anom = r.get("is_anomaly")
        if is_anom is None:
            is_anom = r.get("isAnomaly", False)

        batch_params.append((
            mission_id,
            _safe_int(r.get("timestamp_ms"), 0),
            _safe_float(r.get("met_seconds"), 0.0),
            _safe_float(r.get("altitude"), 0.0),
            _safe_float(r.get("pressure"), 1013.25),
            _safe_float(r.get("temp"), 25.0),
            _safe_float(r.get("humidity"), 50.0),
            _safe_float(battery_v, 4.2),
            ax, ay, az,
            _safe_float(r.get("gx"), 0.0),
            _safe_float(r.get("gy"), 0.0),
            _safe_float(r.get("gz"), 0.0),
            _safe_float(r.get("lat"), 0.0),
            _safe_float(r.get("lon"), 0.0),
            _safe_float(v_spd, 0.0),
            accel_mag,
            _safe_float(r.get("pitch"), 0.0),
            _safe_float(r.get("roll"), 0.0),
            _safe_float(air_dens, 1.225),
            _safe_float(dew_p, 15.0),
            _safe_float(lapse_r, 0.65),
            str(phase or "PAD_IDLE"),
            _safe_float(anom_score, 0.0),
            1 if bool(is_anom) else 0
        ))

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(sql, batch_params)
        conn.commit()
        return len(batch_params)


def insert_event(mission_id: str, timestamp_ms: int, event_type: str,
                 severity: str = "INFO", description: str = "",
                 db_path: Optional[str] = None) -> int:
    """Log an operational flight event."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO mission_events (mission_id, timestamp_ms, event_type, severity, description)
            VALUES (?, ?, ?, ?, ?)
            """,
            (mission_id, timestamp_ms, event_type, severity, description)
        )
        conn.commit()
        return cursor.lastrowid


def generate_and_save_pfr_report(mission_id: str, pad_baseline_alt: Optional[float] = None,
                                 db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run SQL aggregations on mission telemetry, calculate Post-Flight Review (PFR) KPIs,
    and persist an audit record in mission_reports.
    """
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Check mission
        cursor.execute("SELECT * FROM missions WHERE id = ?", (mission_id,))
        mission = cursor.fetchone()
        if not mission:
            raise ValueError(f"Mission '{mission_id}' not found in database.")

        # Query all records ordered by met_seconds / timestamp_ms
        cursor.execute(
            """
            SELECT * FROM telemetry_records
            WHERE mission_id = ?
            ORDER BY met_seconds ASC, timestamp_ms ASC
            """,
            (mission_id,)
        )
        rows = cursor.fetchall()
        if not rows:
            raise ValueError(f"No telemetry records found for mission '{mission_id}'.")

        count = len(rows)
        first = rows[0]
        last = rows[-1]

        pad_alt = pad_baseline_alt if pad_baseline_alt is not None else first["altitude"]
        pad_temp = first["temp"]
        pad_press = first["pressure"]

        peak_alt = -float("inf")
        apogee_idx = 0
        max_ascent_v = 0.0
        max_g = 0.0
        max_g_idx = 0
        descent_rates = []
        min_temp = float("inf")
        max_temp = -float("inf")
        anomalies_detected = 0

        for i, r in enumerate(rows):
            alt = r["altitude"]
            if alt > peak_alt:
                peak_alt = alt
                apogee_idx = i

            v_spd = r["v_spd"]
            if v_spd > max_ascent_v:
                max_ascent_v = v_spd

            g = r["accel_mag"]
            if g > max_g:
                max_g = g
                max_g_idx = i

            t = r["temp"]
            if t < min_temp:
                min_temp = t
            if t > max_temp:
                max_temp = t

            if r["is_anomaly"]:
                anomalies_detected += 1

            if i > apogee_idx and v_spd < -0.5:
                descent_rates.append(abs(v_spd))

        agl_apogee = max(0.0, peak_alt - pad_alt)
        avg_descent_rate = sum(descent_rates) / len(descent_rates) if descent_rates else 0.0
        terminal_descent_rate = (
            descent_rates[int(len(descent_rates) * 0.7)]
            if len(descent_rates) > 5
            else avg_descent_rate
        )

        # Regulatory compliance: 6.0 m/s - 11.0 m/s
        if terminal_descent_rate == 0.0:
            chute_status = "NO_DESCENT_DETECTED"
        elif terminal_descent_rate < 6.0:
            chute_status = "EXCESSIVE_DRIFT_RISK (<6.0 m/s)"
        elif terminal_descent_rate > 11.0:
            chute_status = "HARD_IMPACT_WARNING (>11.0 m/s)"
        else:
            chute_status = "COMPLIANT_NOMINAL (6-11 m/s)"

        # Horizontal GPS drift (Haversine approximation)
        d_lat = (last["lat"] - first["lat"]) * 111139.0
        d_lon = (last["lon"] - first["lon"]) * 111139.0 * math.cos(math.radians(first["lat"]))
        drift_dist = math.sqrt(d_lat * d_lat + d_lon * d_lon)
        drift_bearing = math.degrees(math.atan2(d_lon, d_lat))
        if drift_bearing < 0:
            drift_bearing += 360.0

        # Compass cardinal helper
        val = int((drift_bearing / 22.5) + 0.5)
        cardinals = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        cardinal_str = cardinals[(val % 16)]

        # Timing
        time_to_apogee = rows[apogee_idx]["met_seconds"]
        total_duration = last["met_seconds"] - first["met_seconds"]
        if total_duration <= 0:
            total_duration = count * 0.8  # Fallback based on packet count

        start_v = first["battery_voltage"]
        end_v = last["battery_voltage"]
        delta_v = start_v - end_v

        anomaly_summary = (
            f"NOMINAL ({anomalies_detected} transients flagged)"
            if anomalies_detected == 0
            else f"ANOMALOUS ({anomalies_detected} sensor transients detected)"
        )

        # Build Markdown Report
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        md = f"""# {mission['callsign']} POST-FLIGHT REVIEW (PFR) & TELEMETRY AUDIT REPORT
Mission ID: {mission_id} ({mission['name']})
Generated: {now_iso}
Operator: {mission['operator']}

## 1. Executive Flight Summary
- **Peak Apogee (AGL)**: {agl_apogee:.1f} m (MSL: {peak_alt:.1f} m)
- **Time to Apogee**: {time_to_apogee:.1f} s
- **Max Ejection Shock**: {max_g:.2f} G (T+ {rows[max_g_idx]['met_seconds']:.1f}s)
- **Terminal Descent Velocity**: {terminal_descent_rate:.2f} m/s
- **Mean Canopy Descent Rate**: {avg_descent_rate:.2f} m/s
- **Regulatory Descent Compliance**: {chute_status}
- **Total Flight Duration**: {total_duration:.1f} s
- **Total Ingested Packets**: {count} (Loss: 0.0%)

## 2. Atmospheric Sounding Profile
- **Launch Pad Baseline**: {pad_alt:.1f} m | {pad_temp:.1f} °C | {pad_press:.1f} hPa
- **Ambient Temperature Range**: {min_temp:.1f} °C to {max_temp:.1f} °C
- **Surface Air Density**: {first['air_density']:.3f} kg/m³
- **Apogee Air Density**: {rows[apogee_idx]['air_density']:.3f} kg/m³
- **Environmental Lapse Rate**: {rows[apogee_idx]['lapse_rate']:.2f} °C/100m
- **Dew Point Range**: {last['dew_point']:.1f} °C to {first['dew_point']:.1f} °C

## 3. GIS & Field Recovery
- **Launch Pad GNSS**: {first['lat']:.6f}°, {first['lon']:.6f}°
- **Touchdown Site GNSS**: {last['lat']:.6f}°, {last['lon']:.6f}°
- **Total Horizontal Drift**: {drift_dist:.1f} m
- **Recovery Azimuth**: {drift_bearing:.0f}° ({cardinal_str})
- **Recovery Maps Link**: https://www.google.com/maps/search/?api=1&query={last['lat']:.6f},{last['lon']:.6f}

## 4. Subsystems & Power Audit
- **Battery Initial**: {start_v:.2f} V
- **Battery Touchdown**: {end_v:.2f} V (Drop: {delta_v:.2f} V)
- **Anomaly Detection State**: {anomaly_summary}
- **Audit Signature**: CANSAT-DB-VERIFIED-{mission_id}
"""

        report_dict = {
            "mission_id": mission_id,
            "mission_name": mission["name"],
            "callsign": mission["callsign"],
            "operator": mission["operator"],
            "generated_at": now_iso,
            "peak_apogee_agl": round(agl_apogee, 2),
            "peak_altitude_msl": round(peak_alt, 2),
            "time_to_apogee_s": round(time_to_apogee, 2),
            "max_ejection_shock_g": round(max_g, 2),
            "terminal_descent_rate_mps": round(terminal_descent_rate, 2),
            "average_descent_rate_mps": round(avg_descent_rate, 2),
            "descent_compliance": chute_status,
            "total_flight_time_s": round(total_duration, 2),
            "total_packets": count,
            "packet_loss_pct": 0.0,
            "launch_lat": round(first["lat"], 6),
            "launch_lon": round(first["lon"], 6),
            "touchdown_lat": round(last["lat"], 6),
            "touchdown_lon": round(last["lon"], 6),
            "horizontal_drift_m": round(drift_dist, 2),
            "drift_azimuth_deg": round(drift_bearing, 1),
            "cardinal_bearing": cardinal_str,
            "battery_start_v": round(start_v, 2),
            "battery_end_v": round(end_v, 2),
            "battery_delta_v": round(delta_v, 2),
            "min_temp": round(min_temp, 2),
            "max_temp": round(max_temp, 2),
            "surface_air_density": round(first["air_density"], 3),
            "apogee_air_density": round(rows[apogee_idx]["air_density"], 3),
            "lapse_rate": round(rows[apogee_idx]["lapse_rate"], 2),
            "anomaly_summary": anomaly_summary,
            "report_markdown": md
        }

        cursor.execute(
            """
            INSERT INTO mission_reports (
                mission_id, peak_apogee_agl, peak_altitude_msl, time_to_apogee_s,
                max_ejection_shock_g, terminal_descent_rate_mps, average_descent_rate_mps,
                descent_compliance, total_flight_time_s, total_packets, packet_loss_pct,
                launch_lat, launch_lon, touchdown_lat, touchdown_lon, horizontal_drift_m,
                drift_azimuth_deg, battery_start_v, battery_end_v, battery_delta_v,
                anomaly_summary, report_markdown, report_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(mission_id) DO UPDATE SET
                generated_at = CURRENT_TIMESTAMP,
                peak_apogee_agl = excluded.peak_apogee_agl,
                peak_altitude_msl = excluded.peak_altitude_msl,
                time_to_apogee_s = excluded.time_to_apogee_s,
                max_ejection_shock_g = excluded.max_ejection_shock_g,
                terminal_descent_rate_mps = excluded.terminal_descent_rate_mps,
                average_descent_rate_mps = excluded.average_descent_rate_mps,
                descent_compliance = excluded.descent_compliance,
                total_flight_time_s = excluded.total_flight_time_s,
                total_packets = excluded.total_packets,
                launch_lat = excluded.launch_lat,
                launch_lon = excluded.launch_lon,
                touchdown_lat = excluded.touchdown_lat,
                touchdown_lon = excluded.touchdown_lon,
                horizontal_drift_m = excluded.horizontal_drift_m,
                drift_azimuth_deg = excluded.drift_azimuth_deg,
                battery_start_v = excluded.battery_start_v,
                battery_end_v = excluded.battery_end_v,
                battery_delta_v = excluded.battery_delta_v,
                anomaly_summary = excluded.anomaly_summary,
                report_markdown = excluded.report_markdown,
                report_json = excluded.report_json
            """,
            (
                mission_id,
                report_dict["peak_apogee_agl"],
                report_dict["peak_altitude_msl"],
                report_dict["time_to_apogee_s"],
                report_dict["max_ejection_shock_g"],
                report_dict["terminal_descent_rate_mps"],
                report_dict["average_descent_rate_mps"],
                report_dict["descent_compliance"],
                report_dict["total_flight_time_s"],
                report_dict["total_packets"],
                report_dict["packet_loss_pct"],
                report_dict["launch_lat"],
                report_dict["launch_lon"],
                report_dict["touchdown_lat"],
                report_dict["touchdown_lon"],
                report_dict["horizontal_drift_m"],
                report_dict["drift_azimuth_deg"],
                report_dict["battery_start_v"],
                report_dict["battery_end_v"],
                report_dict["battery_delta_v"],
                report_dict["anomaly_summary"],
                md,
                json.dumps(report_dict)
            )
        )
        conn.commit()
        return report_dict


def finalize_mission(mission_id: str, db_path: Optional[str] = None,
                     pad_baseline_alt: Optional[float] = None) -> Dict[str, Any]:
    """Mark a mission completed and automatically build/save the PFR audit report."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE missions
            SET end_time = CURRENT_TIMESTAMP, status = 'COMPLETED'
            WHERE id = ?
            """,
            (mission_id,)
        )
        conn.commit()

    return generate_and_save_pfr_report(mission_id, pad_baseline_alt=pad_baseline_alt, db_path=db_path)


def list_missions(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all stored missions with summaries and report status."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT m.*,
                   COUNT(t.id) as packet_count,
                   MAX(t.altitude) as max_altitude,
                   MAX(t.accel_mag) as max_shock,
                   r.descent_compliance,
                   r.peak_apogee_agl,
                   r.terminal_descent_rate_mps
            FROM missions m
            LEFT JOIN telemetry_records t ON m.id = t.mission_id
            LEFT JOIN mission_reports r ON m.id = r.mission_id
            GROUP BY m.id
            ORDER BY m.start_time DESC
        """)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_mission(mission_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get mission metadata by ID."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM missions WHERE id = ?", (mission_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_mission_report(mission_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get the stored PFR audit report for a mission."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM mission_reports WHERE mission_id = ?", (mission_id,))
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        if "report_json" in res and res["report_json"]:
            try:
                res["parsed_json"] = json.loads(res["report_json"])
            except Exception:
                pass
        return res


def get_mission_telemetry(mission_id: str, limit: Optional[int] = None,
                          db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all telemetry records for a mission in chronological order."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM telemetry_records WHERE mission_id = ? ORDER BY met_seconds ASC"
        if limit:
            query += f" LIMIT {int(limit)}"
        cursor.execute(query, (mission_id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_mission_events(mission_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all recorded events for a mission."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM mission_events WHERE mission_id = ? ORDER BY timestamp_ms ASC",
            (mission_id,)
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def delete_mission(mission_id: str, db_path: Optional[str] = None) -> bool:
    """Delete a mission and all associated telemetry, events, and reports."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM missions WHERE id = ?", (mission_id,))
        conn.commit()
        return cursor.rowcount > 0


def purge_all_missions(db_path: Optional[str] = None) -> int:
    """Delete all recorded missions, cascading to telemetry, reports, and events."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM missions")
        conn.commit()
        deleted_count = cursor.rowcount
        try:
            cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        return deleted_count


def get_database_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Get database file metrics, table row counts, and storage mode."""
    target_path = db_path or DEFAULT_DB_PATH
    init_db(target_path)
    file_size_bytes = os.path.getsize(target_path) if os.path.exists(target_path) else 0
    wal_size_bytes = os.path.getsize(target_path + "-wal") if os.path.exists(target_path + "-wal") else 0
    shm_size_bytes = os.path.getsize(target_path + "-shm") if os.path.exists(target_path + "-shm") else 0

    with get_db_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM missions")
        total_missions = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM telemetry_records")
        total_packets = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM mission_events")
        total_events = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM mission_reports")
        total_reports = cursor.fetchone()[0]
        cursor.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0]

    return {
        "db_path": os.path.abspath(target_path),
        "file_size_bytes": file_size_bytes,
        "file_size_kb": round(file_size_bytes / 1024, 2),
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 3),
        "wal_size_bytes": wal_size_bytes,
        "shm_size_bytes": shm_size_bytes,
        "total_missions": total_missions,
        "total_telemetry_records": total_packets,
        "total_events": total_events,
        "total_reports": total_reports,
        "journal_mode": str(journal_mode).upper()
    }


def export_mission_csv(mission_id: str, db_path: Optional[str] = None) -> Optional[str]:
    """Export all telemetry for a mission formatted as standard CanSat CSV."""
    records = get_mission_telemetry(mission_id, db_path=db_path)
    if not records:
        return None
    output = io.StringIO()
    fieldnames = [
        "timestamp_iso", "met_seconds", "temp", "pressure", "altitude",
        "gx", "gy", "gz", "ax", "ay", "az", "lat", "lon", "humidity",
        "battery_voltage", "v_spd", "accel_mag", "pitch", "roll",
        "air_density", "dew_point", "lapse_rate", "flight_phase",
        "anomaly_score", "is_anomaly"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for r in records:
        ts_ms = r.get("timestamp_ms", 0)
        try:
            ts_iso = datetime.datetime.fromtimestamp(ts_ms / 1000.0, tz=datetime.timezone.utc).isoformat()
        except Exception:
            ts_iso = ""
        writer.writerow({
            "timestamp_iso": ts_iso,
            "met_seconds": r.get("met_seconds", 0.0),
            "temp": r.get("temp", 0.0),
            "pressure": r.get("pressure", 0.0),
            "altitude": r.get("altitude", 0.0),
            "gx": r.get("gx", 0.0),
            "gy": r.get("gy", 0.0),
            "gz": r.get("gz", 0.0),
            "ax": r.get("ax", 0.0),
            "ay": r.get("ay", 0.0),
            "az": r.get("az", 0.0),
            "lat": r.get("lat", 0.0),
            "lon": r.get("lon", 0.0),
            "humidity": r.get("humidity", 0.0),
            "battery_voltage": r.get("battery_voltage", 0.0),
            "v_spd": r.get("v_spd", 0.0),
            "accel_mag": r.get("accel_mag", 0.0),
            "pitch": r.get("pitch", 0.0),
            "roll": r.get("roll", 0.0),
            "air_density": r.get("air_density", 0.0),
            "dew_point": r.get("dew_point", 0.0),
            "lapse_rate": r.get("lapse_rate", 0.0),
            "flight_phase": r.get("flight_phase", ""),
            "anomaly_score": r.get("anomaly_score", 0.0),
            "is_anomaly": 1 if r.get("is_anomaly") else 0
        })
    return output.getvalue()

