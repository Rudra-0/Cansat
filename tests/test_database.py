"""
Unit test suite for Cognitive CanSat Database & Post-Flight Review Report Engine
Tests SQLite WAL initialization, batch telemetry ingestion, event logging, and PFR audit persistence.
"""

import os
import unittest
import tempfile
from pathlib import Path

from backend.core.database import (
    init_db,
    get_db_connection,
    create_mission,
    insert_telemetry_batch,
    insert_event,
    finalize_mission,
    generate_and_save_pfr_report,
    get_mission,
    list_missions,
    get_mission_report,
    get_mission_telemetry,
    get_mission_events,
    delete_mission,
    get_database_stats,
    purge_all_missions,
    export_mission_csv
)


class TestCanSatDatabase(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.test_dir.name, "test_cansat.db")
        init_db(self.db_path)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_wal_mode_and_tables(self):
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode;")
            mode = cursor.fetchone()[0]
            self.assertEqual(mode.lower(), "wal")

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r[0] for r in cursor.fetchall()]
            self.assertIn("missions", tables)
            self.assertIn("telemetry_records", tables)
            self.assertIn("mission_reports", tables)
            self.assertIn("mission_events", tables)

    def test_mission_lifecycle_and_pfr_report(self):
        # 1. Create Mission
        m = create_mission(
            name="Autonomous Sounding Test Alpha",
            callsign="CANSAT-TEST-01",
            operator="Rishi Flight Ops",
            notes="Testing EKF fusion and PFR database persistence",
            db_path=self.db_path
        )
        mission_id = m["id"]
        self.assertEqual(m["status"], "ACTIVE_FLIGHT")
        self.assertEqual(m["callsign"], "CANSAT-TEST-01")

        # 2. Log Initial Event
        evt_id = insert_event(
            mission_id=mission_id,
            timestamp_ms=0,
            event_type="MISSION_ARMED",
            severity="INFO",
            description="Ground station armed for launch pad static test",
            db_path=self.db_path
        )
        self.assertGreater(evt_id, 0)

        # 3. Simulate Flight Telemetry Sequence (Pad -> Ascent -> Apogee -> Descent -> Touchdown)
        records = []
        # Pad Idle (0 - 5s)
        for i in range(6):
            records.append({
                "timestamp_ms": i * 1000,
                "met_seconds": float(i),
                "altitude": 200.0 + (i * 0.05),
                "pressure": 990.0,
                "temp": 24.0,
                "humidity": 45.0,
                "battery_voltage": 4.18 - (i * 0.002),
                "ax": 0.01, "ay": 0.02, "az": 1.0,
                "gx": 0.1, "gy": -0.1, "gz": 0.05,
                "lat": 28.613939, "lon": 77.209021,
                "v_spd": 0.0,
                "accel_mag": 1.0,
                "air_density": 1.205,
                "dew_point": 11.5,
                "lapse_rate": 0.65,
                "flight_phase": "PAD_IDLE"
            })

        # Balloon Ascent (6 - 40s) up to 850m
        for i in range(6, 41):
            t = float(i)
            alt = 200.0 + (t - 5.0) * 18.5
            records.append({
                "timestamp_ms": i * 1000,
                "met_seconds": t,
                "altitude": alt,
                "pressure": 990.0 - ((alt - 200.0) * 0.11),
                "temp": 24.0 - ((alt - 200.0) * 0.0065),
                "humidity": 45.0 + ((alt - 200.0) * 0.02),
                "battery_voltage": 4.15 - (i * 0.003),
                "ax": 0.05, "ay": 0.08, "az": 1.35,
                "gx": 0.5, "gy": -0.4, "gz": 0.2,
                "lat": 28.613939 + (i * 0.00001),
                "lon": 77.209021 + (i * 0.00002),
                "v_spd": 18.5,
                "accel_mag": 1.38,
                "air_density": 1.15,
                "dew_point": 9.2,
                "lapse_rate": 0.65,
                "flight_phase": "BALLOON_ASCENT"
            })

        # Apogee Separation & Shock (41 - 43s) Peak 860m, 12.5G shock
        for i in range(41, 44):
            records.append({
                "timestamp_ms": i * 1000,
                "met_seconds": float(i),
                "altitude": 860.0 - (i - 41) * 2.0,
                "pressure": 915.0,
                "temp": 19.7,
                "humidity": 58.0,
                "battery_voltage": 4.02,
                "ax": 4.5, "ay": 6.2, "az": 9.8,
                "gx": 25.0, "gy": -18.0, "gz": 15.0,
                "lat": 28.6143, "lon": 77.2098,
                "v_spd": 0.5 - (i - 41) * 3.0,
                "accel_mag": 12.5,
                "air_density": 1.12,
                "dew_point": 8.5,
                "lapse_rate": 0.65,
                "flight_phase": "APOGEE_BURST"
            })

        # Parachute Descent (44 - 120s) steady 8.2 m/s descent
        for i in range(44, 121):
            t = float(i)
            alt = max(200.0, 854.0 - (t - 43.0) * 8.2)
            records.append({
                "timestamp_ms": i * 1000,
                "met_seconds": t,
                "altitude": alt,
                "pressure": 915.0 + ((860.0 - alt) * 0.11),
                "temp": 19.7 + ((860.0 - alt) * 0.0065),
                "humidity": 58.0,
                "battery_voltage": 3.95 - (i * 0.002),
                "ax": 0.05, "ay": 0.06, "az": 1.02,
                "gx": 2.1, "gy": -1.8, "gz": 0.9,
                "lat": 28.6143 + (i * 0.000015),
                "lon": 77.2098 + (i * 0.00003),
                "v_spd": -8.2,
                "accel_mag": 1.05,
                "air_density": 1.18,
                "dew_point": 10.1,
                "lapse_rate": 0.65,
                "flight_phase": "PARACHUTE_DESCENT"
            })

        # Touchdown (121 - 125s)
        for i in range(121, 126):
            records.append({
                "timestamp_ms": i * 1000,
                "met_seconds": float(i),
                "altitude": 200.0,
                "pressure": 990.0,
                "temp": 24.1,
                "humidity": 46.0,
                "battery_voltage": 3.75,
                "ax": 0.01, "ay": 0.01, "az": 1.0,
                "gx": 0.0, "gy": 0.0, "gz": 0.0,
                "lat": 28.6162, "lon": 77.2135,
                "v_spd": 0.0,
                "accel_mag": 1.0,
                "air_density": 1.205,
                "dew_point": 11.6,
                "lapse_rate": 0.65,
                "flight_phase": "TOUCHDOWN_RECOVERY"
            })

        # 4. Batch Insert
        inserted = insert_telemetry_batch(mission_id, records, db_path=self.db_path)
        self.assertEqual(inserted, len(records))

        # 5. Finalize Mission and Build Report
        report = finalize_mission(mission_id, db_path=self.db_path)

        # 6. Verify Calculated PFR Metrics
        self.assertEqual(report["mission_id"], mission_id)
        self.assertAlmostEqual(report["peak_altitude_msl"], 860.0, delta=0.5)
        self.assertAlmostEqual(report["peak_apogee_agl"], 660.0, delta=1.0)
        self.assertEqual(report["time_to_apogee_s"], 41.0)
        self.assertAlmostEqual(report["max_ejection_shock_g"], 12.5, delta=0.1)
        self.assertAlmostEqual(report["terminal_descent_rate_mps"], 8.2, delta=0.2)
        self.assertEqual(report["descent_compliance"], "COMPLIANT_NOMINAL (6-11 m/s)")
        self.assertEqual(report["total_packets"], len(records))
        self.assertGreater(report["horizontal_drift_m"], 100.0)
        self.assertIn("POST-FLIGHT REVIEW", report["report_markdown"])

        # 7. Query Mission and Report from DB
        stored_m = get_mission(mission_id, db_path=self.db_path)
        self.assertEqual(stored_m["status"], "COMPLETED")

        stored_report = get_mission_report(mission_id, db_path=self.db_path)
        self.assertIsNotNone(stored_report)
        self.assertEqual(stored_report["mission_id"], mission_id)
        self.assertEqual(stored_report["descent_compliance"], "COMPLIANT_NOMINAL (6-11 m/s)")
        self.assertIn("parsed_json", stored_report)

        # 8. Query Telemetry
        telemetry = get_mission_telemetry(mission_id, db_path=self.db_path)
        self.assertEqual(len(telemetry), len(records))
        self.assertEqual(telemetry[0]["flight_phase"], "PAD_IDLE")

        # 9. Query Events
        events = get_mission_events(mission_id, db_path=self.db_path)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "MISSION_ARMED")

        # 10. Test Mission Listing
        all_missions = list_missions(db_path=self.db_path)
        self.assertEqual(len(all_missions), 1)
        self.assertEqual(all_missions[0]["packet_count"], len(records))
        self.assertAlmostEqual(all_missions[0]["max_altitude"], 860.0, delta=0.5)

        # 11. Test Delete Mission (Cascades)
        deleted = delete_mission(mission_id, db_path=self.db_path)
        self.assertTrue(deleted)
        self.assertIsNone(get_mission(mission_id, db_path=self.db_path))
        self.assertIsNone(get_mission_report(mission_id, db_path=self.db_path))
        self.assertEqual(len(get_mission_telemetry(mission_id, db_path=self.db_path)), 0)

    def test_database_stats_csv_export_and_purge(self):
        # Create 2 test missions
        m1 = create_mission(name="Mission One", callsign="CANSAT-01", db_path=self.db_path)
        m2 = create_mission(name="Mission Two", callsign="CANSAT-02", db_path=self.db_path)
        
        insert_telemetry_batch(m1["id"], [
            {
                "timestamp_ms": 1000, "met_seconds": 1.0, "altitude": 150.0,
                "pressure": 1000.0, "temp": 20.0, "humidity": 50.0, "battery_voltage": 4.1,
                "ax": 0.0, "ay": 0.0, "az": 1.0, "gx": 0.0, "gy": 0.0, "gz": 0.0,
                "lat": 10.0, "lon": 20.0, "v_spd": 5.0, "accel_mag": 1.0, "pitch": 0.0, "roll": 0.0,
                "air_density": 1.2, "dew_point": 10.0, "lapse_rate": 0.65, "flight_phase": "ASCENT"
            }
        ], db_path=self.db_path)

        # Test CSV Export
        csv_str = export_mission_csv(m1["id"], db_path=self.db_path)
        self.assertIsNotNone(csv_str)
        self.assertIn("timestamp_iso", csv_str)
        self.assertIn("met_seconds", csv_str)
        self.assertIn("150.0", csv_str)

        # Non-existent mission CSV should return None
        self.assertIsNone(export_mission_csv("MSN-NON-EXISTENT", db_path=self.db_path))

        # Test Stats
        stats = get_database_stats(db_path=self.db_path)
        self.assertEqual(stats["total_missions"], 2)
        self.assertEqual(stats["total_telemetry_records"], 1)
        self.assertEqual(stats["journal_mode"], "WAL")
        self.assertGreater(stats["file_size_bytes"], 0)

        # Test Purge
        purged = purge_all_missions(db_path=self.db_path)
        self.assertEqual(purged, 2)
        stats_after = get_database_stats(db_path=self.db_path)
        self.assertEqual(stats_after["total_missions"], 0)
        self.assertEqual(stats_after["total_telemetry_records"], 0)

    def test_api_db_endpoints(self):
        try:
            from fastapi.testclient import TestClient
            from backend.app import app
            client = TestClient(app)
            use_client = True
        except (ImportError, RuntimeError):
            try:
                from backend.app import (
                    start_db_mission, get_db_stats, ingest_mission_telemetry,
                    export_mission_telemetry_csv, delete_db_mission,
                    MissionStartRequest, TelemetryBatchRequest
                )
                use_client = False
            except (ImportError, RuntimeError) as e:
                self.skipTest(f"FastAPI or backend dependencies not installed: {e}")
                return

        if use_client:
            # Operator endpoints (download/delete) require the API key when one is configured.
            os.environ.setdefault("CANSAT_API_KEY", "test-key")
            auth = {"X-API-Key": os.environ["CANSAT_API_KEY"]}

            # 1. Stats
            resp = client.get("/api/db/stats")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("stats", resp.json())

            # 2. Download
            resp = client.get("/api/db/download", headers=auth)
            self.assertEqual(resp.status_code, 200)
            self.assertIn("application/x-sqlite3", resp.headers["content-type"])

            # 3. Create, Ingest, CSV Export & Delete
            s_resp = client.post("/api/db/missions/start", json={"name": "API Test Flight"})
            self.assertEqual(s_resp.status_code, 200)
            m_id = s_resp.json()["mission_id"]

            t_resp = client.post("/api/db/missions/telemetry", json={
                "mission_id": m_id,
                "records": [{
                    "timestamp_ms": 1000, "met_seconds": 0.1, "altitude": 75.0,
                    "pressure": 1010.0, "temp": 21.0, "humidity": 45.0, "battery_voltage": 4.15,
                    "ax": 0.0, "ay": 0.0, "az": 1.0, "gx": 0.0, "gy": 0.0, "gz": 0.0,
                    "lat": 12.9, "lon": 77.5, "v_spd": 8.0, "accel_mag": 1.0,
                    "pitch": 0.0, "roll": 0.0, "air_density": 1.2, "dew_point": 10.0,
                    "lapse_rate": 0.65, "flight_phase": "ASCENT"
                }]
            })
            self.assertEqual(t_resp.status_code, 200)

            csv_resp = client.get(f"/api/db/missions/{m_id}/export/csv")
            self.assertEqual(csv_resp.status_code, 200)
            self.assertIn("text/csv", csv_resp.headers["content-type"])
            self.assertIn("75.0", csv_resp.text)

            d_resp = client.delete(f"/api/db/missions/{m_id}", headers=auth)
            self.assertEqual(d_resp.status_code, 200)
            self.assertTrue(d_resp.json()["deleted"])
        else:
            # 1. Stats
            stats_resp = get_db_stats()
            self.assertEqual(stats_resp["status"], "success")
            self.assertIn("stats", stats_resp)

            # 2. Create, Ingest, CSV Export & Delete
            s_resp = start_db_mission(MissionStartRequest(name="API Test Flight"))
            self.assertEqual(s_resp["status"], "success")
            m_id = s_resp["mission_id"]

            t_resp = ingest_mission_telemetry(TelemetryBatchRequest(
                mission_id=m_id,
                records=[{
                    "timestamp_ms": 1000, "met_seconds": 0.1, "altitude": 75.0,
                    "pressure": 1010.0, "temp": 21.0, "humidity": 45.0, "battery_voltage": 4.15,
                    "ax": 0.0, "ay": 0.0, "az": 1.0, "gx": 0.0, "gy": 0.0, "gz": 0.0,
                    "lat": 12.9, "lon": 77.5, "v_spd": 8.0, "accel_mag": 1.0,
                    "pitch": 0.0, "roll": 0.0, "air_density": 1.2, "dew_point": 10.0,
                    "lapse_rate": 0.65, "flight_phase": "ASCENT"
                }]
            ))
            self.assertEqual(t_resp["status"], "success")

            csv_resp = export_mission_telemetry_csv(m_id)
            self.assertEqual(csv_resp.media_type, "text/csv")
            self.assertIn("75.0", csv_resp.body.decode("utf-8"))

            d_resp = delete_db_mission(m_id)
            self.assertEqual(d_resp["status"], "success")
            self.assertTrue(d_resp["deleted"])
 
    def test_telemetry_batch_safe_parsing_and_camel_case(self):
        """Verify insert_telemetry_batch handles None values, NaN, and camelCase keys gracefully."""
        m = create_mission(
            name="Safe Parsing Test Flight",
            callsign="SAFE-01",
            db_path=self.db_path
        )
        mission_id = m["id"]

        records = [
            {
                # Record with None values (e.g. before GPS lock)
                "timestamp_ms": None,
                "met_seconds": None,
                "altitude": 150.5,
                "pressure": 1005.2,
                "temp": 22.0,
                "humidity": None,
                "battery_voltage": None,
                "ax": None, "ay": None, "az": None,
                "gx": None, "gy": None, "gz": None,
                "lat": None, "lon": None,
                "v_spd": None,
                "accel_mag": None,
                "pitch": None, "roll": None,
                "air_density": None, "dew_point": None, "lapse_rate": None,
                "flight_phase": None,
                "anomaly_score": None,
                "is_anomaly": None
            },
            {
                # Record using camelCase keys (common from JS frontend)
                "timestamp_ms": 2000,
                "met_seconds": 1.0,
                "altitude": 160.0,
                "pressure": 1004.0,
                "temp": 21.8,
                "humidity": 55.0,
                "batteryVoltage": 4.12,
                "ax": 0.05, "ay": 0.02, "az": 1.01,
                "gx": 0.1, "gy": 0.0, "gz": -0.1,
                "lat": 28.6139, "lon": 77.2090,
                "vSpd": 9.5,
                "accelMag": 1.02,
                "pitch": 1.5, "roll": -0.8,
                "airDensity": 1.21, "dewPoint": 12.0, "lapseRate": 0.65,
                "flightPhase": "ASCENT",
                "anomalyScore": 0.05,
                "isAnomaly": False
            }
        ]

        inserted = insert_telemetry_batch(mission_id, records, db_path=self.db_path)
        self.assertEqual(inserted, 2)

        stored = get_mission_telemetry(mission_id, db_path=self.db_path)
        self.assertEqual(len(stored), 2)
        # Verify defaults were applied for None values
        self.assertEqual(stored[0]["lat"], 0.0)
        self.assertEqual(stored[0]["lon"], 0.0)
        self.assertEqual(stored[0]["battery_voltage"], 4.2)
        self.assertEqual(stored[0]["flight_phase"], "PAD_IDLE")
        # Verify camelCase was mapped correctly
        self.assertAlmostEqual(stored[1]["battery_voltage"], 4.12, places=2)
        self.assertAlmostEqual(stored[1]["v_spd"], 9.5, places=2)
        self.assertEqual(stored[1]["flight_phase"], "ASCENT")


if __name__ == "__main__":
    unittest.main()
