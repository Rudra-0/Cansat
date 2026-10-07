"""
API smoke tests for the FastAPI ground-station server (backend/app.py + backend/routers/).
Runs against an isolated temporary SQLite database via FastAPI's TestClient.
"""

import os
import tempfile
import unittest

_TMP_DIR = tempfile.mkdtemp(prefix="cansat_api_test_")
os.environ["CANSAT_DB_PATH"] = os.path.join(_TMP_DIR, "test_missions.db")
os.environ.setdefault("CANSAT_API_KEY", "test-key")

from fastapi.testclient import TestClient  # noqa: E402

from backend import app as server  # noqa: E402
from backend import ml_service  # noqa: E402
from backend.routers import hardware as hw  # noqa: E402

AUTH = {"X-API-Key": os.environ["CANSAT_API_KEY"]}

NOMINAL_PACKET = {
    "temp": 24.1, "pressure": 1001.5, "altitude": 100.0,
    "gx": 1.2, "gy": -0.4, "gz": 0.3,
    "ax": 0.02, "ay": 0.01, "az": 1.01,
    "humidity": 55.0, "batteryVoltage": 4.05, "vSpd": 4.2,
}


class TestApiSmoke(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(server.app)

    def test_pages_served(self):
        for path in ["/", "/index.html", "/dashboard.html", "/tinyml.html",
                     "/models.html", "/analysis.html", "/results.html"]:
            r = self.client.get(path)
            self.assertEqual(r.status_code, 200, path)
            self.assertIn("<html", r.text.lower(), path)

    def test_static_mounts(self):
        self.assertEqual(self.client.get("/css/resend-theme.css").status_code, 200)
        self.assertEqual(self.client.get("/js/cansat-3d.js").status_code, 200)
        self.assertEqual(self.client.get("/reports/all_missions_summary.csv").status_code, 200)

    def test_health_and_models(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["models_loaded"])
        r = self.client.get("/api/models/info")
        self.assertEqual(r.status_code, 200)
        self.assertIn("flight_phase_classifier", r.json()["models"])

    def test_predict(self):
        r = self.client.post("/api/predict", json=NOMINAL_PACKET)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "success")
        self.assertIn(body["predicted_phase"], ml_service.phase_classifier["classes"])
        self.assertGreaterEqual(body["anomaly_score"], 0.0)
        self.assertLessEqual(body["anomaly_score"], 1.0)

    def test_telemetry_websocket(self):
        with self.client.websocket_connect("/ws/telemetry") as ws:
            ws.send_json(NOMINAL_PACKET)
            self.assertEqual(ws.receive_json()["status"], "success")
            ws.send_text("not json")
            self.assertEqual(ws.receive_json()["status"], "error")

    def test_analysis_endpoints(self):
        r = self.client.get("/api/analysis/scenarios")
        self.assertEqual(r.status_code, 200)
        r = self.client.get("/api/analysis/scenario/01_nominal_sounding_flight")
        self.assertEqual(r.status_code, 200)
        self.assertGreater(r.json()["kpis"]["max_altitude_m"], 100)
        self.assertEqual(self.client.get("/api/analysis/scenario/does_not_exist").status_code, 404)
        self.assertEqual(self.client.get("/api/analysis/scenario/..").status_code, 404)
        r = self.client.get("/api/analysis/ablation-data")
        self.assertEqual(r.status_code, 200)
        r = self.client.post("/api/analysis/run-ablation", json={})
        self.assertEqual(r.status_code, 200)
        r = self.client.get("/api/analysis/reports-zip")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content[:2], b"PK")

    def test_mission_lifecycle(self):
        r = self.client.post("/api/db/missions/start", json={"name": "API Test"})
        self.assertEqual(r.status_code, 200)
        mid = r.json()["mission_id"]
        records = [dict(NOMINAL_PACKET, timestamp_ms=i * 500, altitude=100.0 + i) for i in range(10)]
        r = self.client.post("/api/db/missions/telemetry", json={"mission_id": mid, "records": records})
        self.assertEqual(r.json()["inserted"], 10)
        r = self.client.post(f"/api/db/missions/{mid}/events",
                             json={"timestamp_ms": 1000, "event_type": "TEST", "description": "smoke"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.post(f"/api/db/missions/{mid}/finish").status_code, 200)
        self.assertEqual(self.client.get(f"/api/db/missions/{mid}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/db/missions/{mid}/report").status_code, 200)
        r = self.client.get(f"/api/db/missions/{mid}/telemetry")
        self.assertEqual(r.json()["count"], 10)
        r = self.client.get(f"/api/db/missions/{mid}/export/csv")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.get("/api/db/missions").status_code, 200)
        self.assertEqual(self.client.get("/api/db/stats").status_code, 200)
        r = self.client.delete(f"/api/db/missions/{mid}", headers=AUTH)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.get(f"/api/db/missions/{mid}").status_code, 404)

    def test_finish_honours_pad_baseline(self):
        mid = self.client.post("/api/db/missions/start", json={"name": "Pad Test"}).json()["mission_id"]
        records = [dict(NOMINAL_PACKET, timestamp_ms=i * 500, altitude=100.0 + 10 * i) for i in range(10)]
        self.client.post("/api/db/missions/telemetry", json={"mission_id": mid, "records": records})
        default = self.client.post(f"/api/db/missions/{mid}/finish").json()["report"]
        self.assertAlmostEqual(default["peak_apogee_agl"], 90.0)
        custom = self.client.post(f"/api/db/missions/{mid}/finish", json={"pad_baseline_alt": 50.0}).json()["report"]
        self.assertAlmostEqual(custom["peak_apogee_agl"], 140.0)

    def test_operator_endpoints_require_key(self):
        for method, path in [("delete", "/api/db/missions"), ("delete", "/api/db/missions/x"),
                             ("get", "/api/db/download")]:
            r = getattr(self.client, method)(path)
            self.assertEqual(r.status_code, 401, path)
            r = getattr(self.client, method)(path, headers={"X-API-Key": "wrong"})
            self.assertEqual(r.status_code, 401, path)
        self.assertEqual(self.client.post("/hardware/config", json={}).status_code, 401)
        r = self.client.get("/api/db/download", params={"api_key": AUTH["X-API-Key"]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.delete("/api/db/missions/nope", headers=AUTH).status_code, 404)

    def test_operator_endpoints_localhost_only_without_key(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CANSAT_API_KEY")
            # TestClient's peer address is "testclient", i.e. not this machine.
            self.assertEqual(self.client.get("/api/db/download").status_code, 403)
            local = TestClient(server.app, client=("127.0.0.1", 50000))
            self.assertEqual(local.get("/api/db/download").status_code, 200)

    def test_cors_policy(self):
        def preflight(origin):
            return self.client.options("/api/health", headers={
                "Origin": origin, "Access-Control-Request-Method": "GET"})
        for origin in ["http://localhost:5500", "http://127.0.0.1:8000", "null"]:
            r = preflight(origin)
            self.assertEqual(r.headers.get("access-control-allow-origin"), origin, origin)
            self.assertNotIn("access-control-allow-credentials", r.headers)
        r = preflight("https://evil.example.com")
        self.assertNotIn("access-control-allow-origin", r.headers)

    def test_scenario_name_validation(self):
        for bad in ["..%2F..%2Freports%2Fall_missions_summary", "%2E%2E", "README", "01_nominal_sounding_flight.csv.bak"]:
            self.assertEqual(self.client.get(f"/api/analysis/scenario/{bad}").status_code, 404, bad)
        self.assertEqual(self.client.get("/api/analysis/scenario/01_nominal_sounding_flight.csv").status_code, 200)

    def test_hardware_endpoints(self):
        self.assertEqual(self.client.get("/hardware/ports").status_code, 200)
        self.assertEqual(self.client.get("/hardware/tare").status_code, 200)
        self.assertEqual(self.client.get("/analytics/kinematics").status_code, 200)

    def test_hardware_disabled_mode(self):
        from unittest import mock
        from starlette.websockets import WebSocketDisconnect
        with mock.patch.dict(os.environ, {"CANSAT_HARDWARE": "off"}):
            self.assertFalse(hw.hardware_bridge_enabled())
            self.assertEqual(self.client.get("/hardware/ports").json(), {"ports": [], "hardware_enabled": False})
            self.assertEqual(self.client.get("/hardware/config").json()["status"], "disabled")
            self.assertEqual(self.client.post("/hardware/config", json={}, headers=AUTH).status_code, 409)
            with self.client.websocket_connect("/ws/serial") as ws:
                self.assertIn("Serial bridge disabled", ws.receive_text())
                with self.assertRaises(WebSocketDisconnect) as ctx:
                    ws.receive_text()
                self.assertEqual(ctx.exception.code, hw.WS_CLOSE_NO_HARDWARE)
        self.assertIsNone(hw.dual_serial_manager)

    def test_hardware_enabled_bridge(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {"CANSAT_HARDWARE": "on",
                                          "CANSAT_TELEMETRY_PORT": "/dev/nonexistent0",
                                          "CANSAT_VIDEO_PORT": "/dev/nonexistent1"}):
            self.assertTrue(hw.hardware_bridge_enabled())
            with self.client.websocket_connect("/ws/serial") as ws:
                self.assertIn("Active dual bridge", ws.receive_text())
                self.assertTrue(hw.dual_serial_manager.is_running)
        hw.dual_serial_manager.stop()
        hw.dual_serial_manager = None

    def test_serial_csv_processor(self):
        line = "24.1,1001.5,100.0,1.2,-0.4,0.3,0.02,0.01,1.01,22.57,88.36,55.0,4.05"
        before = len(ml_service.live_telemetry_history)
        hw.on_telemetry_csv_processor(line)
        self.assertGreaterEqual(len(ml_service.live_telemetry_history), min(before + 1, 20))
        self.assertEqual(self.client.get("/analytics/kinematics").json()["status"], "active")

    def test_malformed_serial_line_is_logged(self):
        with self.assertLogs("cansat.server", level="WARNING") as logs:
            hw.on_telemetry_csv_processor("24.1,abc,100,1,1,1,0,0,1,22,88,55,4.0")
        self.assertIn("malformed telemetry line", logs.output[0])

    def test_dead_serial_subscriber_is_dropped(self):
        import asyncio

        class DeadSocket:
            async def send_text(self, _):
                raise RuntimeError("socket closed")

        class LiveSocket:
            def __init__(self):
                self.sent = []

            async def send_text(self, line):
                self.sent.append(line)

        dead, live = DeadSocket(), LiveSocket()
        hw.serial_subscribers[:] = [dead, live]
        try:
            asyncio.run(hw.broadcast_serial_line("hello"))
            self.assertEqual(live.sent, ["hello"])
            self.assertEqual(hw.serial_subscribers, [live])
        finally:
            hw.serial_subscribers.clear()


if __name__ == "__main__":
    unittest.main()
