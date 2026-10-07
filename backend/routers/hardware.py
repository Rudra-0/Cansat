"""
Real-time dual-port hardware serial bridge (LoRa telemetry + ESP32-CAM feed),
pad tare calibration and the live kinematics / sounding engines.
"""

import os
import time
import asyncio
import logging
import threading
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.core.serial_manager import DualSerialManager, list_available_ports, HAS_SERIAL
from backend.core.kinematics import KinematicsEngine, KinematicState
from backend.core.atmospheric import AtmosphericEngine, AtmosphericSounding
from backend.ml_service import live_telemetry_history
from backend.security import require_operator

logger = logging.getLogger("cansat.server.hardware")

router = APIRouter(tags=["hardware"])


# Close code sent on /ws/serial when this host has no serial hardware bridge.
WS_CLOSE_NO_HARDWARE = 4404


def hardware_bridge_enabled() -> bool:
    """CANSAT_HARDWARE=off disables the server-side serial bridge (cloud / Docker hosts)."""
    mode = os.environ.get("CANSAT_HARDWARE", "on").strip().lower()
    return HAS_SERIAL and mode not in ("off", "0", "false", "no")

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

@router.get("/hardware/ports")
def get_hardware_ports():
    """Enumerate all serial ports on the host system."""
    if not hardware_bridge_enabled():
        return {"ports": [], "hardware_enabled": False}
    return {"ports": list_available_ports(), "hardware_enabled": True}

@router.get("/hardware/config")
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

@router.post("/hardware/config", dependencies=[Depends(require_operator)])
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

@router.get("/analytics/kinematics")
def get_latest_kinematics():
    """Return latest real-time Kinematics state (Kalman altitude, attitude, alarms)."""
    with state_lock:
        if latest_kinematic_state is None:
            return {"status": "waiting_for_data"}
        return {"status": "active", "data": latest_kinematic_state.__dict__}

@router.get("/analytics/sounding")
def get_latest_sounding():
    """Return latest Atmospheric Thermodynamics state (dew point, air density, ELR)."""
    with state_lock:
        if latest_sounding_state is None:
            return {"status": "waiting_for_data"}
        return {"status": "active", "data": latest_sounding_state.__dict__}

@router.post("/hardware/tare")
def trigger_hardware_tare(samples: int = 15):
    """Trigger stationary pad tare calibration across baro and IMU sensors."""
    kinematics_engine.start_tare(num_samples=samples)
    return {
        "status": "taring_started",
        "samples_target": samples,
        "is_taring": kinematics_engine.is_taring
    }

@router.get("/hardware/tare")
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

@router.websocket("/ws/serial")
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
