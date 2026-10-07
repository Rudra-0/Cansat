"""
Cognitive CanSat - Dual-Port Serial Manager
-------------------------------------------
Thread-safe, asynchronous multi-port serial communication manager for:
  - Port 1: Primary Flight Telemetry (LoRa transceiver @ 9600 baud)
  - Port 2: Airborne Video Stream (ESP32-CAM via ESP-NOW receiver @ 460800 baud)
Features automatic port discovery, exponential reconnect backoff,
thread lifecycle control, and real-time observer callback dispatching.
"""

import time
import threading
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Dict

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    serial = None
    HAS_SERIAL = False

logger = logging.getLogger("DualSerialManager")


@dataclass
class SerialPortConfig:
    port: str
    baud_rate: int
    label: str
    timeout: float = 1.0
    reconnect_interval: float = 2.0
    is_video: bool = False


def list_available_ports() -> List[Dict[str, str]]:
    """Enumerate all physical and virtual serial ports on host OS."""
    if not HAS_SERIAL:
        return []
    results = []
    for p in serial.tools.list_ports.comports():
        results.append({
            "device": p.device,
            "description": p.description or "Unknown",
            "hwid": p.hwid or "N/A"
        })
    return results


class DualSerialManager:
    """
    Manages concurrent, non-blocking serial communication across dual hardware ports.
    """

    def __init__(
        self,
        telemetry_port: str = "COM4",
        telemetry_baud: int = 9600,
        video_port: str = "COM5",
        video_baud: int = 460800,
        on_line_received: Optional[Callable[[str, str], None]] = None,
        on_camera_frame: Optional[Callable[[int, int, int, str], None]] = None,
        on_telemetry_csv: Optional[Callable[[str], None]] = None,
        on_status_change: Optional[Callable[[str, bool], None]] = None,
    ):
        self.telemetry_cfg = SerialPortConfig(
            port=telemetry_port,
            baud_rate=telemetry_baud,
            label="CanSat LoRa Telemetry",
            is_video=False,
        )
        self.video_cfg = SerialPortConfig(
            port=video_port,
            baud_rate=video_baud,
            label="ESP32 Aerial Video Feed",
            is_video=True,
        )

        self.on_line_received = on_line_received
        self.on_camera_frame = on_camera_frame
        self.on_telemetry_csv = on_telemetry_csv
        self.on_status_change = on_status_change

        self._running = False
        self._lock = threading.Lock()
        self._threads: List[threading.Thread] = []
        self._connection_states: Dict[str, bool] = {
            self.telemetry_cfg.label: False,
            self.video_cfg.label: False,
        }

    @property
    def is_running(self) -> bool:
        return self._running

    def get_connection_status(self) -> Dict[str, bool]:
        with self._lock:
            return dict(self._connection_states)

    def reconfigure(
        self,
        telemetry_port: Optional[str] = None,
        telemetry_baud: Optional[int] = None,
        video_port: Optional[str] = None,
        video_baud: Optional[int] = None,
    ) -> None:
        """Dynamically update port/baud settings and restart active workers."""
        was_running = self._running
        if was_running:
            self.stop()

        with self._lock:
            if telemetry_port is not None:
                self.telemetry_cfg.port = telemetry_port
            if telemetry_baud is not None:
                self.telemetry_cfg.baud_rate = telemetry_baud
            if video_port is not None:
                self.video_cfg.port = video_port
            if video_baud is not None:
                self.video_cfg.baud_rate = video_baud

        if was_running:
            self.start()

    def _resolve_port(self, requested_port: str, is_video: bool) -> str:
        """
        Intelligently resolve requested port against available hardware ports.
        If requested port exists, return it.
        If COM3/COM4/AUTO is requested, auto-fallback between available non-video ports.
        """
        if not HAS_SERIAL:
            return requested_port

        available = [p.device.upper() for p in serial.tools.list_ports.comports()]
        if not available:
            return requested_port

        if requested_port.upper() in available:
            return requested_port

        # Auto-fallback between COM4 and COM3 or when AUTO is requested
        if requested_port.upper() in ("COM3", "COM4", "AUTO"):
            other_port = (self.video_cfg.port if not is_video else self.telemetry_cfg.port).upper()
            candidates = [p for p in available if p != other_port]
            if candidates:
                if requested_port.upper() in candidates:
                    return requested_port.upper()
                if "COM4" in candidates:
                    return "COM4"
                if "COM3" in candidates:
                    return "COM3"
                return candidates[0]

        return requested_port

    def start(self) -> None:
        """Start background listener threads for both ports."""
        with self._lock:
            if self._running:
                return
            self._running = True

            t1 = threading.Thread(
                target=self._worker_loop,
                args=(self.telemetry_cfg,),
                name="SerialWorker-Telemetry",
                daemon=True,
            )
            t2 = threading.Thread(
                target=self._worker_loop,
                args=(self.video_cfg,),
                name="SerialWorker-Video",
                daemon=True,
            )
            self._threads = [t1, t2]
            t1.start()
            t2.start()
            logger.info(
                f"[DualSerialManager] Worker threads started. "
                f"Telemetry: {self.telemetry_cfg.port} @ {self.telemetry_cfg.baud_rate} baud. "
                f"Video: {self.video_cfg.port} @ {self.video_cfg.baud_rate} baud."
            )

    def stop(self) -> None:
        """Gracefully terminate all background workers."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        for t in self._threads:
            if t.is_alive():
                t.join(timeout=1.5)
        self._threads.clear()

        with self._lock:
            for k in self._connection_states:
                self._connection_states[k] = False

        logger.info("[DualSerialManager] All serial listener workers terminated cleanly.")

    def _update_status(self, label: str, connected: bool) -> None:
        with self._lock:
            prev = self._connection_states.get(label, False)
            self._connection_states[label] = connected
        if prev != connected and self.on_status_change:
            try:
                self.on_status_change(label, connected)
            except Exception as e:
                logger.error(f"Error in on_status_change callback: {e}")

    def _worker_loop(self, cfg: SerialPortConfig) -> None:
        """Worker loop for an individual serial port with auto-reconnect resilience."""
        while self._running:
            ser = None
            try:
                if not HAS_SERIAL:
                    time.sleep(cfg.reconnect_interval)
                    continue

                active_port = self._resolve_port(cfg.port, cfg.is_video)
                logger.info(f"[{cfg.label}] Opening {active_port} @ {cfg.baud_rate} baud...")
                ser = serial.Serial(active_port, cfg.baud_rate, timeout=cfg.timeout)
                ser.dtr = True
                ser.rts = True
                self._update_status(cfg.label, True)
                logger.info(f"[{cfg.label}] [+] Successfully connected to {active_port} @ {cfg.baud_rate}.")

                while self._running:
                    if ser.in_waiting:
                        raw = ser.readline()
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if line:
                            self._dispatch_line(cfg, line)
                    else:
                        time.sleep(0.005 if cfg.is_video else 0.020)

            except Exception as e:
                logger.debug(f"[{cfg.label}] Serial read/connect error on {cfg.port}: {e}")
                self._update_status(cfg.label, False)
                time.sleep(cfg.reconnect_interval)
            finally:
                if ser:
                    try:
                        ser.close()
                    except Exception:
                        pass
                self._update_status(cfg.label, False)

        logger.info(f"[{cfg.label}] Listener on {cfg.port} cleanly stopped.")

    def _dispatch_line(self, cfg: SerialPortConfig, line: str) -> None:
        """Parse and dispatch incoming line to registered subscribers."""
        if self.on_line_received:
            try:
                self.on_line_received(cfg.label, line)
            except Exception as e:
                logger.error(f"Error in on_line_received callback: {e}")

        if line.startswith("$CAM_FRAME"):
            parts = line.split(",")
            if len(parts) >= 5:
                try:
                    f_id = int(parts[1])
                    f_len = int(parts[2])
                    f_dur = int(parts[3])
                    b64_data = parts[4]
                    if self.on_camera_frame:
                        self.on_camera_frame(f_id, f_len, f_dur, b64_data)
                except Exception as e:
                    logger.warning(f"Malformed $CAM_FRAME line: {e}")
            return

        if not line.startswith("#") and not line.startswith("$"):
            parts = line.split(",")
            if len(parts) >= 13:
                if self.on_telemetry_csv:
                    try:
                        self.on_telemetry_csv(line)
                    except Exception as e:
                        logger.error(f"Error in on_telemetry_csv callback: {e}")
