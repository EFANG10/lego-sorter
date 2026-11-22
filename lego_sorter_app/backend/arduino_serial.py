#!/usr/bin/env python3
"""
Arduino Serial Controller for LEGO Sorter
Communicates via USB serial with Arduino running gate control firmware.
Supports automatic port discovery, reconnect logic, per-bin travel delay scheduling,
and transparent simulation/mock mode.
"""

import time
import threading
import logging
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger("lego_sorter.serial")

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    logger.warning("pyserial not available, running in mock serial mode.")


class ArduinoController:
    def __init__(self, port: str = "MOCK", baud_rate: int = 115200, pulse_ms: int = 350):
        self.port_name = port
        self.baud_rate = baud_rate
        self.default_pulse_ms = pulse_ms
        self.conn: Optional[serial.Serial] = None
        self.is_connected = False
        self.is_mock = (port == "MOCK") or (not HAS_SERIAL)
        self.lock = threading.Lock()
        self.history: List[Dict[str, Any]] = []

        self.connect()

    def get_available_ports(self) -> List[Dict[str, str]]:
        ports = [{"port": "MOCK", "description": "Virtual / Simulated Arduino"}]
        if HAS_SERIAL:
            try:
                for p in serial.tools.list_ports.comports():
                    ports.append({
                        "port": p.device,
                        "description": f"{p.device} ({p.description or 'USB Serial'})"
                    })
            except Exception as e:
                logger.error(f"Error enumerating serial ports: {e}")
        return ports

    def connect(self, port: Optional[str] = None, baud: Optional[int] = None) -> bool:
        with self.lock:
            if port is not None:
                self.port_name = port
            if baud is not None:
                self.baud_rate = baud

            self.close()

            if self.port_name == "MOCK" or not HAS_SERIAL:
                self.is_mock = True
                self.is_connected = True
                logger.info("Arduino Controller initialized in SIMULATED/MOCK mode.")
                return True

            try:
                logger.info(f"Connecting to Arduino on {self.port_name} @ {self.baud_rate} baud...")
                self.conn = serial.Serial(
                    port=self.port_name,
                    baudrate=self.baud_rate,
                    timeout=0.5,
                    write_timeout=0.5
                )
                time.sleep(1.8) # Wait for Arduino reset / bootloader
                self.is_mock = False
                self.is_connected = True
                logger.info(f"Successfully connected to Arduino on {self.port_name}")
                return True
            except Exception as e:
                logger.warning(f"Could not open serial port {self.port_name}: {e}. Falling back to MOCK mode.")
                self.is_mock = True
                self.is_connected = False
                return False

    def close(self):
        if self.conn:
            try:
                self.conn.close()
                logger.info("Arduino serial port closed cleanly.")
            except Exception:
                pass
            self.conn = None
        self.is_connected = False

    def send_bin(
        self,
        bin_id: int,
        delay_ms: int = 0,
        pulse_ms: Optional[int] = None,
        spot: Optional[int] = None,
        row: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Schedules or immediately fires gate actuation for bin_id.
        If delay_ms > 0, schedules actuation after delay_ms has elapsed
        to account for the physical conveyor travel time to that specific bin.
        """
        pulse = pulse_ms if pulse_ms is not None else self.default_pulse_ms

        if bin_id == -1:
            msg = "Pass-through to Bin -1 (Reject / End-of-Belt). No gate actuated."
            logger.info(f"[ROUTING] {msg}")
            return True, msg

        if delay_ms > 0:
            delay_sec = delay_ms / 1000.0
            timer = threading.Timer(
                delay_sec,
                self._execute_send,
                args=[bin_id, pulse, spot, row]
            )
            timer.daemon = True
            timer.start()

            msg = f"Gate for Bin {bin_id} (Spot {spot}, Row {row}) scheduled in {delay_ms}ms (pulse {pulse}ms)."
            logger.info(f"[SERIAL SCHEDULED] {msg}")
            return True, msg
        else:
            return self._execute_send(bin_id, pulse, spot, row)

    def _execute_send(
        self,
        bin_id: int,
        pulse_ms: int,
        spot: Optional[int] = None,
        row: Optional[int] = None
    ) -> Tuple[bool, str]:
        with self.lock:
            now = time.time()
            cmd = f"BIN:{bin_id}:{pulse_ms}\n"
            entry = {
                "timestamp": now,
                "bin_id": bin_id,
                "pulse_ms": pulse_ms,
                "spot": spot,
                "row": row,
                "command": cmd.strip(),
                "mode": "MOCK" if self.is_mock else "REAL"
            }

            if self.is_mock or not self.conn or not self.is_connected:
                msg = f"[MOCK SERIAL] FIRED Gate {bin_id} (Spot {spot}, Row {row}) pulsed for {pulse_ms}ms"
                logger.info(msg)
                entry["status"] = "SIMULATED"
                self.history.append(entry)
                if len(self.history) > 100:
                    self.history.pop(0)
                return True, msg

            try:
                self.conn.write(cmd.encode("ascii"))
                self.conn.flush()
                line = ""
                try:
                    if self.conn.in_waiting > 0:
                        line = self.conn.readline().decode("ascii", errors="ignore").strip()
                except Exception:
                    pass

                msg = f"Sent to Arduino: {cmd.strip()} (Response: {line or 'ACK'})"
                logger.info(msg)
                entry["status"] = "SENT"
                entry["response"] = line
                self.history.append(entry)
                if len(self.history) > 100:
                    self.history.pop(0)
                return True, msg
            except Exception as e:
                err_msg = f"Serial transmission error on {self.port_name}: {e}"
                logger.error(err_msg)
                entry["status"] = f"ERROR: {e}"
                self.history.append(entry)
                self.is_connected = False
                return False, err_msg

    def ping(self) -> Tuple[bool, str]:
        with self.lock:
            if self.is_mock or not self.conn:
                return True, "PONG (MOCK)"
            try:
                self.conn.write(b"PING\n")
                self.conn.flush()
                res = self.conn.readline().decode("ascii", errors="ignore").strip()
                return True, res if res else "PONG"
            except Exception as e:
                return False, str(e)
