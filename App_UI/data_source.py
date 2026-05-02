"""
data_source.py — IMU data source abstraction.

Three concrete sources:
  SerialSource  — USB-C serial port (pyserial)
  BLESource     — BLE UART via Nordic UART Service (bleak, optional)
  ReplaySource  — replay a saved .imu.json file at configurable speed

All sources run in daemon threads and emit Qt signals.  Because Qt
auto-queues cross-thread signals, all signal deliveries land on the
main thread — no explicit locking needed.

Serial line format: comma-separated numbers.  Handles both:
  • 6-column  ax,ay,az,gx,gy,gz            (simplified firmware output)
  • 17-column time,ax,ay,az,gx,gy,gz,...   (full STEVAL CSV format)
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

# Nordic UART Service TX characteristic (device → host)
_NUS_TX_UUID = "6e400003-b5b3-f393-e0a9-e50e24dcca9e"


class DataSource(QObject):
    sample_received = pyqtSignal(list)   # [ax, ay, az, gx, gy, gz]
    status_changed = pyqtSignal(str)     # human-readable connection status
    error_occurred = pyqtSignal(str)     # error message

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


# ── Serial ───────────────────────────────────────────────────────────────────

class SerialSource(DataSource):
    def __init__(self, port: str, baud: int = 115200) -> None:
        super().__init__()
        self._port = port
        self._baud = baud
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="serial")
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _run(self) -> None:
        try:
            import serial
            with serial.Serial(self._port, self._baud, timeout=1) as ser:
                self.status_changed.emit(f"USB  {self._port}")
                buf = ""
                while self._running:
                    chunk = ser.read(256).decode("utf-8", errors="ignore")
                    if not chunk:
                        continue
                    buf += chunk
                    lines = buf.split("\n")
                    buf = lines[-1]
                    for line in lines[:-1]:
                        sample = self._parse(line.strip())
                        if sample:
                            self.sample_received.emit(sample)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            self.status_changed.emit("Disconnected")

    @staticmethod
    def _parse(line: str) -> list | None:
        if not line or line.startswith("#"):
            return None
        parts = line.split(",")
        nums = []
        for p in parts:
            try:
                nums.append(float(p.strip()))
            except ValueError:
                pass
        if len(nums) == 6:
            return nums
        if len(nums) >= 7:
            # Assume first column is timestamp — skip it
            return nums[1:7]
        return None


# ── BLE ──────────────────────────────────────────────────────────────────────

class BLESource(DataSource):
    """
    BLE connection using Nordic UART Service.
    Set TX_CHAR_UUID class attribute if your firmware uses different UUIDs.
    """
    TX_CHAR_UUID = _NUS_TX_UUID

    def __init__(
        self,
        device_name: Optional[str] = None,
        device_address: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._device_name = device_name
        self._device_address = device_address
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run_sync, daemon=True, name="ble")
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _run_sync(self) -> None:
        import asyncio
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        try:
            from bleak import BleakClient, BleakScanner
        except ImportError:
            self.error_occurred.emit(
                "bleak not installed.\nRun:  pip install bleak"
            )
            return

        try:
            if self._device_address:
                address = self._device_address
            else:
                self.status_changed.emit("BLE  Scanning…")
                dev = await BleakScanner.find_device_by_name(
                    self._device_name, timeout=10.0
                )
                if dev is None:
                    self.error_occurred.emit(
                        f"BLE device '{self._device_name}' not found within 10 s."
                    )
                    return
                address = dev.address

            async with BleakClient(address) as client:
                self.status_changed.emit(f"BLE  {address}")
                buf = ""

                def _on_notify(_sender, data: bytearray) -> None:
                    nonlocal buf
                    buf += data.decode("utf-8", errors="ignore")
                    lines = buf.split("\n")
                    buf = lines[-1]
                    for line in lines[:-1]:
                        sample = SerialSource._parse(line.strip())
                        if sample:
                            self.sample_received.emit(sample)

                await client.start_notify(self.TX_CHAR_UUID, _on_notify)
                import asyncio as _aio
                while self._running:
                    await _aio.sleep(0.05)
                await client.stop_notify(self.TX_CHAR_UUID)

        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            self.status_changed.emit("Disconnected")


# ── Replay ───────────────────────────────────────────────────────────────────

class ReplaySource(DataSource):
    """
    Replays a saved .imu.json file at the original sample rate scaled by speed.
    speed=1.0 → real-time, speed=2.0 → 2× faster, speed=100.0 → near-instant.
    """

    def __init__(self, path: str, speed: float = 1.0) -> None:
        super().__init__()
        self._path = path
        self._speed = max(speed, 0.1)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="replay"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _run(self) -> None:
        try:
            data = json.loads(Path(self._path).read_text())
            samples = data["samples"]
            name = Path(self._path).stem
            self.status_changed.emit(f"Replay  {name}")

            prev_t = 0.0
            for row in samples:
                if not self._running:
                    break
                t = row[0]
                delay = (t - prev_t) / self._speed
                if delay > 0.001:
                    time.sleep(delay)
                prev_t = t
                self.sample_received.emit(row[1:7])   # skip timestamp

            self.status_changed.emit("Replay complete")
        except Exception as exc:
            self.error_occurred.emit(str(exc))
