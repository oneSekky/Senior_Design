# IMU Streaming — Complete Software Integration Handoff

Single source of truth for consuming IMU data from the STM32WB55 + LSM6DSO16IS board, both today (USB CDC) and once BLE comes online. Includes data format spec, architecture, working code skeleton, and migration plan.

---

## Contents

1. [Overview](#overview)
2. [Connecting to the device](#connecting-to-the-device)
3. [Data format specification](#data-format-specification)
4. [Sensor configuration](#sensor-configuration-firmware-side)
5. [Sample-rate behavior](#sample-rate-behavior)
6. [Sanity-check values](#sanity-check-values)
7. [Format match with pipeline reference data](#format-match-with-pipeline-reference-data)
8. [Software architecture](#software-architecture-build-once-swap-transports)
9. [Project layout](#project-layout)
10. [The code (5 files)](#the-code)
11. [Running it](#running-it)
12. [What changes when BLE goes live](#what-changes-when-ble-goes-live)
13. [Testing without hardware](#testing-without-hardware)
14. [Failure modes](#failure-modes)
15. [Important: serial port exclusivity](#important-only-one-program-can-hold-the-serial-port)

---

## Overview

Custom STM32WB55 board with an onboard LSM6DSO16IS 6-axis IMU streams accelerometer + gyroscope samples at 104 Hz over USB CDC (virtual serial port). When the BLE firmware is finished, the same data stream becomes available over Bluetooth GATT notifications — same 7-column CSV format, just delivered via radio instead of cable.

```
[LSM6DSO16IS sensor]
        ↓ I²C (12 bytes per sample, 104 Hz)
[STM32WB55 MCU — formats as CSV]
        ↓
   ┌────────┴────────┐
   ↓                 ↓
[USB CDC NOW]    [BLE GATT LATER]
   ↓                 ↓
/dev/tty.usbmodem*   bleak.BleakClient
   ↓                 ↓
   └────────┬────────┘
            ↓
       parse + process (your code — same regardless of transport)
```

No drivers required for USB on Mac/Linux. Windows: same on Win10+. BLE will use `bleak` (cross-platform Python BLE library).

---

## Connecting to the device

### Today (USB CDC)

**Find the port:**
```bash
# Mac
ls /dev/tty.usbmodem*
# → /dev/tty.usbmodem205E309550431

# Linux
ls /dev/ttyACM*
# → /dev/ttyACM0

# Windows
# Device Manager → Ports (COM & LPT) → look for "STM32 Virtual ComPort"
```

**Quick sanity check (any platform):**
```bash
screen /dev/tty.usbmodem205E309550431 115200    # Mac/Linux
# Should immediately see header line + ~104 lines/sec of data
# Exit: Ctrl-A, then K, then y
```

Baud rate is **ignored** by USB CDC — pass any value (115200 is conventional). No flow control, parity, or stop-bit configuration matters.

### Later (BLE)

Will require:
- Board's BLE MAC address (firmware sets this at boot — TBD)
- Custom GATT service UUID (assigned in firmware — TBD)
- Custom GATT characteristic UUID for IMU notifications (assigned in firmware — TBD)
- `pip install bleak`

These values will be filled in when the BLE firmware lands. See [What changes when BLE goes live](#what-changes-when-ble-goes-live).

---

## Data format specification

**Format:** ASCII text, CSV, line-terminated with `\r\n`.

**First line is always the header:**
```
time[us],acc_x[mg],acc_y[mg],acc_z[mg],gyro_x[mdps],gyro_y[mdps],gyro_z[mdps]
```

**Subsequent lines are data:**
```
48047948,52.155,-5.368,1004.243,2572.500,-3211.250,-3228.750
48057159,49.410,-3.843,1001.437,2546.250,-3176.250,-3272.500
48066371,51.911,-5.185,1003.633,2546.250,-3115.000,-3158.750
```

**7 columns, in fixed order:**

| # | Column         | Unit                 | Type    | Range                       | Notes                                   |
|---|----------------|----------------------|---------|-----------------------------|-----------------------------------------|
| 1 | `time[us]`     | microseconds         | uint64  | 0 → ∞ (monotonic)           | Resets to 0 each time the board reboots |
| 2 | `acc_x[mg]`    | milli-g              | float   | -2000.000 → +2000.000       | 1 g = 1000 mg                           |
| 3 | `acc_y[mg]`    | milli-g              | float   | -2000.000 → +2000.000       |                                         |
| 4 | `acc_z[mg]`    | milli-g              | float   | -2000.000 → +2000.000       |                                         |
| 5 | `gyro_x[mdps]` | milli-degrees/second | float   | -250000.000 → +250000.000   | 1 dps = 1000 mdps                       |
| 6 | `gyro_y[mdps]` | milli-degrees/second | float   | -250000.000 → +250000.000   |                                         |
| 7 | `gyro_z[mdps]` | milli-degrees/second | float   | -250000.000 → +250000.000   |                                         |

All numeric fields use **exactly 3 decimal places** (firmware prints them that way).

This format is preserved byte-for-byte when the transport changes from USB to BLE.

---

## Sensor configuration (firmware-side)

| Setting                  | Value                  | Why software cares                                |
|--------------------------|------------------------|---------------------------------------------------|
| Accelerometer range      | ±2 g                   | Values clip beyond ±2000 mg                       |
| Accelerometer rate (ODR) | 104 Hz                 | One sample every ~9.6 ms                          |
| Accelerometer LSB        | 0.061 mg               | Sets resolution; 16-bit raw                       |
| Gyroscope range          | ±250 dps               | Values clip beyond ±250000 mdps                   |
| Gyroscope rate (ODR)     | 104 Hz                 | Same cadence as accel                             |
| Gyroscope LSB            | 8.75 mdps              | Sets resolution; 16-bit raw                       |
| Sample sync              | BDU + DRDY-polled      | Accel and gyro samples are time-aligned per row   |

**Axes:** right-handed coordinate system per the LSM6DSO16IS datasheet (X/Y in plane of chip, Z perpendicular to chip top surface).

---

## Sample-rate behavior

- ~104 lines per second
- ~9612 μs between consecutive `time[us]` values (close to ideal 9615 μs = 1/104 s)
- Slight jitter (~1 ms) from USB CDC bursts is normal — **don't assume exact uniform spacing**, use the timestamp column as truth
- Timestamps are monotonically increasing within a session
- If the board resets mid-stream (USB unplug, brownout, or firmware update), `time[us]` jumps back near 0 and the header line reappears — handle gracefully (the parser below filters out the header automatically)

---

## Sanity-check values

With the board sitting flat, chip-side facing up, completely stationary:

| Channel               | Expected                    | Why                                              |
|-----------------------|-----------------------------|--------------------------------------------------|
| `acc_z`               | ~+1000 mg                   | Gravity points down through Z-axis               |
| `acc_x`, `acc_y`      | ~0 mg (within ±50 mg)       | No horizontal force; small mounting/sensor offset |
| `gyro_*`              | All within ±5000 mdps       | Stationary; small zero-rate offset is normal     |

When you tilt the board onto a side, gravity migrates from `acc_z` to whichever axis is now pointing down (and changes sign accordingly).

---

## Format match with pipeline reference data

Both formats below show the same 7 columns in the same order:

**Pipeline reference data** (from MEMS Studio export):
```
time[us],acc_x[mg],acc_y[mg],acc_z[mg],gyro_x[mdps],gyro_y[mdps],gyro_z[mdps]
3269011733,813.923,214.415,554.185,3342.500,-3167.500,-595.000
3269021140,817.583,208.681,567.971,2992.500,-5118.750,-822.500
```

**This board's live output:**
```
time[us],acc_x[mg],acc_y[mg],acc_z[mg],gyro_x[mdps],gyro_y[mdps],gyro_z[mdps]
48047948,52.155,-5.368,1004.243,2572.500,-3211.250,-3228.750
48057159,49.410,-3.843,1001.437,2546.250,-3176.250,-3272.500
```

| Aspect            | Reference                               | This board                              | Match? |
|-------------------|-----------------------------------------|-----------------------------------------|--------|
| Column count      | 7                                       | 7                                       | ✓      |
| Column names      | `time[us]`, `acc_x[mg]`, ...            | identical                               | ✓      |
| Column order      | time, accX, accY, accZ, gyrX, gyrY, gyrZ | identical                              | ✓      |
| Units             | mg, mdps, μs                            | mg, mdps, μs                            | ✓      |
| Decimal precision | 3 digits                                | 3 digits                                | ✓      |
| Sample rate       | ~104 Hz                                 | ~104 Hz                                 | ✓      |

**Single behavioral difference:**

The reference data shows timestamps like `3269011733` (~54 minutes worth of microseconds), suggesting it's measured against a long-running reference. This board's `time[us]` resets to ~0 every time it reboots, since it's measured from chip startup using the ARM DWT cycle counter. **For relative timing (sample-to-sample deltas), both formats give identical ~9612 μs spacing — no change needed.** If your pipeline needs absolute timestamps tied to wall clock, layer it on the host side (record `datetime.now()` when the first sample arrives, add the offset to each `time[us]` value).

---

## Software architecture: build once, swap transports

```
       ┌─────────────────────────┐
USB ──▶│                         │
       │   on_line(text) ──▶ parse ──▶ dict ──▶ processor.on_sample(dict)
BLE ──▶│                         │
       └─────────────────────────┘
       transport (swappable)        parsing + processing (frozen)
```

Each transport is a thin function that pulls strings off a wire and hands them to a single callback. The callback parses + processes. **Parser and processor never know which transport delivered the data — and they don't need to.**

When BLE comes online: drop in a UUID and a MAC address, flip a CLI flag, run. Zero changes to model code.

**Why this matters:**

| Risk                                              | Mitigation                                              |
|---------------------------------------------------|---------------------------------------------------------|
| BLE firmware takes longer than expected           | USB still works, demo proceeds                          |
| BLE has range/reliability issues during demo      | Fall back to `--transport usb` on the spot              |
| Need to compare USB vs BLE data quality           | Run twice with different `--transport` flags, diff logs |
| Teammates iterate on the model while you work on BLE | Independent — neither blocks the other               |
| Model code accidentally depends on transport quirks | Forced separation makes this hard to do by accident   |

---

## Project layout

```
imu_app/
├── parser.py         # bytes → dict   (no I/O — pure function)
├── processor.py      # the actual pipeline / model (your team's work goes here)
├── transport_usb.py  # USB CDC source (works today)
├── transport_ble.py  # BLE source (stub now, fill in when firmware is ready)
└── main.py           # CLI entry, wires it all together
```

Five small files. Each has one job.

---

## The code

### `parser.py` — pure parsing, no I/O

```python
"""Parse one line of IMU CSV into a sample dict. Transport-agnostic."""

def parse_line(line: str) -> dict | None:
    """
    Returns a sample dict, or None if the line is malformed / a header / empty.
    Never raises.

    Keys in the returned dict:
        time_us       int    — microseconds since board boot
        acc_x_mg      float  — accelerometer X, milli-g
        acc_y_mg      float
        acc_z_mg      float
        gyro_x_mdps   float  — gyroscope X, milli-degrees/sec
        gyro_y_mdps   float
        gyro_z_mdps   float
    """
    line = line.strip()
    if not line or line.startswith("time["):
        return None  # empty or header line
    parts = line.split(",")
    if len(parts) != 7:
        return None
    try:
        return {
            "time_us":     int(parts[0]),
            "acc_x_mg":    float(parts[1]),
            "acc_y_mg":    float(parts[2]),
            "acc_z_mg":    float(parts[3]),
            "gyro_x_mdps": float(parts[4]),
            "gyro_y_mdps": float(parts[5]),
            "gyro_z_mdps": float(parts[6]),
        }
    except ValueError:
        return None
```

Pure function. No imports beyond the stdlib. Testable without any hardware. Identical behavior whether the line came from USB or BLE.

### `processor.py` — your pipeline

```python
"""IMU sample processor. Replace stub logic with your model / inference / etc."""


class IMUProcessor:
    def __init__(self, window_size: int = 104):
        self.buffer = []
        self.window_size = window_size  # 104 = 1 second of data at 104 Hz
        self.sample_count = 0

    def on_sample(self, sample: dict) -> None:
        """Called once per sample, ~104x per second."""
        self.buffer.append(sample)
        self.sample_count += 1

        # Once-per-second example: print stats and run inference on last second
        if len(self.buffer) >= self.window_size:
            window = self.buffer[-self.window_size:]
            self.buffer.clear()

            # ===========================================================
            # YOUR MODEL CODE GOES HERE
            #
            # `window` is a list of `window_size` sample dicts
            # (1 second @ 104 Hz)
            #
            # Example: pass to your trained classifier
            #   prediction = your_model.predict(window)
            #   print(f"Predicted gesture: {prediction}")
            # ===========================================================

            last = window[-1]
            print(f"[{self.sample_count} samples] "
                  f"acc=({last['acc_x_mg']:+8.2f}, {last['acc_y_mg']:+8.2f}, "
                  f"{last['acc_z_mg']:+8.2f}) mg")
```

This file gets touched a lot during model development. The other files stay frozen.

### `transport_usb.py` — USB CDC source (works today)

```python
"""Pull CSV lines off a virtual COM port (USB CDC) and hand each to a callback."""

import glob
import sys
from typing import Callable, Optional

import serial  # pip install pyserial


def find_serial_port() -> str:
    """Auto-detect the first USB CDC device on Mac/Linux."""
    candidates = sorted(glob.glob("/dev/tty.usbmodem*") + glob.glob("/dev/ttyACM*"))
    if not candidates:
        sys.exit("No USB CDC device found. Is the board plugged in?\n"
                 "  Mac:   ls /dev/tty.usbmodem*\n"
                 "  Linux: ls /dev/ttyACM*")
    return candidates[0]


def stream_from_usb(on_line: Callable[[str], None],
                    port: Optional[str] = None) -> None:
    """
    Open the USB CDC device and call on_line(text) for every line received.
    Blocks forever; raise KeyboardInterrupt (Ctrl+C) to stop.

    @param on_line  Callback invoked once per line, raw text including any line
                    endings. Pass the line straight to parse_line() — it filters
                    out the header automatically.
    @param port     Optional explicit path (e.g. "/dev/tty.usbmodem205E309550431").
                    If None, auto-detects the first one found.
    """
    port = port or find_serial_port()
    print(f"USB transport: opening {port}")
    ser = serial.Serial(port, 115200, timeout=1)
    try:
        while True:
            try:
                raw = ser.readline().decode("ascii", errors="replace")
            except serial.SerialException as e:
                print(f"USB disconnected: {e}")
                return
            if raw:
                on_line(raw)
    finally:
        ser.close()
```

### `transport_ble.py` — BLE source (stub for now)

```python
"""
BLE transport — fill this in once the firmware advertises a GATT service.

You'll need three values from the firmware side:
  - DEVICE_ADDRESS:  the board's BLE MAC (firmware sets this; ask hardware side)
  - IMU_SERVICE_UUID + IMU_CHAR_UUID: custom UUIDs assigned in the firmware

Until then this file just raises NotImplementedError. The rest of the code
is structured so swapping it in later requires no other file changes.
"""

import asyncio
from typing import Callable

# from bleak import BleakClient, BleakScanner  # pip install bleak

# TODO: fill these in when BLE firmware is ready
DEVICE_ADDRESS    = "XX:XX:XX:XX:XX:XX"
IMU_SERVICE_UUID  = "00000000-0000-0000-0000-000000000000"
IMU_CHAR_UUID     = "00000000-0000-0000-0000-000000000000"


async def stream_from_ble(on_line: Callable[[str], None]) -> None:
    """
    Connect to the board over BLE, subscribe to IMU notifications,
    call on_line(text) for every notification.

    Same signature contract as stream_from_usb: hand each line of CSV text
    to on_line(). The header line is sent on connect, then samples flow.
    """
    raise NotImplementedError(
        "BLE transport not implemented yet — firmware not ready. "
        "Use --transport usb for now."
    )

    # Reference implementation (uncomment when ready):
    #
    # def _callback(_sender, data: bytearray):
    #     line = data.decode("ascii", errors="replace")
    #     on_line(line)
    #
    # async with BleakClient(DEVICE_ADDRESS) as client:
    #     print(f"BLE transport: connected to {DEVICE_ADDRESS}")
    #     await client.start_notify(IMU_CHAR_UUID, _callback)
    #     await asyncio.Future()  # run forever until cancelled
```

### `main.py` — pick a transport, run

```python
"""Entry point. Picks transport at startup; everything downstream is identical."""

import argparse
import asyncio

from parser import parse_line
from processor import IMUProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", choices=["usb", "ble"], default="usb",
                    help="Where to read IMU samples from (default: usb)")
    args = ap.parse_args()

    processor = IMUProcessor(window_size=104)

    def on_line(line: str) -> None:
        sample = parse_line(line)
        if sample is not None:
            processor.on_sample(sample)

    try:
        if args.transport == "usb":
            from transport_usb import stream_from_usb
            stream_from_usb(on_line)
        else:
            from transport_ble import stream_from_ble
            asyncio.run(stream_from_ble(on_line))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
```

---

## Running it

**Today (USB working, BLE stub):**
```bash
pip install pyserial
python3 main.py --transport usb
# (or just `python3 main.py` since usb is the default)
```

You should see something like:
```
USB transport: opening /dev/tty.usbmodem205E309550431
[104 samples] acc=(  +49.35,   -3.04, +1004.84) mg
[208 samples] acc=(  +50.21,   -4.16, +1003.92) mg
...
```

**Once BLE firmware is ready:**
```bash
pip install bleak
python3 main.py --transport ble
```

`parser.py` and `processor.py` don't change. That's the whole point.

---

## What changes when BLE goes live

When the BLE firmware lands, three things change in `transport_ble.py`:

1. Set `DEVICE_ADDRESS` to the board's BLE MAC (get from `bleak`'s scanner or from the hardware side)
2. Set `IMU_CHAR_UUID` to whatever the firmware assigned
3. Uncomment the `BleakClient` block at the bottom of the file

`pip install bleak` if it's not already installed.

That's it. Run `python3 main.py --transport ble` — works.

**No other file is touched.** `parser.py`, `processor.py`, `transport_usb.py`, and `main.py` all stay byte-identical to what's running today.

---

## Testing without hardware

`parser.py` and `processor.py` are pure Python with no I/O — they can be unit-tested without a board attached:

```python
# test_parser.py
from parser import parse_line


def test_valid_line():
    line = "9612,-23.973,12.749,998.426,8.750,-26.250,17.500\r\n"
    s = parse_line(line)
    assert s is not None
    assert s["time_us"] == 9612
    assert s["acc_z_mg"] == 998.426
    assert s["gyro_y_mdps"] == -26.250


def test_header_line():
    assert parse_line("time[us],acc_x[mg],...") is None


def test_malformed():
    assert parse_line("not a csv line") is None
    assert parse_line("1,2,3") is None
    assert parse_line("") is None


def test_negative_zero():
    s = parse_line("100,-0.061,0.000,1000.000,0.000,0.000,0.000")
    assert s is not None
    assert s["acc_x_mg"] == -0.061


# test_processor.py
from processor import IMUProcessor


def make_sample(t_us=0):
    return {
        "time_us": t_us,
        "acc_x_mg": 0.0, "acc_y_mg": 0.0, "acc_z_mg": 1000.0,
        "gyro_x_mdps": 0.0, "gyro_y_mdps": 0.0, "gyro_z_mdps": 0.0,
    }


def test_buffer_fills_and_clears():
    p = IMUProcessor(window_size=10)
    for i in range(15):
        p.on_sample(make_sample(t_us=i))
    # After 15 samples with window=10, buffer cleared once at sample 10,
    # leaving 5 samples in buffer
    assert len(p.buffer) == 5
    assert p.sample_count == 15
```

Run with `pytest`. Gives confidence in the parsing/processing layer without needing a working board, useful when iterating on the model in parallel with hardware work.

---

## Failure modes

| Symptom                                              | Cause / fix                                                                                  |
|------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `/dev/tty.usbmodem*` doesn't appear                  | Board not plugged in, USB-C cable is power-only, or firmware crashed before USB init. Try a different cable. |
| Connects but only the header appears, no data lines  | IMU init succeeded but data-ready polling stuck. Power-cycle the board.                      |
| Lines appear with `0.000,0.000,0.000,...` everywhere | I²C glitch on read. One-off lines: ignore (parser handles). Persistent: power-cycle.        |
| Stream stops mid-session                             | USB cable jiggle, host went to sleep, or another program grabbed the port. Reconnect.        |
| Lines occasionally split across two reads            | Normal — `readline()` handles transparently.                                                 |
| Garbage characters                                   | Should not happen on USB CDC; if it does, suspect non-ASCII data corruption upstream.        |
| `BLE transport not implemented yet`                  | Expected — firmware not ready. Use `--transport usb`.                                        |
| `bleak` import error                                 | `pip install bleak` (only needed once BLE is live)                                           |

---

## Important: only one program can hold the serial port

If `screen` (or any other terminal) has the port open, your Python script will fail to connect with a "Resource busy" error. Always quit terminals (`Ctrl-A K y` for screen) before running the Python pipeline.

This restriction goes away with BLE — multiple BLE centrals can technically scan, but only one can actively connect to the peripheral at a time. Same idea, different mechanism.

---

## Future-proofing checklist

When writing your code, ask:

- ☐ Does my parser depend on the source of the data? (It shouldn't.)
- ☐ Does my processor know whether USB or BLE delivered the sample? (It shouldn't.)
- ☐ If I added a third transport tomorrow (UDP socket, file replay, simulated data), would I have to change parser.py or processor.py? (You shouldn't have to.)

If the answer to any of those is "yes," the abstraction has leaked. Refactor before going further — it's cheaper now than after the model is written around it.
