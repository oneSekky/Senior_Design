# Handwriting Demo App

Real-time handwriting recognition display for the Senior Design demo.
Receives raw IMU data from a pen-mounted accelerometer (over BLE or USB),
runs the trained PyTorch model, and renders the predicted letter strokes
on a fullscreen white canvas — designed for display on a large TV.

---

## Quick Start

```bash
cd App_UI
pip install -r requirements.txt   # one-time setup
python main.py
```

The app opens **fullscreen** immediately. Press `F11` or `Escape` to exit fullscreen.

---

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `PyQt6` | GUI framework |
| `torch` | PyTorch — model inference |
| `scipy` | Butterworth filters for IMU preprocessing |
| `numpy` | Numerical arrays |
| `scikit-learn` | StandardScaler (saved with trained model) |
| `pyserial` | USB serial connection to IMU |
| `Pillow` | PDF export |
| `bleak` | BLE connection (optional — only needed for wireless) |
| `pandas` | Reading CSV files |

> **GPU not required.** The model is small (~3 MB) and runs on CPU in under 50 ms per letter.

---

## Connecting to the IMU

Click **Connect** in the bottom toolbar. Three connection modes are available:

### USB Serial (recommended for demo backup)
1. Select **USB Serial** tab
2. Choose the COM port from the dropdown (click **Refresh** if it doesn't appear)
3. Set baud rate (default `115200`)
4. Click **OK**

The IMU firmware should output comma-separated values:
```
acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
```
or the full STEVAL CSV format — the app handles both automatically.

### BLE (wireless)
1. Select **BLE** tab
2. Enter the device name (e.g. `IMU_SENSOR`) or its MAC address
3. Click **OK** — the app scans for up to 10 seconds
4. Uses Nordic UART Service (NUS) by default

> If BLE is unavailable, `bleak` gives a clear error message in a popup.

### Replay a Recording
1. Select **Replay** tab
2. Click **Browse** and select a `.imu.json` file
3. Choose playback speed (`1×` = real-time, `Instant` = all at once)
4. Click **OK** — the canvas clears and letters appear as they were written

A test recording (`test_replay.imu.json`) is included in this folder.

---

## Writing Letters

Once connected, just write. The app automatically:

- **Detects when a stroke starts** — based on accelerometer magnitude rising above a motion threshold
- **Detects when a stroke ends** — ~300 ms of stillness after movement
- **Runs inference** — the PyTorch model converts the ~2 s IMU window to a 64×64 letter image
- **Displays the letter** — scaled up to 320 px, placed at the current cursor position
- **Detects word gaps** — ~800 ms of stillness inserts a space between words
- **Wraps to next line** — automatically when the current line fills the screen width

---

## All Controls

### Bottom Toolbar

| Button | Keyboard | Action |
|---|---|---|
| Connect | — | Open connection dialog (USB / BLE / Replay) |
| Disconnect | — | Stop receiving IMU data |
| ↩ Undo | `Ctrl+Z` | Remove the last predicted letter |
| ✕ Clear | `Ctrl+L` | Erase the entire canvas and reset the recording |
| Save PDF | `Ctrl+S` | Save the current canvas as a PDF file |
| Export | `Ctrl+E` | Export the raw IMU recording for later replay |
| Threshold | slider | Adjust binarization cutoff (0.05–0.70). Lower = more ink, higher = cleaner |
| Scale | spinbox | Letter display size in pixels (64–512, default 320) |
| F11 / Escape | — | Toggle fullscreen |

### Activity Indicator
The green dot `●` in the top-right status bar pulses every time an IMU sample arrives.
If it stops pulsing, the connection has dropped.

---

## Saving and Exporting

### Save PDF
`Ctrl+S` → choose a file path → saves the current canvas exactly as displayed.
Good for printing or sharing what was written.

### Export IMU Recording
`Ctrl+E` → choose a file path → saves a `.imu.json` file containing every raw IMU
sample received since the last **Connect** or **Clear**.

The recording can be loaded later via **Connect → Replay** to replay the exact
page that was written, letter by letter, at the original speed or faster.

**Recording starts automatically** whenever you connect or clear — no manual action needed.

---

## Replay File Format

Recordings are plain JSON:

```json
{
  "version": 1,
  "created": "2026-05-01T12:00:00",
  "sample_rate": 104,
  "columns": ["t", "acc_x[mg]", "acc_y[mg]", "acc_z[mg]", "gyro_x[mdps]", "gyro_y[mdps]", "gyro_z[mdps]"],
  "samples": [[0.0, 10.2, -3.1, 980.1, 0.0, 0.0, 0.0], ...],
  "events":  [{"t": 1.23, "type": "stroke_complete", "n": 128}, ...]
}
```

---

## File Structure

```
App_UI/
├── main.py           Main window — UI, toolbar, keyboard shortcuts, orchestration
├── inference.py      PyTorch model definition + preprocessing pipeline
├── stroke_buffer.py  Real-time stroke detection (causal HP filter + activity threshold)
├── canvas.py         White canvas widget — letter layout, undo, threshold, PDF grab
├── recorder.py       Saves raw IMU stream to .imu.json
├── data_source.py    USB serial, BLE, and replay data sources (Qt signals)
├── requirements.txt  Python dependencies
└── test_replay.imu.json  Sample recording for testing
```

---

## Model Details

| Property | Value |
|---|---|
| File | `Signal_Processing_Algorithm/models/best_side_mount_model.pt` |
| Architecture | 1D ResNet encoder → global avg-pool → ConvTranspose2d decoder |
| Input | `(T, 11)` — any number of timesteps, 11 features |
| Output | `(64, 64)` logits → sigmoid → threshold |
| 11 Features | filtered acc xyz, filtered gyro xyz, acc magnitude, velocity xy, position xy |
| Preprocessing | Butterworth HP filter (gravity removal) + double integration + StandardScaler |
| Inference device | CPU (no GPU needed) |

---

## Tuning Tips

- **Letters look too faint / sparse** → lower the Threshold slider (try 0.15–0.20)
- **Letters look blobby / filled in** → raise the Threshold slider (try 0.40–0.50)
- **Letters are too small for the TV** → increase Scale (try 384 or 448)
- **Stroke cuts off early** → the motion threshold may be too high; check IMU placement
- **Word gaps firing too often** → the pen is being held still too long between letters; this is expected behavior

---

## People Working On This

Senior Design Team — embedded pen, IMU firmware, ML model, and this UI.
