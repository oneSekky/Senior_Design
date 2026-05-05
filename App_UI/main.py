"""
main.py — Handwriting recognition demo application.

Run with:   python main.py

Controls:
  Ctrl+Z        Undo last letter
  Ctrl+L        Clear canvas
  Ctrl+S        Save PDF
  Ctrl+E        Export IMU recording
  F11 / Escape  Toggle fullscreen
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from PyQt6.QtCore import QEvent, QPointF, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QKeySequence, QPainter, QPalette, QPen, QPolygonF, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Calibration phrase — 3 words (long + short + proper noun) that together cover:
#   below-baseline loops (g, j, p), intra-letter gaps (i×2, j, t crossbar),
#   open curves (u, r), bumps (m, n), circles (o, a), diagonal (z),
#   straight (l), capital (B), and 2 word gaps for word-gap calibration.
_CAL_PHRASE   = "jumping to Brazil"
_PROFILE_PATH = Path(__file__).with_name("profile.json")

# Flat-on-paper calibration protocol constants
_FLAT_THRESHOLD = 5.0    # mg/sample: acc-delta below this = pen motionless
_FLAT_DURATION  = 130    # samples (~1.25 s) of stillness to confirm flat state
_PICKUP_SETTLE  = 52     # samples (~0.5 s) to skip after pen is lifted (removes
                         # pickup transient before writing begins)

from canvas import CanvasWidget
from data_source import BLESource, DataSource, MEMSStudioSource, ReplaySource, SerialSource
from recorder import IMURecorder
from stroke_buffer import (
    ACTIVITY_THRESHOLD,
    MIN_ABSOLUTE,
    VALLEY_FRACTION,
    PEAK_WINDOW,
    SMOOTH_WINDOW,
    STROKE_END_SAMPLES,
    WORD_GAP_SAMPLES,
    _CAL_THRESHOLD,
    FS,
    StrokeBuffer,
)
# InferenceEngine imported lazily inside _load_model to avoid blocking startup


# ── Connect dialog ────────────────────────────────────────────────────────────

class ConnectDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connect to IMU")
        self.setMinimumWidth(420)
        self._source: DataSource | None = None

        tabs = QTabWidget()

        # USB tab
        usb_w = QWidget()
        usb_f = QFormLayout(usb_w)
        self._port_combo = QComboBox()
        self._port_combo.setEditable(True)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_ports)
        port_row = QHBoxLayout()
        port_row.addWidget(self._port_combo, stretch=1)
        port_row.addWidget(refresh)
        self._baud_combo = QComboBox()
        for b in ["9600", "57600", "115200", "230400", "460800"]:
            self._baud_combo.addItem(b)
        self._baud_combo.setCurrentText("115200")
        usb_f.addRow("Port:", port_row)
        usb_f.addRow("Baud rate:", self._baud_combo)
        tabs.addTab(usb_w, "USB Serial")
        self._refresh_ports()

        # BLE tab
        ble_w = QWidget()
        ble_f = QFormLayout(ble_w)
        self._ble_name = QLineEdit()
        self._ble_name.setPlaceholderText("e.g.  IMU_SENSOR")
        self._ble_addr = QLineEdit()
        self._ble_addr.setPlaceholderText("AA:BB:CC:DD:EE:FF  (optional)")
        ble_f.addRow("Device name:", self._ble_name)
        ble_f.addRow("MAC address:", self._ble_addr)
        tabs.addTab(ble_w, "BLE")

        # Replay tab
        rep_w = QWidget()
        rep_f = QFormLayout(rep_w)
        self._rep_path = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self._rep_path, stretch=1)
        path_row.addWidget(browse)
        self._speed_combo = QComboBox()
        for s in ["0.5×", "1×", "2×", "5×", "Instant"]:
            self._speed_combo.addItem(s)
        self._speed_combo.setCurrentText("1×")
        rep_f.addRow("File:", path_row)
        rep_f.addRow("Speed:", self._speed_combo)
        tabs.addTab(rep_w, "Replay")

        # MEMS Studio tab
        mems_w = QWidget()
        mems_f = QFormLayout(mems_w)
        self._mems_path = QLineEdit()
        self._mems_path.setPlaceholderText("Path to MEMS Studio log CSV…")
        mems_browse = QPushButton("Browse…")
        mems_browse.clicked.connect(self._browse_mems)
        mems_path_row = QHBoxLayout()
        mems_path_row.addWidget(self._mems_path, stretch=1)
        mems_path_row.addWidget(mems_browse)
        mems_note = QLabel(
            "1. In MEMS Studio: Save to File → Browse → set this same path → Start\n"
            "2. Check Timestamp + Accelerometer + Gyroscope columns\n"
            "3. Click OK here — the app will wait for data to arrive"
        )
        mems_note.setStyleSheet("color:#888; font-size:11px;")
        mems_note.setWordWrap(True)
        mems_f.addRow("Log file:", mems_path_row)
        mems_f.addRow("", mems_note)
        tabs.addTab(mems_w, "MEMS Studio")

        self._tabs = tabs

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(btns)

    def _refresh_ports(self) -> None:
        self._port_combo.clear()
        try:
            import serial.tools.list_ports
            ports = [p.device for p in serial.tools.list_ports.comports()]
        except Exception:
            ports = ["COM3", "COM4"]
        for p in ports:
            self._port_combo.addItem(p)
        if not ports:
            self._port_combo.addItem("COM3")

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open IMU Recording", "",
            "IMU Recording (*.imu.json);;All Files (*)"
        )
        if path:
            self._rep_path.setText(path)

    def _browse_mems(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "MEMS Studio Log File", "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self._mems_path.setText(path)

    def _accept(self) -> None:
        idx = self._tabs.currentIndex()
        if idx == 0:  # USB
            port = self._port_combo.currentText().strip()
            if not port:
                return
            baud = int(self._baud_combo.currentText())
            self._source = SerialSource(port, baud)
        elif idx == 1:  # BLE
            name = self._ble_name.text().strip() or None
            addr = self._ble_addr.text().strip() or None
            if not name and not addr:
                QMessageBox.warning(self, "BLE", "Enter a device name or MAC address.")
                return
            self._source = BLESource(device_name=name, device_address=addr)
        elif idx == 2:  # Replay
            path = self._rep_path.text().strip()
            if not path:
                QMessageBox.warning(self, "Replay", "Select a recording file.")
                return
            speed_map = {"0.5×": 0.5, "1×": 1.0, "2×": 2.0, "5×": 5.0, "Instant": 200.0}
            speed = speed_map.get(self._speed_combo.currentText(), 1.0)
            self._source = ReplaySource(path, speed)
        else:  # MEMS Studio
            path = self._mems_path.text().strip()
            if not path:
                QMessageBox.warning(self, "MEMS Studio", "Enter the log file path.")
                return
            self._source = MEMSStudioSource(path)
        self.accept()

    @property
    def source(self) -> DataSource | None:
        return self._source

    @property
    def is_replay(self) -> bool:
        return self._tabs.currentIndex() == 2


# ── Activity dot ─────────────────────────────────────────────────────────────

class _ActivityDot(QLabel):
    _ACTIVE = "color:#22cc55; font-size:18px;"
    _IDLE   = "color:#555555; font-size:18px;"

    def __init__(self, parent=None) -> None:
        super().__init__("●", parent)
        self.setStyleSheet(self._IDLE)
        self._dim_timer = QTimer(self)
        self._dim_timer.setSingleShot(True)
        self._dim_timer.timeout.connect(lambda: self.setStyleSheet(self._IDLE))

    def pulse(self) -> None:
        self.setStyleSheet(self._ACTIVE)
        self._dim_timer.start(250)


# ── Accelerometer graph ───────────────────────────────────────────────────────

class AccelGraphWidget(QWidget):
    """Stacked real-time IMU plots: accelerometer (top) + gyroscope (bottom)."""

    _FS  = 104
    _WIN = _FS * 8   # 832 samples ≈ 8 s

    _ACC_LO, _ACC_HI = -2000.0,  2000.0   # mg
    _GYR_LO, _GYR_HI = -2000.0,  2000.0   # dps (raw mdps ÷ 1000)
    _GYR_SCALE        = 1.0 / 1000.0

    _ACC_GRID = [(-2000, "-2g"), (-1000, "-1g"), (0, "0"), (1000, "1g"), (2000, "2g")]
    _GYR_GRID = [(-2000, "-2k"), (-1000, "-1k"), (0, "0"), (1000, "1k"), (2000, "2k")]

    _COLORS = (
        QColor(255, 85,  85),
        QColor(75,  215, 75),
        QColor(100, 155, 255),
    )
    _LABELS = ("X", "Y", "Z")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(280)
        self.setStyleSheet("background:#111;")
        self._bufs: list[deque] = [deque(maxlen=self._WIN) for _ in range(6)]
        self._dirty = False
        timer = QTimer(self)
        timer.setInterval(33)
        timer.timeout.connect(self._tick)
        timer.start()

    def push_sample(self, sample: list) -> None:
        for i in range(min(6, len(sample))):
            self._bufs[i].append(float(sample[i]))
        self._dirty = True

    def _tick(self) -> None:
        if self._dirty:
            self._dirty = False
            self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        W, H = self.width(), self.height()
        GAP  = 4
        half = (H - GAP) // 2

        self._draw_plot(p, W, 0,          half, self._bufs[:3],
                        self._ACC_LO, self._ACC_HI, self._ACC_GRID, 1.0,
                        "Accel (mg)")
        self._draw_plot(p, W, half + GAP, half, self._bufs[3:],
                        self._GYR_LO, self._GYR_HI, self._GYR_GRID, self._GYR_SCALE,
                        "Gyro (dps)")

        p.setPen(QPen(QColor(45, 45, 45), 1))
        p.drawLine(0, half + GAP // 2, W, half + GAP // 2)

    def _draw_plot(self, p: QPainter, W: int, y_off: int, H: int,
                   bufs: list, y_lo: float, y_hi: float, grid: list,
                   scale: float, title: str) -> None:
        ML, MR, MT, MB = 38, 8, 21, 22
        PW = W - ML - MR
        PH = H - MT - MB
        if PW < 10 or PH < 10:
            return

        p.fillRect(0, y_off, W, H, QColor(17, 17, 17))

        font = p.font()
        font.setPointSize(8)
        p.setFont(font)
        p.setPen(QColor(138, 138, 138))
        p.drawText(ML, y_off + 3, PW, 15, Qt.AlignmentFlag.AlignHCenter, title)

        font.setPointSize(7)
        p.setFont(font)
        for v, lbl in grid:
            gy = y_off + MT + int(PH * (1.0 - (v - y_lo) / (y_hi - y_lo)))
            p.setPen(QPen(QColor(65, 65, 65) if v == 0 else QColor(37, 37, 37), 1))
            p.drawLine(ML, gy, ML + PW, gy)
            p.setPen(QColor(86, 86, 86))
            p.drawText(0, gy - 7, ML - 3, 14,
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, lbl)

        p.setPen(QPen(QColor(60, 60, 60), 1))
        p.drawRect(ML, y_off + MT, PW, PH)

        p.setClipRect(ML, y_off + MT, PW + 1, PH + 1)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        n_total = self._WIN
        n_fill  = len(bufs[0])
        offset  = n_total - n_fill

        for ch in range(3):
            buf = list(bufs[ch])
            if len(buf) < 2:
                continue
            poly = QPolygonF()
            for i, raw in enumerate(buf):
                v = raw * scale
                x = ML + PW * (offset + i) / (n_total - 1)
                y = y_off + MT + PH * (1.0 - (v - y_lo) / (y_hi - y_lo))
                poly.append(QPointF(x, y))
            p.setPen(QPen(self._COLORS[ch], 1))
            p.drawPolyline(poly)

        p.setClipping(False)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Legend
        ly  = y_off + H - MB + 7
        seg = PW // 3
        for i in range(3):
            lx = ML + i * seg + 2
            p.setPen(QPen(self._COLORS[i], 2))
            p.drawLine(lx, ly, lx + 12, ly)
            p.setPen(self._COLORS[i])
            font.setPointSize(7)
            p.setFont(font)
            p.drawText(lx + 15, ly - 5, 22, 12, 0, self._LABELS[i])


# ── Main window ───────────────────────────────────────────────────────────────

class HandwritingApp(QMainWindow):
    _inference_result = pyqtSignal(object)  # delivers (64,64) array from worker thread

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Handwriting Demo")

        self._inference = None  # type: ignore[assignment]  # InferenceEngine loaded lazily
        self._recorder = IMURecorder()
        self._stroke_buf = StrokeBuffer(
            on_stroke_complete=self._on_stroke_complete,
            on_word_gap=self._on_word_gap,
        )
        self._source: DataSource | None = None
        self._is_replay = False       # True while a ReplaySource is active
        self._is_calibrating = False
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._inference_result.connect(self._on_inference_result)
        # Calibration state machine (flat-baseline protocol)
        self._cal_state: str = "idle"          # idle | flat_pre | post_pickup | writing
        self._cal_prev_acc: np.ndarray | None = None
        self._cal_mag_win: deque = deque(maxlen=SMOOTH_WINDOW)
        self._cal_flat_count: int = 0
        self._cal_counter: int = 0             # reused for settle / post-pickup counts
        self._cal_baseline: list[float] = []   # smoothed values during flat period
        self._cal_samples: list = []           # raw samples during writing

        self._build_ui()
        self._build_shortcuts()
        self._load_profile()

        self._recorder.start()

        # Show window immediately, then load model after event loop starts
        self.show()
        QTimer.singleShot(50, self._load_model)

    # ── Calibration overlay ───────────────────────────────────────────────────

    class _CalibrationOverlay(QWidget):
        """
        Flat-baseline calibration overlay — no buttons needed.

        Protocol:
          1. Lay pen flat and still  →  baseline captured automatically (~1.25 s)
          2. Pick up and write phrase →  recording starts after 0.5 s settle
          3. Lay pen flat again      →  recording stops, analysis runs
        """
        cancelled = pyqtSignal()

        def __init__(self, parent, phrase: str) -> None:
            super().__init__(parent)
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
            self.setStyleSheet("background-color: rgba(0,0,0,215);")

            lay = QVBoxLayout(self)
            lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.setSpacing(12)

            title = QLabel("CALIBRATION")
            title.setStyleSheet("color:#aaa; font-size:13px; letter-spacing:4px;")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)

            self._step_lbl = QLabel("Step 1 of 3")
            self._step_lbl.setStyleSheet("color:#666; font-size:13px;")
            self._step_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

            self._instr = QLabel("Lay the pen flat on the paper and hold still.")
            self._instr.setStyleSheet("color:white; font-size:18px;")
            self._instr.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._instr.setWordWrap(True)

            self._phrase_lbl = QLabel(phrase)
            self._phrase_lbl.setStyleSheet(
                "color:#555; font-size:52px; font-weight:bold; letter-spacing:10px;"
            )
            self._phrase_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

            self._status_lbl = QLabel("Waiting for stillness...")
            self._status_lbl.setStyleSheet("color:#888; font-size:14px;")
            self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._status_lbl.setFixedHeight(26)

            cancel_btn = QPushButton("Cancel")
            cancel_btn.setFixedWidth(100)
            cancel_btn.setStyleSheet(
                "QPushButton{background:#444;color:white;border-radius:5px;"
                "font-size:13px;padding:7px;}"
                "QPushButton:hover{background:#555;}"
            )
            cancel_btn.clicked.connect(self.cancelled.emit)

            lay.addWidget(title)
            lay.addWidget(self._step_lbl)
            lay.addSpacing(6)
            lay.addWidget(self._instr)
            lay.addSpacing(10)
            lay.addWidget(self._phrase_lbl)
            lay.addSpacing(8)
            lay.addWidget(self._status_lbl)
            lay.addSpacing(14)
            lay.addWidget(cancel_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        def set_state(self, state: str) -> None:
            if state == "flat_pre":
                self._step_lbl.setText("Step 1 complete  -  Baseline captured")
                self._step_lbl.setStyleSheet("color:#55ff55; font-size:13px;")
                self._instr.setText("Pick up the pen and write the phrase.")
                self._phrase_lbl.setStyleSheet(
                    "color:white; font-size:52px; font-weight:bold; letter-spacing:10px;"
                )
                self._status_lbl.setText("Lift pen to begin...")
                self._status_lbl.setStyleSheet("color:#aaa; font-size:14px;")
            elif state == "post_pickup":
                self._step_lbl.setText("Step 2 of 3  -  Getting ready")
                self._status_lbl.setText("Getting ready...")
                self._status_lbl.setStyleSheet("color:#ffcc44; font-size:14px;")
            elif state == "writing":
                self._step_lbl.setText("Step 2 of 3  -  Recording")
                self._instr.setText("Write the phrase.  Lay pen flat when done.")
                self._status_lbl.setText("Recording...")
                self._status_lbl.setStyleSheet("color:#ff5555; font-size:14px;")
            elif state == "analyzing":
                self._step_lbl.setText("Step 3 of 3  -  Analyzing")
                self._instr.setText("Analyzing...")
                self._status_lbl.setText("")

        def show_done(self, threshold: float, stroke_end: int, word_gap: int) -> None:
            se_ms = stroke_end * 1000 // 104
            wg_ms = word_gap   * 1000 // 104
            self._step_lbl.setText("Complete  -  Profile saved")
            self._step_lbl.setStyleSheet("color:#55ff55; font-size:13px;")
            self._instr.setText("Calibration complete!")
            self._phrase_lbl.setText("")
            self._status_lbl.setText(
                f"Threshold: {threshold:.1f}     "
                f"Stroke end: {stroke_end} ({se_ms} ms)     "
                f"Word gap: {word_gap} ({wg_ms} ms)"
            )
            self._status_lbl.setStyleSheet("color:#55ff55; font-size:14px;")

        def showEvent(self, e) -> None:
            super().showEvent(e)
            self._fit()
            if self.parent():
                self.parent().installEventFilter(self)

        def hideEvent(self, e) -> None:
            super().hideEvent(e)
            if self.parent():
                self.parent().removeEventFilter(self)

        def eventFilter(self, obj, event) -> bool:
            if obj is self.parent() and event.type() == QEvent.Type.Resize:
                self._fit()
            return False

        def _fit(self) -> None:
            if self.parent():
                self.setGeometry(self.parent().rect())

    # ── Calibration logic ────────────────────────────────────────────────────

    def _start_calibration(self) -> None:
        if self._is_replay or self._source is None:
            return
        self._is_calibrating = True
        self._cal_state      = "idle"
        self._cal_prev_acc   = None
        self._cal_mag_win.clear()
        self._cal_flat_count = 0
        self._cal_counter    = 0
        self._cal_baseline   = []
        self._cal_samples    = []

        self._cal_overlay = HandwritingApp._CalibrationOverlay(self.canvas, _CAL_PHRASE)
        self._cal_overlay.cancelled.connect(self._cancel_calibration)
        self._cal_overlay.show()
        self._cal_btn.setEnabled(False)

    def _cal_process_sample(self, sample: list) -> None:
        """Drive the flat-baseline calibration state machine on each sample."""
        s = np.asarray(sample, dtype=np.float32)

        # Rolling acc-delta (same signal as StrokeBuffer)
        if self._cal_prev_acc is None:
            self._cal_prev_acc = s[:3].copy()
            return
        delta = s[:3] - self._cal_prev_acc
        self._cal_prev_acc = s[:3].copy()
        mag = float(np.sqrt((delta ** 2).sum()))
        self._cal_mag_win.append(mag)
        smoothed = float(np.mean(self._cal_mag_win))

        state = self._cal_state

        if state == "idle":
            # Count consecutive "flat" samples
            if smoothed < _FLAT_THRESHOLD:
                self._cal_flat_count += 1
                if self._cal_flat_count >= _FLAT_DURATION:
                    self._cal_state      = "flat_pre"
                    self._cal_baseline   = []
                    self._cal_flat_count = 0
                    self._cal_overlay.set_state("flat_pre")
            else:
                self._cal_flat_count = 0

        elif state == "flat_pre":
            # Still flat — keep accumulating baseline values
            if smoothed < _FLAT_THRESHOLD:
                self._cal_baseline.append(smoothed)
            else:
                # Pen lifted — enter fixed post-pickup settle
                self._cal_state   = "post_pickup"
                self._cal_counter = 0
                self._cal_overlay.set_state("post_pickup")

        elif state == "post_pickup":
            # Skip _PICKUP_SETTLE samples to absorb the pickup transient
            self._cal_counter += 1
            if self._cal_counter >= _PICKUP_SETTLE:
                self._cal_state   = "writing"
                self._cal_samples = []
                self._cal_flat_count = 0
                self._cal_overlay.set_state("writing")

        elif state == "writing":
            self._cal_samples.append(sample)
            if smoothed < _FLAT_THRESHOLD:
                self._cal_flat_count += 1
                if self._cal_flat_count >= _FLAT_DURATION:
                    # Pen is flat again — trim putdown transient and analyze
                    trim     = self._cal_flat_count
                    writing  = self._cal_samples[:-trim] if trim < len(self._cal_samples) else self._cal_samples
                    self._cal_state = "idle"
                    self._is_calibrating = False
                    self._cal_overlay.set_state("analyzing")
                    self._on_cal_done(writing, self._cal_baseline)
            else:
                self._cal_flat_count = 0

    def _on_cal_done(self, samples: list, baseline: list[float]) -> None:
        threshold, stroke_end, word_gap = self._analyze_cal_phrase(
            samples, baseline, _CAL_PHRASE
        )
        self._stroke_buf.set_threshold(threshold)
        self._stroke_buf.set_stroke_end(stroke_end)
        self._stroke_buf.set_word_gap(word_gap)
        self._save_profile(threshold, stroke_end, word_gap)
        self._cal_overlay.show_done(threshold, stroke_end, word_gap)
        QTimer.singleShot(3000, self._close_cal_overlay)

    def _analyze_cal_phrase(
        self, samples: list, baseline: list[float], phrase: str
    ) -> tuple[float, int, int]:
        """
        Derive stroke_end_samples and word_gap_samples from the calibration phrase.

        Uses the same adaptive valley-detection logic as StrokeBuffer so that
        gap measurements during calibration match what the live detector sees.
        The returned threshold value is kept for API compatibility but is no
        longer applied (StrokeBuffer.set_threshold is a no-op).

        Gap tiers (sorted longest-first):
          Tier 1 (n_word_gaps largest)       → word gaps
          Tier 2 (next n_inter_letter gaps)  → inter-letter gaps
          Tier 3 (remainder)                 → intra-letter (dots, crossbars)

        Position sanity check: validates word-gap candidates against expected
        temporal position assuming constant writing rate (±30% tolerance).
        """
        words          = phrase.split()
        n_word_gaps    = len(words) - 1
        n_inter_letter = sum(max(0, len(w) - 1) for w in words)
        total_letters  = sum(len(w) for w in words)

        cumulative = 0
        expected_frac: list[float] = []
        for w in words[:-1]:
            cumulative += len(w)
            expected_frac.append(cumulative / total_letters)

        threshold = ACTIVITY_THRESHOLD   # returned for compat; not used by StrokeBuffer

        if len(samples) < 20:
            return threshold, STROKE_END_SAMPLES, WORD_GAP_SAMPLES

        arr    = np.array(samples, dtype=np.float32)
        acc    = arr[:, :3]
        deltas = np.diff(acc, axis=0)
        mags   = np.concatenate([[0.0], np.sqrt((deltas ** 2).sum(axis=1))])
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        smoothed = np.convolve(mags, kernel, mode="same")

        # ── Build adaptive threshold trace (mirrors StrokeBuffer exactly) ─────
        from collections import deque as _deque
        peak_win  = _deque(maxlen=PEAK_WINDOW)
        adap_thr  = np.zeros(len(smoothed), dtype=np.float32)
        for i, v in enumerate(smoothed):
            if v >= MIN_ABSOLUTE:
                peak_win.append(v)
            rp = float(max(peak_win)) if peak_win else 0.0
            adap_thr[i] = max(MIN_ABSOLUTE, VALLEY_FRACTION * rp)

        # ── Dynamic trim: drop leading/trailing pickup transient ──────────────
        active_mask = smoothed > adap_thr
        if not active_mask.any():
            return threshold, STROKE_END_SAMPLES, WORD_GAP_SAMPLES
        first_i = int(np.argmax(active_mask))
        last_i  = int(len(active_mask) - 1 - np.argmax(active_mask[::-1]))
        sm = smoothed[first_i : last_i + 1]
        at = adap_thr[first_i : last_i + 1]
        N  = len(sm)
        if N < 10:
            return threshold, STROKE_END_SAMPLES, WORD_GAP_SAMPLES

        # ── Collect gap runs below adaptive threshold ─────────────────────────
        gaps_len:   list[int]   = []
        gaps_start: list[float] = []
        count = g_start = 0
        for i, (v, t) in enumerate(zip(sm, at)):
            if v < t:
                if count == 0:
                    g_start = i
                count += 1
            else:
                if count >= 3:
                    gaps_len.append(count)
                    gaps_start.append(g_start / N)
                count = 0
        if count >= 3:
            gaps_len.append(count)
            gaps_start.append(g_start / N)

        stroke_end = STROKE_END_SAMPLES
        word_gap   = WORD_GAP_SAMPLES

        if len(gaps_len) < n_word_gaps + 1:
            return threshold, stroke_end, word_gap

        # ── Tier assignment ───────────────────────────────────────────────────
        order      = sorted(range(len(gaps_len)), key=lambda i: gaps_len[i], reverse=True)
        sorted_len = [gaps_len[i] for i in order]
        sorted_pos = [gaps_start[i] for i in order]

        wg_indices: list[int] = []
        for rank in range(min(n_word_gaps * 3, len(sorted_len))):
            if len(wg_indices) == n_word_gaps:
                break
            pos = sorted_pos[rank]
            exp = expected_frac[len(wg_indices)]
            if abs(pos - exp) <= 0.30:
                wg_indices.append(rank)

        if len(wg_indices) < n_word_gaps:
            wg_indices = list(range(n_word_gaps))

        wg_set    = set(wg_indices)
        tier1     = [sorted_len[i] for i in wg_indices]
        tier_rest = [sorted_len[i] for i in range(len(sorted_len)) if i not in wg_set]

        if tier_rest:
            word_gap = max(
                STROKE_END_SAMPLES + 5,
                int(tier_rest[0] + (min(tier1) - tier_rest[0]) * 0.5),
            )

        if len(tier_rest) >= n_inter_letter:
            inter     = tier_rest[:n_inter_letter]
            intra     = tier_rest[n_inter_letter:]
            min_inter = min(inter)
            max_intra = max(intra) if intra else 0
            if max_intra < min_inter:
                stroke_end = max(3, int(max_intra + (min_inter - max_intra) * 0.4))
            else:
                stroke_end = max(3, int(min_inter * 0.70))
        elif tier_rest:
            stroke_end = max(3, int(min(tier_rest) * 0.70))

        # ── Hard bounds ───────────────────────────────────────────────────────
        stroke_end = int(np.clip(stroke_end, 3, int(1.5 * FS)))
        word_gap   = int(np.clip(word_gap, stroke_end + 5, int(3.0 * FS)))
        if word_gap <= stroke_end + 2:
            word_gap = stroke_end * 3

        return threshold, stroke_end, word_gap

    # ── Profile save / load ──────────────────────────────────────────────────

    def _save_profile(self, threshold: float, stroke_end: int, word_gap: int) -> None:
        data = {
            "activity_threshold": round(threshold, 3),
            "stroke_end_samples": stroke_end,
            "word_gap_samples":   word_gap,
            "calibration_phrase": _CAL_PHRASE,
            "calibrated_at":      datetime.now().isoformat(timespec="seconds"),
        }
        try:
            _PROFILE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _apply_profile(self, data: dict) -> tuple[float, int, int]:
        thr  = float(data.get("activity_threshold", ACTIVITY_THRESHOLD))
        se   = int(data.get("stroke_end_samples",   STROKE_END_SAMPLES))
        wg   = int(data.get("word_gap_samples",     WORD_GAP_SAMPLES))
        self._stroke_buf.set_threshold(thr)
        self._stroke_buf.set_stroke_end(se)
        self._stroke_buf.set_word_gap(wg)
        return thr, se, wg

    def _load_profile(self) -> None:
        if not _PROFILE_PATH.exists():
            return
        try:
            data  = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
            thr, se, wg = self._apply_profile(data)
            se_ms = se * 1000 // 104
            wg_ms = wg * 1000 // 104
            stamp = data.get("calibrated_at", "")
            self._conn_label.setText(
                f"Profile loaded — thr={thr:.1f}  "
                f"stroke_end={se} ({se_ms} ms)  word_gap={wg} ({wg_ms} ms)"
                + (f"  [{stamp}]" if stamp else "")
            )
        except Exception:
            pass

    def _browse_profile_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Profile", str(_PROFILE_PATH.parent),
            "Profile (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            thr, se, wg = self._apply_profile(data)
            se_ms = se * 1000 // 104
            wg_ms = wg * 1000 // 104
            QMessageBox.information(
                self, "Profile Loaded",
                f"Threshold:  {thr:.1f} mg/s\n"
                f"Stroke end: {se} samp ({se_ms} ms)\n"
                f"Word gap:   {wg} samp ({wg_ms} ms)"
            )
        except Exception as exc:
            QMessageBox.warning(self, "Load Error", str(exc))

    def _browse_profile_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Profile", str(_PROFILE_PATH),
            "Profile (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            src = _PROFILE_PATH.read_text(encoding="utf-8")
            Path(path).write_text(src, encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", str(exc))

    def _cancel_calibration(self) -> None:
        self._is_calibrating = False
        self._cal_state = "idle"
        self._close_cal_overlay()

    def _close_cal_overlay(self) -> None:
        if hasattr(self, "_cal_overlay"):
            self._cal_overlay.hide()
        self._cal_btn.setEnabled(True)

    def _load_model(self) -> None:
        self._conn_label.setText("Loading model…")
        self._connect_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            from inference import InferenceEngine  # deferred: triggers torch import
            self._inference = InferenceEngine()
            self._conn_label.setText("Ready — click Connect to begin")
            self._connect_btn.setEnabled(True)
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "Missing model files", str(exc))
            self.close()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        vbox.addWidget(self._make_status_bar())

        middle = QWidget()
        hbox = QHBoxLayout(middle)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)

        self.canvas = CanvasWidget()
        hbox.addWidget(self.canvas, stretch=1)

        self._accel_graph = AccelGraphWidget()
        hbox.addWidget(self._accel_graph)

        vbox.addWidget(middle, stretch=1)

        vbox.addWidget(self._make_toolbar())

    def _make_status_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(34)
        bar.setStyleSheet("background:#111; color:white;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 0, 14, 0)

        self._conn_label = QLabel("Not connected")
        self._conn_label.setStyleSheet("color:#888; font-size:13px;")

        self._rec_label = QLabel("⏺ Recording")
        self._rec_label.setStyleSheet("color:#cc3333; font-size:13px;")

        self._count_label = QLabel("0 letters")
        self._count_label.setStyleSheet("color:#888; font-size:13px;")

        self._dot = _ActivityDot()

        lay.addWidget(self._conn_label)
        lay.addStretch()
        lay.addWidget(self._count_label)
        lay.addSpacing(20)
        lay.addWidget(self._rec_label)
        lay.addSpacing(8)
        lay.addWidget(self._dot)
        return bar

    def _make_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(58)
        bar.setStyleSheet("background:#222;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 6, 12, 6)
        lay.setSpacing(8)

        _btn_style = (
            "QPushButton{background:#444;color:white;border-radius:6px;"
            "font-size:14px;padding:0 14px;min-width:80px;}"
            "QPushButton:hover{background:#555;}"
            "QPushButton:pressed{background:#333;}"
            "QPushButton:disabled{background:#2a2a2a;color:#555;}"
        )

        def btn(label: str, tip: str) -> QPushButton:
            b = QPushButton(label)
            b.setToolTip(tip)
            b.setFixedHeight(40)
            b.setStyleSheet(_btn_style)
            return b

        def sep() -> QFrame:
            f = QFrame()
            f.setFrameShape(QFrame.Shape.VLine)
            f.setStyleSheet("color:#444;")
            f.setFixedWidth(1)
            return f

        lbl_style = "color:#aaa; font-size:13px;"

        # Connection
        self._connect_btn = btn("Connect", "Connect to IMU  (USB, BLE, or Replay)")
        self._connect_btn.clicked.connect(self._connect)
        self._disc_btn = btn("Disconnect", "Disconnect from IMU")
        self._disc_btn.clicked.connect(self._disconnect)
        self._disc_btn.setEnabled(False)
        self._cal_btn = btn("Calibrate", f'Write "{_CAL_PHRASE}" to calibrate timing')
        self._cal_btn.clicked.connect(self._start_calibration)
        self._cal_btn.setEnabled(False)
        load_prof_btn = btn("Load Profile", "Load saved calibration profile")
        load_prof_btn.clicked.connect(self._browse_profile_load)
        save_prof_btn = btn("Save Profile", "Save current calibration profile")
        save_prof_btn.clicked.connect(self._browse_profile_save)

        # Edit
        undo_btn = btn("↩ Undo", "Remove last letter  (Ctrl+Z)")
        undo_btn.clicked.connect(self.canvas.undo_last)
        clear_btn = btn("✕ Clear", "Erase entire canvas  (Ctrl+L)")
        clear_btn.clicked.connect(self._clear)

        # Export
        pdf_btn = btn("Save PDF", "Save page as PDF  (Ctrl+S)")
        pdf_btn.clicked.connect(self._save_pdf)
        export_btn = btn("Export", "Export IMU recording for replay  (Ctrl+E)")
        export_btn.clicked.connect(self._export_recording)

        # Threshold slider
        self._thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self._thresh_slider.setRange(5, 70)
        self._thresh_slider.setValue(30)
        self._thresh_slider.setFixedWidth(110)
        self._thresh_slider.setToolTip("Binarization threshold (×0.01)")
        self._thresh_val_lbl = QLabel("0.30")
        self._thresh_val_lbl.setStyleSheet(lbl_style)
        self._thresh_val_lbl.setFixedWidth(34)
        self._thresh_slider.valueChanged.connect(self._on_thresh_changed)

        # Scale spinbox
        self._scale_spin = QSpinBox()
        self._scale_spin.setRange(64, 512)
        self._scale_spin.setSingleStep(32)
        self._scale_spin.setValue(320)
        self._scale_spin.setFixedWidth(68)
        self._scale_spin.setStyleSheet(
            "QSpinBox{background:#444;color:white;border:none;font-size:13px;padding:2px;}"
        )
        self._scale_spin.valueChanged.connect(self.canvas.set_letter_size)

        thresh_lbl = QLabel("Threshold:")
        thresh_lbl.setStyleSheet(lbl_style)
        scale_lbl = QLabel("Scale:")
        scale_lbl.setStyleSheet(lbl_style)

        lay.addWidget(self._connect_btn)
        lay.addWidget(self._disc_btn)
        lay.addWidget(self._cal_btn)
        lay.addWidget(load_prof_btn)
        lay.addWidget(save_prof_btn)
        lay.addWidget(sep())
        lay.addWidget(undo_btn)
        lay.addWidget(clear_btn)
        lay.addWidget(sep())
        lay.addWidget(pdf_btn)
        lay.addWidget(export_btn)
        lay.addWidget(sep())
        lay.addWidget(thresh_lbl)
        lay.addWidget(self._thresh_slider)
        lay.addWidget(self._thresh_val_lbl)
        lay.addWidget(sep())
        lay.addWidget(scale_lbl)
        lay.addWidget(self._scale_spin)
        lay.addStretch()
        return bar

    def _build_shortcuts(self) -> None:
        pairs = [
            ("Ctrl+Z", self.canvas.undo_last),
            ("Ctrl+L", self._clear),
            ("Ctrl+S", self._save_pdf),
            ("Ctrl+E", self._export_recording),
            ("F11",    self._toggle_fullscreen),
            ("Escape", self._on_escape),
        ]
        for key, slot in pairs:
            QShortcut(QKeySequence(key), self).activated.connect(slot)

    # ── Slots ────────────────────────────────────────────────────────────────

    @pyqtSlot(list)
    def _on_sample(self, sample: list) -> None:
        self._recorder.record_sample(sample)
        self._dot.pulse()
        self._accel_graph.push_sample(sample)
        if self._is_calibrating:
            self._cal_process_sample(sample)
        # Replay drives inference via stroke_complete events — skip stroke buffer
        if not self._is_replay:
            self._stroke_buf.feed(sample)

    def _on_stroke_complete(self, stroke: np.ndarray) -> None:
        """Called by StrokeBuffer for live (USB/BLE) data."""
        self._recorder.record_event("stroke_complete")
        if not self._is_calibrating:
            self._run_inference(stroke)

    @pyqtSlot(object)
    def _on_stroke_from_replay(self, stroke) -> None:
        """Called by ReplaySource for each recorded stroke event."""
        self._run_inference(np.asarray(stroke, dtype=np.float32))

    def _run_inference(self, stroke: np.ndarray) -> None:
        if self._inference is None:
            return
        stroke_copy = stroke.copy()
        future = self._executor.submit(self._inference.predict, stroke_copy)
        future.add_done_callback(self._inference_done)

    def _inference_done(self, future) -> None:
        try:
            pred = future.result()
        except Exception:
            return
        if pred is not None:
            self._inference_result.emit(pred)

    @pyqtSlot(object)
    def _on_inference_result(self, pred) -> None:
        self.canvas.add_letter(pred)
        self._count_label.setText(f"{self.canvas.letter_count} letters")

    def _on_word_gap(self) -> None:
        self._recorder.record_event("word_gap")
        self.canvas.add_word_gap()

    @pyqtSlot(str)
    def _on_status(self, status: str) -> None:
        self._conn_label.setText(status)
        alive = "Disconnected" not in status and "complete" not in status.lower()
        self._disc_btn.setEnabled(alive)
        self._connect_btn.setEnabled(not alive)
        # Calibrate only available for live (non-replay) sources
        self._cal_btn.setEnabled(alive and not self._is_replay)

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Connection Error", msg)
        self._on_status("Disconnected")

    @pyqtSlot(int)
    def _on_thresh_changed(self, val: int) -> None:
        t = val / 100.0
        self._thresh_val_lbl.setText(f"{t:.2f}")
        self.canvas.set_threshold(t)

    # ── Actions ──────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        dlg = ConnectDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        src = dlg.source
        if src is None:
            return
        self._disconnect()
        self._source = src
        self._is_replay = dlg.is_replay
        if dlg.is_replay:
            self.canvas.clear_all()
            self._stroke_buf.reset()
        src.sample_received.connect(self._on_sample)
        src.stroke_complete.connect(self._on_stroke_from_replay)
        src.status_changed.connect(self._on_status)
        src.error_occurred.connect(self._on_error)
        self._recorder.start()
        self._rec_label.setVisible(True)
        src.start()

    def _disconnect(self) -> None:
        if self._source is not None:
            self._stroke_buf.flush()
            self._source.stop()
            self._source = None
        self._conn_label.setText("Not connected")
        self._disc_btn.setEnabled(False)
        self._connect_btn.setEnabled(True)
        self._is_calibrating = False
        self._cal_state = "idle"
        self._cal_btn.setEnabled(False)
        if hasattr(self, "_cal_overlay"):
            self._cal_overlay.hide()

    def _clear(self) -> None:
        reply = QMessageBox.question(
            self, "Clear Canvas", "Erase all letters?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.canvas.clear_all()
        self._stroke_buf.reset()
        self._recorder.start()
        self._count_label.setText("0 letters")

    def _save_pdf(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save as PDF", "", "PDF Files (*.pdf)"
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            self._export_pdf(path)
            QMessageBox.information(self, "Saved", f"PDF saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "PDF Error", str(exc))

    def _export_pdf(self, path: str) -> None:
        from PIL import Image
        pixmap = self.canvas.get_canvas_image()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        try:
            pixmap.save(tmp, "PNG")
            img = Image.open(tmp).convert("RGB")
            img.save(path, "PDF", resolution=150.0)
        finally:
            os.unlink(tmp)

    def _export_recording(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export IMU Recording", "",
            "IMU Recording (*.imu.json);;All Files (*)"
        )
        if not path:
            return
        if not path.endswith(".imu.json"):
            path += ".imu.json"
        self._recorder.save(path)
        n = self._recorder.sample_count
        dur = self._recorder.duration_s
        QMessageBox.information(
            self, "Exported",
            f"Saved {n:,} samples  ({dur:.1f} s)\nto:\n{path}"
        )

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _on_escape(self) -> None:
        if self.isFullScreen():
            self.showNormal()

    def closeEvent(self, event) -> None:
        self._disconnect()
        self._executor.shutdown(wait=False)
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#ffffff"))
    app.setPalette(palette)

    window = HandwritingApp()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
