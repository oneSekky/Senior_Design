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

import os
import sys
import tempfile
import time

import numpy as np

from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QColor, QKeySequence, QPalette, QShortcut
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

from canvas import CanvasWidget
from data_source import BLESource, DataSource, ReplaySource, SerialSource
from recorder import IMURecorder
from stroke_buffer import StrokeBuffer
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
        else:  # Replay
            path = self._rep_path.text().strip()
            if not path:
                QMessageBox.warning(self, "Replay", "Select a recording file.")
                return
            speed_map = {"0.5×": 0.5, "1×": 1.0, "2×": 2.0, "5×": 5.0, "Instant": 200.0}
            speed = speed_map.get(self._speed_combo.currentText(), 1.0)
            self._source = ReplaySource(path, speed)
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


# ── Main window ───────────────────────────────────────────────────────────────

class HandwritingApp(QMainWindow):
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
        self._is_replay = False   # True while a ReplaySource is active

        self._build_ui()
        self._build_shortcuts()

        self._recorder.start()

        # Show window immediately, then load model after event loop starts
        self.show()
        QTimer.singleShot(50, self._load_model)

    # ── Commented-out calibration overlay ────────────────────────────────────
    # Uncomment this entire block once the physical pen is working.
    # Calibration asks the user to write a circle/letter on connect so the
    # activity threshold is auto-set to their specific pen and writing style.
    #
    # class _CalibrationOverlay(QWidget):
    #     """Semi-transparent overlay shown during calibration."""
    #     finished = pyqtSignal()
    #
    #     def __init__(self, parent):
    #         super().__init__(parent)
    #         self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
    #         self.setStyleSheet("background: rgba(0,0,0,180);")
    #         lay = QVBoxLayout(self)
    #         lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
    #         self._lbl = QLabel("Write a circle to calibrate", self)
    #         self._lbl.setStyleSheet("color:white; font-size:28px; font-weight:bold;")
    #         self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    #         lay.addWidget(self._lbl)
    #
    #     def update_status(self, text: str) -> None:
    #         self._lbl.setText(text)
    #
    #     def resizeEvent(self, e):
    #         super().resizeEvent(e)
    #         self.setGeometry(self.parent().rect())
    #
    # def _start_calibration(self) -> None:
    #     """Show overlay and run one calibration stroke through the stroke buffer."""
    #     self._cal_overlay = HandwritingApp._CalibrationOverlay(self)
    #     self._cal_overlay.show()
    #     self._stroke_buf.start_calibration(self._on_calibration_done)
    #
    # def _on_calibration_done(self, suggested_threshold: float) -> None:
    #     self._cal_overlay.update_status(
    #         f"Calibrated  (threshold = {suggested_threshold:.1f} mg)"
    #     )
    #     self._stroke_buf.set_threshold(suggested_threshold)
    #     self._thresh_slider.setValue(int(suggested_threshold))
    #     QTimer.singleShot(1500, self._cal_overlay.hide)

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

        self.canvas = CanvasWidget()
        vbox.addWidget(self.canvas, stretch=1)

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
        # Replay drives inference via stroke_complete events — skip stroke buffer
        if not self._is_replay:
            self._stroke_buf.feed(sample)

    def _on_stroke_complete(self, stroke: np.ndarray) -> None:
        """Called by StrokeBuffer for live (USB/BLE) data."""
        self._recorder.record_event("stroke_complete")
        self._run_inference(stroke)

    @pyqtSlot(object)
    def _on_stroke_from_replay(self, stroke) -> None:
        """Called by ReplaySource for each recorded stroke event."""
        self._run_inference(np.asarray(stroke, dtype=np.float32))

    def _run_inference(self, stroke: np.ndarray) -> None:
        if self._inference is None:
            return
        pred = self._inference.predict(stroke)
        if pred is not None:
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
        # To enable calibration on connect for live sources, uncomment:
        # if not dlg.is_replay:
        #     QTimer.singleShot(200, self._start_calibration)

    def _disconnect(self) -> None:
        if self._source is not None:
            self._stroke_buf.flush()
            self._source.stop()
            self._source = None
        self._conn_label.setText("Not connected")
        self._disc_btn.setEnabled(False)
        self._connect_btn.setEnabled(True)

    def _clear(self) -> None:
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
