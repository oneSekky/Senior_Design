"""
canvas.py — White-background letter canvas widget.

Each letter is a (64, 64) sigmoid prediction from the model.  Letters are
binarised, inverted (stroke=black on white), scaled up, and placed left-to-right
with automatic line wrapping.

Supports:
  - Variable display scale (set_letter_size)
  - Live threshold adjustment (set_threshold) — rebuilds all pixmaps from stored
    raw predictions so changes are instantaneous
  - Undo last item (letter or word gap)
  - Full clear
  - get_canvas_image() → QPixmap for PDF export
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QWidget

_MARGIN_L = 60
_MARGIN_T = 40
_MARGIN_R = 60
_LETTER_GAP = 8    # px between adjacent letters
_WORD_GAP = 52     # extra px for a word space


class CanvasWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setStyleSheet("background-color: white;")

        self._items: list[dict] = []    # {'type': 'letter'|'gap', 'pred': arr|None,
                                        #  'pixmap': QPixmap|None, 'rect': QRect|None}
        self._letter_size: int = 320    # display px (scaled from 64)
        self._line_height: int = 340
        self._threshold: float = 0.30
        self._cursor_x: int = _MARGIN_L
        self._cursor_y: int = _MARGIN_T
        self._scroll_y: int = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def add_letter(self, pred: np.ndarray) -> None:
        """pred: (64, 64) float32 sigmoid in [0, 1].  Strokes = high values."""
        self._wrap_if_needed()
        pixmap = self._pred_to_pixmap(pred)
        rect = QRect(self._cursor_x, self._cursor_y, self._letter_size, self._letter_size)
        self._items.append({
            "type": "letter",
            "pred": pred.copy(),
            "pixmap": pixmap,
            "rect": rect,
        })
        self._cursor_x += self._letter_size + _LETTER_GAP
        self._scroll_to_bottom()
        self.update()

    def add_word_gap(self) -> None:
        self._items.append({"type": "gap", "pred": None, "pixmap": None, "rect": None})
        self._cursor_x += _WORD_GAP
        self._wrap_if_needed()
        self._scroll_to_bottom()
        self.update()

    def undo_last(self) -> None:
        if not self._items:
            return
        item = self._items.pop()
        if item["type"] == "letter" and item["rect"] is not None:
            self._cursor_x = item["rect"].x()
            self._cursor_y = item["rect"].y()
        else:
            self._recalc_cursor()
        self.update()

    def clear_all(self) -> None:
        self._items.clear()
        self._cursor_x = _MARGIN_L
        self._cursor_y = _MARGIN_T
        self._scroll_y = 0
        self.update()

    def set_threshold(self, value: float) -> None:
        if value == self._threshold:
            return
        self._threshold = value
        for item in self._items:
            if item["type"] == "letter" and item["pred"] is not None:
                item["pixmap"] = self._pred_to_pixmap(item["pred"])
        self.update()

    def set_letter_size(self, size: int) -> None:
        if size == self._letter_size:
            return
        self._letter_size = size
        self._line_height = size + 18
        self._relayout()
        self.update()

    def get_canvas_image(self) -> QPixmap:
        pm = QPixmap(self.size())
        pm.fill(Qt.GlobalColor.white)
        p = QPainter(pm)
        self.render(p)
        p.end()
        return pm

    @property
    def letter_count(self) -> int:
        return sum(1 for i in self._items if i["type"] == "letter")

    # ── Private helpers ──────────────────────────────────────────────────────

    def _pred_to_pixmap(self, pred: np.ndarray) -> QPixmap:
        binary = (pred > self._threshold).astype(np.uint8)
        display = (1 - binary) * 255   # invert: stroke → black (0) on white (255)
        h, w = display.shape
        img = QImage(display.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(img).scaled(
            self._letter_size,
            self._letter_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _wrap_if_needed(self) -> None:
        if self._cursor_x + self._letter_size > self.width() - _MARGIN_R:
            self._cursor_x = _MARGIN_L
            self._cursor_y += self._line_height

    def _recalc_cursor(self) -> None:
        x, y = _MARGIN_L, _MARGIN_T
        for item in self._items:
            if item["type"] == "letter":
                if x + self._letter_size > self.width() - _MARGIN_R:
                    x = _MARGIN_L
                    y += self._line_height
                x += self._letter_size + _LETTER_GAP
            elif item["type"] == "gap":
                x += _WORD_GAP
                if x > self.width() - _MARGIN_R:
                    x = _MARGIN_L
                    y += self._line_height
        self._cursor_x = x
        self._cursor_y = y

    def _relayout(self) -> None:
        x, y = _MARGIN_L, _MARGIN_T
        for item in self._items:
            if item["type"] == "letter":
                if x + self._letter_size > self.width() - _MARGIN_R:
                    x = _MARGIN_L
                    y += self._line_height
                item["rect"] = QRect(x, y, self._letter_size, self._letter_size)
                item["pixmap"] = self._pred_to_pixmap(item["pred"])
                x += self._letter_size + _LETTER_GAP
            elif item["type"] == "gap":
                x += _WORD_GAP
                if x > self.width() - _MARGIN_R:
                    x = _MARGIN_L
                    y += self._line_height
        self._cursor_x = x
        self._cursor_y = y

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()   # typically ±120 per notch
        step = self._line_height
        if delta > 0:
            self._scroll_y = max(0, self._scroll_y - step)
        else:
            self._scroll_y = min(self._max_scroll(), self._scroll_y + step)
        self.update()
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._items:
            self._relayout()
        self._scroll_y = min(self._scroll_y, self._max_scroll())

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), Qt.GlobalColor.white)
        p.translate(0, -self._scroll_y)
        for item in self._items:
            if item["type"] == "letter" and item["pixmap"] and item["rect"]:
                p.drawPixmap(item["rect"].topLeft(), item["pixmap"])
        p.end()

    # ── Scroll helpers ───────────────────────────────────────────────────────

    def _max_scroll(self) -> int:
        content_bottom = self._cursor_y + self._letter_size
        return max(0, content_bottom - self.height())

    def _scroll_to_bottom(self) -> None:
        self._scroll_y = self._max_scroll()
