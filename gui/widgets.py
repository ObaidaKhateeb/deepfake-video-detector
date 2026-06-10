"""
gui/widgets.py
Reusable custom PyQt5 widgets:
  - ScoreDial      : circular gauge showing the overall fake score
  - ParameterBar   : horizontal bar for a single analyzer score
  - VerdictBadge   : colored label showing the verdict text
"""

import math
from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout, QFrame
from PyQt5.QtCore import Qt, QRectF, QSize
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QBrush, QPainterPath
from gui.styles import score_color, TEXT_PRIMARY, TEXT_MUTED, BG_ELEVATED, BORDER


class ScoreDial(QWidget):
    """
    Circular arc gauge.
    Arc sweeps from 225° (bottom-left) clockwise to 315° (bottom-right).
    Color transitions from green → orange → red based on score.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._score = 0.0
        self.setMinimumSize(200, 200)

    def set_score(self, score: float):
        self._score = max(0.0, min(1.0, score))
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(200, 200)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        size = min(w, h) - 20
        x = (w - size) / 2
        y = (h - size) / 2
        rect = QRectF(x, y, size, size)

        # Track arc (background)
        track_pen = QPen(QColor(BORDER))
        track_pen.setWidth(14)
        track_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(track_pen)
        painter.drawArc(rect, int(225 * 16), int(-270 * 16))

        # Value arc
        sweep = -int(270 * self._score * 16)
        if sweep != 0:
            color = QColor(score_color(self._score))
            val_pen = QPen(color)
            val_pen.setWidth(14)
            val_pen.setCapStyle(Qt.RoundCap)
            painter.setPen(val_pen)
            painter.drawArc(rect, int(225 * 16), sweep)

        # Center text: score percentage
        painter.setPen(QColor(TEXT_PRIMARY))
        pct_font = QFont("Segoe UI", int(size * 0.18), QFont.Bold)
        painter.setFont(pct_font)
        painter.drawText(rect, Qt.AlignCenter, f"{int(self._score * 100)}%")

        # Sub-label
        painter.setPen(QColor(TEXT_MUTED))
        sub_font = QFont("Segoe UI", int(size * 0.07))
        painter.setFont(sub_font)
        label_rect = QRectF(x, y + size * 0.58, size, size * 0.15)
        painter.drawText(label_rect, Qt.AlignCenter, "FAKE SCORE")


class ParameterBar(QWidget):
    """
    One row: parameter name | colored bar | percentage
    """

    def __init__(self, label: str, score: float = 0.0,
                 confidence: float = 1.0, parent=None):
        super().__init__(parent)
        self._score = score
        self._label = label
        self._confidence = confidence
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(10)

        # Label
        lbl = QLabel(self._label)
        lbl.setFixedWidth(170)
        lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px;")
        layout.addWidget(lbl)

        # Bar widget
        self._bar = _BarWidget(self._score)
        self._bar.setFixedHeight(10)
        layout.addWidget(self._bar, stretch=1)

        # Percentage
        color = score_color(self._score)
        pct_lbl = QLabel(f"{int(self._score * 100)}%")
        pct_lbl.setFixedWidth(40)
        pct_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        pct_lbl.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 12px;")
        layout.addWidget(pct_lbl)

        # Confidence indicator
        conf_lbl = QLabel(f"conf {int(self._confidence * 100)}%")
        conf_lbl.setFixedWidth(55)
        conf_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        conf_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        layout.addWidget(conf_lbl)


class _BarWidget(QWidget):
    def __init__(self, score: float, parent=None):
        super().__init__(parent)
        self._score = score

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        r = h / 2

        # Background
        painter.setBrush(QBrush(QColor(BORDER)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, w, h, r, r)

        # Fill
        fill_w = int(w * self._score)
        if fill_w > 0:
            painter.setBrush(QBrush(QColor(score_color(self._score))))
            painter.drawRoundedRect(0, 0, fill_w, h, r, r)


class VerdictBadge(QLabel):
    """Large colored label for the verdict text."""

    def __init__(self, verdict: str = "—", score: float = 0.0, parent=None):
        super().__init__(verdict, parent)
        self.set_verdict(verdict, score)

    def set_verdict(self, verdict: str, score: float):
        color = score_color(score)
        self.setText(verdict)
        self.setStyleSheet(f"""
            color: {color};
            font-size: 22px;
            font-weight: bold;
            letter-spacing: 1px;
            background: transparent;
        """)
        self.setAlignment(Qt.AlignCenter)
