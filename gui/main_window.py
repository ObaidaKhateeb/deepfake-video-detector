"""
gui/main_window.py
Main application window.
Layout:
  Left panel  — drop zone, file info, analyze button, progress
  Right panel — score dial, verdict, parameter bars, detail panel
"""

import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFileDialog, QProgressBar,
    QFrame, QScrollArea, QSizePolicy, QSpacerItem
)
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QFont, QIcon

from gui.styles import (
    MAIN_STYLE, BG_CARD, BG_ELEVATED, BORDER,
    TEXT_PRIMARY, TEXT_MUTED, ACCENT
)
from gui.widgets import ScoreDial, ParameterBar, VerdictBadge
from gui.analysis_worker import AnalysisWorker
from core.result import AggregatedResult


SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


class DropZone(QFrame):
    """Drag-and-drop area for video files."""

    def __init__(self, on_file_dropped, parent=None):
        super().__init__(parent)
        self._callback = on_file_dropped
        self.setAcceptDrops(True)
        self.setObjectName("card")
        self.setMinimumHeight(140)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(8)

        icon = QLabel("⬇")
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet(f"font-size: 32px; color: {TEXT_MUTED};")
        layout.addWidget(icon)

        msg = QLabel("Drop a video here")
        msg.setAlignment(Qt.AlignCenter)
        msg.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px;")
        layout.addWidget(msg)

        sub = QLabel("MP4, AVI, MOV, MKV, WebM …")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(sub)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(f"background: {BG_ELEVATED}; border: 1px solid {ACCENT}; border-radius: 10px;")

    def dragLeaveEvent(self, event):
        self.setStyleSheet("")

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet("")
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self._callback(path)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fake Video Detector")
        self.setMinimumSize(960, 640)
        self.resize(1100, 720)
        self.setStyleSheet(MAIN_STYLE)

        self._video_path = None
        self._worker = None

        self._build_ui()

    # ------------------------------------------------------------------ build

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        root.addWidget(self._build_left_panel(), stretch=2)
        root.addWidget(self._build_right_panel(), stretch=3)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        # Title
        title = QLabel("Fake Video\nDetector")
        title.setStyleSheet(f"""
            color: {TEXT_PRIMARY};
            font-size: 22px;
            font-weight: bold;
            line-height: 1.3;
        """)
        layout.addWidget(title)

        sub = QLabel("Analyze any video for signs of\ndeepfake or synthetic manipulation.")
        sub.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        layout.addWidget(sub)

        # Divider
        div = QFrame()
        div.setObjectName("divider")
        div.setFixedHeight(1)
        layout.addWidget(div)

        # Drop zone
        self._drop_zone = DropZone(self._on_file_selected)
        layout.addWidget(self._drop_zone)

        # Browse button
        browse_btn = QPushButton("Browse for File…")
        browse_btn.clicked.connect(self._browse)
        layout.addWidget(browse_btn)

        # File info card
        self._file_info = QLabel("No file selected")
        self._file_info.setWordWrap(True)
        self._file_info.setStyleSheet(f"""
            color: {TEXT_MUTED};
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 6px;
            padding: 10px;
            font-size: 11px;
            font-family: monospace;
        """)
        layout.addWidget(self._file_info)

        # Analyze button
        self._analyze_btn = QPushButton("Analyze Video")
        self._analyze_btn.setObjectName("primary")
        self._analyze_btn.setEnabled(False)
        self._analyze_btn.clicked.connect(self._start_analysis)
        layout.addWidget(self._analyze_btn)

        # Progress bar + status
        self._progress_bar = QProgressBar()
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(self._status_label)

        layout.addStretch()

        # Footer
        footer = QLabel("Analysis is heuristic.\nResults are indicative, not definitive.")
        footer.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        footer.setWordWrap(True)
        layout.addWidget(footer)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        # ── Top: dial + verdict side by side ──────────────────────────────
        top_row = QFrame()
        top_row.setObjectName("card")
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(20, 20, 20, 20)
        top_layout.setSpacing(20)

        self._dial = ScoreDial()
        top_layout.addWidget(self._dial)

        verdict_block = QVBoxLayout()
        verdict_block.setAlignment(Qt.AlignVCenter)

        self._verdict_badge = VerdictBadge("Awaiting Analysis", 0.0)
        verdict_block.addWidget(self._verdict_badge)

        self._overall_pct = QLabel("—")
        self._overall_pct.setAlignment(Qt.AlignCenter)
        self._overall_pct.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        verdict_block.addWidget(self._overall_pct)

        top_layout.addLayout(verdict_block, stretch=1)
        layout.addWidget(top_row)

        # ── Parameter bars ────────────────────────────────────────────────
        bars_card = QFrame()
        bars_card.setObjectName("card")
        bars_layout = QVBoxLayout(bars_card)
        bars_layout.setContentsMargins(18, 14, 18, 14)
        bars_layout.setSpacing(2)

        bars_title = QLabel("Parameter Breakdown")
        bars_title.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        bars_layout.addWidget(bars_title)

        self._param_bars_layout = QVBoxLayout()
        self._param_bars_layout.setSpacing(4)
        bars_layout.addLayout(self._param_bars_layout)

        self._placeholder_label = QLabel("Run an analysis to see parameter scores.")
        self._placeholder_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px; padding: 10px 0;")
        self._param_bars_layout.addWidget(self._placeholder_label)

        layout.addWidget(bars_card)

        # ── Detail panel (scrollable) ─────────────────────────────────────
        detail_card = QFrame()
        detail_card.setObjectName("card")
        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setContentsMargins(18, 14, 18, 14)

        detail_title = QLabel("Analysis Details")
        detail_title.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        detail_layout.addWidget(detail_title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(160)

        self._detail_text = QLabel("No details yet.")
        self._detail_text.setWordWrap(True)
        self._detail_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._detail_text.setStyleSheet(f"""
            color: {TEXT_MUTED};
            font-family: monospace;
            font-size: 11px;
            padding: 4px;
            background: transparent;
        """)
        scroll.setWidget(self._detail_text)
        detail_layout.addWidget(scroll)

        layout.addWidget(detail_card)

        return panel

    # ------------------------------------------------------------------ slots

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv);;All Files (*)"
        )
        if path:
            self._on_file_selected(path)

    def _on_file_selected(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTS:
            self._file_info.setText(f"⚠ Unsupported format: {ext}")
            return

        self._video_path = path
        size_mb = os.path.getsize(path) / (1024 * 1024)
        name = os.path.basename(path)
        self._file_info.setText(
            f"File: {name}\n"
            f"Size: {size_mb:.2f} MB\n"
            f"Path: {path}"
        )
        self._analyze_btn.setEnabled(True)
        self._reset_results()

    def _start_analysis(self):
        if not self._video_path:
            return

        self._analyze_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._reset_results()

        self._worker = AnalysisWorker(self._video_path)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, pct: int, step: str):
        self._progress_bar.setValue(pct)
        self._status_label.setText(f"Running: {step}…")

    def _on_finished(self, result: AggregatedResult):
        self._analyze_btn.setEnabled(True)
        self._progress_bar.setValue(100)
        self._status_label.setText("Analysis complete.")
        self._render_results(result)

    def _on_error(self, msg: str):
        self._analyze_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._status_label.setText(f"Error: {msg}")
        self._detail_text.setText(f"Error during analysis:\n{msg}")

    # ------------------------------------------------------------------ render

    def _reset_results(self):
        self._dial.set_score(0.0)
        self._verdict_badge.set_verdict("Awaiting Analysis", 0.0)
        self._overall_pct.setText("—")

        # Clear param bars
        while self._param_bars_layout.count():
            item = self._param_bars_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._placeholder_label = QLabel("Run an analysis to see parameter scores.")
        self._placeholder_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px; padding: 10px 0;")
        self._param_bars_layout.addWidget(self._placeholder_label)
        self._detail_text.setText("No details yet.")

    def _render_results(self, result: AggregatedResult):
        # Dial + verdict
        self._dial.set_score(result.overall_score)
        self._verdict_badge.set_verdict(result.verdict, result.overall_score)
        self._overall_pct.setText(
            f"Overall fake probability: {int(result.overall_score * 100)}%"
        )

        # Clear placeholder
        while self._param_bars_layout.count():
            item = self._param_bars_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add parameter bars
        for label, analyzer_result in result.components.items():
            bar = ParameterBar(
                label=label,
                score=analyzer_result.clamped_score(),
                confidence=analyzer_result.confidence,
            )
            self._param_bars_layout.addWidget(bar)

        # Build detail text
        lines = []
        for label, analyzer_result in result.components.items():
            lines.append(f"── {label} ──")
            for d in analyzer_result.details:
                lines.append(f"  {d}")
            lines.append("")

        self._detail_text.setText("\n".join(lines))
