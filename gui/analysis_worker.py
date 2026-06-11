"""
gui/analysis_worker.py
QThread worker that runs all analyzers off the main thread
so the GUI stays responsive.
Emits progress updates and the final AggregatedResult.
"""

from PyQt5.QtCore import QThread, pyqtSignal
from core.video_loader import load_video
from core.aggregator import aggregate
from core.result import AggregatedResult, AnalyzerResult

import analyzers.visual_analysis      as visual_analysis
import analyzers.metadata             as metadata
import analyzers.content_verification as content_verification


ANALYZER_STEPS = [
    "Loading video",
    "Visual Analysis",
    "Metadata",
    "Content Verification",
    "Aggregating results",
]


class AnalysisWorker(QThread):
    progress    = pyqtSignal(int, str)          # (percent, step_name)
    finished    = pyqtSignal(object)            # AggregatedResult
    error       = pyqtSignal(str)               # error message

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.video_path = video_path

    def run(self):
        try:
            total = len(ANALYZER_STEPS)

            def step(i, name):
                self.progress.emit(int(i / total * 100), name)

            # Step 0 — load video
            step(0, "Loading video")
            video = load_video(self.video_path)

            results = {}

            # Step 1 — AI visual analysis (EfficientNet-B4 ONNX)
            step(1, "Visual Analysis")
            results["Visual Analysis"] = visual_analysis.analyze(video.frames)

            # Step 2 — metadata
            step(2, "Metadata")
            results["Metadata"], meta_dict = metadata.analyze(self.video_path)

            # Step 3 — content verification
            step(3, "Content Verification")
            results["Content Verification"] = content_verification.analyze(
                video.frames, self.video_path, meta_dict
            )

            # Step 4 — aggregate
            step(4, "Aggregating results")
            result = aggregate(results)

            self.progress.emit(100, "Done")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))
