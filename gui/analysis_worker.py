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

import analyzers.temporal_consistency  as temporal
import analyzers.face_texture          as face_texture
import analyzers.compression_artifacts as compression
import analyzers.noise_pattern         as noise
import analyzers.brightness_flicker    as flicker
import analyzers.edge_sharpness        as sharpness
import analyzers.metadata              as metadata
import analyzers.content_verification  as content_verification


ANALYZER_STEPS = [
    "Loading video",
    "Temporal Consistency",
    "Face Texture",
    "Compression Artifacts",
    "Noise Pattern",
    "Brightness Flicker",
    "Edge Sharpness",
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

            # Step 1 — temporal
            step(1, "Temporal Consistency")
            results["Temporal Consistency"] = temporal.analyze(video.frames)

            # Step 2 — face texture
            step(2, "Face Texture")
            results["Face Texture"] = face_texture.analyze(video.frames)

            # Step 3 — compression
            step(3, "Compression Artifacts")
            results["Compression Artifacts"] = compression.analyze(video.frames)

            # Step 4 — noise
            step(4, "Noise Pattern")
            results["Noise Pattern"] = noise.analyze(video.frames)

            # Step 5 — flicker
            step(5, "Brightness Flicker")
            results["Brightness Flicker"] = flicker.analyze(video.frames)

            # Step 6 — sharpness
            step(6, "Edge Sharpness")
            results["Edge Sharpness"] = sharpness.analyze(video.frames)

            # Step 7 — metadata
            step(7, "Metadata")
            results["Metadata"], meta_dict = metadata.analyze(self.video_path)

            # Step 8 — content verification (audio transcript + frames + web search + metadata cross-checks)
            step(8, "Content Verification")
            results["Content Verification"] = content_verification.analyze(
                video.frames, self.video_path, meta_dict
            )

            # Step 9 — aggregate
            step(9, "Aggregating results")
            result = aggregate(results)

            self.progress.emit(100, "Done")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))
