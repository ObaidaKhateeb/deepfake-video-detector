"""
analyzers/brightness_flicker.py
Measures frame-to-frame luminance variation.
Deepfake generators normalize each frame independently, causing subtle
luminance inconsistencies that don't match natural lighting changes.

Score: 0 = smooth luminance flow, 1 = erratic flickering
"""

import cv2
import numpy as np
from typing import List
from core.result import AnalyzerResult


def analyze(frames: List[np.ndarray]) -> AnalyzerResult:
    label = "Brightness Flicker"

    if len(frames) < 3:
        return AnalyzerResult(
            label=label, score=0.0, confidence=0.0,
            details=["Not enough frames."]
        )

    luminances = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        luminances.append(float(np.mean(hsv[:, :, 2])))

    lum = np.array(luminances)
    diffs = np.abs(np.diff(lum))

    mean_diff = float(np.mean(diffs))
    std_diff  = float(np.std(diffs))

    # Coefficient of variation of luminance diffs
    cv_ratio = std_diff / (mean_diff + 1e-6)

    # Sudden spikes: diffs > 3x mean
    spikes = int(np.sum(diffs > 3.0 * mean_diff))
    spike_ratio = spikes / len(diffs)

    # Real video: CV low, spikes rare
    # Deepfake: CV higher, more spikes
    score = min(1.0, (cv_ratio / 3.0) * 0.5 + spike_ratio * 0.5)

    details = [
        f"Frames analyzed: {len(frames)}",
        f"Mean luminance: {float(np.mean(lum)):.1f}",
        f"Mean inter-frame luminance diff: {mean_diff:.2f}",
        f"Std of diffs: {std_diff:.2f}  (CV: {cv_ratio:.3f})",
        f"Luminance spike frames: {spikes} / {len(diffs)} ({spike_ratio*100:.1f}%)",
    ]

    return AnalyzerResult(label=label, score=score, confidence=0.85, details=details)
