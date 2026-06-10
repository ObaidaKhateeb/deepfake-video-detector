"""
analyzers/temporal_consistency.py
Detects unnatural frame-to-frame instability in the face/scene region.
Deepfakes are generated per-frame and often flicker slightly between frames.

Score: 0 = stable/real, 1 = highly unstable/fake
"""

import cv2
import numpy as np
from typing import List
from core.result import AnalyzerResult


def analyze(frames: List[np.ndarray]) -> AnalyzerResult:
    label = "Temporal Consistency"

    if len(frames) < 4:
        return AnalyzerResult(
            label=label, score=0.0, confidence=0.0,
            details=["Not enough frames for temporal analysis."]
        )

    # Convert frames to grayscale and compute absolute diff between consecutive frames
    diffs = []
    for i in range(1, len(frames)):
        g1 = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(frames[i],     cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.mean(np.abs(g2 - g1))
        diffs.append(diff)

    diffs = np.array(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff  = float(np.std(diffs))

    # High std relative to mean = irregular flickering = suspicious
    # Coefficient of variation
    cv_ratio = std_diff / (mean_diff + 1e-6)

    # Spike detection: frames where diff is >2.5x the mean
    spikes = int(np.sum(diffs > 2.5 * mean_diff))
    spike_ratio = spikes / len(diffs)

    # Normalize to 0–1 score
    # cv_ratio > 1.5 and spike_ratio > 0.15 push toward fake
    score = min(1.0, (cv_ratio / 2.0) * 0.5 + spike_ratio * 0.5)

    details = [
        f"Frames analyzed: {len(frames)}",
        f"Mean inter-frame diff: {mean_diff:.2f}",
        f"Std of diffs: {std_diff:.2f}",
        f"Variation coefficient: {cv_ratio:.3f}",
        f"Spike frames: {spikes} / {len(diffs)} ({spike_ratio*100:.1f}%)",
    ]

    return AnalyzerResult(label=label, score=score, confidence=0.9, details=details)
