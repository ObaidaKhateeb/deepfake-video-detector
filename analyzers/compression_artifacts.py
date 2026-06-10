"""
analyzers/compression_artifacts.py
Analyzes DCT-domain blocking artifacts.
Deepfakes are often compressed twice (generation + re-encoding),
leaving detectable layered blocking patterns in the 8x8 DCT grid.

Score: 0 = clean compression, 1 = heavy/suspicious artifacts
"""

import cv2
import numpy as np
from typing import List
from core.result import AnalyzerResult


def _blocking_score(gray: np.ndarray) -> float:
    """
    Estimate JPEG blocking by comparing pixel differences at 8-pixel boundaries
    versus differences at non-boundary positions.
    A high ratio of boundary/non-boundary diff = strong blocking = suspicious.
    """
    h, w = gray.shape
    g = gray.astype(np.float32)

    # Horizontal boundary diffs (at columns 8, 16, 24 ...)
    boundary_cols = np.arange(8, w - 1, 8)
    non_boundary_cols = np.arange(9, w - 1, 8)

    if len(boundary_cols) == 0 or len(non_boundary_cols) == 0:
        return 0.0

    bd_h = np.mean(np.abs(g[:, boundary_cols] - g[:, boundary_cols - 1]))
    nb_h = np.mean(np.abs(g[:, non_boundary_cols] - g[:, non_boundary_cols - 1]))

    # Vertical boundary diffs
    boundary_rows = np.arange(8, h - 1, 8)
    non_boundary_rows = np.arange(9, h - 1, 8)

    bd_v = np.mean(np.abs(g[boundary_rows, :] - g[boundary_rows - 1, :]))
    nb_v = np.mean(np.abs(g[non_boundary_rows, :] - g[non_boundary_rows - 1, :]))

    boundary_diff     = (bd_h + bd_v) / 2.0
    non_boundary_diff = (nb_h + nb_v) / 2.0

    ratio = boundary_diff / (non_boundary_diff + 1e-6)
    return float(ratio)


def analyze(frames: List[np.ndarray]) -> AnalyzerResult:
    label = "Compression Artifacts"

    sample = frames[::max(1, len(frames) // 15)][:15]
    ratios = []

    for frame in sample:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ratios.append(_blocking_score(gray))

    if not ratios:
        return AnalyzerResult(label=label, score=0.0, confidence=0.0,
                              details=["No frames to analyze."])

    mean_ratio = float(np.mean(ratios))
    std_ratio  = float(np.std(ratios))

    # ratio ≈ 1.0 → no blocking. ratio > 1.4 → noticeable blocking.
    # Deepfake double-compression typically pushes ratio > 1.5
    score = max(0.0, min(1.0, (mean_ratio - 1.0) / 1.0))

    details = [
        f"Frames sampled: {len(sample)}",
        f"Mean boundary/non-boundary diff ratio: {mean_ratio:.3f}",
        f"Std across frames: {std_ratio:.3f}",
        f"Ratio > 1.4 suggests double-compression artifacts",
    ]

    return AnalyzerResult(label=label, score=score, confidence=0.85, details=details)
