"""
analyzers/noise_pattern.py
Real cameras produce consistent sensor noise (photon noise, read noise).
Synthetic/deepfake regions often have a different or absent noise floor,
and the noise pattern may be inconsistent across frame regions.

Score: 0 = consistent natural noise, 1 = abnormal noise pattern
"""

import cv2
import numpy as np
from typing import List
from core.result import AnalyzerResult


def _estimate_noise(gray: np.ndarray) -> float:
    """
    Estimate noise level using the Laplacian-based method.
    Returns the mean absolute deviation of the high-frequency residual.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float32) - blurred.astype(np.float32)
    return float(np.mean(np.abs(residual)))


def _region_noise_std(gray: np.ndarray, n_regions: int = 9) -> float:
    """
    Split the frame into a grid of regions, compute noise per region,
    then return std of those noise values.
    High std = inconsistent noise = suspicious (face-swap boundary).
    """
    h, w = gray.shape
    rh, rw = h // 3, w // 3
    noises = []
    for i in range(3):
        for j in range(3):
            region = gray[i * rh:(i + 1) * rh, j * rw:(j + 1) * rw]
            noises.append(_estimate_noise(region))
    return float(np.std(noises))


def analyze(frames: List[np.ndarray]) -> AnalyzerResult:
    label = "Noise Pattern"

    sample = frames[::max(1, len(frames) // 15)][:15]

    global_noises = []
    region_stds   = []

    for frame in sample:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        global_noises.append(_estimate_noise(gray))
        region_stds.append(_region_noise_std(gray))

    if not global_noises:
        return AnalyzerResult(label=label, score=0.0, confidence=0.0,
                              details=["No frames to analyze."])

    mean_noise      = float(np.mean(global_noises))
    noise_temporal_std = float(np.std(global_noises))   # should be low for real video
    mean_region_std = float(np.mean(region_stds))

    # Very low global noise = possibly synthetic
    low_noise_score = max(0.0, 1.0 - mean_noise / 4.0)

    # High temporal variation in noise = suspicious
    temporal_score = min(1.0, noise_temporal_std / 2.0)

    # High spatial inconsistency = suspicious
    spatial_score = min(1.0, mean_region_std / 3.0)

    score = 0.4 * low_noise_score + 0.3 * temporal_score + 0.3 * spatial_score

    details = [
        f"Mean global noise level: {mean_noise:.3f}",
        f"Temporal noise std (frame-to-frame): {noise_temporal_std:.3f}",
        f"Mean spatial noise inconsistency: {mean_region_std:.3f}",
        f"Low noise or high inconsistency = suspicious",
    ]

    return AnalyzerResult(label=label, score=score, confidence=0.8, details=details)
