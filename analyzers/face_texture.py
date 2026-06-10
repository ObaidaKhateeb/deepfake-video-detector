"""
analyzers/face_texture.py
Analyzes skin texture in detected face regions.
GAN-generated faces often have unnaturally smooth or overly uniform texture.

Score: 0 = natural texture, 1 = suspiciously smooth/fake
"""

import cv2
import numpy as np
from typing import List
from core.result import AnalyzerResult


def _detect_faces(frame: np.ndarray):
    """Use OpenCV Haar cascade to find face bounding boxes."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return faces


def _texture_score(roi: np.ndarray) -> float:
    """
    Compute a texture richness score for a face ROI using Laplacian variance.
    Low variance = smooth = suspicious.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))


def analyze(frames: List[np.ndarray]) -> AnalyzerResult:
    label = "Face Texture"

    texture_scores = []
    faces_found = 0

    # Sample up to 20 frames for face analysis (expensive)
    sample = frames[::max(1, len(frames) // 20)][:20]

    for frame in sample:
        faces = _detect_faces(frame)
        for (x, y, w, h) in faces:
            roi = frame[y:y + h, x:x + w]
            if roi.size == 0:
                continue
            faces_found += 1
            texture_scores.append(_texture_score(roi))

    if not texture_scores:
        return AnalyzerResult(
            label=label, score=0.3, confidence=0.2,
            details=["No faces detected — texture analysis skipped.",
                     "Score defaulted to 0.3 (neutral)."]
        )

    mean_texture = float(np.mean(texture_scores))

    # Calibration: real faces typically have Laplacian variance > 200
    # Very smooth/fake faces: < 80
    # We map: 0 variance → score 1.0 (fake), 300+ variance → score 0.0 (real)
    score = max(0.0, min(1.0, 1.0 - (mean_texture / 300.0)))

    details = [
        f"Faces analyzed across {len(sample)} sampled frames: {faces_found}",
        f"Mean Laplacian variance (texture richness): {mean_texture:.1f}",
        f"Low variance = smooth skin = more likely fake",
        f"Texture score range: 0 (rich/real) → 1 (smooth/fake)",
    ]

    return AnalyzerResult(
        label=label, score=score,
        confidence=min(1.0, faces_found / 10.0),
        details=details
    )
