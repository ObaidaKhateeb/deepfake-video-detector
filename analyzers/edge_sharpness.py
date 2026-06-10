"""
analyzers/edge_sharpness.py
Compares edge sharpness between face regions and background.
Face-swap deepfakes often produce a face that is sharper or blurrier
than the surrounding scene — a resolution/quality mismatch.

Score: 0 = consistent sharpness, 1 = strong face/background mismatch
"""

import cv2
import numpy as np
from typing import List
from core.result import AnalyzerResult


def _sharpness(region: np.ndarray) -> float:
    """Laplacian variance as sharpness metric."""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def analyze(frames: List[np.ndarray]) -> AnalyzerResult:
    label = "Edge Sharpness"

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    mismatches = []
    faces_found = 0
    sample = frames[::max(1, len(frames) // 15)][:15]

    for frame in sample:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                         minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            continue

        frame_sharpness = _sharpness(frame)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            if face_roi.size == 0:
                continue
            faces_found += 1
            face_sharpness = _sharpness(face_roi)

            # Build a background mask: full frame minus the face region
            mask = np.ones(frame.shape[:2], dtype=bool)
            mask[y:y + h, x:x + w] = False
            bg_pixels = frame[mask]
            if bg_pixels.size < 100:
                continue

            # Reconstruct background region for sharpness estimate
            bg_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bg_roi[y:y + h, x:x + w] = int(np.mean(bg_roi))  # blank face area
            bg_sharpness = float(cv2.Laplacian(bg_roi, cv2.CV_64F).var())

            # Mismatch ratio
            ratio = face_sharpness / (bg_sharpness + 1e-6)
            # ratio near 1.0 = consistent. Far from 1 (either too sharp or too blurry) = suspicious
            mismatch = abs(np.log(ratio + 1e-6))   # log scale, symmetric
            mismatches.append(mismatch)

    if not mismatches:
        return AnalyzerResult(
            label=label, score=0.2, confidence=0.1,
            details=["No faces detected — sharpness mismatch analysis skipped."]
        )

    mean_mismatch = float(np.mean(mismatches))
    # log mismatch > 1.5 = very strong mismatch
    score = min(1.0, mean_mismatch / 2.0)

    details = [
        f"Faces analyzed: {faces_found} across {len(sample)} sampled frames",
        f"Mean log sharpness mismatch (face vs background): {mean_mismatch:.3f}",
        f"0 = perfectly matched sharpness, >1.5 = strong mismatch",
    ]

    return AnalyzerResult(
        label=label, score=score,
        confidence=min(1.0, faces_found / 8.0),
        details=details
    )
