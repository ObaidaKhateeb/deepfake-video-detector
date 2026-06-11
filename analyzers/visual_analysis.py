"""
analyzers/visual_analysis.py
Runs the trained EfficientNet-B4 ONNX model on video frames.
Score: 0 = real, 1 = fake

Model file: models/deepfake_detector.onnx
To update the model, replace that file — nothing else needs to change.

onnxruntime is run in a subprocess to isolate DLL issues on Windows.
"""

import os
import sys
import json
import tempfile
import subprocess
import numpy as np
import cv2
from typing import List
from core.result import AnalyzerResult

_MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "deepfake_detector.onnx")
)

_IMG_SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (_IMG_SIZE, _IMG_SIZE)).astype(np.float32) / 255.0
    normalized = (resized - _MEAN) / _STD
    return normalized.transpose(2, 0, 1)  # (3, H, W)


def analyze(frames: List[np.ndarray]) -> AnalyzerResult:
    label = "Visual Analysis"
    details = []

    if not os.path.isfile(_MODEL_PATH):
        details.append(f"Model not found at {_MODEL_PATH}")
        details.append("Place deepfake_detector.onnx in models/")
        return AnalyzerResult(label=label, score=0.5, confidence=0.0, details=details)

    if not frames:
        details.append("No frames provided")
        return AnalyzerResult(label=label, score=0.5, confidence=0.0, details=details)

    # Sample up to 16 evenly spaced frames
    indices = np.linspace(0, len(frames) - 1, min(16, len(frames)), dtype=int)
    sampled = [_preprocess(frames[i]) for i in indices]
    batch = np.stack(sampled).astype(np.float32)  # (N, 3, H, W)

    # Save frames to temp file, run inference in subprocess to avoid DLL crash
    try:
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            frames_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            result_path = f.name

        np.save(frames_path, batch)

        script = (
            "import numpy as np, json, sys\n"
            "import onnxruntime as ort\n"
            f"sess = ort.InferenceSession({repr(_MODEL_PATH)}, providers=['CPUExecutionProvider'])\n"
            f"batch = np.load({repr(frames_path)})\n"
            "probs = []\n"
            "inp = sess.get_inputs()[0].name\n"
            "for i in range(len(batch)):\n"
            "    raw = sess.run(None, {inp: batch[i:i+1]})[0]\n"
            "    e = np.exp(raw - raw.max()); p = e / e.sum()\n"
            "    probs.append(float(p[0][1]))\n"
            f"json.dump(probs, open({repr(result_path)}, 'w'))\n"
        )

        # Use Python 3.13 because onnxruntime doesn't support Python 3.14 yet
        _PY313 = r"C:\Users\okhatib\AppData\Local\Programs\Python\Python313\python.exe"
        py_exe = _PY313 if os.path.isfile(_PY313) else sys.executable
        proc = subprocess.run(
            [py_exe, "-c", script],
            capture_output=True, text=True, timeout=120
        )

        if proc.returncode != 0:
            details.append(f"Inference subprocess failed: {proc.stderr.strip()[:300]}")
            return AnalyzerResult(label=label, score=0.5, confidence=0.0, details=details)

        fake_probs = json.load(open(result_path))

    finally:
        for p in [frames_path, result_path]:
            try:
                os.unlink(p)
            except Exception:
                pass

    score = float(np.mean(fake_probs))
    score_std = float(np.std(fake_probs))

    details.append(f"Frames analyzed: {len(fake_probs)}")
    details.append(f"Fake probability: {score:.3f} (±{score_std:.3f})")
    details.append(f"Min: {min(fake_probs):.3f}  Max: {max(fake_probs):.3f}")

    confidence = max(0.3, 1.0 - score_std * 2)

    return AnalyzerResult(label=label, score=score, confidence=confidence, details=details)
