# Fake Video Detector

A desktop GUI tool that analyzes a video file and outputs a **fake probability score** (0–100%) based on seven independent heuristic analyzers.

---

## Project Structure

    fake_video_detector/
    ├── main.py                          # Entry point
    ├── core/
    │   ├── video_loader.py              # Loads video, extracts frames
    │   ├── result.py                    # AnalyzerResult / AggregatedResult dataclasses
    │   └── aggregator.py               # Weighted combination of all scores
    ├── analyzers/
    │   ├── temporal_consistency.py      # Frame-to-frame instability
    │   ├── face_texture.py              # Skin texture smoothness (Laplacian variance)
    │   ├── compression_artifacts.py     # DCT blocking / double-compression
    │   ├── noise_pattern.py             # Sensor noise consistency
    │   ├── brightness_flicker.py        # Luminance spike detection
    │   ├── edge_sharpness.py            # Face vs background sharpness mismatch
    │   └── metadata.py                  # File metadata anomalies (ffprobe)
    └── gui/
        ├── styles.py                    # Color tokens + Qt stylesheet
        ├── widgets.py                   # ScoreDial, ParameterBar, VerdictBadge
        ├── analysis_worker.py           # QThread that runs analyzers off main thread
        └── main_window.py               # Full application window

---

## Install & Run

    pip install opencv-python-headless numpy scipy Pillow PyQt5

    # Optional — enables deep metadata analysis
    # Ubuntu/Debian: sudo apt install ffmpeg
    # Windows: download from https://ffmpeg.org

    python main.py

---

## Analyzers & Weights

| Analyzer              | Weight | What it measures |
|-----------------------|--------|-----------------|
| Temporal Consistency  | 25%    | Frame-to-frame flickering / instability |
| Face Texture          | 20%    | Skin smoothness (GANs over-smooth faces) |
| Compression Artifacts | 15%    | DCT blocking from double-compression |
| Noise Pattern         | 15%    | Sensor noise consistency across regions |
| Brightness Flicker    | 10%    | Unnatural luminance jumps |
| Edge Sharpness        | 10%    | Face vs background sharpness mismatch |
| Metadata              | 5%     | Missing/suspicious file metadata |

---

## Score Interpretation

| Score Range | Verdict       |
|-------------|---------------|
| 0 – 34%     | Likely Real   |
| 35 – 64%    | Suspicious    |
| 65 – 100%   | Likely Fake   |

---

## Notes

- All analysis is **heuristic** — no trained ML model is bundled.
- Results are **indicative**, not forensically definitive.
- Face-dependent analyzers (texture, sharpness) fall back gracefully when no face is detected.
- Install `ffmpeg` for the metadata analyzer to report full encoding details.
