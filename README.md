# Fake Video Detector

A desktop GUI tool that analyzes a video file and outputs a **fake probability score** (0–100%) based on eight independent analyzers — seven heuristic and one AI-powered content verification step.

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
    │   ├── metadata.py                  # File metadata anomalies (ffprobe)
    │   └── content_verification.py      # AI vision + web search claim verification
    └── gui/
        ├── styles.py                    # Color tokens + Qt stylesheet
        ├── widgets.py                   # ScoreDial, ParameterBar, VerdictBadge
        ├── analysis_worker.py           # QThread that runs analyzers off main thread
        └── main_window.py               # Full application window

---

## Run

    python main.py

All required packages are installed automatically on first run — no manual `pip install` needed.
Restart after the first run is handled automatically as well.

> **Python 3.14+:** audio transcription via Whisper is skipped because PyTorch does not yet
> support Python 3.14. All other analyzers work normally.

### ffmpeg (optional)

Installing [ffmpeg](https://ffmpeg.org) enables deep metadata analysis and audio transcription:

    # Ubuntu / Debian
    sudo apt install ffmpeg

    # Windows — download from https://ffmpeg.org and add the bin/ folder to PATH

### Content Verification

The **Content Verification** analyzer uses Claude AI to extract verifiable factual claims
from the video and cross-checks them against live web search results and the file's own
metadata. It uses two complementary sources:

- **Audio** — the spoken transcript (via Whisper, requires ffmpeg)
- **Visual** — frames sampled at visually distinct intervals to capture on-screen text,
  lower-thirds, and name plates

Claims can be anything world-verifiable: a named person's role, a news event, a statistic,
a quote, a location. Each claim is checked against DuckDuckGo. For person + company claims,
Apollo.io is tried first (more precise) before falling back to DuckDuckGo.

The analyzer also cross-checks the video's file metadata (GPS location, creation timestamp,
device info, encoder) against what the content claims — a mismatch is a strong manipulation signal.

To enable Content Verification, set your Anthropic API key before launching:

    # Windows (PowerShell)
    $env:ANTHROPIC_API_KEY = "your_key_here"

    # macOS / Linux
    export ANTHROPIC_API_KEY=your_key_here

Get a key at https://console.anthropic.com

To also enable Apollo.io person verification (optional):

    $env:APOLLO_API_KEY = "your_key_here"   # PowerShell
    export APOLLO_API_KEY=your_key_here     # macOS / Linux

Get a free key at https://app.apollo.io

Each component degrades gracefully: if Whisper is unavailable only visual frames are used;
if API keys are missing those features are skipped and the remaining analyzers still run.

---

## Analyzers & Weights

| Analyzer              | Weight | What it measures |
|-----------------------|--------|-----------------|
| Temporal Consistency  | 25%    | Frame-to-frame flickering / instability |
| Face Texture          | 20%    | Skin smoothness (GANs over-smooth faces) |
| Compression Artifacts | 15%    | DCT blocking from double-compression |
| Noise Pattern         | 15%    | Sensor noise consistency across regions |
| Brightness Flicker    | 5%     | Unnatural luminance jumps |
| Edge Sharpness        | 5%     | Face vs background sharpness mismatch |
| Metadata              | 5%     | Missing/suspicious file metadata |
| Content Verification  | 10%    | AI vision + web search claim verification *(requires API key)* |

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
- Install `ffmpeg` for full metadata analysis and audio transcription.
