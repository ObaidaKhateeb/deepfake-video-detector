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

## Install & Run

    pip install opencv-python-headless numpy scipy Pillow PyQt5

    # Optional — enables deep metadata analysis
    # Ubuntu/Debian: sudo apt install ffmpeg
    # Windows: download from https://ffmpeg.org

    python main.py

### Content Verification (optional but recommended)

The **Content Verification** analyzer extracts factual claims from the video using two
complementary approaches that run side by side:

- **Approach 1 — Audio:** the full audio track is transcribed locally by
  [Whisper](https://github.com/openai/whisper), capturing every spoken claim
  (names, titles, organizations said out loud).
- **Approach 2 — Visual:** frames are sampled at ~1 fps (instead of 5 arbitrary images)
  so that on-screen text, lower-thirds, and name plates are reliably captured throughout
  the video.

Both the transcript and the dense frames are sent together to Claude AI, which extracts every
verifiable factual claim. Those claims are then checked against live web search results.
For example, if someone says *"I'm John Smith, CEO of Acme Corp"* or a lower-third displays
that text, the analyzer will verify that the person, company, and role are real and consistent.

To enable it:

1. Install the extra dependencies:

       pip install anthropic duckduckgo_search openai-whisper

   Whisper also requires **ffmpeg** to decode the audio track:

       # Ubuntu/Debian
       sudo apt install ffmpeg
       # Windows — download from https://ffmpeg.org and add to PATH

2. Set your Anthropic API key as an environment variable before launching the app:

       # Windows (Command Prompt)
       set ANTHROPIC_API_KEY=your_key_here

       # Windows (PowerShell)
       $env:ANTHROPIC_API_KEY="your_key_here"

       # macOS / Linux
       export ANTHROPIC_API_KEY=your_key_here

   Get a key at https://console.anthropic.com

Each component degrades gracefully: if Whisper is not installed only the visual
approach runs; if the API key is missing the analyzer is skipped entirely and the
remaining seven analyzers still produce a result.

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
- Install `ffmpeg` for the metadata analyzer to report full encoding details.
