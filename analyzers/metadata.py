"""
analyzers/metadata.py
Analyzes video file metadata for suspicious patterns.
Missing metadata, unusual encoding parameters, or
atypical file characteristics can indicate synthetic origin.

Score: 0 = normal metadata, 1 = suspicious metadata
"""

import os
import subprocess
import json
from core.result import AnalyzerResult


def _run_ffprobe(path: str) -> dict:
    """Run ffprobe if available and return parsed JSON."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return {}


def analyze(path: str) -> AnalyzerResult:
    label = "Metadata"
    score = 0.0
    details = []
    suspicious_flags = 0

    # Basic file checks
    file_size = os.path.getsize(path)
    ext = os.path.splitext(path)[1].lower()
    details.append(f"File: {os.path.basename(path)}")
    details.append(f"Size: {file_size / (1024*1024):.2f} MB")
    details.append(f"Extension: {ext}")

    # Unusually small file for a video = possibly synthetic/compressed
    if file_size < 500_000:   # < 500 KB
        details.append("⚠ Very small file size — may indicate synthetic or heavily compressed content")
        suspicious_flags += 1

    # Try ffprobe for deep metadata
    meta = _run_ffprobe(path)
    if meta:
        fmt = meta.get("format", {})
        streams = meta.get("streams", [])

        # Check for missing encoder tag (real cameras usually write encoder info)
        encoder = fmt.get("tags", {}).get("encoder", "") or fmt.get("tags", {}).get("ENCODER", "")
        if not encoder:
            details.append("⚠ No encoder tag in metadata (common in synthetic media)")
            suspicious_flags += 1
        else:
            details.append(f"Encoder: {encoder}")

        # Check creation time
        creation = fmt.get("tags", {}).get("creation_time", "")
        if creation:
            details.append(f"Creation time: {creation}")
        else:
            details.append("⚠ No creation timestamp in metadata")
            suspicious_flags += 1

        # Video stream check
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

        if video_streams:
            vs = video_streams[0]
            codec = vs.get("codec_name", "unknown")
            details.append(f"Video codec: {codec}")
            # Unusual codecs can indicate re-encoding
            if codec not in ("h264", "hevc", "vp9", "vp8", "av1", "mpeg4"):
                details.append(f"⚠ Uncommon video codec: {codec}")
                suspicious_flags += 1

        if not audio_streams:
            details.append("⚠ No audio stream detected")
            suspicious_flags += 1
        else:
            details.append(f"Audio streams: {len(audio_streams)}")

    else:
        details.append("ffprobe not available — deep metadata analysis skipped")
        details.append("Install ffmpeg for full metadata inspection")

    # Score: each flag adds ~0.2
    score = min(1.0, suspicious_flags * 0.22)

    confidence = 0.9 if meta else 0.4   # lower confidence without ffprobe

    return AnalyzerResult(label=label, score=score,
                          confidence=confidence, details=details)
