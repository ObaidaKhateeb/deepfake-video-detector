"""
analyzers/metadata.py
Analyzes video file metadata for suspicious patterns.
Uses ffmpeg (bundled via imageio-ffmpeg) — no separate ffprobe needed.
"""

import os
import re
import subprocess
import sys
from typing import Dict, Optional, Tuple
from core.result import AnalyzerResult


def _get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _run_ffmpeg_info(path: str) -> str:
    """Run ffmpeg -i and return stderr output (where ffmpeg prints stream info)."""
    try:
        result = subprocess.run(
            [_get_ffmpeg_exe(), "-i", path],
            capture_output=True, text=True, timeout=10
        )
        return result.stderr
    except Exception:
        return ""


def _parse_ffmpeg_output(output: str) -> dict:
    """Parse ffmpeg -i stderr output into a structured dict."""
    data = {
        "encoder": "",
        "creation_time": "",
        "video_codec": "",
        "audio_streams": 0,
        "gps": None,
        "make": "",
        "model": "",
        "audio_language": "",
    }

    for line in output.splitlines():
        line = line.strip()

        m = re.search(r'encoder\s*:\s*(.+)', line, re.IGNORECASE)
        if m:
            data["encoder"] = m.group(1).strip()

        m = re.search(r'creation_time\s*:\s*(.+)', line, re.IGNORECASE)
        if m:
            data["creation_time"] = m.group(1).strip()

        m = re.search(r'com\.apple\.quicktime\.make\s*:\s*(.+)', line, re.IGNORECASE)
        if m:
            data["make"] = m.group(1).strip()

        m = re.search(r'com\.apple\.quicktime\.model\s*:\s*(.+)', line, re.IGNORECASE)
        if m:
            data["model"] = m.group(1).strip()

        m = re.search(r'location\s*:\s*([+-]\d+\.?\d*[+-]\d+\.?\d*)', line, re.IGNORECASE)
        if m:
            coords = re.match(r'([+-]\d+\.?\d*)([+-]\d+\.?\d*)', m.group(1))
            if coords:
                data["gps"] = {"lat": float(coords.group(1)), "lon": float(coords.group(2))}

        if re.search(r'Stream.*Video:\s*(\w+)', line):
            m = re.search(r'Stream.*Video:\s*(\w+)', line)
            if m:
                data["video_codec"] = m.group(1).lower()

        if re.search(r'Stream.*Audio:', line):
            data["audio_streams"] += 1
            m = re.search(r'\((\w{3})\).*Audio:', line)
            if m and not data["audio_language"]:
                data["audio_language"] = m.group(1)

    return data


def analyze(path: str) -> Tuple[AnalyzerResult, Dict]:
    label = "Metadata"
    details = []
    suspicious_flags = 0

    file_size = os.path.getsize(path)
    ext = os.path.splitext(path)[1].lower()
    details.append(f"File: {os.path.basename(path)}")
    details.append(f"Size: {file_size / (1024*1024):.2f} MB")
    details.append(f"Extension: {ext}")

    if file_size < 500_000:
        details.append("⚠ Very small file size — may indicate synthetic or heavily compressed content")
        suspicious_flags += 1

    output = _run_ffmpeg_info(path)
    parsed_raw = _parse_ffmpeg_output(output) if output else {}

    if output and parsed_raw:
        encoder = parsed_raw.get("encoder", "")
        if not encoder:
            details.append("⚠ No encoder tag in metadata (common in synthetic media)")
            suspicious_flags += 1
        else:
            details.append(f"Encoder: {encoder}")

        creation = parsed_raw.get("creation_time", "")
        if creation:
            details.append(f"Creation time: {creation}")
        else:
            details.append("⚠ No creation timestamp in metadata")
            suspicious_flags += 1

        codec = parsed_raw.get("video_codec", "")
        if codec:
            details.append(f"Video codec: {codec}")
            if codec not in ("h264", "hevc", "vp9", "vp8", "av1", "mpeg4"):
                details.append(f"⚠ Uncommon video codec: {codec}")
                suspicious_flags += 1

        audio = parsed_raw.get("audio_streams", 0)
        if audio == 0:
            details.append("⚠ No audio stream detected")
            suspicious_flags += 1
        else:
            details.append(f"Audio streams: {audio}")
    else:
        details.append("Could not read metadata")

    score = min(1.0, suspicious_flags * 0.22)
    confidence = 0.9 if output else 0.4

    parsed = {
        "creation_time":  parsed_raw.get("creation_time", ""),
        "gps":            parsed_raw.get("gps"),
        "make":           parsed_raw.get("make", ""),
        "model":          parsed_raw.get("model", ""),
        "encoder":        parsed_raw.get("encoder", ""),
        "audio_language": parsed_raw.get("audio_language", ""),
    }

    return AnalyzerResult(label=label, score=score, confidence=confidence, details=details), parsed
