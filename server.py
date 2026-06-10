"""
server.py
Local API server for the browser extension.
Accepts a video URL, downloads it, runs the full analysis pipeline,
and returns JSON results.

Usage:
    pip install flask flask-cors requests
    python server.py

The server listens on http://127.0.0.1:7177
"""

import os
import sys
import tempfile
import urllib.parse
import threading

import requests
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from core.video_loader import load_video
from core.aggregator import aggregate

import analyzers.temporal_consistency  as temporal
import analyzers.face_texture          as face_texture
import analyzers.compression_artifacts as compression
import analyzers.noise_pattern         as noise
import analyzers.brightness_flicker    as flicker
import analyzers.edge_sharpness        as sharpness
import analyzers.metadata              as metadata
import analyzers.content_verification  as content_verification

app = Flask(__name__)
CORS(app, origins=["*"])

PORT = 7177

# Headers to pass when downloading videos so sites don't block the request
_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "video/webm,video/mp4,video/*;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_SUPPORTED_SCHEMES = ("http://", "https://")


def _download_video(url: str) -> str:
    """
    Download a video URL to a temporary file.
    Returns the path to the temp file; caller is responsible for deletion.
    Raises ValueError for unsupported URL types, IOError for download failures.
    """
    if not any(url.startswith(s) for s in _SUPPORTED_SCHEMES):
        raise ValueError(
            "Only http:// and https:// URLs are supported. "
            "Blob URLs and local files cannot be fetched by the server."
        )

    parsed = urllib.parse.urlparse(url)
    raw_ext = os.path.splitext(parsed.path)[1]
    # Keep only the extension part (strip query string artifacts like ".mp4?t=123")
    ext = raw_ext.split("?")[0].lower() if raw_ext else ".mp4"
    if ext not in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"):
        ext = ".mp4"

    resp = requests.get(url, stream=True, headers=_DOWNLOAD_HEADERS, timeout=60)
    resp.raise_for_status()

    # Use content-type to pick extension if the URL path didn't give us one
    ct = resp.headers.get("Content-Type", "")
    if "webm" in ct:
        ext = ".webm"
    elif "mp4" in ct or "mpeg4" in ct:
        ext = ".mp4"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                tmp.write(chunk)
        return tmp.name


def _run_analysis(video_path: str) -> dict:
    """Run all analyzers and return a serialisable results dict."""
    video = load_video(video_path)

    results = {
        "Temporal Consistency":  temporal.analyze(video.frames),
        "Face Texture":          face_texture.analyze(video.frames),
        "Compression Artifacts": compression.analyze(video.frames),
        "Noise Pattern":         noise.analyze(video.frames),
        "Brightness Flicker":    flicker.analyze(video.frames),
        "Edge Sharpness":        sharpness.analyze(video.frames),
        "Metadata":              metadata.analyze(video_path),
        "Content Verification":  content_verification.analyze(video.frames, video_path),
    }

    agg = aggregate(results)

    return {
        "overall_score": round(agg.overall_score, 4),
        "verdict": agg.verdict,
        "components": {
            label: {
                "score":      round(r.clamped_score(), 4),
                "confidence": round(r.confidence, 4),
                "details":    r.details,
            }
            for label, r in agg.components.items()
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "ok", "version": "1.0"})


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()

    if not url:
        return jsonify({"error": "No 'url' field in request body."}), 400

    tmp_path = None
    try:
        print(f"[server] Downloading: {url[:120]}", flush=True)
        tmp_path = _download_video(url)
        print(f"[server] Analysing: {tmp_path}", flush=True)
        result = _run_analysis(tmp_path)
        print(f"[server] Done — verdict: {result['verdict']}", flush=True)
        return jsonify(result)

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except requests.HTTPError as exc:
        return jsonify({"error": f"Could not download video: {exc}"}), 502

    except Exception as exc:
        print(f"[server] Error: {exc}", flush=True)
        return jsonify({"error": str(exc)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Deepfake Detector API server starting on http://127.0.0.1:{PORT}")
    print("Keep this window open while using the browser extension.")
    print("Press Ctrl+C to stop.\n")
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
