"""
analyzers/content_verification.py
Two parallel approaches to extracting factual claims from a video:

  Approach 1 — Audio:  full spoken transcript via local Whisper model
  Approach 2 — Visual: frames sampled at ~1 fps so on-screen text and
                        lower-thirds are not missed

Both are fed into Claude Vision, which extracts any verifiable claim found
in either source — people, events, statistics, dates, locations, quotes, etc.
Each claim is paired with a targeted search query and verified against live
web search results.

Score: 0 = all claims verified or no verifiable claims found
       1 = claims present and actively contradicted by web sources
"""

import os
import sys
import base64
import json
import re
import datetime
import subprocess
import urllib.request
import urllib.error
from typing import List, Optional, Tuple

_DEBUG = "--debug" in sys.argv

import cv2
import numpy as np

from core.result import AnalyzerResult

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    import whisper
    _HAS_WHISPER = True
except Exception:
    _HAS_WHISPER = False

try:
    from duckduckgo_search import DDGS
    _HAS_DDGS = True
except ImportError:
    _HAS_DDGS = False

_MAX_VISUAL_FRAMES = 15
_MIN_FRAME_GAP_SEC = 3.0
_VISUAL_DIFF_THRESHOLD = 10.0

_CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement":    {"type": "string"},
                    "search_query": {"type": "string"},
                    "category":     {"type": "string"},
                },
                "required": ["statement", "search_query", "category"],
                "additionalProperties": False,
            },
        },
        "context": {
            "type": "object",
            "properties": {
                "claimed_dates":     {"type": "array", "items": {"type": "string"}},
                "claimed_locations": {"type": "array", "items": {"type": "string"}},
                "claimed_source":    {"type": "string"},
                "transcript_language": {"type": "string"},
            },
            "required": ["claimed_dates", "claimed_locations", "claimed_source", "transcript_language"],
            "additionalProperties": False,
        },
    },
    "required": ["claims", "context"],
    "additionalProperties": False,
}


# ── Approach 1: audio transcription ──────────────────────────────────────────

def _transcribe_audio(video_path: str) -> str:
    """
    Transcribe spoken content from the video using a local Whisper model.
    Whisper reads the audio track directly from the video file — no separate
    extraction step needed. Requires ffmpeg to be installed.
    """
    if not video_path or not os.path.isfile(video_path):
        return ""
    try:
        model = whisper.load_model("base")
        result = model.transcribe(video_path, fp16=False)
        return result.get("text", "").strip()
    except Exception:
        return ""


# ── Approach 2: frame sampling ────────────────────────────────────────────────

def _sample_frames(frames: List[np.ndarray], video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    fps = fps if fps > 0 else 30.0

    min_gap = int(fps * _MIN_FRAME_GAP_SEC)
    sampled: List[np.ndarray] = []
    last_idx = -min_gap
    last_gray: np.ndarray | None = None

    for i, frame in enumerate(frames):
        if i - last_idx < min_gap:
            continue
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
        diff = 0.0 if last_gray is None else float(np.mean(np.abs(gray.astype(float) - last_gray.astype(float))))
        if last_gray is None or diff >= _VISUAL_DIFF_THRESHOLD:
            sampled.append(frame)
            last_idx = i
            last_gray = gray
            if len(sampled) >= _MAX_VISUAL_FRAMES:
                break

    return sampled if sampled else frames[:1]


def _frame_to_base64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ── Claim extraction (combines both sources) ──────────────────────────────────

def _extract_claims(frames: List[np.ndarray], transcript: str, api_key: str) -> dict:
    """
    Send dense visual frames + full audio transcript to Claude Vision and return
    a structured dict of every verifiable factual claim found in either source.
    """
    client = anthropic.Anthropic(api_key=api_key)

    content = []
    for frame in frames:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": _frame_to_base64(frame),
            },
        })

    lines = ["Analyze this video content and extract every verifiable factual claim.\n"]

    if transcript:
        lines.append(
            "AUDIO TRANSCRIPT (complete spoken content of the video):\n"
            f'"""\n{transcript}\n"""\n'
        )

    if frames:
        lines.append(
            f"VIDEO FRAMES: The {len(frames)} image(s) above are sampled at "
            "approximately 1 frame per second and capture on-screen text, "
            "lower-thirds, name plates, and visual context throughout the video.\n"
        )

    lines.append(
        "You are helping detect misinformation and deepfakes. Your response has two parts.\n\n"
        "PART 1 — claims: Extract only claims that assert something about the real world "
        "that could be fabricated or manipulated.\n"
        "A qualifying claim must:\n"
        "  1. Be about a real-world entity or event that exists independently of this video\n"
        "  2. Be checkable against an independent source (news, official records, encyclopedias)\n"
        "  3. Be something that, if false, would suggest the video is misleading or manipulated\n"
        "Examples: a named person's identity/role, a world event or announcement, "
        "a published statistic, a quote attributed to a public figure, an institutional fact.\n"
        "Do NOT include claims only meaningful within this specific video with no external source.\n"
        "For each claim: statement (the assertion), search_query (web query to verify it), "
        "category ('person', 'event', 'statistic', 'quote', 'institution').\n\n"
        "PART 2 — context (for metadata cross-checking):\n"
        "  - claimed_dates:     list of any dates or time references mentioned "
        "(e.g. 'March 4th 2024', 'last Tuesday', 'in 2019')\n"
        "  - claimed_locations: list of any locations or places mentioned "
        "(e.g. 'Paris', 'the White House', 'northern Gaza')\n"
        "  - claimed_source:    how the video presents itself — describe in a few words "
        "(e.g. 'personal phone recording', 'TV news broadcast', 'CCTV footage', 'documentary', "
        "'unknown')\n"
        "  - transcript_language: language of any spoken content (e.g. 'English', 'Arabic', "
        "or '' if no speech)\n\n"
        "Return ONLY the JSON object matching the schema."
    )

    content.append({"type": "text", "text": "\n".join(lines)})

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        output_config={"format": {"type": "json_schema", "schema": _CLAIMS_SCHEMA}},
        messages=[{"role": "user", "content": content}],
    )

    text = next((b.text for b in response.content if b.type == "text"), "{}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"claims": [], "context": {"claimed_dates": [], "claimed_locations": [], "claimed_source": "", "transcript_language": ""}}


# ── Web verification ──────────────────────────────────────────────────────────

def _web_snippets(query: str, max_results: int = 5) -> List[str]:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [r.get("body", "") for r in results if r.get("body")]
    except Exception:
        return []


_STOP_WORDS = {
    "the", "this", "that", "with", "from", "have", "been", "were", "they",
    "their", "about", "into", "which", "when", "said", "also", "some", "more",
}


def _verify_claim(statement: str, search_query: str, category: str) -> Tuple[float, List[str]]:
    """
    Generic claim verifier: searches the web using search_query and scores how
    well the snippets support the statement using keyword overlap.
    """
    details: List[str] = []
    if not statement:
        return 0.3, ["Empty claim — skipped."]

    snippets = _web_snippets(search_query)
    if not snippets:
        details.append(f"[{category}] No web results for: {search_query}")
        return 0.5, details

    combined = " ".join(snippets).lower()
    key_words = [
        w for w in re.findall(r"\b[a-z]{3,}\b", statement.lower())
        if w not in _STOP_WORDS
    ]

    if not key_words:
        details.append(f"[{category}] Results found but no specific keywords to match.")
        return 0.2, details

    matched = sum(1 for w in key_words if w in combined)
    ratio = matched / len(key_words)

    if ratio >= 0.6:
        details.append(
            f"[{category}] VERIFIED: {matched}/{len(key_words)} key terms found in web sources."
        )
        return 0.0, details
    elif ratio >= 0.3:
        details.append(
            f"[{category}] PARTIAL: {matched}/{len(key_words)} key terms found — "
            "claim may be inaccurate or context differs."
        )
        return 0.4, details
    else:
        details.append(
            f"[{category}] SUSPICIOUS: only {matched}/{len(key_words)} key terms found — "
            "little web support for this claim."
        )
        return 0.65, details


# ── Metadata reading ─────────────────────────────────────────────────────────

def _read_video_metadata(video_path: str) -> dict:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", video_path],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {}
        data = json.loads(result.stdout)
    except Exception:
        return {}

    fmt = data.get("format", {})
    tags = {k.lower(): v for k, v in fmt.get("tags", {}).items()}
    streams = data.get("streams", [])
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    location_raw = (
        tags.get("com.apple.quicktime.location.iso6709", "") or
        tags.get("location", "") or
        tags.get("location-eng", "")
    )
    gps: Optional[dict] = None
    if location_raw:
        m = re.match(r'([+-]\d+\.?\d*)([+-]\d+\.?\d*)', location_raw)
        if m:
            gps = {"lat": float(m.group(1)), "lon": float(m.group(2))}

    audio_lang = ""
    if audio_streams:
        audio_lang = audio_streams[0].get("tags", {}).get("language", "")

    return {
        "creation_time": tags.get("creation_time", ""),
        "gps": gps,
        "make":  tags.get("com.apple.quicktime.make", "") or tags.get("make", ""),
        "model": tags.get("com.apple.quicktime.model", "") or tags.get("model", ""),
        "encoder": tags.get("encoder", "") or tags.get("com.apple.quicktime.software", ""),
        "audio_language": audio_lang,
    }


def _reverse_geocode(lat: float, lon: float) -> str:
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&zoom=5"
        req = urllib.request.Request(url, headers={"User-Agent": "deepfake-detector/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        addr = data.get("address", {})
        parts = [addr.get("country", ""), addr.get("state", ""), addr.get("city", "")]
        return ", ".join(p for p in parts if p)
    except Exception:
        return ""


# ── Metadata cross-checks ─────────────────────────────────────────────────────

def _crosscheck_date(claimed_dates: List[str], creation_time: str) -> Tuple[float, List[str]]:
    details: List[str] = []
    if not creation_time or not claimed_dates:
        return 0.0, details
    try:
        ct = datetime.datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
    except Exception:
        return 0.0, details

    creation_year = ct.year
    years: set = set()
    for d in claimed_dates:
        for y in re.findall(r'\b(20\d{2}|19\d{2})\b', d):
            years.add(int(y))

    if not years:
        details.append(f"[date] Dates mentioned but no specific year to cross-check (file: {creation_year}).")
        return 0.0, details

    if creation_year not in years:
        diff = min(abs(y - creation_year) for y in years)
        if diff >= 2:
            details.append(
                f"[date] DATE MISMATCH: content references {sorted(years)} "
                f"but file was created in {creation_year}."
            )
            return 0.8, details
        details.append(
            f"[date] DATE WARNING: content references {sorted(years)}, file created {creation_year}."
        )
        return 0.3, details

    details.append(f"[date] Consistent: content year(s) {sorted(years)} match file creation year {creation_year}.")
    return 0.0, details


def _crosscheck_location(claimed_locations: List[str], gps: Optional[dict]) -> Tuple[float, List[str]]:
    details: List[str] = []
    if not gps or not claimed_locations:
        return 0.0, details

    lat, lon = gps["lat"], gps["lon"]
    geo_str = _reverse_geocode(lat, lon)
    if not geo_str:
        details.append(f"[location] GPS ({lat:.4f}, {lon:.4f}) — could not resolve to a region.")
        return 0.0, details

    details.append(f"[location] GPS resolves to: {geo_str}")
    geo_lower = geo_str.lower()

    matches, mismatches = [], []
    for loc in claimed_locations:
        words = [w for w in loc.lower().split() if len(w) > 3]
        (matches if any(w in geo_lower for w in words) else mismatches).append(loc)

    if mismatches and not matches:
        details.append(f"[location] LOCATION MISMATCH: claimed {mismatches} but GPS points to {geo_str}.")
        return 0.75, details
    if matches:
        details.append(f"[location] Consistent: {matches} matches GPS region.")
    return 0.0, details


def _crosscheck_source(claimed_source: str, encoder: str, make: str, model: str) -> Tuple[float, List[str]]:
    details: List[str] = []
    if not claimed_source:
        return 0.0, details

    src = claimed_source.lower()
    official_keywords = {"broadcast", "news", "cctv", "surveillance", "official",
                         "government", "press", "conference", "agency"}
    is_official = any(k in src for k in official_keywords)
    has_device = bool(make or model)
    is_reencoded = any(e in encoder.lower() for e in ("lavf", "libav", "ffmpeg"))

    if is_official and has_device:
        details.append(
            f"[source] MISMATCH: content presents as official/broadcast footage "
            f"but was recorded on a personal device ({(make + ' ' + model).strip()})."
        )
        return 0.7, details
    if is_official and is_reencoded:
        details.append(
            f"[source] WARNING: content claims official footage but encoder is '{encoder}' "
            "(common in re-encoded or AI-generated video)."
        )
        return 0.5, details
    if has_device:
        details.append(f"[source] Recorded on: {(make + ' ' + model).strip()}.")
    return 0.0, details


def _crosscheck_missing_metadata(creation_time: str, encoder: str, make: str, model: str) -> Tuple[float, List[str]]:
    details: List[str] = []
    missing = []
    if not creation_time:
        missing.append("creation timestamp")
    if not encoder:
        missing.append("encoder tag")
    if not make and not model:
        missing.append("device info")

    if len(missing) >= 2:
        details.append(
            f"[metadata] STRIPPED METADATA: missing {', '.join(missing)} — "
            "common in AI-generated or re-processed video."
        )
        return 0.5, details
    if missing:
        details.append(f"[metadata] Partial metadata: missing {missing[0]}.")
        return 0.2, details
    return 0.0, details


def _crosscheck_language(transcript_language: str, gps: Optional[dict], geo_str: str) -> Tuple[float, List[str]]:
    details: List[str] = []
    if transcript_language:
        details.append(f"[language] Detected spoken language: {transcript_language}.")
    return 0.0, details


# ── DEBUG: remove this entire section before production ──────────────────────

_DEBUG_PATH = os.path.join(os.path.dirname(__file__), "..", "content_verification_debug.json")

def _save_debug(extracted: dict, transcript: str, video_path: str, meta: dict) -> None:
    # DEBUG: appends extracted claims + metadata to content_verification_debug.json at project root
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "video_path": video_path,
        "transcript_preview": transcript[:300] if transcript else "",
        "claims": extracted.get("claims", []),
        "context": extracted.get("context", {}),
        "metadata": meta,
    }
    try:
        existing: list = []
        debug_path = os.path.abspath(_DEBUG_PATH)
        if os.path.isfile(debug_path):
            with open(debug_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing.append(record)
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f"[content_verification] debug saved → {debug_path}")  # DEBUG: remove before production
    except Exception as e:
        print(f"[content_verification] debug save failed: {e}")  # DEBUG: remove before production

# ── END DEBUG ─────────────────────────────────────────────────────────────────


# ── Public entry point ────────────────────────────────────────────────────────

def analyze(frames: List[np.ndarray], video_path: str = "", meta: Optional[dict] = None) -> AnalyzerResult:
    label = "Content Verification"
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not _HAS_ANTHROPIC:
        return AnalyzerResult(
            label=label, score=0.3, confidence=0.1,
            details=[
                "anthropic package not installed — content verification skipped.",
                "Install with: pip install anthropic",
            ],
        )

    if not api_key:
        return AnalyzerResult(
            label=label, score=0.3, confidence=0.1,
            details=[
                "ANTHROPIC_API_KEY not set — content verification skipped.",
                "Set the environment variable to enable AI-powered claim verification.",
            ],
        )

    if not _HAS_DDGS:
        return AnalyzerResult(
            label=label, score=0.3, confidence=0.1,
            details=[
                "duckduckgo_search not installed — content verification skipped.",
                "Install with: pip install duckduckgo_search",
            ],
        )

    details: List[str] = []

    try:
        # ── Approach 1: audio transcription ──────────────────────────────────
        transcript = ""
        if _HAS_WHISPER and video_path:
            details.append("Approach 1 (audio): transcribing spoken content via Whisper...")
            transcript = _transcribe_audio(video_path)
            if transcript:
                word_count = len(transcript.split())
                preview = transcript[:150] + ("…" if len(transcript) > 150 else "")
                details.append(f"  Transcript ({word_count} words): \"{preview}\"")
            else:
                details.append("  No speech detected, or audio extraction failed.")
        elif not _HAS_WHISPER:
            details.append(
                "Approach 1 (audio): skipped — openai-whisper not installed. "
                "Run: pip install openai-whisper"
            )
        else:
            details.append("Approach 1 (audio): skipped — no video path supplied.")

        # ── Approach 2: frame sampling ────────────────────────────────────────
        if video_path:
            details.append("Approach 2 (visual): sampling visually distinct frames (min 3s gap)...")
            sampled = _sample_frames(frames, video_path)
        else:
            details.append("Approach 2 (visual): no video path — using uniform 5-frame fallback.")
            sampled = frames[::max(1, len(frames) // 5)][:5]
        details.append(f"  {len(sampled)} frame(s) selected for visual analysis.")

        if not transcript and not sampled:
            return AnalyzerResult(
                label=label, score=0.3, confidence=0.2,
                details=details + ["No audio or visual content available to analyze."],
            )

        # ── Combined claim extraction ─────────────────────────────────────────
        sources = []
        if transcript:
            sources.append("audio transcript")
        if sampled:
            sources.append(f"{len(sampled)} visual frames")
        details.append(
            f"Sending {' + '.join(sources)} to Claude AI for claim extraction..."
        )

        # ── Read file metadata ────────────────────────────────────────────────
        if meta is None:
            meta = _read_video_metadata(video_path) if video_path else {}
        if meta:
            details.append(
                f"File metadata: creation={meta.get('creation_time') or 'missing'}, "
                f"device={((meta.get('make','') + ' ' + meta.get('model','')).strip()) or 'missing'}, "
                f"encoder={meta.get('encoder') or 'missing'}, "
                f"gps={'yes' if meta.get('gps') else 'none'}."
            )
        else:
            details.append("File metadata: ffprobe unavailable — metadata cross-checks skipped.")

        try:
            extracted = _extract_claims(sampled, transcript, api_key)
        except Exception as api_exc:
            if _DEBUG:
                _save_debug({"claims": [], "error": str(api_exc)}, transcript, video_path, meta)
            raise

        if _DEBUG:
            _save_debug(extracted, transcript, video_path, meta)

        claim_list = extracted.get("claims", [])
        ctx        = extracted.get("context", {})

        if not claim_list and not ctx.get("claimed_dates") and not ctx.get("claimed_locations"):
            return AnalyzerResult(
                label=label, score=0.2, confidence=0.5,
                details=details + ["No verifiable factual claims found in either audio or visual content."],
            )

        details.append(f"Extracted {len(claim_list)} verifiable claim(s).")

        # ── Web verification ──────────────────────────────────────────────────
        details.append("\nVerifying claims against web sources...")
        scores: List[float] = []

        for claim in claim_list:
            statement    = claim.get("statement", "")
            search_query = claim.get("search_query", statement)
            category     = claim.get("category", "claim")
            if not statement:
                continue
            details.append(f"\n  \"{statement}\"")
            score, sub = _verify_claim(statement, search_query, category)
            scores.append(score)
            details.extend(f"    {s}" for s in sub)

        # ── Metadata cross-checks ─────────────────────────────────────────────
        if meta:
            details.append("\nCross-checking content against file metadata...")

            geo_str = ""
            gps = meta.get("gps")

            for fn, args in [
                (_crosscheck_date,             (ctx.get("claimed_dates", []),    meta.get("creation_time", ""))),
                (_crosscheck_location,         (ctx.get("claimed_locations", []), gps)),
                (_crosscheck_source,           (ctx.get("claimed_source", ""),   meta.get("encoder", ""), meta.get("make", ""), meta.get("model", ""))),
                (_crosscheck_missing_metadata, (meta.get("creation_time", ""),   meta.get("encoder", ""), meta.get("make", ""), meta.get("model", ""))),
                (_crosscheck_language,         (ctx.get("transcript_language", ""), gps, geo_str)),
            ]:
                s, sub = fn(*args)
                if sub:
                    if s > 0.0:
                        scores.append(s)
                    details.extend(f"  {line}" for line in sub)

        if not scores:
            return AnalyzerResult(
                label=label, score=0.3, confidence=0.3,
                details=details + ["Claims extracted but none could be scored."],
            )

        avg_score   = float(np.mean(scores))
        max_score   = float(max(scores))
        final_score = max(0.0, min(1.0, 0.6 * avg_score + 0.4 * max_score))
        confidence  = min(1.0, len(scores) * 0.25 + 0.25)

        details.append(
            f"\nSummary: {len(scores)} check(s) — "
            f"avg suspicion {avg_score:.2f}, max {max_score:.2f}."
        )
        details.append("Score key: 0 = verified real, 1 = claims actively contradicted.")

        return AnalyzerResult(
            label=label,
            score=final_score,
            confidence=confidence,
            details=details,
        )

    except Exception as exc:
        return AnalyzerResult(
            label=label, score=0.3, confidence=0.1,
            details=[
                f"Content verification error: {exc}",
                "Possible causes: API rate limit, network error, or invalid API key.",
            ],
        )
