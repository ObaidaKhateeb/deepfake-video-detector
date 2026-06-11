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
_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cv_debug.log")


def _dlog(msg: str) -> None:
    if not _DEBUG:
        return
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n"
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


import cv2
import numpy as np

from core.result import AnalyzerResult

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

_FasterWhisperModel = None
_HAS_WHISPER   = False
_WHISPER_ERROR = ""
try:
    import importlib.util as _ilu
    if _ilu.find_spec("faster_whisper") is not None:
        _HAS_WHISPER = True  # defer actual import to _load_whisper_model()
except Exception as _e:
    _WHISPER_ERROR = str(_e)



try:
    from ddgs import DDGS
    _HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS  # legacy fallback
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
                    "person_name":  {"type": "string"},
                    "organization": {"type": "string"},
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

def _get_ffmpeg_exe() -> str:
    """Return the full path to the ffmpeg binary (bundled via imageio-ffmpeg or system)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return "ffmpeg"  # fall back to system PATH


def _extract_audio_wav(video_path: str) -> Optional[str]:
    """
    Extract audio from video to a temp WAV file using the bundled ffmpeg binary.
    Returns the temp file path, or None on failure.
    """
    import tempfile
    ffmpeg_exe = _get_ffmpeg_exe()
    _dlog(f"ffmpeg path: {ffmpeg_exe}")
    _dlog(f"ffmpeg exists: {os.path.isfile(ffmpeg_exe)}")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-y", "-i", video_path,
             "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
             tmp.name],
            capture_output=True, timeout=120,
        )
        _dlog(f"ffmpeg returncode: {result.returncode}")
        if result.returncode != 0:
            _dlog(f"ffmpeg stderr: {result.stderr.decode(errors='replace')[:500]}")
        if result.returncode == 0 and os.path.getsize(tmp.name) > 0:
            _dlog(f"WAV extracted ok, size={os.path.getsize(tmp.name)}")
            return tmp.name
    except Exception as exc:
        _dlog(f"audio extraction exception: {exc}")
    try:
        os.unlink(tmp.name)
    except Exception:
        pass
    return None


def _transcribe_audio(video_path: str) -> str:
    """
    Run faster-whisper in a child process to isolate DLL/torch crashes from the GUI.
    Audio is first extracted to a temp WAV via the bundled ffmpeg binary.
    """
    if not _HAS_WHISPER or not video_path or not os.path.isfile(video_path):
        _dlog(f"_transcribe_audio skipped: has_whisper={_HAS_WHISPER} path_exists={os.path.isfile(video_path) if video_path else False}")
        return ""
    wav_path = _extract_audio_wav(video_path)
    if not wav_path:
        _dlog("audio extraction returned None — no WAV to transcribe")
        return ""
    script = (
        "import os, sys\n"
        "os.environ['CUDA_VISIBLE_DEVICES']=''\n"
        "from faster_whisper import WhisperModel\n"
        f"m=WhisperModel('base',device='cpu',compute_type='int8')\n"
        f"segs,_=m.transcribe({repr(wav_path)})\n"
        "print(' '.join(s.text for s in segs).strip())"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=300,
        )
        _dlog(f"whisper subprocess returncode={result.returncode}")
        if result.returncode != 0:
            _dlog(f"whisper subprocess stderr: {result.stderr[:300]}")
        text = result.stdout.strip()
        _dlog(f"whisper transcript ({len(text)} chars): {text[:100]}")
        return text
    except Exception as exc:
        _dlog(f"whisper subprocess failed: {exc}")
        return ""
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass


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
        "category ('person', 'event', 'statistic', 'quote', 'institution').\n"
        "Additionally, for category 'person' claims, also fill:\n"
        "  - person_name:  the individual's full name (e.g. 'Barack Obama')\n"
        "  - organization: the company or institution they are associated with (e.g. 'US Government'), "
        "or omit if none is mentioned.\n\n"
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


def _verify_person_apollo(name: str, org: str) -> Tuple[Optional[float], List[str]]:
    """
    Look up a person by name (+ optional org) in Apollo.io.
    Returns (None, []) to signal DuckDuckGo fallback when Apollo is not
    configured or returns no matching record.
    """
    api_key = os.environ.get("APOLLO_API_KEY", "")
    if not api_key or not name:
        return None, []

    try:
        payload: dict = {"api_key": api_key, "q_person_name": name, "page": 1, "per_page": 3}
        if org:
            payload["q_organization_name"] = org

        req = urllib.request.Request(
            "https://api.apollo.io/v1/people/search",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Cache-Control": "no-cache"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return None, []

    people = data.get("people", [])
    if not people:
        return None, []

    name_words = [w for w in name.lower().split() if len(w) > 2]

    for person in people:
        found_name = person.get("name", "").lower()
        found_org  = (person.get("organization") or {}).get("name", "").lower()
        found_title = person.get("title", "")

        name_match = any(w in found_name for w in name_words)
        org_match  = not org or org.lower() in found_org or found_org in org.lower()

        if name_match and org_match:
            label = f"{person.get('name')} — {found_title} at {(person.get('organization') or {}).get('name', '')}"
            return 0.0, [f"[Apollo] VERIFIED: {label}"]

        if name_match and org:
            return 0.7, [
                f"[Apollo] MISMATCH: {person.get('name')} found but linked to "
                f"'{found_org}', not '{org}'."
            ]

    return None, []


_STOP_WORDS = {
    "the", "this", "that", "with", "from", "have", "been", "were", "they",
    "their", "about", "into", "which", "when", "said", "also", "some", "more",
}


def _verify_claim(
    statement: str,
    search_query: str,
    category: str,
    person_name: str = "",
    organization: str = "",
    api_key: str = "",
) -> Tuple[float, List[str]]:
    details: List[str] = []
    if not statement:
        return 0.3, ["Empty claim — skipped."]

    # For person claims, try Apollo first
    if category == "person" and person_name:
        score, sub = _verify_person_apollo(person_name, organization)
        if score is not None:
            return score, sub

    snippets = _web_snippets(search_query)
    if not snippets:
        details.append(f"[{category}] No web results for: {search_query}")
        return 0.5, details

    # Ask Claude to judge whether the snippets confirm or contradict the claim
    if api_key:
        evidence = "\n".join(f"- {s}" for s in snippets[:5])
        prompt = (
            f"Claim: \"{statement}\"\n\n"
            f"Web evidence:\n{evidence}\n\n"
            "Based only on the evidence above, does it CONFIRM, CONTRADICT, or give "
            "INSUFFICIENT information about the claim?\n"
            "Reply with exactly one word: CONFIRM, CONTRADICT, or INSUFFICIENT."
        )
        try:
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            verdict = resp.content[0].text.strip().upper()
            _dlog(f"claim verdict: {verdict} — {statement[:60]}")
            if "CONFIRM" in verdict:
                details.append(f"[{category}] VERIFIED: web evidence confirms this claim.")
                return 0.0, details
            elif "CONTRADICT" in verdict:
                details.append(f"[{category}] CONTRADICTED: web evidence contradicts this claim.")
                return 0.9, details
            else:
                details.append(f"[{category}] INSUFFICIENT: web evidence neither confirms nor contradicts.")
                return 0.5, details
        except Exception as exc:
            _dlog(f"Claude verify failed: {exc}")

    # Fallback: keyword overlap (less accurate)
    combined = " ".join(snippets).lower()
    key_words = [w for w in re.findall(r"\b[a-z]{3,}\b", statement.lower()) if w not in _STOP_WORDS]
    if not key_words:
        return 0.2, details
    ratio = sum(1 for w in key_words if w in combined) / len(key_words)
    if ratio >= 0.6:
        details.append(f"[{category}] VERIFIED (keyword match).")
        return 0.0, details
    elif ratio >= 0.3:
        details.append(f"[{category}] PARTIAL (keyword match).")
        return 0.4, details
    else:
        details.append(f"[{category}] SUSPICIOUS (keyword match).")
        return 0.65, details


# ── Metadata reading ─────────────────────────────────────────────────────────

def _get_ffprobe_exe() -> str:
    """Return the best available ffprobe path."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        candidate = os.path.join(
            os.path.dirname(ffmpeg_exe),
            "ffprobe.exe" if sys.platform == "win32" else "ffprobe",
        )
        if os.path.isfile(candidate):
            return candidate
    except Exception:
        pass
    return "ffprobe"


def _read_video_metadata(video_path: str) -> dict:
    try:
        result = subprocess.run(
            [_get_ffprobe_exe(), "-v", "quiet", "-print_format", "json",
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


_DEBUG_PATH = os.path.join(os.path.dirname(__file__), "..", "content_verification_debug.json")


def _save_debug(extracted: dict, transcript: str, video_path: str, meta: dict) -> None:
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
        print(f"[content_verification] debug saved → {debug_path}")
    except Exception as e:
        print(f"[content_verification] debug save failed: {e}")


# ── Public entry point ────────────────────────────────────────────────────────

def _load_api_key() -> str:
    api_json = os.path.join(os.path.dirname(__file__), "..", "api.json")
    try:
        import json
        with open(os.path.normpath(api_json)) as f:
            return json.load(f).get("anthropic_api_key", "")
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY", "")


def analyze(frames: List[np.ndarray], video_path: str = "", meta: Optional[dict] = None) -> AnalyzerResult:
    label = "Content Verification"
    api_key = _load_api_key()

    if not _HAS_ANTHROPIC:
        _dlog("analyze: early return — anthropic not installed")
        return AnalyzerResult(
            label=label, score=0.3, confidence=0.1,
            details=[
                "anthropic package not installed — content verification skipped.",
                "Install with: pip install anthropic",
            ],
        )

    if not api_key:
        _dlog("analyze: early return — ANTHROPIC_API_KEY not set")
        return AnalyzerResult(
            label=label, score=0.3, confidence=0.1,
            details=[
                "ANTHROPIC_API_KEY not set — content verification skipped.",
                "Set the environment variable to enable AI-powered claim verification.",
            ],
        )

    if not _HAS_DDGS:
        _dlog("analyze: early return — ddgs/duckduckgo_search not installed")
        return AnalyzerResult(
            label=label, score=0.3, confidence=0.1,
            details=[
                "duckduckgo_search not installed — content verification skipped.",
                "Install with: pip install ddgs",
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
                "Approach 1 (audio): skipped — faster-whisper could not be loaded. "
                + (f"Reason: {_WHISPER_ERROR}" if _WHISPER_ERROR else "Run: pip install faster-whisper")
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
                label=label, score=0.2, confidence=0.1,
                details=details + ["No verifiable factual claims found — content verification has minimal weight in final score."],
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
            score, sub = _verify_claim(
                statement, search_query, category,
                claim.get("person_name", ""),
                claim.get("organization", ""),
                api_key=api_key,
            )
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
                label=label, score=0.3, confidence=0.1,
                details=details + ["Claims extracted but none could be scored."],
            )

        avg_score   = float(np.mean(scores))
        max_score   = float(max(scores))
        final_score = max(0.0, min(1.0, 0.6 * avg_score + 0.4 * max_score))
        confidence  = min(1.0, len(scores) * 0.20)

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
