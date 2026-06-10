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
import base64
import json
import re
from typing import List, Tuple

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
except ImportError:
    _HAS_WHISPER = False

try:
    from duckduckgo_search import DDGS
    _HAS_DDGS = True
except ImportError:
    _HAS_DDGS = False

_MAX_VISUAL_FRAMES = 15

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
    },
    "required": ["claims"],
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


# ── Approach 2: dense frame sampling ─────────────────────────────────────────

def _sample_frames_dense(frames: List[np.ndarray], video_path: str) -> List[np.ndarray]:
    """
    Sample ~one frame per second so that lower-thirds and text overlays that
    appear briefly are not skipped. Falls back to a uniform 5-frame sample when
    no video path is provided. Capped at _MAX_VISUAL_FRAMES to stay within the
    API's context limits.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    fps = fps if fps > 0 else 30.0

    step = max(1, int(fps))
    sampled = frames[::step]

    if len(sampled) > _MAX_VISUAL_FRAMES:
        step2 = max(1, len(sampled) // _MAX_VISUAL_FRAMES)
        sampled = sampled[::step2][:_MAX_VISUAL_FRAMES]

    return sampled if len(sampled) > 0 else frames[:1]


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
        "Extract every verifiable factual claim found in EITHER the transcript OR the frames.\n"
        "A claim can be anything checkable against real-world sources, for example:\n"
        "  - A person's name, title, or role (e.g. 'John Smith is the CEO of Acme Corp')\n"
        "  - A statistic or number (e.g. 'unemployment is at 3.5%')\n"
        "  - An event or date (e.g. 'the earthquake struck on March 4th')\n"
        "  - A location or geography (e.g. 'the factory is in Detroit')\n"
        "  - An organizational or institutional fact (e.g. 'WHO declared a pandemic')\n"
        "  - A quote attributed to someone\n"
        "  - Any other assertable fact that could be confirmed or refuted\n\n"
        "For each claim provide:\n"
        "  - statement:    the exact factual assertion as it appears in the content\n"
        "  - search_query: a concise web search query that would verify or refute it\n"
        "  - category:     a short label describing the claim type "
        "(e.g. 'person', 'statistic', 'event', 'location', 'quote', 'organization', etc.)\n\n"
        "Return ONLY the JSON object matching the schema. "
        "If nothing verifiable is present, return an empty claims array."
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
            return {"claims": []}


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


# ── Public entry point ────────────────────────────────────────────────────────

def analyze(frames: List[np.ndarray], video_path: str = "") -> AnalyzerResult:
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

        # ── Approach 2: dense frame sampling ─────────────────────────────────
        if video_path:
            details.append("Approach 2 (visual): sampling frames at ~1 fps...")
            sampled = _sample_frames_dense(frames, video_path)
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

        extracted = _extract_claims(sampled, transcript, api_key)

        claim_list = extracted.get("claims", [])

        if not claim_list:
            return AnalyzerResult(
                label=label, score=0.2, confidence=0.5,
                details=details + [
                    "No verifiable factual claims found in either audio or visual content.",
                ],
            )

        details.append(f"Extracted {len(claim_list)} verifiable claim(s).")

        # ── Web verification ──────────────────────────────────────────────────
        details.append("Verifying claims against web sources...")
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

        if not scores:
            return AnalyzerResult(
                label=label, score=0.3, confidence=0.3,
                details=details + ["Claims extracted but none could be verified."],
            )

        avg_score   = float(np.mean(scores))
        max_score   = float(max(scores))
        final_score = max(0.0, min(1.0, 0.6 * avg_score + 0.4 * max_score))
        confidence  = min(1.0, len(scores) * 0.25 + 0.25)

        details.append(
            f"\nSummary: {len(scores)} entity/entities checked — "
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
