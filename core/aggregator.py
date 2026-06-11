"""
core/aggregator.py
Two-group aggregation:

  Group 1 — Processing (50%): AI visual model + metadata.
  Group 2 — Content Verification (50%): AI-powered claim extraction and
             metadata cross-check.

Each group's contribution is scaled by its effective confidence, so a
low-confidence content verification result (e.g. API key missing) naturally
cedes weight back to the processing group.
"""

from core.result import AnalyzerResult, AggregatedResult
from typing import Dict, Tuple

PROCESSING_WEIGHTS: Dict[str, float] = {
    "Visual Analysis": 0.95,
    "Metadata":        0.05,
}

_PROCESSING_SHARE   = 0.50
_CONTENT_SHARE      = 0.50


def _group_score(results: Dict[str, AnalyzerResult],
                 weights: Dict[str, float]) -> Tuple[float, float]:
    """
    Returns (score, mean_confidence) for a group of analyzers.
    Score is a confidence-weighted average; mean_confidence is the
    plain average of individual confidences (used for inter-group weighting).
    """
    weighted_sum   = 0.0
    total_weight   = 0.0
    conf_sum       = 0.0
    conf_count     = 0

    for label, w in weights.items():
        result = results.get(label)
        if result is None:
            continue
        ew = w * result.confidence
        weighted_sum += result.clamped_score() * ew
        total_weight += ew
        conf_sum     += result.confidence
        conf_count   += 1

    score           = weighted_sum / total_weight if total_weight > 0 else 0.0
    mean_confidence = conf_sum / conf_count if conf_count > 0 else 0.0
    return score, mean_confidence


def aggregate(results: Dict[str, AnalyzerResult]) -> AggregatedResult:
    # ── Group 1: processing analyzers ────────────────────────────────────────
    proc_score, proc_conf = _group_score(results, PROCESSING_WEIGHTS)

    # ── Group 2: content verification ────────────────────────────────────────
    cv = results.get("Content Verification")
    cv_score = cv.clamped_score() if cv else 0.0
    cv_conf  = cv.confidence      if cv else 0.0

    # ── Combine 50 / 50, scaled by each group's confidence ───────────────────
    proc_w = _PROCESSING_SHARE * proc_conf
    cv_w   = _CONTENT_SHARE    * cv_conf
    total  = proc_w + cv_w

    if total > 0:
        overall = (proc_w * proc_score + cv_w * cv_score) / total
    else:
        overall = 0.0

    overall = max(0.0, min(1.0, overall))

    return AggregatedResult(
        overall_score=overall,
        verdict=AggregatedResult.verdict_from_score(overall),
        components=results,
    )
