"""
core/aggregator.py
Combines individual AnalyzerResult objects into one AggregatedResult.
Weights can be tuned per parameter.
"""

from core.result import AnalyzerResult, AggregatedResult
from typing import Dict

# Weight of each analyzer in the final score (must sum to 1.0)
WEIGHTS: Dict[str, float] = {
    "Temporal Consistency": 0.25,
    "Face Texture":         0.20,
    "Compression Artifacts":0.15,
    "Noise Pattern":        0.15,
    "Brightness Flicker":   0.10,
    "Edge Sharpness":       0.10,
    "Metadata":             0.05,
}


def aggregate(results: Dict[str, AnalyzerResult]) -> AggregatedResult:
    """
    Weighted average of all analyzer scores, adjusted by each analyzer's
    confidence. Low-confidence analyzers contribute less.
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for label, result in results.items():
        w = WEIGHTS.get(label, 0.05)
        effective_weight = w * result.confidence
        weighted_sum  += result.clamped_score() * effective_weight
        total_weight  += effective_weight

    overall = weighted_sum / total_weight if total_weight > 0 else 0.0
    overall = max(0.0, min(1.0, overall))

    return AggregatedResult(
        overall_score=overall,
        verdict=AggregatedResult.verdict_from_score(overall),
        components=results,
    )
