"""
core/result.py
Shared result types used by all analyzers and the aggregator.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AnalyzerResult:
    """
    Output of a single analyzer module.

    score       : 0.0 (definitely real) → 1.0 (definitely fake)
    label       : human-readable name of this parameter
    confidence  : 0.0–1.0, how reliable this score is
                  (e.g. low if no faces were found)
    details     : list of human-readable detail strings shown in the GUI
    """
    label: str
    score: float                        # 0 = real, 1 = fake
    confidence: float = 1.0
    details: List[str] = field(default_factory=list)

    def clamped_score(self) -> float:
        return max(0.0, min(1.0, self.score))


@dataclass
class AggregatedResult:
    """
    Final output combining all analyzer results.
    """
    overall_score: float                # 0 = real, 1 = fake
    verdict: str                        # "Likely Real" / "Suspicious" / "Likely Fake"
    components: Dict[str, AnalyzerResult] = field(default_factory=dict)

    @staticmethod
    def verdict_from_score(score: float) -> str:
        if score < 0.35:
            return "Likely Real"
        elif score < 0.65:
            return "Suspicious"
        else:
            return "Likely Fake"
