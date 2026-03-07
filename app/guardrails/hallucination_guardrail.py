"""
Hallucination Guardrail — Detects AI hallucinations and fabricated data.

Identifies unverifiable claims, suspicious statistical precision,
self-contradictions, and entity hallucination in AI reasoning.
"""

import re
from typing import Any, Dict, List

from app.guardrails.base_guardrail import BaseGuardrail, GuardrailResult
from models.decision import AIRecommendation


# Patterns indicating fabricated or unverifiable statements
_FABRICATION_PATTERNS = [
    r"according to (?:our|internal|proprietary) (?:records|data|analysis)",
    r"studies (?:show|prove|confirm) that",
    r"it is (?:well[-\s])?known that",
    r"research (?:has shown|proves|confirms)",
    r"statistics (?:show|indicate|prove)",
    r"experts (?:agree|confirm|recommend)",
]

# Patterns for suspiciously precise statistics without sources
_SUSPICIOUS_STAT_PATTERNS = [
    r"\b\d{1,3}\.\d{2,}%\b",  # Overly precise percentages like 87.43%
    r"\b(?:exactly|precisely|specifically) \d+",
    r"\$[\d,]+\.\d{2}\b.*(?:projected|estimated|expected)",
]

# Contradiction indicators
_CONTRADICTION_PAIRS = [
    ("high risk", "low risk"),
    ("no risk", "significant risk"),
    ("strongly recommend", "not recommend"),
    ("safe", "dangerous"),
    ("approve", "deny"),
    ("certain", "uncertain"),
    ("guaranteed", "no guarantee"),
    ("100%", "0%"),
]


class HallucinationGuardrail(BaseGuardrail):
    """
    Detects potential hallucinations in AI-generated recommendations.

    Checks for:
    - Unverifiable authority claims (e.g., "studies show", "experts agree")
    - Suspiciously precise statistics without cited sources
    - Self-contradictory statements in reasoning
    - Confidence-reasoning mismatch (high confidence with vague reasoning)
    """

    def __init__(self):
        super().__init__("HallucinationGuardrail")

    def check(
        self,
        recommendation: AIRecommendation,
        policy: Dict[str, Any],
    ) -> GuardrailResult:
        config = policy.get("hallucination", {})
        max_fabrication_score = config.get("max_fabrication_score", 0.6)
        max_contradiction_score = config.get("max_contradiction_score", 0.5)
        issues: List[str] = []

        reasoning = recommendation.reasoning
        reasoning_lower = reasoning.lower()

        # ── Check 1: Unverifiable authority claims ───────────────────
        fabrication_hits = []
        for pattern in _FABRICATION_PATTERNS:
            matches = re.findall(pattern, reasoning_lower)
            fabrication_hits.extend(matches)

        fabrication_score = min(len(fabrication_hits) / 3.0, 1.0)
        if fabrication_score > max_fabrication_score:
            issues.append(
                f"Unverifiable claims detected ({len(fabrication_hits)} instances): "
                f"{', '.join(fabrication_hits[:3])}"
            )

        # ── Check 2: Suspiciously precise statistics ─────────────────
        stat_hits = []
        for pattern in _SUSPICIOUS_STAT_PATTERNS:
            matches = re.findall(pattern, reasoning)
            stat_hits.extend(matches)

        if len(stat_hits) >= 2:
            issues.append(
                f"Suspicious statistical precision without sources "
                f"({len(stat_hits)} instances)"
            )

        # ── Check 3: Self-contradictions ──────────────────────────────
        contradictions = []
        for term_a, term_b in _CONTRADICTION_PAIRS:
            if term_a in reasoning_lower and term_b in reasoning_lower:
                contradictions.append(f"'{term_a}' vs '{term_b}'")

        contradiction_score = min(len(contradictions) / 2.0, 1.0)
        if contradiction_score > max_contradiction_score:
            issues.append(
                f"Self-contradictory statements detected: "
                f"{'; '.join(contradictions)}"
            )

        # ── Check 4: Confidence-reasoning mismatch ───────────────────
        word_count = len(reasoning.split())
        if recommendation.confidence > 0.9 and word_count < 15:
            issues.append(
                f"High confidence ({recommendation.confidence:.0%}) "
                f"with minimal reasoning ({word_count} words) — "
                f"possible hallucinated confidence"
            )

        if recommendation.confidence < 0.3 and word_count > 100:
            issues.append(
                f"Low confidence ({recommendation.confidence:.0%}) "
                f"with extensive reasoning ({word_count} words) — "
                f"reasoning may be fabricated to compensate"
            )

        # ── Compute overall hallucination score ──────────────────────
        hallucination_score = min(
            (fabrication_score + contradiction_score + len(stat_hits) * 0.2) / 2.0,
            1.0,
        )

        if issues:
            return GuardrailResult(
                passed=False,
                reason="; ".join(issues),
                score=hallucination_score,
                severity="critical" if hallucination_score > 0.7 else "high",
                metadata={
                    "hallucination_score": round(hallucination_score, 3),
                    "fabrication_hits": fabrication_hits[:5],
                    "contradictions": contradictions,
                    "stat_hits": stat_hits[:5],
                },
            )

        return GuardrailResult(
            passed=True,
            score=round(hallucination_score, 3),
            severity="low",
            metadata={"hallucination_score": round(hallucination_score, 3)},
        )
