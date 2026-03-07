"""
Explainability Guardrail — Ensures AI decisions have sufficient reasoning.

Checks that the AI recommendation includes adequate explanation,
required fields, and minimum confidence levels as defined by policy.
"""

from typing import Any, Dict

from app.guardrails.base_guardrail import BaseGuardrail, GuardrailResult
from models.decision import AIRecommendation


class ExplainabilityGuardrail(BaseGuardrail):
    """Validates that AI recommendations are sufficiently explainable."""

    def __init__(self):
        super().__init__("ExplainabilityGuardrail")

    def check(
        self,
        recommendation: AIRecommendation,
        policy: Dict[str, Any],
    ) -> GuardrailResult:
        config = policy.get("explainability", {})
        min_length = config.get("min_reasoning_length", 50)
        required_fields = config.get("required_fields", [])
        min_confidence = config.get("min_confidence", 0.3)
        issues = []

        # --- Check reasoning length ---
        if len(recommendation.reasoning.strip()) < min_length:
            issues.append(
                f"Reasoning too short ({len(recommendation.reasoning.strip())} chars, "
                f"minimum {min_length})"
            )

        # --- Check required fields ---
        for field_name in required_fields:
            value = getattr(recommendation, field_name, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                issues.append(f"Required field '{field_name}' is missing or empty")

        # --- Check confidence ---
        if recommendation.confidence < min_confidence:
            issues.append(
                f"Confidence too low ({recommendation.confidence:.2f}, "
                f"minimum {min_confidence})"
            )

        if issues:
            return GuardrailResult(
                passed=False,
                reason="; ".join(issues),
                score=recommendation.confidence,
                severity="high" if len(issues) > 1 else "medium",
            )

        return GuardrailResult(
            passed=True,
            score=recommendation.confidence,
            severity="low",
        )
