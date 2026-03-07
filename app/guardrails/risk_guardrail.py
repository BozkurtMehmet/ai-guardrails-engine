"""
Risk Guardrail — Prevents decisions with excessive risk.

Validates the AI recommendation's risk score against policy thresholds
and checks for forbidden risk categories.
"""

from typing import Any, Dict

from app.guardrails.base_guardrail import BaseGuardrail, GuardrailResult
from models.decision import AIRecommendation


class RiskGuardrail(BaseGuardrail):
    """Evaluates risk levels and blocks dangerously risky decisions."""

    def __init__(self):
        super().__init__("RiskGuardrail")

    def check(
        self,
        recommendation: AIRecommendation,
        policy: Dict[str, Any],
    ) -> GuardrailResult:
        config = policy.get("risk", {})
        max_risk = config.get("max_risk_score", 0.7)
        forbidden = config.get("forbidden_categories", [])
        human_review_threshold = config.get("human_review_threshold", 0.5)
        issues = []
        severity = "low"

        # --- Check forbidden risk categories ---
        if recommendation.risk_category and recommendation.risk_category.lower() in [
            c.lower() for c in forbidden
        ]:
            issues.append(
                f"Forbidden risk category: '{recommendation.risk_category}'"
            )
            severity = "critical"

        # --- Check max risk score ---
        if recommendation.risk_score > max_risk:
            issues.append(
                f"Risk score too high ({recommendation.risk_score:.2f}, "
                f"maximum {max_risk})"
            )
            severity = "critical" if severity != "critical" else severity

        # --- Check human review threshold ---
        elif recommendation.risk_score > human_review_threshold:
            issues.append(
                f"Risk score requires human review ({recommendation.risk_score:.2f}, "
                f"threshold {human_review_threshold})"
            )
            severity = "high"

        if issues:
            return GuardrailResult(
                passed=False,
                reason="; ".join(issues),
                score=recommendation.risk_score,
                severity=severity,
            )

        return GuardrailResult(
            passed=True,
            score=recommendation.risk_score,
            severity="low",
        )
