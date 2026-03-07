"""
Fairness Guardrail — Detects and prevents biased or discriminatory outcomes.

Analyses demographic data to identify disparities and checks for
protected-attribute influence on AI decisions.
"""

import re
from typing import Any, Dict

from app.guardrails.base_guardrail import BaseGuardrail, GuardrailResult
from models.decision import AIRecommendation


class FairnessGuardrail(BaseGuardrail):
    """Evaluates fairness and detects potential bias in AI recommendations."""

    def __init__(self):
        super().__init__("FairnessGuardrail")

    def check(
        self,
        recommendation: AIRecommendation,
        policy: Dict[str, Any],
    ) -> GuardrailResult:
        config = policy.get("fairness", {})
        max_disparity = config.get("max_demographic_disparity", 0.15)
        protected_attrs = config.get("protected_attributes", [])
        issues = []

        demographic_data = recommendation.demographic_data

        # --- If no demographic data, pass but note it ---
        if not demographic_data:
            return GuardrailResult(
                passed=True,
                reason="No demographic data provided for fairness evaluation",
                severity="low",
                metadata={"note": "Skipped — no demographic data"},
            )

        # --- Check for protected attribute influence ---
        reasoning_lower = recommendation.reasoning.lower()
        for attr in protected_attrs:
            if re.search(r'\b' + re.escape(attr.lower()) + r'\b', reasoning_lower):
                issues.append(
                    f"Protected attribute '{attr}' appears to influence the decision"
                )

        # --- Check demographic disparity ---
        approval_rates = demographic_data.get("approval_rates", {})
        if approval_rates and len(approval_rates) >= 2:
            rates = list(approval_rates.values())
            max_rate = max(rates)
            min_rate = min(rates)
            disparity = max_rate - min_rate

            if disparity > max_disparity:
                issues.append(
                    f"Demographic disparity too high ({disparity:.2f}, "
                    f"maximum {max_disparity})"
                )

        # --- Check group-level approval rates ---
        min_group_rate = config.get("min_group_approval_rate", 0.6)
        for group, rate in approval_rates.items():
            if rate < min_group_rate:
                issues.append(
                    f"Approval rate for '{group}' is below minimum "
                    f"({rate:.2f}, minimum {min_group_rate})"
                )

        if issues:
            return GuardrailResult(
                passed=False,
                reason="; ".join(issues),
                severity="critical" if len(issues) > 1 else "high",
                metadata={"demographic_data": demographic_data},
            )

        return GuardrailResult(
            passed=True,
            severity="low",
            metadata={"demographic_data": demographic_data},
        )
