"""
Regulation Guardrail — Blocks decisions violating regulatory rules.

Scans AI output for blocked terms and ensures required disclosures
are present in approved decisions.
"""

from typing import Any, Dict

from app.guardrails.base_guardrail import BaseGuardrail, GuardrailResult
from models.decision import AIRecommendation


class RegulationGuardrail(BaseGuardrail):
    """Validates AI outputs against regulatory compliance rules."""

    def __init__(self):
        super().__init__("RegulationGuardrail")

    def check(
        self,
        recommendation: AIRecommendation,
        policy: Dict[str, Any],
    ) -> GuardrailResult:
        config = policy.get("regulation", {})
        blocked_terms = config.get("blocked_terms", [])
        required_disclosures = config.get("required_disclosures", [])
        issues = []

        reasoning_lower = recommendation.reasoning.lower()

        # --- Check for blocked terms ---
        for term in blocked_terms:
            if term.lower() in reasoning_lower:
                issues.append(f"Blocked regulatory term found: '{term}'")

        # --- Check for required disclosures ---
        for disclosure in required_disclosures:
            if disclosure.lower() not in reasoning_lower:
                issues.append(f"Missing required disclosure: '{disclosure}'")

        if issues:
            return GuardrailResult(
                passed=False,
                reason="; ".join(issues),
                severity="critical" if any("Blocked" in i for i in issues) else "high",
            )

        return GuardrailResult(
            passed=True,
            severity="low",
        )
