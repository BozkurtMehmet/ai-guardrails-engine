"""
Decision Orchestrator — Runs constitution enforcement and all guardrails.

Takes an AI recommendation, first validates it against the AI Constitution,
then evaluates it through all registered guardrails, and determines whether
the decision should be APPROVED, sent for HUMAN_REVIEW, or REJECTED.
"""

from typing import Any, Dict, List, Optional

from app.guardrails.base_guardrail import BaseGuardrail, GuardrailResult
from app.guardrails.explainability_guardrail import ExplainabilityGuardrail
from app.guardrails.risk_guardrail import RiskGuardrail
from app.guardrails.regulation_guardrail import RegulationGuardrail
from app.guardrails.fairness_guardrail import FairnessGuardrail
from app.guardrails.prompt_injection_guardrail import PromptInjectionGuardrail
from app.guardrails.hallucination_guardrail import HallucinationGuardrail
from constitution.enforcer import ConstitutionEnforcer
from models.constitution import ConstitutionVerdict
from models.decision import (
    AIRecommendation,
    DecisionVerdict,
    FinalDecision,
    GuardrailOutcome,
)


class DecisionOrchestrator:
    """
    Orchestrates the full guardrail + constitution evaluation pipeline.

    Pipeline order:
    1. Constitution enforcement (pre-check)
    2. Individual guardrail checks
    3. Verdict determination based on all results
    """

    def __init__(self, policy: Dict[str, Any], constitution: Dict[str, Any] = None):
        """
        Initialize the orchestrator with policy and optional constitution.

        Args:
            policy: The loaded policy dictionary.
            constitution: Pre-loaded constitution dict. If None, loads default.
        """
        self.policy = policy
        self.guardrails: List[BaseGuardrail] = [
            ExplainabilityGuardrail(),
            RiskGuardrail(),
            RegulationGuardrail(),
            FairnessGuardrail(),
            PromptInjectionGuardrail(),
            HallucinationGuardrail(),
        ]

        # Initialize constitution enforcer
        try:
            self.constitution_enforcer = ConstitutionEnforcer(constitution)
        except Exception:
            self.constitution_enforcer = None

    def register_guardrail(self, guardrail: BaseGuardrail) -> None:
        """Add a custom guardrail to the pipeline."""
        self.guardrails.append(guardrail)

    def evaluate(self, recommendation: AIRecommendation) -> FinalDecision:
        """
        Run constitution enforcement + all guardrails and produce a verdict.

        Pipeline:
        1. Constitution check → may produce violations
        2. Guardrail checks → each produces pass/fail
        3. Verdict logic:
           - Constitution critical violation → REJECTED
           - Any guardrail fails with severity 'critical' → REJECTED
           - Any guardrail fails (non-critical) → HUMAN_REVIEW
           - All pass → APPROVED

        Args:
            recommendation: The AI recommendation to evaluate.

        Returns:
            FinalDecision with verdict, outcomes, and reasons.
        """
        outcomes: List[GuardrailOutcome] = []
        triggered: List[str] = []
        reasons: List[str] = []
        has_critical = False
        has_failure = False

        # ── Step 1: Constitution Enforcement ─────────────────────────
        constitution_verdict: Optional[ConstitutionVerdict] = None
        if self.constitution_enforcer:
            constitution_verdict = self.constitution_enforcer.enforce(recommendation)

            for violation in constitution_verdict.violations:
                outcome = GuardrailOutcome(
                    guardrail_name=f"Constitution:{violation.principle_title}",
                    passed=False,
                    reason=violation.reason,
                    severity=violation.severity,
                )
                outcomes.append(outcome)
                triggered.append(f"Constitution:{violation.principle_key}")
                reasons.append(violation.reason)
                has_failure = True

                if violation.severity == "critical":
                    has_critical = True

        # ── Step 2: Guardrail Checks ─────────────────────────────────
        for guardrail in self.guardrails:
            result: GuardrailResult = guardrail(recommendation, self.policy)

            outcome = GuardrailOutcome(
                guardrail_name=guardrail.name,
                passed=result.passed,
                reason=result.reason,
                score=result.score,
                severity=result.severity,
            )
            outcomes.append(outcome)

            if not result.passed:
                triggered.append(guardrail.name)
                reasons.append(f"[{guardrail.name}] {result.reason}")
                has_failure = True

                if result.severity == "critical":
                    has_critical = True

        # ── Step 3: Determine Verdict ────────────────────────────────
        if has_critical:
            verdict = DecisionVerdict.REJECTED
        elif has_failure:
            verdict = DecisionVerdict.HUMAN_REVIEW
        else:
            verdict = DecisionVerdict.APPROVED

        return FinalDecision(
            verdict=verdict,
            recommendation=recommendation,
            guardrail_outcomes=outcomes,
            triggered_guardrails=triggered,
            reasons=reasons,
        )
