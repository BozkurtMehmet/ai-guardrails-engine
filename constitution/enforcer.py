"""
Constitution Enforcer — Validates AI decisions against the AI Constitution.

Maps each constitutional principle to an enforcement check and produces
a ConstitutionVerdict indicating compliance or violations.
"""

import re
from typing import Any, Dict, List

from constitution.constitution_loader import get_constitution
from models.constitution import (
    ConstitutionViolation,
    ConstitutionVerdict,
)
from models.decision import AIRecommendation


class ConstitutionEnforcer:
    """
    Enforces the AI Constitution against AI recommendations.

    Each principle in the constitution maps to an enforcement rule.
    The enforcer runs all applicable rules and returns a verdict.
    """

    def __init__(self, constitution: Dict[str, Any] = None):
        """
        Initialize the enforcer.

        Args:
            constitution: Pre-loaded constitution dict. If None, loads default.
        """
        self.constitution = constitution or get_constitution()
        self.principles = self.constitution.get("principles", {})

        # Map enforcement rules to check methods
        self._rule_map = {
            "reject_if_bias_detected": self._check_fairness,
            "require_explanation": self._check_explainability,
            "escalate_if_high_risk": self._check_safety,
            "enforce_audit_logging": self._check_accountability,
            "detect_prompt_injection": self._check_security,
            "block_if_regulation_violation": self._check_regulatory,
            "escalate_if_low_confidence": self._check_uncertainty,
            "require_evidence_based_decision": self._check_proportionality,
            "trigger_failure_playground_tests": self._check_robustness,
            "enforce_policy_visibility": self._check_transparency,
        }

    def enforce(self, recommendation: AIRecommendation) -> ConstitutionVerdict:
        """
        Check an AI recommendation against all constitutional principles.

        Args:
            recommendation: The AI recommendation to validate.

        Returns:
            ConstitutionVerdict with compliance status and violations.
        """
        violations: List[ConstitutionViolation] = []
        checked = 0

        for key, principle in self.principles.items():
            enforcement = principle.get("enforcement", {})
            rule = enforcement.get("rule")

            if rule and rule in self._rule_map:
                checked += 1
                violation = self._rule_map[rule](key, principle, recommendation)
                if violation:
                    violations.append(violation)

            # Handle human_oversight (uses 'conditions' instead of 'enforcement.rule')
            if key == "human_oversight":
                checked += 1
                violation = self._check_human_oversight(key, principle, recommendation)
                if violation:
                    violations.append(violation)

        # Determine required action
        required_action = None
        if violations:
            severities = [v.severity for v in violations]
            if "critical" in severities:
                required_action = "reject"
            else:
                required_action = "human_review"

        return ConstitutionVerdict(
            compliant=len(violations) == 0,
            violations=violations,
            principles_checked=checked,
            required_action=required_action,
        )

    # ── Enforcement Rule Implementations ─────────────────────────────────

    def _check_fairness(
        self, key: str, principle: Dict, rec: AIRecommendation
    ):
        """Reject if bias detected in reasoning or demographic data."""
        protected = principle.get("protected_attributes", [])
        reasoning_lower = rec.reasoning.lower()

        for attr in protected:
            if re.search(r'\b' + re.escape(attr.lower()) + r'\b', reasoning_lower):
                return ConstitutionViolation(
                    principle_key=key,
                    principle_title=principle.get("title", ""),
                    severity=principle.get("enforcement", {}).get("severity", "critical"),
                    reason=f"Constitutional violation: protected attribute '{attr}' "
                           f"influences the decision reasoning",
                    enforcement_action="reject",
                )

        # Check demographic disparity
        if rec.demographic_data:
            rates = rec.demographic_data.get("approval_rates", {})
            if rates and len(rates) >= 2:
                values = list(rates.values())
                if max(values) - min(values) > 0.15:
                    return ConstitutionViolation(
                        principle_key=key,
                        principle_title=principle.get("title", ""),
                        severity="critical",
                        reason="Constitutional violation: demographic disparity "
                               f"exceeds threshold ({max(values) - min(values):.2f})",
                        enforcement_action="reject",
                    )
        return None

    def _check_explainability(
        self, key: str, principle: Dict, rec: AIRecommendation
    ):
        """Require adequate explanation in the recommendation."""
        reqs = principle.get("requirements", {})
        min_len = reqs.get("minimum_reasoning_length", 40)
        required_fields = reqs.get("required_fields", [])

        issues = []
        if len(rec.reasoning.strip()) < min_len:
            issues.append(
                f"Reasoning too short ({len(rec.reasoning.strip())} chars, "
                f"constitution requires {min_len})"
            )

        for field in required_fields:
            val = getattr(rec, field, None)
            if val is None or (isinstance(val, str) and not val.strip()):
                issues.append(f"Required field '{field}' missing")

        if issues:
            return ConstitutionViolation(
                principle_key=key,
                principle_title=principle.get("title", ""),
                severity=principle.get("enforcement", {}).get("severity", "high"),
                reason=f"Constitutional violation: {'; '.join(issues)}",
                enforcement_action="human_review",
            )
        return None

    def _check_safety(
        self, key: str, principle: Dict, rec: AIRecommendation
    ):
        """Escalate if risk score or uncertainty too high."""
        thresholds = principle.get("enforcement", {}).get("thresholds", {})
        max_risk = thresholds.get("max_risk_score", 0.7)
        max_uncertainty = thresholds.get("max_uncertainty", 0.5)

        if rec.risk_score > max_risk:
            return ConstitutionViolation(
                principle_key=key,
                principle_title=principle.get("title", ""),
                severity="critical",
                reason=f"Constitutional violation: risk score {rec.risk_score:.2f} "
                       f"exceeds safety threshold {max_risk}",
                enforcement_action="reject",
            )

        # Uncertainty = 1 - confidence
        uncertainty = 1.0 - rec.confidence
        if uncertainty > max_uncertainty and rec.risk_score > 0.3:
            return ConstitutionViolation(
                principle_key=key,
                principle_title=principle.get("title", ""),
                severity="critical",
                reason=f"Constitutional violation: high uncertainty ({uncertainty:.2f}) "
                       f"combined with moderate risk ({rec.risk_score:.2f})",
                enforcement_action="human_review",
            )
        return None

    def _check_accountability(
        self, key: str, principle: Dict, rec: AIRecommendation
    ):
        """Accountability is enforced by the audit system — always passes."""
        # This is a systemic check, not per-recommendation
        return None

    def _check_security(
        self, key: str, principle: Dict, rec: AIRecommendation
    ):
        """Detect prompt injection patterns using semantic analysis."""
        from app.guardrails.prompt_injection_guardrail import detect_injection

        text = f"{rec.decision} {rec.reasoning}"
        is_injection, score, detections = detect_injection(text, threshold=0.5)

        if is_injection:
            return ConstitutionViolation(
                principle_key=key,
                principle_title=principle.get("title", ""),
                severity="critical",
                reason=f"Constitutional violation: prompt injection "
                       f"detected (score: {score:.2f}): "
                       f"{'; '.join(detections[:2])}",
                enforcement_action="reject",
            )
        return None

    def _check_regulatory(
        self, key: str, principle: Dict, rec: AIRecommendation
    ):
        """Block if regulation-violating terms found."""
        blocked_terms = [
            "guaranteed returns",
            "no risk involved",
            "100% safe",
            "cannot lose",
        ]
        reasoning_lower = rec.reasoning.lower()
        for term in blocked_terms:
            if term in reasoning_lower:
                return ConstitutionViolation(
                    principle_key=key,
                    principle_title=principle.get("title", ""),
                    severity="critical",
                    reason=f"Constitutional violation: regulatory blocked "
                           f"term found: '{term}'",
                    enforcement_action="reject",
                )
        return None

    def _check_uncertainty(
        self, key: str, principle: Dict, rec: AIRecommendation
    ):
        """Escalate if confidence is below threshold."""
        thresholds = principle.get("thresholds", {})
        min_confidence = thresholds.get("min_confidence", 0.6)

        if rec.confidence < min_confidence:
            return ConstitutionViolation(
                principle_key=key,
                principle_title=principle.get("title", ""),
                severity=principle.get("enforcement", {}).get("severity", "medium"),
                reason=f"Constitutional violation: confidence {rec.confidence:.2f} "
                       f"below threshold {min_confidence}, requires human review",
                enforcement_action="human_review",
            )
        return None

    def _check_proportionality(
        self, key: str, principle: Dict, rec: AIRecommendation
    ):
        """Ensure decisions are proportional to evidence."""
        # High confidence + high risk without detailed reasoning is disproportionate
        if (
            rec.confidence > 0.8
            and rec.risk_score > 0.5
            and len(rec.reasoning.strip()) < 100
        ):
            return ConstitutionViolation(
                principle_key=key,
                principle_title=principle.get("title", ""),
                severity="medium",
                reason="Constitutional violation: high-confidence, high-risk "
                       "decision lacks proportional evidence",
                enforcement_action="human_review",
            )
        return None

    def _check_robustness(
        self, key: str, principle: Dict, rec: AIRecommendation
    ):
        """Robustness is validated via the failure playground — passes here."""
        return None

    def _check_transparency(
        self, key: str, principle: Dict, rec: AIRecommendation
    ):
        """Transparency is a system property — always passes."""
        return None

    def _check_human_oversight(
        self, key: str, principle: Dict, rec: AIRecommendation
    ):
        """Ensure human oversight for high-risk decisions."""
        conditions = principle.get("conditions", [])

        if "risk_score_above_threshold" in conditions and rec.risk_score > 0.7:
            return ConstitutionViolation(
                principle_key=key,
                principle_title=principle.get("title", ""),
                severity=principle.get("severity", "critical"),
                reason=f"Constitutional violation: high-risk decision "
                       f"(risk={rec.risk_score:.2f}) requires human oversight",
                enforcement_action="human_review",
            )
        return None
