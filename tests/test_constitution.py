"""
Tests for the Constitutional AI layer.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constitution.constitution_loader import load_constitution
from constitution.enforcer import ConstitutionEnforcer
from models.decision import AIRecommendation


class TestConstitutionLoader:
    def test_loads_constitution(self):
        constitution = load_constitution()
        assert "version" in constitution
        assert "principles" in constitution
        assert "governance" in constitution

    def test_has_all_principles(self):
        constitution = load_constitution()
        principles = constitution["principles"]
        expected = [
            "fairness", "explainability", "safety", "accountability",
            "security", "regulatory_compliance", "uncertainty_management",
            "proportionality", "human_oversight", "robustness", "transparency",
        ]
        for key in expected:
            assert key in principles, f"Missing principle: {key}"


class TestConstitutionEnforcer:
    def test_safe_recommendation_is_compliant(
        self, constitution_enforcer, safe_recommendation
    ):
        verdict = constitution_enforcer.enforce(safe_recommendation)
        assert verdict.compliant is True
        assert len(verdict.violations) == 0

    def test_detects_fairness_violation(
        self, constitution_enforcer, biased_recommendation
    ):
        verdict = constitution_enforcer.enforce(biased_recommendation)
        assert verdict.compliant is False
        fairness_violations = [
            v for v in verdict.violations if v.principle_key == "fairness"
        ]
        assert len(fairness_violations) > 0

    def test_detects_safety_violation(self, constitution_enforcer):
        rec = AIRecommendation(
            decision="approve",
            confidence=0.3,
            reasoning="High risk scenario with significant uncertainty " * 3,
            risk_score=0.9,
            risk_category="catastrophic",
        )
        verdict = constitution_enforcer.enforce(rec)
        assert verdict.compliant is False
        safety_violations = [
            v for v in verdict.violations if v.principle_key == "safety"
        ]
        assert len(safety_violations) > 0

    def test_detects_security_violation(
        self, constitution_enforcer, injection_recommendation
    ):
        verdict = constitution_enforcer.enforce(injection_recommendation)
        assert verdict.compliant is False
        security_violations = [
            v for v in verdict.violations if v.principle_key == "security"
        ]
        assert len(security_violations) > 0

    def test_detects_uncertainty_violation(self, constitution_enforcer):
        rec = AIRecommendation(
            decision="approve",
            confidence=0.3,
            reasoning="Analysis shows moderate indicators " * 5,
            risk_score=0.2,
        )
        verdict = constitution_enforcer.enforce(rec)
        assert verdict.compliant is False
        uncertainty_violations = [
            v for v in verdict.violations
            if v.principle_key == "uncertainty_management"
        ]
        assert len(uncertainty_violations) > 0

    def test_detects_regulatory_violation(self, constitution_enforcer):
        rec = AIRecommendation(
            decision="approve",
            confidence=0.8,
            reasoning="This investment offers guaranteed returns with no risk involved " * 2,
            risk_score=0.2,
        )
        verdict = constitution_enforcer.enforce(rec)
        assert verdict.compliant is False
        reg_violations = [
            v for v in verdict.violations
            if v.principle_key == "regulatory_compliance"
        ]
        assert len(reg_violations) > 0

    def test_required_action_is_reject_for_critical(
        self, constitution_enforcer, injection_recommendation
    ):
        verdict = constitution_enforcer.enforce(injection_recommendation)
        assert verdict.required_action == "reject"

    def test_principles_checked_count(
        self, constitution_enforcer, safe_recommendation
    ):
        verdict = constitution_enforcer.enforce(safe_recommendation)
        assert verdict.principles_checked >= 8
