"""
Tests for the Hallucination Detection Guardrail.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.guardrails.hallucination_guardrail import HallucinationGuardrail
from models.decision import AIRecommendation


class TestHallucinationGuardrail:
    def setup_method(self):
        self.guardrail = HallucinationGuardrail()

    def test_passes_clean_recommendation(self, sample_policy, safe_recommendation):
        result = self.guardrail.check(safe_recommendation, sample_policy)
        assert result.passed is True

    def test_detects_fabricated_claims(self, sample_policy, hallucination_recommendation):
        result = self.guardrail.check(hallucination_recommendation, sample_policy)
        assert result.passed is False
        assert "Unverifiable claims" in result.reason or "contradiction" in result.reason.lower()

    def test_detects_self_contradictions(self, sample_policy):
        rec = AIRecommendation(
            decision="approve",
            confidence=0.7,
            reasoning=(
                "This is a high risk investment but also low risk overall. "
                "There is no risk involved but there is significant risk. "
                "We strongly recommend and also do not recommend this option. "
                "Past performance does not guarantee future results. "
                "Subject to terms and conditions."
            ),
            risk_score=0.3,
        )
        result = self.guardrail.check(rec, sample_policy)
        assert result.passed is False
        assert "contradict" in result.reason.lower()

    def test_detects_confidence_mismatch(self, sample_policy):
        rec = AIRecommendation(
            decision="approve",
            confidence=0.95,
            reasoning="Looks good.",
            risk_score=0.1,
        )
        result = self.guardrail.check(rec, sample_policy)
        assert result.passed is False
        assert "confidence" in result.reason.lower()

    def test_returns_hallucination_score(self, sample_policy, hallucination_recommendation):
        result = self.guardrail.check(hallucination_recommendation, sample_policy)
        assert result.metadata is not None
        assert "hallucination_score" in result.metadata
        assert result.metadata["hallucination_score"] >= 0
