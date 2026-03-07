"""
Tests for the Decision Orchestrator.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.decision import DecisionVerdict


class TestDecisionOrchestrator:
    def test_approves_safe_recommendation(self, orchestrator, safe_recommendation):
        decision = orchestrator.evaluate(safe_recommendation)
        assert decision.verdict == DecisionVerdict.APPROVED
        assert len(decision.triggered_guardrails) == 0

    def test_rejects_risky_recommendation(self, orchestrator, risky_recommendation):
        decision = orchestrator.evaluate(risky_recommendation)
        assert decision.verdict == DecisionVerdict.REJECTED
        assert len(decision.triggered_guardrails) > 0
        assert "RiskGuardrail" in decision.triggered_guardrails

    def test_flags_biased_recommendation(self, orchestrator, biased_recommendation):
        decision = orchestrator.evaluate(biased_recommendation)
        # Should be HUMAN_REVIEW or REJECTED due to bias
        assert decision.verdict in (DecisionVerdict.HUMAN_REVIEW, DecisionVerdict.REJECTED)
        assert "FairnessGuardrail" in decision.triggered_guardrails

    def test_flags_unexplained_recommendation(self, orchestrator, unexplained_recommendation):
        decision = orchestrator.evaluate(unexplained_recommendation)
        assert decision.verdict in (DecisionVerdict.HUMAN_REVIEW, DecisionVerdict.REJECTED)
        assert "ExplainabilityGuardrail" in decision.triggered_guardrails

    def test_rejects_injection_attempt(self, orchestrator, injection_recommendation):
        decision = orchestrator.evaluate(injection_recommendation)
        assert decision.verdict == DecisionVerdict.REJECTED
        assert "PromptInjectionGuardrail" in decision.triggered_guardrails

    def test_all_guardrails_run(self, orchestrator, safe_recommendation):
        decision = orchestrator.evaluate(safe_recommendation)
        guardrail_names = [o.guardrail_name for o in decision.guardrail_outcomes]
        assert "ExplainabilityGuardrail" in guardrail_names
        assert "RiskGuardrail" in guardrail_names
        assert "RegulationGuardrail" in guardrail_names
        assert "FairnessGuardrail" in guardrail_names
        assert "PromptInjectionGuardrail" in guardrail_names
