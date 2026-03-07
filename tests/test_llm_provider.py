"""
Tests for the LLM provider interface, mock provider, and registry.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from typing import Any, Dict, Optional

from app.llm import BaseLLMProvider, MockLLMProvider, LLMProviderRegistry
from app.ai_engine import AIEngine
from models.decision import AIRecommendation


# ── Helper: Custom test provider ────────────────────────────────────────────


class DummyProvider(BaseLLMProvider):
    """Minimal custom provider for testing the interface."""

    @property
    def name(self) -> str:
        return "dummy-test"

    def generate_recommendation(
        self, request: str, context: Optional[Dict[str, Any]] = None
    ) -> AIRecommendation:
        return AIRecommendation(
            decision="dummy_decision",
            confidence=0.75,
            reasoning="This is a dummy provider response for testing purposes.",
            risk_score=0.10,
            risk_category="test",
            metadata={"provider": self.name, "request": request},
        )


# ── BaseLLMProvider Tests ───────────────────────────────────────────────────


class TestBaseLLMProvider:
    def test_cannot_instantiate_directly(self):
        """Abstract class should not be instantiatable."""
        with pytest.raises(TypeError):
            BaseLLMProvider()

    def test_custom_provider_implements_interface(self):
        """A concrete subclass can be instantiated and used."""
        provider = DummyProvider()
        assert provider.name == "dummy-test"
        rec = provider.generate_recommendation("test request")
        assert isinstance(rec, AIRecommendation)
        assert rec.decision == "dummy_decision"

    def test_repr(self):
        """Provider has a useful repr."""
        provider = DummyProvider()
        assert "DummyProvider" in repr(provider)
        assert "dummy-test" in repr(provider)


# ── MockLLMProvider Tests ───────────────────────────────────────────────────


class TestMockLLMProvider:
    def setup_method(self):
        self.provider = MockLLMProvider()

    def test_name(self):
        assert self.provider.name == "mock-llm"

    def test_scenario_response(self):
        """Known scenario returns the expected recommendation."""
        rec = self.provider.generate_recommendation(
            "test", context={"scenario": "credit_application_safe"}
        )
        assert rec.decision == "approve_loan"
        assert rec.confidence == 0.92
        assert rec.risk_score == 0.15

    def test_unknown_scenario_falls_back_to_default(self):
        """Unknown scenario name falls back to default generation."""
        rec = self.provider.generate_recommendation(
            "test", context={"scenario": "nonexistent_scenario"}
        )
        assert rec.decision == "process_request"

    def test_no_context_generates_default(self):
        """No context produces a default recommendation."""
        rec = self.provider.generate_recommendation("evaluate loan")
        assert rec.decision == "process_request"
        assert 0.0 <= rec.confidence <= 1.0
        assert 0.0 <= rec.risk_score <= 1.0

    def test_list_scenarios(self):
        """Should return all pre-defined scenario names."""
        scenarios = MockLLMProvider.list_scenarios()
        assert "credit_application_safe" in scenarios
        assert "prompt_injection_attempt" in scenarios
        assert "hallucination_scenario" in scenarios
        assert len(scenarios) >= 7

    def test_all_scenarios_produce_valid_recommendations(self):
        """Every scenario should return a valid AIRecommendation."""
        for scenario_name in MockLLMProvider.list_scenarios():
            rec = self.provider.generate_recommendation(
                "test", context={"scenario": scenario_name}
            )
            assert isinstance(rec, AIRecommendation)
            assert 0.0 <= rec.confidence <= 1.0
            assert 0.0 <= rec.risk_score <= 1.0
            assert len(rec.reasoning) > 0

    def test_metadata_includes_provider_name(self):
        """Recommendation metadata should include provider name."""
        rec = self.provider.generate_recommendation(
            "test", context={"scenario": "credit_application_safe"}
        )
        assert rec.metadata["provider"] == "mock-llm"


# ── LLMProviderRegistry Tests ──────────────────────────────────────────────


class TestLLMProviderRegistry:
    def test_default_provider_is_mock(self):
        """Fresh registry should have MockLLMProvider as default."""
        registry = LLMProviderRegistry()
        default = registry.get_default()
        assert isinstance(default, MockLLMProvider)
        assert default.name == "mock-llm"

    def test_register_custom_provider(self):
        """Can register a custom provider."""
        registry = LLMProviderRegistry()
        dummy = DummyProvider()
        registry.register(dummy)
        retrieved = registry.get("dummy-test")
        assert retrieved is dummy

    def test_register_duplicate_raises(self):
        """Registering a provider with the same name should raise."""
        registry = LLMProviderRegistry()
        registry.register(DummyProvider())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(DummyProvider())

    def test_register_non_provider_raises(self):
        """Registering a non-BaseLLMProvider should raise TypeError."""
        registry = LLMProviderRegistry()
        with pytest.raises(TypeError, match="BaseLLMProvider"):
            registry.register("not a provider")

    def test_get_unknown_raises(self):
        """Getting an unregistered provider should raise KeyError."""
        registry = LLMProviderRegistry()
        with pytest.raises(KeyError, match="no-such-provider"):
            registry.get("no-such-provider")

    def test_set_default(self):
        """Can switch the default provider."""
        registry = LLMProviderRegistry()
        registry.register(DummyProvider())
        registry.set_default("dummy-test")
        assert registry.default_name == "dummy-test"
        assert isinstance(registry.get_default(), DummyProvider)

    def test_set_default_unknown_raises(self):
        """Setting default to unknown name should raise KeyError."""
        registry = LLMProviderRegistry()
        with pytest.raises(KeyError):
            registry.set_default("nonexistent")

    def test_unregister(self):
        """Can unregister a non-default provider."""
        registry = LLMProviderRegistry()
        registry.register(DummyProvider())
        registry.unregister("dummy-test")
        with pytest.raises(KeyError):
            registry.get("dummy-test")

    def test_unregister_default_raises(self):
        """Cannot unregister the current default provider."""
        registry = LLMProviderRegistry()
        with pytest.raises(ValueError, match="current default"):
            registry.unregister("mock-llm")

    def test_list_providers(self):
        """List should include all registered providers with metadata."""
        registry = LLMProviderRegistry()
        registry.register(DummyProvider())
        providers = registry.list_providers()
        names = [p["name"] for p in providers]
        assert "mock-llm" in names
        assert "dummy-test" in names

        # Check default flag
        for p in providers:
            if p["name"] == "mock-llm":
                assert p["is_default"] is True
            else:
                assert p["is_default"] is False

    def test_full_lifecycle(self):
        """Register → set default → use → switch back → unregister."""
        registry = LLMProviderRegistry()

        # Register custom
        dummy = DummyProvider()
        registry.register(dummy)

        # Switch to custom
        registry.set_default("dummy-test")
        rec = registry.get_default().generate_recommendation("test")
        assert rec.decision == "dummy_decision"

        # Switch back to mock
        registry.set_default("mock-llm")
        rec = registry.get_default().generate_recommendation(
            "test", context={"scenario": "credit_application_safe"}
        )
        assert rec.decision == "approve_loan"

        # Unregister custom
        registry.unregister("dummy-test")
        assert len(registry.list_providers()) == 1


# ── AIEngine Backward Compatibility Tests ───────────────────────────────────


class TestAIEngineBackwardCompat:
    def test_engine_uses_default_provider(self):
        """AIEngine should delegate to the default provider."""
        engine = AIEngine()
        assert engine.model_name == "mock-llm"

    def test_engine_generate_recommendation(self):
        """generate_recommendation should still work as before."""
        engine = AIEngine()
        rec = engine.generate_recommendation(
            "test loan application",
            scenario="credit_application_safe",
        )
        assert rec.decision == "approve_loan"
        assert rec.confidence == 0.92

    def test_engine_default_recommendation(self):
        """Default (no scenario) should produce a valid recommendation."""
        engine = AIEngine()
        rec = engine.generate_recommendation("evaluate something")
        assert isinstance(rec, AIRecommendation)
        assert rec.decision == "process_request"

    def test_engine_list_scenarios(self):
        """list_scenarios should still work."""
        scenarios = AIEngine.list_scenarios()
        assert len(scenarios) >= 7
        assert "credit_application_safe" in scenarios

    def test_engine_with_specific_provider(self):
        """AIEngine can target a specific registered provider."""
        from app.ai_engine import get_registry
        registry = get_registry()

        # Register a dummy for this test
        try:
            registry.register(DummyProvider())
        except ValueError:
            pass  # Already registered in a previous test

        engine = AIEngine(provider_name="dummy-test")
        rec = engine.generate_recommendation("test")
        assert rec.decision == "dummy_decision"
