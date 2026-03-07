"""
AI Engine — Facade over the LLM Provider system.

Provides backward-compatible access to LLM recommendations through
the pluggable provider registry. New code should use LLMProviderRegistry
directly; this module exists for backward compatibility.
"""

from typing import Any, Dict, List, Optional

from app.llm import BaseLLMProvider, LLMProviderRegistry, MockLLMProvider
from models.decision import AIRecommendation


# Module-level shared registry instance
_registry = LLMProviderRegistry()


def get_registry() -> LLMProviderRegistry:
    """Get the shared LLM provider registry."""
    return _registry


class AIEngine:
    """
    AI recommendation engine — facade over the LLM provider registry.

    For backward compatibility, this class wraps the provider system.
    New integrations should use LLMProviderRegistry directly.

    Args:
        provider_name: Optional name of a registered provider to use.
                       If None, uses the registry's current default.
        registry: Optional custom registry. If None, uses the shared one.
    """

    def __init__(
        self,
        provider_name: Optional[str] = None,
        registry: Optional[LLMProviderRegistry] = None,
    ):
        self._registry = registry or _registry
        self._provider_name = provider_name

    @property
    def provider(self) -> BaseLLMProvider:
        """Get the active LLM provider."""
        if self._provider_name:
            return self._registry.get(self._provider_name)
        return self._registry.get_default()

    @property
    def model_name(self) -> str:
        """Name of the active provider (backward compat)."""
        return self.provider.name

    def generate_recommendation(
        self,
        request: str,
        scenario: Optional[str] = None,
    ) -> AIRecommendation:
        """
        Generate an AI recommendation for the given request.

        Args:
            request: The user's request text.
            scenario: Optional scenario name for deterministic testing.

        Returns:
            AIRecommendation with the model's output.
        """
        context: Dict[str, Any] = {}
        if scenario:
            context["scenario"] = scenario

        return self.provider.generate_recommendation(request, context)

    @staticmethod
    def list_scenarios() -> List[str]:
        """Return available test scenario names (from MockLLMProvider)."""
        return MockLLMProvider.list_scenarios()
