"""
LLM Provider Registry — Manages registered LLM providers.

Provides a central registry for discovering, registering, and switching
between different LLM provider implementations at runtime.

Usage:
    from app.llm.registry import LLMProviderRegistry

    registry = LLMProviderRegistry()

    # The MockLLMProvider is registered by default
    provider = registry.get_default()
    recommendation = provider.generate_recommendation("Evaluate loan")

    # Register and switch to a custom provider
    registry.register(MyOpenAIProvider(api_key="sk-..."))
    registry.set_default("openai-gpt4")
"""

from typing import Dict, List, Optional

from app.llm.base_provider import BaseLLMProvider
from app.llm.mock_provider import MockLLMProvider


class LLMProviderRegistry:
    """
    Central registry for LLM provider management.

    Automatically registers the built-in MockLLMProvider as the default.
    Developers can register additional providers and switch the active
    provider at any time.
    """

    def __init__(self):
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._default_name: Optional[str] = None

        # Auto-register the built-in mock provider
        mock = MockLLMProvider()
        self.register(mock)
        self.set_default(mock.name)

    def register(self, provider: BaseLLMProvider) -> None:
        """
        Register an LLM provider.

        Args:
            provider: An instance of a class implementing BaseLLMProvider.

        Raises:
            TypeError: If provider does not extend BaseLLMProvider.
            ValueError: If a provider with the same name is already registered.
        """
        if not isinstance(provider, BaseLLMProvider):
            raise TypeError(
                f"Provider must be an instance of BaseLLMProvider, "
                f"got {type(provider).__name__}"
            )
        if provider.name in self._providers:
            raise ValueError(
                f"Provider '{provider.name}' is already registered. "
                f"Use a different name or unregister the existing one first."
            )
        self._providers[provider.name] = provider

    def unregister(self, name: str) -> None:
        """
        Remove a registered provider.

        Args:
            name: Name of the provider to remove.

        Raises:
            KeyError: If no provider with this name is registered.
            ValueError: If attempting to unregister the current default.
        """
        if name not in self._providers:
            raise KeyError(f"No provider registered with name '{name}'")
        if name == self._default_name:
            raise ValueError(
                f"Cannot unregister the current default provider '{name}'. "
                f"Set a different default first."
            )
        del self._providers[name]

    def get(self, name: str) -> BaseLLMProvider:
        """
        Retrieve a registered provider by name.

        Args:
            name: Name of the provider.

        Returns:
            The registered BaseLLMProvider instance.

        Raises:
            KeyError: If no provider with this name is registered.
        """
        if name not in self._providers:
            available = ", ".join(self._providers.keys()) or "(none)"
            raise KeyError(
                f"No provider registered with name '{name}'. "
                f"Available: {available}"
            )
        return self._providers[name]

    def get_default(self) -> BaseLLMProvider:
        """
        Get the currently active (default) provider.

        Returns:
            The default BaseLLMProvider instance.

        Raises:
            RuntimeError: If no default provider is set.
        """
        if self._default_name is None or self._default_name not in self._providers:
            raise RuntimeError(
                "No default LLM provider is set. "
                "Register a provider and call set_default()."
            )
        return self._providers[self._default_name]

    def set_default(self, name: str) -> None:
        """
        Set the active provider by name.

        Args:
            name: Name of a registered provider to make the default.

        Raises:
            KeyError: If no provider with this name is registered.
        """
        if name not in self._providers:
            available = ", ".join(self._providers.keys()) or "(none)"
            raise KeyError(
                f"Cannot set default to '{name}' — not registered. "
                f"Available: {available}"
            )
        self._default_name = name

    def list_providers(self) -> List[Dict[str, str]]:
        """
        List all registered providers.

        Returns:
            List of dicts with 'name', 'type', and 'is_default' keys.
        """
        return [
            {
                "name": name,
                "type": type(provider).__name__,
                "is_default": name == self._default_name,
            }
            for name, provider in self._providers.items()
        ]

    @property
    def default_name(self) -> Optional[str]:
        """Name of the current default provider."""
        return self._default_name
