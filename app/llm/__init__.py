"""
LLM Provider Package — Pluggable LLM integration for the AI Guardrails Engine.

To integrate your own LLM:
    1. Create a class that extends BaseLLMProvider
    2. Implement the `name` property and `generate_recommendation()` method
    3. Register your provider with LLMProviderRegistry

See MockLLMProvider for a reference implementation.
"""

from app.llm.base_provider import BaseLLMProvider
from app.llm.mock_provider import MockLLMProvider
from app.llm.registry import LLMProviderRegistry

__all__ = ["BaseLLMProvider", "MockLLMProvider", "LLMProviderRegistry"]
