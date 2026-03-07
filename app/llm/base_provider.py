"""
Base LLM Provider — Abstract interface for LLM integrations.

Any developer can implement this interface to connect their preferred LLM
(OpenAI GPT, Anthropic Claude, Ollama, etc.) to the AI Guardrails Engine.

Example:
    from app.llm.base_provider import BaseLLMProvider
    from models.decision import AIRecommendation

    class MyLLMProvider(BaseLLMProvider):
        @property
        def name(self) -> str:
            return "my-custom-llm"

        def generate_recommendation(self, request, context=None):
            # Call your LLM API here
            response = my_llm_api.chat(request)
            return AIRecommendation(
                decision=response.decision,
                confidence=response.confidence,
                reasoning=response.reasoning,
                risk_score=response.risk_score,
            )
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from models.decision import AIRecommendation


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM provider integrations.

    Developers must implement:
        - name (property): A unique identifier for this provider.
        - generate_recommendation(): Core method that sends a request to the
          LLM and returns a structured AIRecommendation.

    The AIRecommendation return value must include:
        - decision (str): The LLM's recommended action (e.g. "approve_loan").
        - confidence (float): Model confidence score, 0.0 to 1.0.
        - reasoning (str): Explanation for the recommendation.
        - risk_score (float): Assessed risk level, 0.0 to 1.0.
        - risk_category (str, optional): Category of risk (e.g. "financial").
        - demographic_data (dict, optional): Data for fairness checks.
        - metadata (dict, optional): Any additional provider-specific data.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this LLM provider (e.g. 'openai-gpt4')."""
        ...

    @abstractmethod
    def generate_recommendation(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AIRecommendation:
        """
        Generate an AI recommendation for the given request.

        Args:
            request: The user's request text to be analyzed.
            context: Optional dictionary with additional context such as:
                - scenario: Pre-built scenario name for deterministic testing.
                - user_id: Identifier for the requesting user.
                - session_id: Current session identifier.
                - Any other provider-specific parameters.

        Returns:
            AIRecommendation with the model's structured output.

        Raises:
            Exception: If the LLM call fails. The guardrails engine will
                       handle exceptions gracefully.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"
