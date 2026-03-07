"""
Base Guardrail — Abstract base class for all guardrail checks.

All guardrails inherit from BaseGuardrail and implement the `check()` method,
which evaluates an AI recommendation against a policy and returns a GuardrailResult.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from models.decision import AIRecommendation


@dataclass
class GuardrailResult:
    """
    Result of a single guardrail check.

    Attributes:
        passed: Whether the guardrail check passed.
        reason: Explanation if the check failed.
        score: Optional numeric score from the evaluation.
        severity: Severity level — 'low', 'medium', 'high', 'critical'.
        metadata: Additional data produced by the guardrail.
    """

    passed: bool
    reason: Optional[str] = None
    score: Optional[float] = None
    severity: str = "medium"
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


class BaseGuardrail(ABC):
    """
    Abstract base class for all guardrail checks.

    Subclasses must implement `check()` to evaluate an AI recommendation
    against a loaded policy dictionary.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def check(
        self,
        recommendation: AIRecommendation,
        policy: Dict[str, Any],
    ) -> GuardrailResult:
        """
        Run the guardrail check.

        Args:
            recommendation: The AI recommendation to evaluate.
            policy: The loaded policy dictionary.

        Returns:
            GuardrailResult with pass/fail and details.
        """
        pass

    def __call__(
        self,
        recommendation: AIRecommendation,
        policy: Dict[str, Any],
    ) -> GuardrailResult:
        """Allow calling the guardrail like a function."""
        result = self.check(recommendation, policy)

        if not isinstance(result, GuardrailResult):
            raise TypeError(
                f"{self.name} guardrail must return a GuardrailResult, "
                f"got {type(result)}"
            )

        return result
