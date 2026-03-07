"""
Pydantic models for the Constitutional AI layer.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConstitutionPrinciple(BaseModel):
    """A single principle from the AI constitution."""

    key: str = Field(..., description="Principle identifier (e.g., 'fairness')")
    title: str = Field(..., description="Human-readable principle title")
    description: str = Field(..., description="Full description of the principle")
    enforcement_rule: str = Field(..., description="The enforcement rule name")
    severity: str = Field(default="medium", description="Enforcement severity")


class ConstitutionViolation(BaseModel):
    """A detected violation of a constitutional principle."""

    principle_key: str = Field(..., description="Which principle was violated")
    principle_title: str = Field(..., description="Title of the violated principle")
    severity: str = Field(..., description="Violation severity")
    reason: str = Field(..., description="Why this is a violation")
    enforcement_action: str = Field(
        default="reject", description="Required action: 'reject' or 'human_review'"
    )


class ConstitutionVerdict(BaseModel):
    """Result of evaluating a decision against the constitution."""

    compliant: bool = Field(
        ..., description="Whether the decision is constitutionally compliant"
    )
    violations: List[ConstitutionViolation] = Field(
        default_factory=list, description="List of constitutional violations"
    )
    principles_checked: int = Field(
        default=0, description="Number of principles checked"
    )
    required_action: Optional[str] = Field(
        default=None,
        description="Required action if violations found: 'reject' or 'human_review'",
    )
