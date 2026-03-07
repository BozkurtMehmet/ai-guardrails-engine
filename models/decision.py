"""
Core data models for the AI Guardrails Engine.

Defines the structures used throughout the pipeline:
AI recommendations, guardrail outcomes, final decisions, and audit records.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DecisionVerdict(str, Enum):
    """Possible final verdicts for a decision."""

    APPROVED = "APPROVED"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    REJECTED = "REJECTED"


class AIRecommendation(BaseModel):
    """
    Raw output from the AI/LLM layer.

    Represents what the AI model recommends before guardrails are applied.
    """

    decision: str = Field(
        ..., description="The AI's recommended action (e.g., 'approve_loan')"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence score (0.0 to 1.0)"
    )
    reasoning: str = Field(
        ..., description="Explanation for why the AI made this recommendation"
    )
    risk_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Assessed risk level (0.0 to 1.0)"
    )
    risk_category: Optional[str] = Field(
        default=None, description="Category of risk (e.g., 'financial', 'safety')"
    )
    demographic_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Demographic information relevant to fairness checks",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata from the AI model"
    )


class GuardrailOutcome(BaseModel):
    """Result of a single guardrail evaluation."""

    guardrail_name: str = Field(..., description="Name of the guardrail that ran")
    passed: bool = Field(..., description="Whether the guardrail check passed")
    reason: Optional[str] = Field(
        default=None, description="Explanation if the guardrail failed"
    )
    score: Optional[float] = Field(
        default=None, description="Numeric score from the guardrail (if applicable)"
    )
    severity: str = Field(
        default="medium",
        description="Severity level: 'low', 'medium', 'high', 'critical'",
    )


class FinalDecision(BaseModel):
    """
    The orchestrated final decision after all guardrails have been evaluated.
    """

    verdict: DecisionVerdict = Field(
        ..., description="Final decision: APPROVED, HUMAN_REVIEW, or REJECTED"
    )
    recommendation: AIRecommendation = Field(
        ..., description="The original AI recommendation"
    )
    guardrail_outcomes: List[GuardrailOutcome] = Field(
        default_factory=list, description="Results from all guardrail checks"
    )
    triggered_guardrails: List[str] = Field(
        default_factory=list, description="Names of guardrails that failed"
    )
    reasons: List[str] = Field(
        default_factory=list, description="Aggregated reasons for the decision"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the decision was made"
    )


class AuditRecord(BaseModel):
    """
    Complete audit log entry capturing the full decision pipeline.
    """

    id: str = Field(..., description="Unique identifier for this audit record")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the record was created"
    )
    request: str = Field(..., description="Original user request")
    recommendation: AIRecommendation = Field(
        ..., description="AI recommendation that was generated"
    )
    guardrail_outcomes: List[GuardrailOutcome] = Field(
        default_factory=list, description="All guardrail evaluation results"
    )
    final_verdict: DecisionVerdict = Field(
        ..., description="The final decision verdict"
    )
    triggered_guardrails: List[str] = Field(
        default_factory=list, description="Guardrails that were triggered"
    )
    reasons: List[str] = Field(
        default_factory=list, description="All reasons for the decision"
    )
