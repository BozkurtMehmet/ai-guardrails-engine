"""
Metrics Collector — Tracks guardrail statistics for observability.

Singleton that collects per-decision metrics including verdict distribution,
per-guardrail trigger rates, constitution violations, and decision history.
"""

import threading
from collections import defaultdict
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

from models.decision import FinalDecision, DecisionVerdict


class MetricsCollector:
    """
    Thread-safe metrics collector for the AI Guardrails Engine.

    Tracks:
    - Total decisions and verdict distribution
    - Per-guardrail trigger counts and pass rates
    - Constitution violation counts
    - Recent decision history
    - Time-series data for dashboard charts
    """

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsCollector":
        """Singleton pattern — only one metrics collector per process."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.total_decisions = 0
        self.verdict_counts: Dict[str, int] = {
            "APPROVED": 0,
            "HUMAN_REVIEW": 0,
            "REJECTED": 0,
        }
        self.guardrail_triggers: Dict[str, int] = defaultdict(int)
        self.guardrail_passes: Dict[str, int] = defaultdict(int)
        self.guardrail_total: Dict[str, int] = defaultdict(int)
        self.constitution_violations: Dict[str, int] = defaultdict(int)
        self.hallucination_detections = 0
        self.recent_decisions: List[Dict[str, Any]] = []
        self.time_series: List[Dict[str, Any]] = []
        self._data_lock = threading.Lock()

    def record_decision(
        self,
        request: str,
        decision: FinalDecision,
        audit_id: str,
    ) -> None:
        """
        Record a decision and update all metrics.

        Args:
            request: The original user request.
            decision: The final decision from the orchestrator.
            audit_id: The audit record ID.
        """
        with self._data_lock:
            self.total_decisions += 1
            self.verdict_counts[decision.verdict.value] += 1

            # Track per-guardrail metrics
            for outcome in decision.guardrail_outcomes:
                name = outcome.guardrail_name
                self.guardrail_total[name] += 1
                if outcome.passed:
                    self.guardrail_passes[name] += 1
                else:
                    self.guardrail_triggers[name] += 1

                    # Track constitution violations separately
                    if name.startswith("Constitution:"):
                        principle = name.replace("Constitution:", "")
                        self.constitution_violations[principle] += 1

                    # Track hallucination detections
                    if name == "HallucinationGuardrail":
                        self.hallucination_detections += 1

            # Add to recent decisions (keep last 100)
            rec = decision.recommendation
            self.recent_decisions.append({
                "timestamp": datetime.now(UTC).isoformat(),
                "request": request[:200],
                "verdict": decision.verdict.value,
                "triggered_count": len(decision.triggered_guardrails),
                "triggered_guardrails": decision.triggered_guardrails,
                "reasons": decision.reasons,
                "audit_id": audit_id,
                "recommendation": {
                    "decision": rec.decision,
                    "confidence": rec.confidence,
                    "reasoning": rec.reasoning,
                    "risk_score": rec.risk_score,
                    "risk_category": rec.risk_category or "general",
                },
                "guardrail_outcomes": [
                    {
                        "name": o.guardrail_name,
                        "passed": o.passed,
                        "reason": o.reason,
                        "score": o.score,
                        "severity": o.severity,
                    }
                    for o in decision.guardrail_outcomes
                ],
            })
            if len(self.recent_decisions) > 100:
                self.recent_decisions = self.recent_decisions[-100:]

            # Add time-series data point
            self.time_series.append({
                "timestamp": datetime.now(UTC).isoformat(),
                "verdict": decision.verdict.value,
                "guardrails_triggered": len(decision.triggered_guardrails),
            })
            if len(self.time_series) > 500:
                self.time_series = self.time_series[-500:]

    def get_metrics(self) -> Dict[str, Any]:
        """Return all metrics as a dictionary for the dashboard API."""
        with self._data_lock:
            # Calculate per-guardrail pass rates
            guardrail_stats = {}
            for name in self.guardrail_total:
                total = self.guardrail_total[name]
                triggers = self.guardrail_triggers.get(name, 0)
                passes = self.guardrail_passes.get(name, 0)
                guardrail_stats[name] = {
                    "total_checks": total,
                    "triggers": triggers,
                    "passes": passes,
                    "trigger_rate": round(triggers / total, 3) if total > 0 else 0,
                    "pass_rate": round(passes / total, 3) if total > 0 else 0,
                }

            return {
                "total_decisions": self.total_decisions,
                "verdict_distribution": dict(self.verdict_counts),
                "approval_rate": round(
                    self.verdict_counts["APPROVED"] / max(self.total_decisions, 1), 3
                ),
                "rejection_rate": round(
                    self.verdict_counts["REJECTED"] / max(self.total_decisions, 1), 3
                ),
                "human_review_rate": round(
                    self.verdict_counts["HUMAN_REVIEW"] / max(self.total_decisions, 1),
                    3,
                ),
                "guardrail_stats": guardrail_stats,
                "constitution_violations": dict(self.constitution_violations),
                "total_constitution_violations": sum(
                    self.constitution_violations.values()
                ),
                "hallucination_detections": self.hallucination_detections,
                "recent_decisions": list(reversed(self.recent_decisions[-20:])),
                "time_series": self.time_series[-50:],
            }

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._data_lock:
            self.total_decisions = 0
            self.verdict_counts = {
                "APPROVED": 0,
                "HUMAN_REVIEW": 0,
                "REJECTED": 0,
            }
            self.guardrail_triggers.clear()
            self.guardrail_passes.clear()
            self.guardrail_total.clear()
            self.constitution_violations.clear()
            self.hallucination_detections = 0
            self.recent_decisions.clear()
            self.time_series.clear()
