"""
Audit Logger — Maintains a traceable log of all AI decisions.

Writes JSON-formatted audit records to disk and provides retrieval
capabilities for compliance and review purposes.
"""

import json
import os
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Optional

from models.decision import (
    AIRecommendation,
    AuditRecord,
    DecisionVerdict,
    FinalDecision,
    GuardrailOutcome,
)


class AuditLogger:
    """
    JSON-based audit logger for AI decision records.

    Each decision is recorded as a JSON file in the logs directory
    with a unique ID, timestamp, and full decision context.
    """

    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize the audit logger.

        Args:
            log_dir: Directory for audit log files.
                     Defaults to audit/logs/ in the project root.
        """
        if log_dir is None:
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "audit",
                "logs",
            )
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log(self, request: str, decision: FinalDecision) -> AuditRecord:
        """
        Record a decision to the audit log.

        Args:
            request: The original user request.
            decision: The final decision from the orchestrator.

        Returns:
            The created AuditRecord.
        """
        record = AuditRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC),
            request=request,
            recommendation=decision.recommendation,
            guardrail_outcomes=decision.guardrail_outcomes,
            final_verdict=decision.verdict,
            triggered_guardrails=decision.triggered_guardrails,
            reasons=decision.reasons,
        )

        # Write to JSON file
        filename = f"{record.timestamp.strftime('%Y%m%d_%H%M%S')}_{record.id[:8]}.json"
        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record.model_dump(mode="json"), f, indent=2, default=str)

        return record

    def get_logs(self, limit: int = 50) -> List[AuditRecord]:
        """
        Retrieve audit log records from disk.

        Args:
            limit: Maximum number of records to return (most recent first).

        Returns:
            List of AuditRecord objects.
        """
        records = []
        log_files = sorted(
            Path(self.log_dir).glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for log_file in log_files[:limit]:
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                records.append(AuditRecord(**data))
            except (json.JSONDecodeError, ValueError) as e:
                # Skip corrupted log files
                continue

        return records

    def get_log_by_id(self, record_id: str) -> Optional[AuditRecord]:
        """
        Retrieve a specific audit record by its ID.

        Args:
            record_id: The unique record identifier.

        Returns:
            AuditRecord if found, None otherwise.
        """
        for log_file in Path(self.log_dir).glob("*.json"):
            if record_id[:8] in log_file.name:
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return AuditRecord(**data)
                except (json.JSONDecodeError, ValueError):
                    return None
        return None

    def clear_logs(self) -> int:
        """
        Remove all audit log files. Returns count of deleted files.
        """
        count = 0
        for log_file in Path(self.log_dir).glob("*.json"):
            log_file.unlink()
            count += 1
        return count
