"""
Tests for the Audit Logger.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.decision import DecisionVerdict


class TestAuditLogger:
    def test_log_creates_record(self, audit_logger, orchestrator, safe_recommendation):
        decision = orchestrator.evaluate(safe_recommendation)
        record = audit_logger.log("Test request", decision)

        assert record.id is not None
        assert record.request == "Test request"
        assert record.final_verdict == DecisionVerdict.APPROVED

    def test_log_writes_json_file(self, audit_logger, orchestrator, safe_recommendation):
        decision = orchestrator.evaluate(safe_recommendation)
        audit_logger.log("Test request", decision)

        # Check that a JSON file was created
        log_files = list(
            f for f in os.listdir(audit_logger.log_dir) if f.endswith(".json")
        )
        assert len(log_files) == 1

        # Validate JSON content
        with open(os.path.join(audit_logger.log_dir, log_files[0])) as f:
            data = json.load(f)
        assert data["request"] == "Test request"
        assert data["final_verdict"] == "APPROVED"

    def test_get_logs(self, audit_logger, orchestrator, safe_recommendation, risky_recommendation):
        decision1 = orchestrator.evaluate(safe_recommendation)
        decision2 = orchestrator.evaluate(risky_recommendation)

        audit_logger.log("Request 1", decision1)
        audit_logger.log("Request 2", decision2)

        logs = audit_logger.get_logs()
        assert len(logs) == 2

    def test_get_logs_limit(self, audit_logger, orchestrator, safe_recommendation):
        decision = orchestrator.evaluate(safe_recommendation)
        for i in range(5):
            audit_logger.log(f"Request {i}", decision)

        logs = audit_logger.get_logs(limit=3)
        assert len(logs) == 3

    def test_clear_logs(self, audit_logger, orchestrator, safe_recommendation):
        decision = orchestrator.evaluate(safe_recommendation)
        audit_logger.log("Test", decision)

        count = audit_logger.clear_logs()
        assert count == 1
        assert len(audit_logger.get_logs()) == 0
