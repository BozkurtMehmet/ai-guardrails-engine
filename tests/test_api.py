"""
Integration tests for the FastAPI API endpoints.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "llm_provider" in data


@pytest.mark.asyncio
async def test_decision_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/decision",
            json={
                "request": "Evaluate credit application",
                "scenario": "credit_application_safe",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["verdict"] == "APPROVED"
        assert "guardrail_outcomes" in data
        assert "audit_id" in data


@pytest.mark.asyncio
async def test_decision_endpoint_risky():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/decision",
            json={
                "request": "High risk application",
                "scenario": "credit_application_risky",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["verdict"] in ("REJECTED", "HUMAN_REVIEW")
        assert len(data["triggered_guardrails"]) > 0


@pytest.mark.asyncio
async def test_policy_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/policy")
        assert response.status_code == 200
        data = response.json()
        assert "policy" in data
        assert "explainability" in data["policy"]
        assert "risk" in data["policy"]


@pytest.mark.asyncio
async def test_playground_list_scenarios():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/playground/scenarios")
        assert response.status_code == 200
        data = response.json()
        assert "scenarios" in data
        assert len(data["scenarios"]) > 0


@pytest.mark.asyncio
async def test_playground_run_scenario():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/playground/high_risk")
        assert response.status_code == 200
        data = response.json()
        assert data["decision"]["verdict"] == "REJECTED"


@pytest.mark.asyncio
async def test_playground_invalid_scenario():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/playground/nonexistent")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_audit_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First create a decision to generate an audit log
        await client.post(
            "/api/decision",
            json={"request": "Test for audit", "scenario": "credit_application_safe"},
        )

        response = await client.get("/api/audit")
        assert response.status_code == 200
        data = response.json()
        assert "records" in data
        assert data["count"] >= 1
