"""
AI Guardrails Engine — FastAPI Application v2.0

REST API providing endpoints for AI decision evaluation, constitutional
governance, hallucination detection, audit logging, failure playground,
policy inspection, LLM provider management, metrics, and visual dashboard.
"""

import sys
import os

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ai_engine import AIEngine, get_registry
from app.llm import LLMProviderRegistry
from audit.audit_logger import AuditLogger
from audit.metrics_collector import MetricsCollector
from constitution.constitution_loader import get_constitution, reload_constitution
from policies.policy_loader import get_policy, reload_policy

# Import from decision-engine using importlib due to the hyphen
import importlib
orchestrator_module = importlib.import_module("decision-engine.orchestrator")
DecisionOrchestrator = orchestrator_module.DecisionOrchestrator

from failure_playground.playground import FailurePlayground


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Guardrails Engine",
    description=(
        "A policy-driven AI decision framework with Constitutional AI governance, "
        "hallucination detection, explainability, risk management, fairness, "
        "and regulatory compliance. Supports pluggable LLM providers."
    ),
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared instances
ai_engine = AIEngine()
audit_logger = AuditLogger()
metrics_collector = MetricsCollector()
llm_registry = get_registry()

# Serve dashboard static files
DASHBOARD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard"
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class DecisionRequest(BaseModel):
    """Request body for the /api/decision endpoint."""

    request: str = Field(
        ..., description="The user's request to be processed by the AI"
    )
    scenario: Optional[str] = Field(
        default=None,
        description="Optional pre-built scenario name for testing",
    )
    provider: Optional[str] = Field(
        default=None,
        description="Optional LLM provider name to use for this request",
    )


class HealthResponse(BaseModel):
    """Response body for the /api/health endpoint."""

    status: str
    version: str
    llm_provider: str
    constitution_version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    constitution = get_constitution()
    provider = llm_registry.get_default()
    return HealthResponse(
        status="healthy",
        version="2.1.0",
        llm_provider=provider.name,
        constitution_version=constitution.get("version", "unknown"),
    )


@app.post("/api/decision", tags=["Decision"])
async def make_decision(body: DecisionRequest):
    """
    Submit a request for AI evaluation with constitutional governance
    and guardrail checks.
    """
    try:
        policy = get_policy()
        constitution = get_constitution()

        # Use specified provider or default
        if body.provider:
            try:
                provider = llm_registry.get(body.provider)
            except KeyError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown LLM provider '{body.provider}'. "
                           f"Use GET /api/llm/providers to see available providers.",
                )
            engine = AIEngine(provider_name=body.provider)
        else:
            engine = ai_engine

        # Generate AI recommendation
        recommendation = engine.generate_recommendation(
            request=body.request,
            scenario=body.scenario,
        )

        # Run through constitution + guardrails
        orchestrator = DecisionOrchestrator(policy, constitution)
        decision = orchestrator.evaluate(recommendation)

        # Audit log
        audit_record = audit_logger.log(body.request, decision)

        # Record metrics
        metrics_collector.record_decision(
            body.request, decision, audit_record.id
        )

        return {
            "audit_id": audit_record.id,
            "verdict": decision.verdict.value,
            "llm_provider": engine.model_name,
            "recommendation": recommendation.model_dump(),
            "guardrail_outcomes": [
                o.model_dump() for o in decision.guardrail_outcomes
            ],
            "triggered_guardrails": decision.triggered_guardrails,
            "reasons": decision.reasons,
            "timestamp": decision.timestamp.isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audit", tags=["Audit"])
async def get_audit_logs(limit: int = 50):
    """Retrieve audit log records (most recent first)."""
    records = audit_logger.get_logs(limit=limit)
    return {
        "count": len(records),
        "records": [r.model_dump(mode="json") for r in records],
    }


@app.get("/api/audit/{record_id}", tags=["Audit"])
async def get_audit_record(record_id: str):
    """Retrieve a specific audit record by ID."""
    record = audit_logger.get_log_by_id(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Audit record not found")
    return record.model_dump(mode="json")


@app.get("/api/policy", tags=["Policy"])
async def get_current_policy():
    """View the currently active policy configuration."""
    try:
        policy = get_policy()
        return {"policy": policy}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Policy file not found")


@app.post("/api/policy/reload", tags=["Policy"])
async def reload_current_policy():
    """Force-reload the policy from disk."""
    try:
        policy = reload_policy()
        return {"message": "Policy reloaded successfully", "policy": policy}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Policy file not found")


@app.get("/api/constitution", tags=["Constitution"])
async def get_active_constitution():
    """View the active AI constitution."""
    try:
        constitution = get_constitution()
        return {"constitution": constitution}
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="Constitution file not found"
        )


@app.post("/api/constitution/reload", tags=["Constitution"])
async def reload_active_constitution():
    """Force-reload the constitution from disk."""
    try:
        constitution = reload_constitution()
        return {
            "message": "Constitution reloaded successfully",
            "constitution": constitution,
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="Constitution file not found"
        )


@app.get("/api/metrics", tags=["Metrics"])
async def get_metrics():
    """Return current guardrail and decision metrics for the dashboard."""
    return metrics_collector.get_metrics()


# ---------------------------------------------------------------------------
# LLM Provider Management
# ---------------------------------------------------------------------------


@app.get("/api/llm/providers", tags=["LLM Providers"])
async def list_llm_providers():
    """List all registered LLM providers."""
    return {
        "providers": llm_registry.list_providers(),
        "default": llm_registry.default_name,
    }


@app.post("/api/llm/provider/{provider_name}/activate", tags=["LLM Providers"])
async def activate_llm_provider(provider_name: str):
    """Set a registered LLM provider as the active default."""
    try:
        llm_registry.set_default(provider_name)
        return {
            "message": f"Provider '{provider_name}' is now the active default.",
            "default": llm_registry.default_name,
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# Playground
# ---------------------------------------------------------------------------


@app.get("/api/playground/scenarios", tags=["Playground"])
async def list_playground_scenarios():
    """List all available failure playground scenarios."""
    return {"scenarios": FailurePlayground.list_scenarios()}


@app.post("/api/playground/{scenario_name}", tags=["Playground"])
async def run_playground_scenario(scenario_name: str):
    """Run a specific failure playground scenario."""
    try:
        policy = get_policy()
        constitution = get_constitution()
        orchestrator = DecisionOrchestrator(policy, constitution)
        playground = FailurePlayground(orchestrator, ai_engine)
        result = playground.run_scenario(scenario_name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/playground/run-all", tags=["Playground"])
async def run_all_playground_scenarios():
    """Run all failure playground scenarios and return results."""
    try:
        policy = get_policy()
        constitution = get_constitution()
        orchestrator = DecisionOrchestrator(policy, constitution)
        playground = FailurePlayground(orchestrator, ai_engine)
        results = playground.run_all()
        return {"scenario_count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard", tags=["Dashboard"])
async def serve_dashboard():
    """Serve the observability dashboard."""
    dashboard_path = os.path.join(DASHBOARD_DIR, "index.html")
    if not os.path.exists(dashboard_path):
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(dashboard_path, media_type="text/html")
