# AI Guardrails Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests: 74 Passing](https://img.shields.io/badge/Tests-74%20Passing-success.svg)](#testing)

A **policy-driven AI decision framework** with Constitutional AI governance, hallucination detection, explainability, risk management, fairness, and regulatory compliance.

## Architecture

```
Client Request
      ↓
🔌 LLM Provider (Pluggable)
      ├─ MockLLMProvider (built-in)
      ├─ OpenAI / ChatGPT (custom)
      ├─ Anthropic / Claude (custom)
      ├─ Ollama (custom)
      └─ Any LLM via BaseLLMProvider
      ↓
AI Recommendation
      ↓
   🏛️ Constitution Enforcement
      ├─ Fairness (Non-Discrimination)
      ├─ Explainability (Transparency)
      ├─ Safety (Harm Prevention)
      ├─ Security (Manipulation Resistance)
      ├─ Regulatory Compliance
      ├─ Uncertainty Escalation
      ├─ Proportionality
      └─ Human Oversight
      ↓
   Guardrail Engine
      ├─ Explainability Check
      ├─ Risk Check
      ├─ Regulation Check
      ├─ Fairness Check
      ├─ Prompt Injection Check
      └─ 🌀 Hallucination Detection
      ↓
   Decision Orchestrator
      ↓
Final Decision (APPROVED / HUMAN_REVIEW / REJECTED)
      ↓
   📊 Metrics → Dashboard
   📋 Audit Logging
   🧪 Failure Playground
```

## Project Structure

```
ai-guardrails-engine/
├── app/
│   ├── main.py                    # FastAPI application (v1.0.0)
│   ├── ai_engine.py               # LLM facade (backward compatible)
│   ├── llm/                       # 🔌 Pluggable LLM provider system
│   │   ├── __init__.py
│   │   ├── base_provider.py       # Abstract interface for LLM integration
│   │   ├── mock_provider.py       # Built-in mock provider (reference impl)
│   │   └── registry.py            # Provider registration & management
│   └── guardrails/                # Modular guardrail checks
│       ├── base_guardrail.py
│       ├── explainability_guardrail.py
│       ├── risk_guardrail.py
│       ├── regulation_guardrail.py
│       ├── fairness_guardrail.py
│       ├── prompt_injection_guardrail.py
│       └── hallucination_guardrail.py
├── constitution/
│   ├── ai_constitution.yaml       # AI Constitutional Policy (v1.0)
│   ├── constitution_loader.py     # Loads & caches constitution
│   └── enforcer.py                # Validates decisions against principles
├── decision-engine/
│   └── orchestrator.py            # Constitution + Guardrails → Verdict
├── models/
│   ├── decision.py                # Pydantic data models
│   └── constitution.py            # Constitutional AI models
├── policies/
│   ├── default_policy.yaml        # YAML policy-as-code
│   └── policy_loader.py           # Policy reader with caching
├── audit/
│   ├── audit_logger.py            # JSON audit log writer
│   └── metrics_collector.py       # Observability metrics
├── dashboard/
│   └── index.html                 # Visual observability dashboard
├── failure_playground/
│   └── playground.py              # Edge-case testing sandbox
├── examples/
│   └── credit_application.py      # End-to-end demo script
├── tests/
│   ├── conftest.py
│   ├── test_guardrails.py
│   ├── test_orchestrator.py
│   ├── test_llm_provider.py       # LLM provider interface tests
│   ├── test_audit.py
│   ├── test_api.py
│   ├── test_constitution.py
│   └── test_hallucination.py
└── requirements.txt
```

## Setup & Installation

You can run the AI Guardrails Engine either using Docker or via a local Python virtual environment.

### Using Docker

1. Clone the repository and navigate to the project folder.
2. Run the following command:

```bash
docker-compose up -d --build
```

The API will be available at http://localhost:8000 and the Dashboard at http://localhost:8000/dashboard.

Useful Docker Commands:

View live logs: docker logs -f ai-guardrails-engine

Stop the engine: docker-compose down

Note: The policies/ and constitution/ directories are mounted as volumes. Any changes you make to the YAML files locally will be reflected in the container immediately.

### Manual Local Setup
If you prefer to run it locally without Docker:

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Start the API Server

```bash
uvicorn app.main:app --reload
```

API docs: `http://localhost:8000/docs`
Dashboard: `http://localhost:8000/dashboard`

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/decision` | Submit request for AI evaluation |
| `GET` | `/api/audit` | Retrieve audit logs |
| `GET` | `/api/audit/{id}` | Get specific audit record |
| `GET` | `/api/policy` | View active policy |
| `POST` | `/api/policy/reload` | Reload policy from disk |
| `GET` | `/api/constitution` | View AI constitution |
| `POST` | `/api/constitution/reload` | Reload constitution |
| `GET` | `/api/metrics` | Guardrail & decision metrics |
| `GET` | `/api/llm/providers` | List registered LLM providers |
| `POST` | `/api/llm/provider/{name}/activate` | Switch active LLM provider |
| `GET` | `/api/playground/scenarios` | List playground scenarios |
| `POST` | `/api/playground/{scenario}` | Run a failure scenario |
| `POST` | `/api/playground/run-all` | Run all failure scenarios |
| `GET` | `/dashboard` | Visual observability dashboard |

### Example: Submit a Decision

```bash
curl -X POST http://localhost:8000/api/decision \
  -H "Content-Type: application/json" \
  -d '{"request": "Evaluate credit application", "scenario": "credit_application_safe"}'

# or for windows
#curl -X POST http://127.0.0.1:8000/api/decision ^
# -H "Content-Type: application/json" ^
# -d "{\"request\":\"Evaluate credit application\",\"scenario\":\"credit_application_safe\"}"
```

### Run the Demo

```bash
python -m examples.credit_application
```

## LLM Integration

The system is designed so any developer can plug in their preferred LLM. Create a provider by extending `BaseLLMProvider`:

```python
from app.llm import BaseLLMProvider
from models.decision import AIRecommendation

class MyOpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "openai-gpt4"

    def generate_recommendation(self, request, context=None):
        # Call your LLM API here
        response = openai.chat(request)
        return AIRecommendation(
            decision=response.decision,
            confidence=response.confidence,
            reasoning=response.reasoning,
            risk_score=response.risk_score,
        )
```

Register and activate it:

```python
from app.llm import LLMProviderRegistry

registry = LLMProviderRegistry()
registry.register(MyOpenAIProvider(api_key="sk-..."))
registry.set_default("openai-gpt4")
```

Or use the API:

```bash
# List providers
curl http://localhost:8000/api/llm/providers

# Switch provider
curl -X POST http://localhost:8000/api/llm/provider/openai-gpt4/activate

# Use a specific provider per request
curl -X POST http://localhost:8000/api/decision \
  -H "Content-Type: application/json" \
  -d '{"request": "Evaluate loan", "provider": "openai-gpt4"}'
```

## Guardrails

| Guardrail             | What It Checks                                         |
|-----------------------|--------------------------------------------------------|
| **Explainability**    | Reasoning length, required fields, minimum confidence  |
| **Risk**              | Risk score thresholds, forbidden risk categories       |
| **Regulation**        | Blocked terms, required disclosures                    |
| **Fairness**          | Demographic disparity, protected attribute influence   |
| **Prompt Injection**  | Suspicious manipulation patterns                       |
| **Hallucination**     | Fabricated claims, fake stats, self-contradictions     |

## Constitutional AI

The system enforces an AI Constitution (`constitution/ai_constitution.yaml`) with 11 governance principles:

| Principle | Enforcement |
|-----------|-------------|
| Non-Discrimination | Reject if bias detected |
| Decision Transparency | Require explanation |
| Harm Prevention | Escalate if high risk |
| Decision Accountability | Enforce audit logging |
| Manipulation Resistance | Detect prompt injection |
| Legal Compliance | Block regulation violations |
| Uncertainty Escalation | Human review if low confidence |
| Proportional Decision | Require evidence-based decisions |
| Human-in-the-Loop | Human oversight for critical decisions |
| Robustness | Failure playground testing |
| System Transparency | Policy visibility |

## Dashboard

The visual observability dashboard at `/dashboard` provides:
- **Summary cards**: total decisions, approval/rejection rates, constitution violations, hallucination detections
- **Donut chart**: verdict distribution (APPROVED / HUMAN_REVIEW / REJECTED)
- **Bar chart**: per-guardrail trigger rates
- **Constitution grid**: principle compliance status
- **Quick test panel**: fire scenarios with one click
- **Live decisions table**: auto-refreshes every 5 seconds
<img width="1903" height="907" alt="Ekran görüntüsü 2026-03-07 021209" src="https://github.com/user-attachments/assets/84fe0061-4104-4e10-84ce-e86df532a2c5" />
<img width="1906" height="910" alt="Ekran görüntüsü 2026-03-07 020950" src="https://github.com/user-attachments/assets/f52783c8-96a5-450b-a11c-6c4fe74906be" />
<img width="1903" height="920" alt="Ekran görüntüsü 2026-03-07 021047" src="https://github.com/user-attachments/assets/a3aca716-98e0-4253-99d7-84acff1d327e" />



## Testing

```bash
python -m pytest tests/ -v
```

## Test on a real LLM

The Guardrails Engine was tested using `arcee-ai/trinity-large-preview:free` via OpenRouter to evaluate three highly realistic scenarios. The system successfully categorized the nuanced financial models:

1. **Prime Residential Mortgage (Approve)**
   - **Scenario**: Senior Software Engineer, 812 FICO score, massive financial safety net.
   - **LLM Assessment**: Confident approval (`approve_loan`), risk score 0.05.
   - **Guardrails Verdict**: `APPROVED` ✅ (No rules tripped).

2. **High-Risk SaaS Startup (Human Review)**
   - **Scenario**: 1-year-old startup requesting a $500k loan. High burn rate and highly uncertain growth projections.
   - **LLM Assessment**: Moderate risk (0.78), noting the founder's investment but extreme market volatility.
   - **Guardrails Verdict**: `HUMAN_REVIEW` ⚠️ (Automatically flagged because the constitution requires human oversight for risk > 0.70).

3. **Active Bankruptcy (Reject)**
   - **Scenario**: Unemployed applicant, 415 FICO score, filed for Chapter 7 bankruptcy recently, massive unsecured debt.
   - **LLM Assessment**: Identified critical risk factors.
   - **Guardrails Verdict**: `REJECTED` ❌ (Blocked due to safety policy and raw risk score violations).

## Tech Stack

- **Backend**: Python 3.11+ with FastAPI
- **AI Layer**: Pluggable LLM providers via `BaseLLMProvider` interface
- **Policy Engine**: YAML-based policy-as-code
- **Constitutional AI**: YAML constitution with Python enforcer
- **Guardrails**: Modular Python classes inheriting from BaseGuardrail
- **Hallucination Detection**: Pattern-based + statistical analysis
- **Logging**: JSON-based audit logs
- **Metrics**: Thread-safe singleton collector
- **Dashboard**: Vanilla HTML/JS/CSS
- **Testing**: Pytest with async support (74 tests)

## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
Copyright (c) 2026 BozkurtMehmet
