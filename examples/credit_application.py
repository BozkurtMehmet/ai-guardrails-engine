"""
Example: Credit Application Flow

Demonstrates the full AI Guardrails Engine pipeline using
a financial credit application scenario.

Shows how to use the LLM provider system for pluggable LLM integration.

Run from the project root:
    python -m examples.credit_application
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
from app.ai_engine import AIEngine, get_registry
from app.llm import BaseLLMProvider, MockLLMProvider, LLMProviderRegistry
from audit.audit_logger import AuditLogger
from models.decision import AIRecommendation
from policies.policy_loader import get_policy
from failure_playground.playground import FailurePlayground

# Import orchestrator from hyphenated module
orchestrator_module = importlib.import_module("decision-engine.orchestrator")
DecisionOrchestrator = orchestrator_module.DecisionOrchestrator


def print_separator():
    print("=" * 70)


def print_decision(label: str, request: str, scenario: str):
    """Run a single scenario and print the results."""
    print_separator()
    print(f"  SCENARIO: {label}")
    print_separator()

    # 1. Generate AI recommendation
    engine = AIEngine()
    recommendation = engine.generate_recommendation(request, scenario=scenario)

    print(f"\n📥 Request: {request}")
    print(f"🤖 AI Decision: {recommendation.decision}")
    print(f"   Provider: {engine.model_name}")
    print(f"   Confidence: {recommendation.confidence:.0%}")
    print(f"   Risk Score: {recommendation.risk_score:.0%}")
    print(f"   Reasoning: {recommendation.reasoning[:100]}...")

    # 2. Run through guardrails
    policy = get_policy()
    orchestrator = DecisionOrchestrator(policy)
    decision = orchestrator.evaluate(recommendation)

    print(f"\n🏁 VERDICT: {decision.verdict.value}")

    if decision.triggered_guardrails:
        print(f"\n🚨 Triggered Guardrails:")
        for name in decision.triggered_guardrails:
            print(f"   ❌ {name}")

    if decision.reasons:
        print(f"\n📋 Reasons:")
        for reason in decision.reasons:
            print(f"   • {reason}")

    # Show all guardrail results
    print(f"\n📊 All Guardrail Results:")
    for outcome in decision.guardrail_outcomes:
        status = "✅" if outcome.passed else "❌"
        print(f"   {status} {outcome.guardrail_name} (severity: {outcome.severity})")

    # 3. Audit log
    logger = AuditLogger()
    record = logger.log(request, decision)
    print(f"\n📝 Audit Record ID: {record.id}")

    print()
    return decision


def demo_provider_system():
    """Demonstrate the pluggable LLM provider system."""
    print_separator()
    print("  LLM PROVIDER SYSTEM DEMO")
    print_separator()

    registry = get_registry()

    # Show registered providers
    print(f"\n🔌 Registered Providers:")
    for p in registry.list_providers():
        default_marker = " (DEFAULT)" if p["is_default"] else ""
        print(f"   • {p['name']} [{p['type']}]{default_marker}")

    # Show how a developer would create a custom provider
    print(f"\n📖 Custom Provider Example:")
    print(f"   To integrate your own LLM, create a class extending BaseLLMProvider:")
    print(f"")
    print(f"   class MyOpenAIProvider(BaseLLMProvider):")
    print(f"       @property")
    print(f"       def name(self): return 'openai-gpt4'")
    print(f"       def generate_recommendation(self, request, context=None):")
    print(f"           # Call OpenAI API here...")
    print(f"           return AIRecommendation(...)")
    print(f"")
    print(f"   Then register it:")
    print(f"   registry.register(MyOpenAIProvider())")
    print(f"   registry.set_default('openai-gpt4')")
    print()


def main():
    print("\n" + "=" * 70)
    print("   AI GUARDRAILS ENGINE — Credit Application Demo")
    print("=" * 70)

    # Show provider system
    demo_provider_system()

    # Scenario 1: Safe application
    print_decision(
        "Safe Credit Application",
        "Evaluate credit application for applicant with 780 credit score",
        "credit_application_safe",
    )

    # Scenario 2: Risky application
    print_decision(
        "High-Risk Credit Application",
        "Evaluate credit application with high debt-to-income ratio",
        "credit_application_risky",
    )

    # Scenario 3: Biased decision
    print_decision(
        "Biased Credit Decision",
        "Evaluate credit application with demographic analysis",
        "credit_application_biased",
    )

    # Scenario 4: Unexplained decision
    print_decision(
        "Unexplained Credit Decision",
        "Process loan with no documentation",
        "credit_application_unexplained",
    )

    # Summary
    print_separator()
    print("   FAILURE PLAYGROUND — Running All Scenarios")
    print_separator()

    policy = get_policy()
    orchestrator = DecisionOrchestrator(policy)
    playground = FailurePlayground(orchestrator)

    scenarios = playground.list_scenarios()
    print(f"\n📋 Available Scenarios ({len(scenarios)}):")
    for s in scenarios:
        print(f"   • {s['id']}: {s['name']}")

    print("\n✅ Demo complete! Check audit/logs/ for audit records.\n")


if __name__ == "__main__":
    main()
