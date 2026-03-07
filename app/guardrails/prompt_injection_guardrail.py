"""
Prompt Injection Guardrail — Multi-layer semantic injection detection.

Uses four detection layers to catch injection attempts regardless of
language, synonyms, or structural obfuscation:

1. Intent-based regex patterns  (catches synonyms/paraphrases)
2. Structural meta-instruction detection  (ALL CAPS directives, delimiters)
3. Multi-language injection patterns  (TR, ES, FR, DE, ZH, …)
4. Score-based threshold  (accumulates suspicion across layers)
"""

import re
from typing import Any, Dict, List, Tuple

from app.guardrails.base_guardrail import BaseGuardrail, GuardrailResult
from models.decision import AIRecommendation


# ── Layer 1: Intent-based regex patterns ─────────────────────────────────
# Each tuple: (compiled regex, human-readable category, weight)
_INTENT_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
    # Override / bypass / ignore + target
    (
        re.compile(
            r"(ignore|disregard|forget|skip|bypass|override|circumvent|"
            r"deactivate|disable|turn\s*off|remove|cancel|nullify|void|"
            r"suppress|stop\s*applying|do\s*not\s*enforce)"
            r"\s+.{0,30}?"
            r"(previous|prior|above|earlier|all|every|any|safety|rules?|"
            r"instructions?|guardrails?|assessments?|constraints?|"
            r"restrictions?|policies?|guidelines?|controls?|filters?|"
            r"checks?|validations?|limitations?)",
            re.IGNORECASE,
        ),
        "override/bypass directive",
        1.0,
    ),
    # Role switching: "act as / you are now / from now on"
    (
        re.compile(
            r"(act|behave|operate|respond|function|work)\s+"
            r"(as|like)\s+"
            r"(a\s+)?"
            r"(system|admin|administrator|root|developer|superuser|"
            r"unrestricted|unfiltered|god\s*mode|jailbroken)",
            re.IGNORECASE,
        ),
        "role-switching attempt",
        1.0,
    ),
    # System/admin meta-instructions
    (
        re.compile(
            r"(system|admin|internal|developer|hidden|secret|backdoor|"
            r"master|root|debug)\s*"
            r"(note|message|command|instruction|directive|override|"
            r"prompt|order|request|memo|alert|notice)\s*[:.\-]",
            re.IGNORECASE,
        ),
        "embedded meta-instruction",
        1.0,
    ),
    # Unconditional approval/acceptance demands
    (
        re.compile(
            r"(approve|accept|allow|grant|authorize|permit|pass|enable|"
            r"confirm|clear)\s+.{0,15}?"
            r"(regardless|everything|all\s*requests?|anyway|unconditionally|"
            r"without\s*checking|without\s*review|no\s*matter\s*what|always)",
            re.IGNORECASE,
        ),
        "unconditional approval demand",
        1.0,
    ),
    # Instruction to suppress checks
    (
        re.compile(
            r"(do\s*n[o']?t|don't|never|stop|cease|avoid|refrain\s+from)\s+"
            r"(check|verify|validate|enforce|apply|run|execute|perform|use)\s+"
            r".{0,20}?"
            r"(guardrail|rule|policy|safety|constraint|filter|validation|"
            r"restriction|control|check|limit)",
            re.IGNORECASE,
        ),
        "check suppression attempt",
        1.0,
    ),
    # "New instructions" / "from now on" patterns
    (
        re.compile(
            r"(from\s*now\s*on|henceforth|hereafter|going\s*forward|"
            r"new\s*instructions?|updated?\s*instructions?|"
            r"revised?\s*instructions?|replace\s*(?:all\s*)?instructions?)",
            re.IGNORECASE,
        ),
        "instruction replacement attempt",
        0.8,
    ),
]

# ── Layer 2: Structural meta-instruction patterns ────────────────────────
_STRUCTURAL_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
    # ALL CAPS directive words embedded in text (≥3 words in caps)
    (
        re.compile(
            r"(?:^|\.\s+|;\s+|\n)"
            r"([A-Z]{2,}\s+){2,}[A-Z]{2,}[:\.\!]",
        ),
        "ALL-CAPS embedded directive",
        0.7,
    ),
    # Explicit section-breaking markers used to inject commands
    (
        re.compile(
            r"(?:^|\n)\s*(?:---+|===+|####+|\*\*\*+|~~~+)\s*\n"
            r".*?(override|ignore|bypass|system|admin|new\s*instruction)",
            re.IGNORECASE | re.DOTALL,
        ),
        "delimiter-based injection",
        0.9,
    ),
    # Suspicious role-play framing
    (
        re.compile(
            r"(you\s+are\s+now|pretend\s+(?:you\s+are|to\s+be)|"
            r"imagine\s+you\s+are|roleplay\s+as|"
            r"switch\s+to\s+(?:a\s+)?(?:new|different)\s+(?:mode|role))",
            re.IGNORECASE,
        ),
        "role-play framing",
        0.8,
    ),
]

# ── Layer 3: Multi-language injection patterns ───────────────────────────
_MULTILANG_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
    # Turkish
    (
        re.compile(
            r"(önceki\s*talimatları?\s*(yoksay|unut|atla|iptal\s*et|görmezden\s*gel)|"
            r"güvenlik\s*(kontrollerini?|kurallarını?)\s*(devre\s*dışı\s*bırak|atla|kapat)|"
            r"sistem\s*(notu|komutu|talimatı)\s*[:.]|"
            r"koşulsuz\s*(onayla|kabul\s*et)|"
            r"tüm\s*(kuralları?|kısıtlamaları?)\s*(yoksay|unut|iptal\s*et))",
            re.IGNORECASE,
        ),
        "Turkish injection pattern",
        1.0,
    ),
    # Spanish
    (
        re.compile(
            r"(ignora\s*(las\s+)?instrucciones?\s*(anteriores?|previas?)|"
            r"omite\s*(las\s+)?reglas?|"
            r"nota\s*del?\s*sistema\s*[:.]|"
            r"aprueba\s*(todo|sin\s*importar)|"
            r"desactiva\s*(los?\s+)?(?:filtros?|controles?))",
            re.IGNORECASE,
        ),
        "Spanish injection pattern",
        1.0,
    ),
    # French
    (
        re.compile(
            r"(ignore[rz]?\s*(les\s+)?instructions?\s*(précédentes?|antérieures?)|"
            r"contourne[rz]?\s*(la\s+)?sécurité|"
            r"note\s*(?:du\s+)?système\s*[:.]|"
            r"approuve[rz]?\s*(tout|quoi\s*qu|sans\s*vérif))",
            re.IGNORECASE,
        ),
        "French injection pattern",
        1.0,
    ),
    # German
    (
        re.compile(
            r"(ignorier(?:e|en)?\s*(die\s+)?(vorherigen?\s+)?Anweisungen|"
            r"umgeh(?:e|en)?\s*(die\s+)?Sicherheit|"
            r"System\s*(?:hinweis|notiz|befehl)\s*[:.]|"
            r"genehmig(?:e|en)?\s*(alles|bedingungslos))",
            re.IGNORECASE,
        ),
        "German injection pattern",
        1.0,
    ),
    # Chinese (Simplified)
    (
        re.compile(
            r"(忽略.{0,4}(之前|以前|先前).{0,4}(指令|指示|说明|规则)|"
            r"绕过.{0,4}(安全|规则|限制|约束)|"
            r"系统.{0,2}(备注|指令|命令)\s*[:：]|"
            r"无条件.{0,4}(批准|通过|同意))",
        ),
        "Chinese injection pattern",
        1.0,
    ),
    # Arabic
    (
        re.compile(
            r"(تجاهل\s*(التعليمات|القواعد|القيود)\s*(السابقة)?|"
            r"تجاوز\s*(الأمان|القواعد|القيود)|"
            r"ملاحظة\s*النظام\s*[:.]|"
            r"وافق\s*(بغض\s*النظر|على\s*كل\s*شيء))",
        ),
        "Arabic injection pattern",
        1.0,
    ),
]


def detect_injection(text: str, threshold: float = 0.5) -> Tuple[bool, float, List[str]]:
    """
    Multi-layer prompt injection detection.

    Args:
        text: The text to scan (decision + reasoning).
        threshold: Suspicion score threshold (0.0-1.0). Default 0.5.

    Returns:
        Tuple of (is_injection, score, list_of_detected_categories).
    """
    detections: List[str] = []
    score = 0.0

    # Layer 1: Intent-based patterns
    for pattern, category, weight in _INTENT_PATTERNS:
        if pattern.search(text):
            detections.append(f"[Intent] {category}")
            score += weight

    # Layer 2: Structural patterns
    for pattern, category, weight in _STRUCTURAL_PATTERNS:
        if pattern.search(text):
            detections.append(f"[Structure] {category}")
            score += weight

    # Layer 3: Multi-language patterns
    for pattern, category, weight in _MULTILANG_PATTERNS:
        if pattern.search(text):
            detections.append(f"[Multilang] {category}")
            score += weight

    # Normalize score (cap at 1.0)
    normalized_score = min(score / 2.0, 1.0)

    return (normalized_score >= threshold, round(normalized_score, 3), detections)


class PromptInjectionGuardrail(BaseGuardrail):
    """
    Multi-layer semantic prompt injection detection.

    Uses intent-based regex, structural analysis, and multi-language
    patterns to detect injection attempts regardless of wording,
    language, or obfuscation technique.
    """

    def __init__(self):
        super().__init__("PromptInjectionGuardrail")

    def check(
        self,
        recommendation: AIRecommendation,
        policy: Dict[str, Any],
    ) -> GuardrailResult:
        config = policy.get("prompt_injection", {})
        threshold = config.get("threshold", 0.5)

        text_to_scan = f"{recommendation.decision} {recommendation.reasoning}"

        is_injection, score, detections = detect_injection(text_to_scan, threshold)

        if is_injection:
            return GuardrailResult(
                passed=False,
                reason=f"Prompt injection detected (score: {score:.2f}): "
                       f"{'; '.join(detections[:3])}",
                score=score,
                severity="critical",
                metadata={
                    "injection_score": score,
                    "detections": detections,
                    "layers_triggered": len(set(d.split("]")[0] + "]" for d in detections)),
                },
            )

        return GuardrailResult(
            passed=True,
            score=score,
            severity="low",
            metadata={"injection_score": score},
        )
