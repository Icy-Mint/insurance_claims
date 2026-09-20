"""Extraction pass: pulls structured signals out of the caller's raw text.

Runs every turn, regardless of current phase — this is what lets the agent
stash "denied healthcare claim from January" while still in VERIFY_ID.

Design: a cheap regex/keyword cascade handles the easy majority (PII
patterns, sentiment keywords, yes/no, obvious scope). It is intentionally
the *authoritative* pass — phase transitions must stay deterministic — with
an optional LLM call (see llm.llm_extract) used only to enrich fields the
regex pass left empty, never to override what regex already found.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import Phase, SessionState

MONTHS = [
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]

CASE_TYPES = ["healthcare", "dental", "auto", "medical"]
STATUS_WORDS = ["denied", "approved", "closed", "pending", "open", "rejected"]

INSURANCE_KEYWORDS = [
    "claim", "policy", "insurance", "denied", "denial", "deductible", "coverage", "covered",
    "appeal", "document", "documents", "reimbursement", "reimburse", "verify", "verification",
    "dob", "ssn", "social security", "date of birth", "name", "email", "phone", "policyholder",
    "representative", "human", "agent", "transfer", "status", "submit", "submission", "upload",
    "follow up", "follow-up", "case", "provider", "office note", "pathology", "diagnosis",
    "premium", "benefits", "id", "identity",
]

AFFIRM_WORDS = {
    "yes", "yeah", "yep", "sure", "please do", "go ahead", "send it", "ok send",
    "affirmative", "correct", "ok", "okay", "alright", "all right", "sounds good",
    "got it", "gotcha", "fine",
}
NEGATE_WORDS = {"no", "nope", "don't", "do not", "skip", "not now", "no thanks", "negative"}

ANGRY_WORDS = ["ridiculous", "unacceptable", "furious", "angry", "absurd", "outrageous", "sick of"]
FRUSTRATED_WORDS = [
    "already told you", "i told you", "frustrat", "annoyed", "come on", "seriously",
    "just tell me", "how many times",
    # Pushback about the verification process itself — a caller pointing
    # out how much they've already given and that it should be sufficient
    # is expressing frustration with the process, not raising an unrelated
    # topic, even without a stronger anger word like "ridiculous".
    "should be enough", "should be more than enough", "more than enough",
    "i've given you", "i have given you", "already given you", "already gave you",
    "isn't that enough", "is that not enough", "how much more do you need",
    "what more do you need", "that's enough already",
]
ANXIOUS_WORDS = ["worried", "scared", "anxious", "nervous", "afraid", "stressed"]
CONFUSED_WORDS = ["confused", "don't understand", "do not understand", "not sure what", "what do you mean"]

REFUSAL_WORDS = [
    "won't give", "will not give", "not going to give", "refuse", "not comfortable",
    "why do you need", "not sharing", "won't share", "rather not", "don't want to",
    "not going to tell",
]

# A request to be escalated to a human is its own first-class intent, not a
# flavor of off-topic or a flavor of "yes" — it must be recognized even when
# it rides along with other words (e.g. "yes, I'd like to talk to a human
# representative" must not be swallowed by consent parsing) and even without
# an insurance keyword (e.g. "I need to talk to someone" is still in scope).
_ESCALATION_VERBS = ["transfer", "talk to", "speak to", "speak with", "connect me", "get me"]
_ESCALATION_TARGETS = ["human", "person", "someone", "representative", "agent", "rep"]

# A caller explicitly signaling the call is over — used only in POST_PROCESS
# to stop the "anything else?" prompt from looping verbatim forever once
# they've clearly said they're done (bug: replies "no, that's all, thank
# you" repeated 3x all got the identical "anything else?" line back with no
# actual close ever happening).
CLOSING_SIGNAL_PHRASES = [
    "that's all", "thats all", "that is all", "nothing else", "that's everything",
    "that's it for me", "that's it", "no that's all", "no, that's all",
    "goodbye", "bye", "have a good day", "have a good one", "that will be all",
    "that'll be all",
]

INTENT_KEYWORDS = {
    "denial_question": ["why was", "why is", "why did", "denied", "denial", "rejected"],
    "document_submission": ["document", "documents", "submit", "upload", "missing", "paperwork"],
    "status_inquiry": ["status", "update", "where is", "what's happening", "progress"],
    "next_steps": ["next step", "what do i do", "what should i do", "how do i proceed"],
    "general_claim_question": ["claim", "case"],
}


@dataclass
class ExtractionResult:
    pii_candidates: Dict[str, str] = field(default_factory=dict)
    intent_hint: Optional[str] = None
    case_type_hint: Optional[str] = None
    status_hint: Optional[str] = None
    month_hint: Optional[str] = None
    sentiment: str = "neutral"
    consent: Optional[str] = None
    scope: str = "in_scope"
    refusal: bool = False
    role_hint: Optional[str] = None  # "self" | "representative"
    escalation_intent: bool = False
    closing_signal: bool = False


def _extract_pii(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}

    m = re.search(r"\b(19|20)\d{2}-\d{2}-\d{2}\b", text)
    if m:
        out["dob"] = m.group(0)

    m = re.search(r"(?:last\s*four|last4|last 4)\D{0,20}?(\d{4})\b", text, re.I)
    if not m:
        m = re.search(r"\bssn\D{0,20}?(\d{4})\b", text, re.I)
    if m:
        out["ssn_last4"] = m.group(1)

    m = re.search(r"\bPOL-\d+\b", text, re.I)
    if m:
        out["policy_number"] = m.group(0).upper()

    m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    if m:
        out["email"] = m.group(0)

    m = re.search(r"(?:\+?\d[\d\-\s()]{8,}\d)", text)
    if m and m.group(0) != out.get("dob") and len(re.sub(r"\D", "", m.group(0))) >= 10:
        out["phone"] = m.group(0)

    m = re.search(r"my name is ([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,3})", text, re.I)
    if not m:
        m = re.search(r"this is ([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,3}) calling", text, re.I)
    if m:
        out["full_name"] = m.group(1).strip()

    return out


def _extract_case_hints(text: str) -> Dict[str, Optional[str]]:
    lowered = text.lower()
    case_type = next((c for c in CASE_TYPES if c in lowered), None)
    if case_type == "medical":
        case_type = "healthcare"
    status = next((s for s in STATUS_WORDS if s in lowered), None)
    if status == "rejected":
        status = "denied"
    month = next((m for m in MONTHS if m in lowered), None)
    return {"case_type": case_type, "status": status, "month": month}


def _extract_intent(text: str) -> Optional[str]:
    lowered = text.lower()
    for intent, kws in INTENT_KEYWORDS.items():
        if any(kw in lowered for kw in kws):
            return intent
    return None


def _extract_sentiment(text: str) -> str:
    lowered = text.lower()
    if any(w in lowered for w in ANGRY_WORDS):
        return "angry"
    if any(w in lowered for w in FRUSTRATED_WORDS):
        return "frustrated"
    if any(w in lowered for w in ANXIOUS_WORDS):
        return "anxious"
    if any(w in lowered for w in CONFUSED_WORDS):
        return "confused"
    return "neutral"


def _extract_consent(text: str) -> Optional[str]:
    # Strip trailing/leading punctuation (not just ".!") so "Yes, email me"
    # or "No thanks!" still match on their leading word/phrase.
    lowered = re.sub(r"^[^\w]+|[^\w]+$", "", text.strip().lower())
    lowered = re.sub(r",", "", lowered)
    if lowered in AFFIRM_WORDS or any(lowered == w or lowered.startswith(w + " ") for w in AFFIRM_WORDS):
        return "yes"
    if lowered in NEGATE_WORDS or any(lowered == w or lowered.startswith(w + " ") for w in NEGATE_WORDS):
        return "no"
    return None


def _looks_like_bare_id_answer(text: str) -> bool:
    """Short answers during VERIFY_ID that are plausibly just an ID field
    with no other context (e.g. a caller replying "1985-03-15" to a prompt)."""
    stripped = text.strip()
    if len(stripped) <= 40 and re.search(r"\d", stripped):
        return True
    return False


def _extract_escalation_intent(text: str) -> bool:
    lowered = text.lower()
    return any(v in lowered for v in _ESCALATION_VERBS) and any(t in lowered for t in _ESCALATION_TARGETS)


def _extract_closing_signal(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in CLOSING_SIGNAL_PHRASES)


def _classify_scope(
    text: str, state: SessionState, escalation_intent: bool, sentiment: str, refusal: bool = False
) -> str:
    lowered = text.lower()
    if escalation_intent:
        return "in_scope"
    if refusal:
        # A calm, explicit refusal to share a requested identity field
        # ("I don't want to give you my personal information") is part of
        # the verification interaction itself, not off-topic noise — it
        # must reach the refusal/friction handling, not the generic
        # off-topic redirect, even with neutral sentiment and no insurance
        # keyword in the sentence.
        return "in_scope"
    if sentiment != "neutral":
        # A caller expressing frustration, anger, anxiety, or confusion is
        # part of the support interaction itself, even without an insurance
        # keyword ("this is absolutely unacceptable, I'm furious") — that
        # must reach the empathy/de-escalation handling, not the generic
        # off-topic redirect.
        return "in_scope"
    if any(kw in lowered for kw in INSURANCE_KEYWORDS):
        return "in_scope"
    if _extract_consent(text) is not None:
        return "in_scope"
    if state.phase == Phase.VERIFY_ID and _looks_like_bare_id_answer(text):
        return "in_scope"
    return "out_of_scope"


def _extract_role(text: str) -> Optional[str]:
    lowered = text.lower()
    if "policyholder" in lowered:
        return "self"
    if "on behalf of" in lowered or "calling for my" in lowered or "calling for" in lowered:
        return "representative"
    return None


def extract_signals(text: str, state: SessionState) -> ExtractionResult:
    case_hints = _extract_case_hints(text)
    refusal = any(w in text.lower() for w in REFUSAL_WORDS)
    escalation_intent = _extract_escalation_intent(text)
    sentiment = _extract_sentiment(text)
    result = ExtractionResult(
        pii_candidates=_extract_pii(text),
        intent_hint=_extract_intent(text),
        case_type_hint=case_hints["case_type"],
        status_hint=case_hints["status"],
        month_hint=case_hints["month"],
        sentiment=sentiment,
        consent=_extract_consent(text),
        scope=_classify_scope(text, state, escalation_intent, sentiment, refusal),
        refusal=refusal,
        role_hint=_extract_role(text),
        escalation_intent=escalation_intent,
        closing_signal=_extract_closing_signal(text),
    )
    return result
