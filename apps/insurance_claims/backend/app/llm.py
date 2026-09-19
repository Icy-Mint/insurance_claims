"""Anthropic API integration.

Two roles, matching the two-pass-per-turn design:
  * llm_extract_enrich  -> optional enrichment of the regex extraction pass
  * generate_reply       -> phase-specific response generation pass

Capability gating lives in build_tool_defs(): claim-lookup tool *definitions*
are only ever placed in the request sent to the model once state.verified is
True. Before that, the model has no schema for get_claim_detail /
get_required_documents at all — it cannot call what it was never told
exists. execute_tool() additionally hard-refuses those calls if verified is
somehow False, as defense in depth.

If ANTHROPIC_API_KEY is not set, or any API call fails, everything here
falls back to the deterministic templates module so the app (and the test
suite) keeps working offline.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from . import templates, tools
from .extraction import ExtractionResult
from .models import Phase, SessionState
from . import prompts

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
MAX_TOOL_ROUNDS = 3
# Extended thinking isn't needed for these short phrasing/extraction calls,
# and leaving it on the model's default risks the reasoning tokens consuming
# the entire max_tokens budget before any visible text is produced — which
# silently returns an empty reply to the caller. Disable it outright.
THINKING_DISABLED = {"type": "disabled"}

_client = None
_client_checked = False


def get_client():
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _client = None
        return None
    try:
        import anthropic

        _client = anthropic.Anthropic(api_key=api_key)
    except Exception:
        _client = None
    return _client


class ToolNotAllowedError(PermissionError):
    pass


CLAIM_TOOL_DEFS = [
    {
        "name": "get_claim_detail",
        "description": (
            "Look up the full detail of the caller's already-identified claim: "
            "status, denial reason, appeal deadline, amounts."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_required_documents",
        "description": "Get guidance on which documents are still missing for the caller's claim and how to submit them.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def build_tool_defs(state: SessionState) -> List[Dict[str, Any]]:
    if not state.verified:
        return []
    return CLAIM_TOOL_DEFS


def execute_tool(name: str, state: SessionState) -> Any:
    if name in {"get_claim_detail", "get_required_documents"} and not state.verified:
        raise ToolNotAllowedError(f"{name} called before verification")
    case_id = state.memory.get("resolved_case_id")
    if name == "get_claim_detail":
        return tools.get_claim_detail(case_id)
    if name == "get_required_documents":
        claim = tools.get_claim_detail(case_id)
        if not claim:
            return {"documents": []}
        return tools.get_required_documents(claim["case_type"], claim.get("documents_needed", []))
    raise ValueError(f"unknown tool {name}")


def _extract_text(resp) -> str:
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return " ".join(parts).strip()


# Post-generation safety net: the phase gate controls what data/tools a live
# model call can access, but nothing stops free text from contradicting the
# gate it just passed through (e.g. asking for more ID while also disclosing
# facts). These marker lists catch that after the fact so exactly one
# coherent branch is ever returned, regardless of what the model wrote.
VERIFICATION_REQUEST_MARKERS = [
    "verify your identity", "need to verify", "need one more", "one more piece of identif",
    "additional identifying information", "confirm your identity", "before i can discuss",
    "before we can discuss", "still need", "date of birth, phone", "last 4 digits of your ssn",
    "phone number on your account", "email address on file",
]

CLAIM_DISCLOSURE_MARKERS = [
    "denial reason", "denied because", "appeal deadline", "claim id:",
    "expected reimbursement", "office note", "pathology report", "diagnosis report",
] + [c["case_id"] for c in tools.CLAIMS]

# RESOLVE_INTENT only ever runs once state.verified is True, but a model can
# still hedge and cast doubt on that fact in its own prose (e.g. "I don't
# have confirmation these details matched a record") — confusing for a
# caller who has, in fact, already been verified.
VERIFICATION_DOUBT_MARKERS = [
    "don't have confirmation", "do not have confirmation", "haven't come back",
    "has not come back", "verification system", "can't confirm", "cannot confirm",
    "not confirmed", "no tool results", "no account data", "genuinely can't",
    "i don't have a way to confirm", "i do not have a way to confirm",
    "nothing has come back",
]

# Bug 1 (identity verification): a live model call has been observed
# hedging about whether it can even check identity fields at all — "I don't
# have a live system connection... Assuming this matches what's on file" —
# instead of relying on the real verify_identity() result it was handed in
# VERIFICATION_CHECK_RESULT. This must never reach a caller in either
# VERIFY_ID (where it wrongly implies unverified info was accepted) or
# RESOLVE_INTENT (where verification has, in fact, already succeeded).
IDENTITY_HEDGE_MARKERS = [
    "live system connection", "live connection", "independently confirm",
    "independently verify", "assuming this matches", "assuming that matches",
    "i'll assume", "i will assume", "assume this matches", "assume that matches",
    "can't verify this myself", "cannot verify this myself",
    "can't check this myself", "cannot check this myself",
    "don't have the ability to", "do not have the ability to",
    "no way to verify", "no way to check", "unable to verify this myself",
    "unable to check this myself",
]

NO_MATCH_CONFIRMATION_MARKERS = [
    "doesn't match", "does not match", "don't match", "do not match",
    "didn't match", "did not match", "no match", "wasn't able to match",
    "was not able to match", "couldn't match", "could not match",
    "not on file", "no account on file", "no record",
]

# A model call during VERIFY_ID is only ever made when state.verified is
# guaranteed False (the deterministic cascade in phases.py already advanced
# past this phase if verification actually completed this turn) — so any
# claim of completed/confirmed verification here is necessarily false and
# will mislead the caller into thinking they're done when they aren't
# (observed live: "Thank you — that matches. I've now confirmed your
# identity" after only 2 of 3 required fields had ever matched).
FALSE_VERIFICATION_CONFIRMATION_MARKERS = [
    "confirmed your identity", "identity confirmed", "your identity has been confirmed",
    "your identity has been verified", "you're verified", "you have been verified",
    "you've been verified", "verification is complete", "verification complete",
    "fully verified", "i've now confirmed", "i have now confirmed", "you're all set",
    "verified your identity", "identity is verified", "identity has been verified",
]

# Exact-phrase lists are brittle against paraphrase (observed live, in the
# same session: "I now have enough to verify your identity... Let me pull
# up your account" — a different paraphrase of the same false claim that
# the literal marker list above didn't catch). These regex patterns catch
# the general *shape* of a premature-completion claim rather than one
# specific wording, as a second layer behind the literal marker list.
FALSE_VERIFICATION_CONFIRMATION_PATTERNS = [
    r"now have enough[^.]{0,40}verif",
    r"(i have|i've|that'?s|that is) enough[^.]{0,20}to verify",
    r"enough (information|details) to verify",
    r"can now (proceed|move forward|go ahead)[^.]{0,20}(your|with)",
    r"let me pull up your (account|claim)",
    r"pull up your (account|claim) now",
    r"i can (now )?(pull up|access|look into) your (account|claim)",
]


def _contains_any(text: str, markers: List[str]) -> bool:
    lowered = text.lower()
    return any(m.lower() in lowered for m in markers)


def _matches_any_pattern(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _build_messages(state: SessionState, user_message: str) -> List[Dict[str, Any]]:
    messages = list(state.history)
    messages.append({"role": "user", "content": user_message})
    return messages


EXTRACTION_TOOL_DEF = {
    "name": "record_extraction",
    "description": "Record any identity fields and sentiment found in the caller's message.",
    "input_schema": {
        "type": "object",
        "properties": {
            "full_name": {"type": "string"},
            "dob": {"type": "string", "description": "ISO format YYYY-MM-DD"},
            "phone": {"type": "string"},
            "email": {"type": "string"},
            "ssn_last4": {"type": "string"},
            "policy_number": {"type": "string"},
            "sentiment": {
                "type": "string",
                "enum": ["neutral", "frustrated", "anxious", "angry", "confused"],
            },
        },
    },
}


# tool_choice below forces the model to call record_extraction on every
# invocation, even for a message that contains no identity content at all
# (e.g. "I'm calling about my denied healthcare claim from January"). A
# forced tool call cannot simply decline to answer, and has been observed
# live filling in a placeholder value (e.g. full_name="<UNKNOWN>") rather
# than omitting the field — which then gets merged into the caller's
# accumulated candidate fields as if it were a real, freshly-submitted
# value, and can trigger a false "that doesn't match" rejection on a field
# that was never actually mentioned this turn. Two independent defenses:
# (1) the prompt explicitly forbids placeholders, (2) every value is
# format-validated before being trusted, regardless of what the prompt says.
_JUNK_VALUES = {
    "unknown", "n/a", "na", "none", "null", "not provided", "not given",
    "not sure", "not specified", "n a", "not applicable", "",
}


def _looks_like_valid_field_value(field: str, value: Any) -> bool:
    """Format-validates an LLM-extracted field before it's trusted — the
    single point of defense against a forced tool call fabricating a
    placeholder for a field that isn't actually present in the message."""
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not v or v.strip("<>").strip().lower() in _JUNK_VALUES:
        return False
    if field == "dob":
        return bool(re.match(r"^(19|20)\d{2}-\d{2}-\d{2}$", v))
    if field == "ssn_last4":
        return bool(re.match(r"^\d{4}$", v))
    if field == "phone":
        return len(re.sub(r"\D", "", v)) >= 10
    if field == "email":
        return bool(re.match(r"^[\w.+-]+@[\w-]+\.[\w.-]+$", v))
    if field == "policy_number":
        return bool(re.match(r"^POL-\d+$", v, re.I))
    if field == "full_name":
        return bool(re.match(r"^[A-Za-z][A-Za-z'\- ]{1,60}$", v))
    return True


def llm_extract_enrich(text: str, state: SessionState, result: ExtractionResult) -> None:
    """Fill gaps the regex pass left, only when it found nothing useful for
    an in-flight VERIFY_ID turn. Never overrides an already-found field, and
    never trusts a value that doesn't pass format validation for its field."""
    if state.phase != Phase.VERIFY_ID or result.pii_candidates:
        return
    client = get_client()
    if client is None:
        return
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=300,
            system=(
                "Extract identity fields and sentiment from the caller's message. Always respond "
                "via the record_extraction tool. If a field is not clearly and explicitly stated in "
                "the message, omit that field entirely from your tool call — do not guess, infer, "
                "or fill in a placeholder value such as 'unknown' or 'N/A'. It is normal and "
                "expected for most fields to be omitted; only include a field when the caller "
                "actually stated that specific piece of information in this message."
            ),
            messages=[{"role": "user", "content": text}],
            tools=[EXTRACTION_TOOL_DEF],
            tool_choice={"type": "tool", "name": "record_extraction"},
            thinking=THINKING_DISABLED,
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                data = block.input or {}
                for k in ("full_name", "dob", "phone", "email", "ssn_last4", "policy_number"):
                    val = data.get(k)
                    if val and _looks_like_valid_field_value(k, val):
                        result.pii_candidates[k] = val
                if data.get("sentiment") and result.sentiment == "neutral":
                    result.sentiment = data["sentiment"]
    except Exception:
        pass


def _fallback_reply(
    state: SessionState,
    phase: Phase,
    rejected_fields: Optional[List[str]] = None,
    no_identity_match: bool = False,
    refusal: bool = False,
) -> str:
    sentiment = state.sentiment_history[-1] if state.sentiment_history else "neutral"
    if phase == Phase.VERIFY_ID:
        return templates.verify_id_reply(state, sentiment, rejected_fields, no_identity_match, refusal)
    if phase == Phase.RESOLVE_INTENT:
        return templates.resolve_intent_reply(state)
    return "Let me look into that for you."


def _tool_results_from_calls(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    tool_results: Dict[str, Any] = {}
    for t in tool_calls:
        if t["tool"] == "get_claim_detail":
            tool_results["claim"] = t["result"]
        elif t["tool"] == "get_required_documents":
            tool_results["required_documents"] = t["result"]
    return tool_results


def _deterministic_process_case(state: SessionState) -> Tuple[str, List[Dict[str, Any]]]:
    case_id = state.memory.get("resolved_case_id")
    claim = tools.get_claim_detail(case_id)
    tool_calls = [{"tool": "get_claim_detail", "result": claim}]
    tool_results: Dict[str, Any] = {"claim": claim}
    if claim and claim["status"] == "denied":
        req_docs = tools.get_required_documents(claim["case_type"], claim.get("documents_needed", []))
        tool_results["required_documents"] = req_docs
        tool_calls.append({"tool": "get_required_documents", "result": req_docs})
    return templates.process_case_reply(tool_results, case_id), tool_calls


def _generate_process_case_reply(state: SessionState, user_message: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Wraps the raw LLM tool-loop with a hard grounding+consistency check:
    the loop already forces at least one tool call before accepting an
    answer, but that only guarantees the *facts* are real — it says nothing
    about whether the model's *prose* contradicts the fact that verification
    is already complete in this phase (e.g. mixing disclosure with "please
    verify" language). If it does, discard the prose and re-render a clean
    reply from the same already-fetched tool data instead of returning
    contradictory text."""
    text, tool_calls = _generate_process_case_reply_raw(state, user_message)
    if _contains_any(text, VERIFICATION_REQUEST_MARKERS):
        tool_results = _tool_results_from_calls(tool_calls)
        case_id = state.memory.get("resolved_case_id")
        if "claim" in tool_results:
            text = templates.process_case_reply(tool_results, case_id)
        else:
            text, tool_calls = _deterministic_process_case(state)
    return text, tool_calls


def _generate_process_case_reply_raw(state: SessionState, user_message: str) -> Tuple[str, List[Dict[str, Any]]]:
    client = get_client()
    if client is None:
        return _deterministic_process_case(state)

    tool_defs = build_tool_defs(state)
    messages = _build_messages(state, user_message)
    tool_calls_made: List[Dict[str, Any]] = []

    for _ in range(MAX_TOOL_ROUNDS):
        system = prompts.process_case_prompt(
            state,
            {"already_fetched": [t["tool"] for t in tool_calls_made]} if tool_calls_made else {},
        )
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=500,
                system=system,
                messages=messages,
                tools=tool_defs,
                thinking=THINKING_DISABLED,
            )
        except Exception:
            return _deterministic_process_case(state)

        if resp.stop_reason == "tool_use":
            tool_result_blocks = []
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use":
                    try:
                        result = execute_tool(block.name, state)
                    except ToolNotAllowedError:
                        result = {"error": "not authorized"}
                    tool_calls_made.append({"tool": block.name, "result": result})
                    tool_result_blocks.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result, default=str)}
                    )
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content": tool_result_blocks})
            continue

        text = _extract_text(resp)
        if not tool_calls_made or not text:
            # Refuse to accept an un-grounded or empty answer; force a retry.
            messages.append({"role": "assistant", "content": resp.content})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please call the available tool(s) to get the real claim facts before answering."}
                    ],
                }
            )
            continue
        return text, tool_calls_made

    return _deterministic_process_case(state)


MISMATCH_CONFIRMATION_MARKERS = [
    "doesn't match", "does not match", "don't match", "do not match",
    "didn't match", "did not match", "no match",
]


def generate_reply(
    state: SessionState,
    phase: Phase,
    user_message: str,
    rejected_fields: Optional[List[str]] = None,
    no_identity_match: bool = False,
    refusal: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    if phase == Phase.PROCESS_CASE:
        return _generate_process_case_reply(state, user_message)

    if phase == Phase.POST_PROCESS:
        # The email-consent gate is safety-critical (it decides whether a
        # real send action fires and what the agent claims about it), so it
        # is handled entirely deterministically in phases.py — never by a
        # live model call that could hallucinate a send that never happened.
        raise AssertionError("POST_PROCESS is handled deterministically in phases.py, not via generate_reply")

    def fallback() -> Tuple[str, List[Dict[str, Any]]]:
        return _fallback_reply(state, phase, rejected_fields, no_identity_match, refusal), []

    client = get_client()
    if client is None:
        return fallback()

    system = (
        prompts.verify_id_prompt(state, rejected_fields, no_identity_match, refusal)
        if phase == Phase.VERIFY_ID
        else prompts.resolve_intent_prompt(state)
    )

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=400,
            system=system,
            messages=_build_messages(state, user_message),
            thinking=THINKING_DISABLED,
        )
        text = _extract_text(resp)
    except Exception:
        return fallback()

    if not text:
        return fallback()

    if phase == Phase.VERIFY_ID and _contains_any(text, CLAIM_DISCLOSURE_MARKERS):
        # state.verified is guaranteed False here (VERIFY_ID's own gate), so
        # any claim-specific fact in the text is necessarily hallucinated —
        # there is no TOOL_RESULTS block this phase could have grounded it in.
        return fallback()

    if phase in (Phase.VERIFY_ID, Phase.RESOLVE_INTENT) and _contains_any(text, IDENTITY_HEDGE_MARKERS):
        # The model claimed it has no way to check identity, or told the
        # caller it's just "assuming" a match, instead of using the real
        # match_count/verified state it was handed. Never trust that prose —
        # the deterministic reply always reflects the actual verified state.
        return fallback()

    if phase == Phase.VERIFY_ID and (
        _contains_any(text, FALSE_VERIFICATION_CONFIRMATION_MARKERS)
        or _matches_any_pattern(text, FALSE_VERIFICATION_CONFIRMATION_PATTERNS)
    ):
        # generate_reply is only ever called for VERIFY_ID when
        # state.verified is guaranteed False — any claim that verification
        # is done (however it's phrased) is necessarily fabricated.
        return fallback()

    if (
        phase == Phase.VERIFY_ID
        and not rejected_fields
        and not no_identity_match
        and _contains_any(text, MISMATCH_CONFIRMATION_MARKERS + NO_MATCH_CONFIRMATION_MARKERS)
    ):
        # No field was actually checked-and-rejected this turn (no
        # rejected_fields, no total non-match) — so any "that doesn't
        # match" / "no match" language in the text is a fabricated claim,
        # not a real result (observed live: the model claimed an
        # already-matched full name "doesn't match" on a turn where no new
        # identity field was even submitted).
        return fallback()

    if phase == Phase.VERIFY_ID and rejected_fields and not _contains_any(text, MISMATCH_CONFIRMATION_MARKERS):
        # A field was actually checked against the fixture data and failed —
        # the model must say so plainly. If it hedged, glossed over it, or
        # implied acceptance instead, don't trust the prose; use the
        # deterministic reply that always states the mismatch outright.
        return fallback()

    if phase == Phase.VERIFY_ID and no_identity_match and not _contains_any(text, NO_MATCH_CONFIRMATION_MARKERS):
        # The submitted identity matched NO record at all (a total
        # non-match, e.g. a fabricated identity) — the model must state
        # that plainly rather than implying it was noted or accepted.
        return fallback()

    if phase == Phase.RESOLVE_INTENT and _contains_any(text, VERIFICATION_DOUBT_MARKERS):
        # state.verified is guaranteed True here — any hedging about whether
        # verification "really" succeeded is false and only confuses a
        # caller who has, in fact, already been verified.
        return fallback()

    if phase == Phase.RESOLVE_INTENT and _contains_any(
        text, MISMATCH_CONFIRMATION_MARKERS + NO_MATCH_CONFIRMATION_MARKERS + VERIFICATION_REQUEST_MARKERS
    ):
        # state.verified is guaranteed True and verify_identity() never runs
        # again once RESOLVE_INTENT is reached — so a "that doesn't match" /
        # "no match" / "please verify again" claim here is always
        # fabricated (observed live: the model claimed an already-matched
        # full name "doesn't match" and asked to re-verify, on a turn that
        # contained no identity content at all). Identity is a one-time,
        # permanent gate; nothing said afterward can legitimately unverify it.
        return fallback()

    return text, []
