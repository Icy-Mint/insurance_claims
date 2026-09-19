"""Deterministic reply templates.

Used whenever no ANTHROPIC_API_KEY is configured (so the demo and automated
tests work fully offline), and as the safe fallback if a live LLM call
fails. They enforce the same grounding rule as the real prompts: PROCESS_CASE
text is built only from the tool_results dict, never invented.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .models import SessionState

MISSING_FIELD_PROMPTS = {
    "full_name": "your full name",
    "dob": "your date of birth",
    "phone": "the phone number on file",
    "email": "the email on file",
    "ssn_last4": "the last 4 digits of your SSN or national ID",
}

FIELD_LABELS = {
    "full_name": "full name",
    "dob": "date of birth",
    "phone": "phone number",
    "email": "email on file",
    "ssn_last4": "last 4 digits of your SSN/ID",
}


def _mismatch_prefix(rejected_fields: Optional[list]) -> str:
    """A field was actually looked up against the fixture record and did not
    match — this must always be stated as a plain fact, never glossed over
    as "let me note that down" or "I can't check that right now"."""
    if not rejected_fields:
        return ""
    labels = " and ".join(FIELD_LABELS.get(f, f) for f in rejected_fields)
    return f"That {labels} you just gave doesn't match what we have on file. "


NO_MATCH_PREFIX = (
    "I wasn't able to match the details you gave to any account on file — that's a real check "
    "that just ran, not something still pending. "
)


def verify_id_reply(
    state: SessionState,
    sentiment: str,
    rejected_fields: Optional[list] = None,
    no_identity_match: bool = False,
    refusal: bool = False,
) -> str:
    remaining = [v for k, v in MISSING_FIELD_PROMPTS.items() if k not in state.verified_fields]
    ask = f"Could you double-check and provide {remaining[0]}, plus one more identifying detail such as {remaining[1] if len(remaining) > 1 else 'your policy number'}?"
    # A total non-match (nothing the caller gave locates ANY real record) is
    # reported distinctly from a partial mismatch (some fields matched a
    # located record, one specific field didn't) — both must be stated
    # plainly, never silently treated as "still collecting info".
    prefix = NO_MATCH_PREFIX if no_identity_match else _mismatch_prefix(rejected_fields)

    if sentiment in ("frustrated", "angry"):
        return (
            prefix
            + "I hear you, and I'm sorry for the extra steps this is taking. "
            "I do need to verify your identity first — it's what keeps your claim details safe from anyone else calling in. "
            "You're welcome to give me a different identifying detail than before, or I can transfer you to a human representative if you'd prefer. "
            f"When you're ready, {ask}"
        )
    if sentiment == "anxious":
        return f"{prefix}No worries, we'll get this sorted out together. To pull up your account safely, {ask}"
    if refusal:
        alt = remaining[0] if remaining else "a different identifying detail"
        return (
            f"{prefix}That's okay, I won't press you on that one. I do still need to confirm a couple of "
            f"identifying details to keep your account protected, though — would you be comfortable "
            f"providing {alt} instead, or would you rather I transfer you to a human representative?"
        )
    if prefix:
        return prefix + ask
    if not state.verified_fields:
        return (
            "I can help with that. First I need to verify your identity — "
            f"{ask}"
        )
    return f"Thanks, that's helpful. {ask}"


def resolve_intent_reply(state: SessionState) -> str:
    return (
        "You're verified — thanks. Could you tell me a bit more about what you need help with "
        "(for example, a denied claim, submitting documents, or a status update), and which claim it concerns?"
    )


def process_case_reply(tool_results: Dict[str, Any], case_id: str) -> str:
    claim = tool_results.get("claim")
    if not claim:
        return f"I couldn't find a claim on file with ID {case_id}. Could you double-check the claim number?"

    lines = [f"I found claim {claim['case_id']}, a {claim['case_type']} claim currently marked as {claim['status']}."]

    if claim["status"] == "denied":
        lines.append(f"It was denied because {claim['denial_reason']}.")
        docs = tool_results.get("required_documents", {}).get("documents", [])
        if docs:
            doc_names = ", ".join(d["name"] for d in docs)
            lines.append(f"To move forward, we still need: {doc_names}.")
        if claim.get("appeal_deadline"):
            lines.append(f"The appeal deadline is {claim['appeal_deadline']}.")
    elif claim["status"] == "closed":
        lines.append(f"It settled with a net payment of ${claim['net_pay']}.")
    elif claim["status"] == "open":
        lines.append("It's still in progress.")

    follow_up = tool_results.get("follow_up_text")
    if follow_up:
        lines.append(follow_up)

    return " ".join(lines)


def post_process_offer_reply(prior_case_text: Optional[str] = None) -> str:
    prefix = f"{prior_case_text} " if prior_case_text else ""
    return prefix + "Would you like me to email you a summary of what we discussed today, including the claim status and next steps? (yes/no)"


def post_process_sent_reply(summary: str) -> str:
    return f"Done — I've sent an email summary: \"{summary}\". Is there anything else I can help with?"


def post_process_declined_reply() -> str:
    return "No problem, I won't send anything. Is there anything else I can help with?"
