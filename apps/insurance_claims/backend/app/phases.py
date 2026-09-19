"""Deterministic phase-transition function and per-turn orchestration.

`next_phase` is plain Python — it never asks the LLM what phase to be in.
The LLM only ever reports extracted data (extraction pass) or produces
caller-facing phrasing (generation pass); this module is what turns that
data into a phase decision.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import prompts, templates, tools
from .extraction import ExtractionResult, extract_signals
from .llm import generate_reply, llm_extract_enrich
from .models import (
    COUNTABLE_VERIFICATION_FIELDS,
    FRICTION_ESCALATION_THRESHOLD,
    FRICTION_HARD_ESCALATION_THRESHOLD,
    OFF_TOPIC_ESCALATION_THRESHOLD,
    OFF_TOPIC_HARD_ESCALATION_THRESHOLD,
    Phase,
    SessionState,
    ToolCallRecord,
    VERIFICATION_THRESHOLD,
)

ID_FIELD_LABELS = {
    "full_name": "Full name",
    "dob": "Date of birth",
    "phone": "Phone number",
    "email": "Email",
    "ssn_last4": "Last 4 of SSN",
}


def next_phase(state: SessionState) -> Phase:
    """Pure deterministic phase-transition function."""
    if state.phase == Phase.VERIFY_ID:
        if len(state.verified_fields) >= VERIFICATION_THRESHOLD:
            return Phase.RESOLVE_INTENT
        return Phase.VERIFY_ID

    if state.phase == Phase.RESOLVE_INTENT:
        if state.memory.get("resolved_case_id") and state.memory.get("resolved_intent"):
            return Phase.PROCESS_CASE
        return Phase.RESOLVE_INTENT

    if state.phase == Phase.PROCESS_CASE:
        if state.memory.get("case_query_resolved"):
            return Phase.POST_PROCESS
        return Phase.PROCESS_CASE

    return Phase.POST_PROCESS


def _record(tool: str, args: Dict[str, Any], result: Any) -> ToolCallRecord:
    return ToolCallRecord(tool=tool, args=args, result=result)


def _apply_extraction(state: SessionState, extraction: ExtractionResult, text: str) -> None:
    state.sentiment_history.append(extraction.sentiment)

    for k, v in extraction.pii_candidates.items():
        if k in state.verified_fields:
            # This field already matched a real record on a previous turn —
            # verification of a field is a one-time gate, never re-litigated.
            # Accepting a new value here (even a well-formatted one) would
            # let a later turn silently overwrite a confirmed match and then
            # have it "fail" against the new value, producing a false
            # un-verification the caller never actually triggered.
            continue
        state.pii_candidates[k] = v

    if extraction.case_type_hint:
        state.memory["case_type_hint"] = extraction.case_type_hint
    if extraction.status_hint:
        state.memory["status_hint"] = extraction.status_hint
    if extraction.month_hint:
        state.memory["month_hint"] = extraction.month_hint
    if (extraction.case_type_hint or extraction.status_hint) and "case_hint_text" not in state.memory:
        state.memory["case_hint_text"] = text

    if extraction.intent_hint and "intent_hint" not in state.memory:
        state.memory["intent_hint"] = extraction.intent_hint
    if extraction.role_hint and not state.caller_role:
        state.caller_role = extraction.role_hint


def _handle_verify_id(state: SessionState, extraction: ExtractionResult) -> Tuple[List[str], bool]:
    """Runs the real verify_identity() lookup against policyholders.json and
    logs input/output on every call. state.verified / state.verified_fields
    are set here from that function's return value alone — never from the
    LLM's judgment or from conversational consistency.

    Returns (rejected_fields, no_identity_match):
      * rejected_fields — fields the caller *just* submitted this turn that
        were checked against a *located* record and did not match. This is
        what lets the reply explicitly say "that doesn't match" instead of
        silently ignoring a wrong SSN/DOB.
      * no_identity_match — True when this turn submitted at least one
        countable identity field but verify_identity() matched it against
        NO policyholder at all (not "not enough fields yet" — a hard,
        total non-match). A fully fabricated identity falls in this bucket
        forever, no matter how many internally-consistent details are
        added, because nothing it contains will ever match a real record.
    """
    if state.verified:
        # Verification is a one-time gate: once fully verified, the identity
        # lookup must never run again for the rest of the session, no matter
        # what phase-transition edge case might otherwise call this.
        return [], False
    if not state.pii_candidates:
        return [], False
    result = tools.verify_identity(state.pii_candidates)
    state.tool_log.append(_record("verify_identity", dict(state.pii_candidates), result))
    state.add_verified_fields(result["matched_fields"])
    if result["party_id"] and not state.party_id:
        state.party_id = result["party_id"]
    if len(state.verified_fields) >= VERIFICATION_THRESHOLD:
        state.verified = True

    this_turn_countable = [f for f in extraction.pii_candidates if f in COUNTABLE_VERIFICATION_FIELDS]

    if not result["party_id"]:
        return [], bool(this_turn_countable)

    rejected = [f for f in this_turn_countable if f not in result["matched_fields"]]
    return rejected, False


def _handle_resolve_intent(state: SessionState) -> None:
    if state.memory.get("resolved_case_id") or not state.party_id:
        return

    claims = tools.get_claims_for_party(state.party_id)
    state.tool_log.append(_record("get_claims_for_party", {"party_id": state.party_id}, claims))

    candidates = claims
    case_type = state.memory.get("case_type_hint")
    status = state.memory.get("status_hint")
    if case_type:
        candidates = [c for c in candidates if c["case_type"] == case_type]
    if status:
        candidates = [c for c in candidates if c["status"] == status]

    if len(candidates) == 1:
        state.memory["resolved_case_id"] = candidates[0]["case_id"]
        state.memory["resolved_intent"] = state.memory.get("intent_hint") or "general_claim_question"
    elif len(candidates) > 1:
        state.memory["ambiguous_candidates"] = [c["case_id"] for c in candidates]


def _next_terminal_followup_reply(
    variants: List[str],
    last_input: Optional[str],
    count: int,
    user_message: str,
) -> Tuple[str, bool]:
    """Shared cycling logic for both post-escalation and post-closing
    follow-ups: a repeat of the same input gets the same reply back
    (state didn't change, so nothing new needs saying); a genuinely
    different input advances to the next canned variant. Never a live
    model call, so it can never fabricate progress or contradict itself.
    Returns (reply, advanced) where advanced tells the caller whether to
    bump its stored count."""
    if user_message == last_input and count > 0:
        return variants[(count - 1) % len(variants)], False
    return variants[count % len(variants)], True


def process_turn(state: SessionState, user_message: str) -> Tuple[str, Dict[str, Any]]:
    # A session that has already escalated is done: no further extraction,
    # no further LLM calls, no further SOP progression. The FIRST turn that
    # triggers escalation shows ESCALATION_TERMINAL_MESSAGE (below); every
    # turn after that lands here and gets a short, distinct closing
    # acknowledgment instead — never the original "please hold" sentence
    # repeated verbatim, and never a fresh live-generated "in progress"
    # variant that could contradict itself. A repeated identical follow-up
    # gets the same acknowledgment back; a different one advances to the
    # next canned variant.
    if state.escalated:
        reply, advanced = _next_terminal_followup_reply(
            prompts.ESCALATION_FOLLOWUP_VARIANTS,
            state.last_escalation_input,
            state.escalation_followup_count,
            user_message,
        )
        if advanced:
            state.escalation_followup_count += 1
        state.last_escalation_input = user_message
        state.last_escalation_reply = reply
        state.history.append({"role": "user", "content": user_message})
        state.history.append({"role": "assistant", "content": reply})
        return reply, {**state.as_public_dict(), "ui_hints": {}}

    # Same idea for a session the caller has already explicitly closed out
    # (see the POST_PROCESS block below for where this first gets set).
    if state.session_closed:
        reply, advanced = _next_terminal_followup_reply(
            prompts.SESSION_CLOSED_FOLLOWUP_VARIANTS,
            state.last_closing_input,
            state.closing_followup_count,
            user_message,
        )
        if advanced:
            state.closing_followup_count += 1
        state.last_closing_input = user_message
        state.last_closing_reply = reply
        state.history.append({"role": "user", "content": user_message})
        state.history.append({"role": "assistant", "content": reply})
        return reply, {**state.as_public_dict(), "ui_hints": {}}

    extraction = extract_signals(user_message, state)
    llm_extract_enrich(user_message, state, extraction)
    _apply_extraction(state, extraction, user_message)

    ui_hints: Dict[str, Any] = {}

    # --- escalation intent: highest priority, before scope/consent/anything
    # else. This is what lets "yes, I'd like to talk to a human
    # representative" escalate instead of being swallowed as email consent,
    # and "I need to talk to someone" escalate instead of being refused as
    # off-topic. Fires exactly once — state.escalated then short-circuits
    # every subsequent turn above.
    if extraction.escalation_intent:
        state.escalated = True
        state.history.append({"role": "user", "content": user_message})
        state.history.append({"role": "assistant", "content": prompts.ESCALATION_TERMINAL_MESSAGE})
        return prompts.ESCALATION_TERMINAL_MESSAGE, {**state.as_public_dict(), "ui_hints": {}}

    # --- scope gate ----------------------------------------------------
    if extraction.scope == "out_of_scope":
        state.off_topic_count += 1
        escalate_offer = state.off_topic_count >= OFF_TOPIC_ESCALATION_THRESHOLD
        if escalate_offer:
            state.escalation_offered = True
            ui_hints["quick_replies"] = ["Transfer me to a human", "Continue with my claim"]

        # Repeated off-topic turns after the transfer has already been
        # offered: stop re-offering forever (which reads as the same stall
        # repeating) and escalate for real, via the same state.escalated
        # mechanism the explicit escalation intent uses.
        if state.off_topic_count >= OFF_TOPIC_HARD_ESCALATION_THRESHOLD:
            state.escalated = True
            state.history.append({"role": "user", "content": user_message})
            state.history.append({"role": "assistant", "content": prompts.ESCALATION_TERMINAL_MESSAGE})
            return prompts.ESCALATION_TERMINAL_MESSAGE, {**state.as_public_dict(), "ui_hints": {}}

        reply = prompts.out_of_scope_reply(escalate_offer)
        state.history.append({"role": "user", "content": user_message})
        state.history.append({"role": "assistant", "content": reply})
        return reply, {**state.as_public_dict(), "ui_hints": ui_hints}

    state.off_topic_count = 0

    # --- verification friction tracking --------------------------------
    if state.phase == Phase.VERIFY_ID and not state.verified:
        if extraction.refusal or extraction.sentiment in ("frustrated", "angry"):
            state.verification_friction_count += 1
        if state.verification_friction_count >= FRICTION_ESCALATION_THRESHOLD:
            state.escalation_offered = True
            ui_hints["quick_replies"] = ["Try a different ID field", "Transfer me to a human"]

        # A transfer has already been offered and the caller still hasn't
        # cooperated or explicitly asked for it after further attempts —
        # stop persuading and escalate rather than offering indefinitely.
        if state.verification_friction_count >= FRICTION_HARD_ESCALATION_THRESHOLD:
            state.escalated = True
            state.history.append({"role": "user", "content": user_message})
            state.history.append({"role": "assistant", "content": prompts.ESCALATION_TERMINAL_MESSAGE})
            return prompts.ESCALATION_TERMINAL_MESSAGE, {**state.as_public_dict(), "ui_hints": {}}

    # --- deterministic cascade through data-only phases -----------------
    rejected_fields: List[str] = []
    no_identity_match = False
    for _ in range(4):
        if state.phase == Phase.VERIFY_ID:
            rejected_fields, no_identity_match = _handle_verify_id(state, extraction)
        elif state.phase == Phase.RESOLVE_INTENT:
            _handle_resolve_intent(state)
        else:
            break
        new_phase = next_phase(state)
        if new_phase == state.phase:
            break
        state.phase = new_phase

    if state.phase == Phase.VERIFY_ID and "quick_replies" not in ui_hints:
        missing = [ID_FIELD_LABELS[f] for f in ID_FIELD_LABELS if f not in state.verified_fields]
        ui_hints["quick_replies"] = missing[:4]

    # --- response generation for the phase we landed on ------------------
    # POST_PROCESS is handled entirely deterministically below — the email
    # send is a real, consequential action, so it (and any claim that it
    # happened) must never come from a free-text model call. Skip the LLM
    # call for that phase rather than generating text we'd only discard.
    just_entered_post_process = False
    if state.phase == Phase.POST_PROCESS:
        reply, tool_calls = "", []
    else:
        reply, tool_calls = generate_reply(
            state,
            state.phase,
            user_message,
            rejected_fields if state.phase == Phase.VERIFY_ID else None,
            no_identity_match if state.phase == Phase.VERIFY_ID else False,
            extraction.refusal if state.phase == Phase.VERIFY_ID else False,
        )

    if state.phase == Phase.PROCESS_CASE and tool_calls:
        state.tool_log.extend(_record(t["tool"], {}, t["result"]) for t in tool_calls)
        if any(t["tool"] == "get_claim_detail" for t in tool_calls):
            state.memory["case_query_resolved"] = True
            new_phase = next_phase(state)
            if new_phase != state.phase:
                state.phase = new_phase
                just_entered_post_process = True

    # --- POST_PROCESS: deterministic consent gate --------------------------
    # `awaiting_email_consent` is set only by this code, exactly when *this*
    # code poses the exact yes/no question below — so on the next turn we
    # can reliably tell whether the caller's "yes" is actually answering the
    # email offer, versus an unrelated affirmative reply elsewhere in the
    # conversation. A "yes" is only ever treated as consent when this flag is
    # set; otherwise it's ignored for consent purposes (and nothing is sent).
    if state.phase == Phase.POST_PROCESS:
        awaiting = state.memory.get("awaiting_email_consent", False)

        if awaiting and extraction.consent == "yes" and not state.email_sent:
            summary = _build_summary(state)
            sent = tools.send_email_summary(state.session_id, summary)
            state.tool_log.append(_record("send_email_summary", {}, sent))
            state.consent["email"] = "yes"
            state.email_sent = True
            state.memory["awaiting_email_consent"] = False
            post_process_reply = templates.post_process_sent_reply(summary)
        elif awaiting and extraction.consent == "no":
            state.consent["email"] = "no"
            state.memory["awaiting_email_consent"] = False
            post_process_reply = templates.post_process_declined_reply()
        elif state.email_sent or state.consent.get("email") == "no":
            if extraction.closing_signal:
                # A clear signal the caller considers the call over — give a
                # genuine one-time sign-off and mark the session closed so
                # further messages don't loop the same "anything else?"
                # prompt (or re-trigger this branch) indefinitely.
                state.session_closed = True
                post_process_reply = prompts.SESSION_CLOSING_SIGNOFF
            else:
                post_process_reply = "Understood — is there anything else I can help with regarding your claim?"
        else:
            # Either we're posing the question for the first time, or the
            # caller's reply didn't clearly answer the pending yes/no —
            # re-ask the exact same deterministic question rather than
            # letting an unrelated "yes" drift into being treated as consent.
            post_process_reply = templates.post_process_offer_reply()
            state.memory["awaiting_email_consent"] = True
            ui_hints["quick_replies"] = ["Yes, email me", "No thanks"]

        reply = f"{reply} {post_process_reply}".strip() if just_entered_post_process else post_process_reply

    state.history.append({"role": "user", "content": user_message})
    state.history.append({"role": "assistant", "content": reply})

    return reply, {**state.as_public_dict(), "ui_hints": ui_hints}


def _build_summary(state: SessionState) -> str:
    case_id = state.memory.get("resolved_case_id")
    claim = tools.get_claim_detail(case_id) if case_id else None
    if not claim:
        return "Summary of your call today."
    parts = [f"Claim {claim['case_id']} ({claim['case_type']}) is currently {claim['status']}."]
    if claim["status"] == "denied":
        parts.append(f"Denial reason: {claim['denial_reason']}.")
        docs = claim.get("documents_needed", [])
        if docs:
            parts.append(f"Documents still needed: {', '.join(docs)}.")
        if claim.get("appeal_deadline"):
            parts.append(f"Appeal deadline: {claim['appeal_deadline']}.")
    return " ".join(parts)
