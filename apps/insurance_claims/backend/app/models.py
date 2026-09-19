"""Session state model. This is the single source of truth the deterministic
phase-transition function reads and writes. Nothing about *which phase comes
next* lives anywhere else.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Phase(str, Enum):
    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"


# Fields on a policyholder record that count toward the identity-verification
# threshold. policy_number is a useful lookup key but is intentionally left
# out of this set per spec.
COUNTABLE_VERIFICATION_FIELDS = {"full_name", "dob", "phone", "email", "ssn_last4"}
VERIFICATION_THRESHOLD = 3
FRICTION_ESCALATION_THRESHOLD = 2
OFF_TOPIC_ESCALATION_THRESHOLD = 2
# After the transfer has been offered and the caller still hasn't cooperated
# or explicitly asked for it, stop persuading and escalate automatically
# rather than offering forever with no way out.
FRICTION_HARD_ESCALATION_THRESHOLD = 4
# Same idea as FRICTION_HARD_ESCALATION_THRESHOLD but for repeated off-topic
# turns: after enough of them we stop re-offering a transfer forever and
# just escalate, so "narrated progress" never loops indefinitely.
OFF_TOPIC_HARD_ESCALATION_THRESHOLD = 4

# Short labels used in the debug panel's "2/3: name, dob" style summary.
VERIFICATION_FIELD_SHORT_LABELS = {
    "full_name": "name",
    "dob": "dob",
    "phone": "phone",
    "email": "email",
    "ssn_last4": "ssn",
}


@dataclass
class ToolCallRecord:
    tool: str
    args: Dict[str, Any]
    result: Any


@dataclass
class SessionState:
    session_id: str
    phase: Phase = Phase.VERIFY_ID
    verified: bool = False
    verified_fields: List[str] = field(default_factory=list)
    pii_candidates: Dict[str, str] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    party_id: Optional[str] = None
    consent: Dict[str, Any] = field(default_factory=dict)
    off_topic_count: int = 0
    verification_friction_count: int = 0
    sentiment_history: List[str] = field(default_factory=list)
    escalation_offered: bool = False
    escalated: bool = False
    # Bookkeeping for the deterministic post-escalation terminal-reply
    # cycling: a repeat of the same follow-up input gets the same reply
    # back; a genuinely different follow-up gets a different one, so the
    # session never both (a) parrots the original transfer sentence forever
    # or (b) fabricates new "in progress" narration via a live model call.
    escalation_followup_count: int = 0
    last_escalation_input: Optional[str] = None
    last_escalation_reply: Optional[str] = None

    # A caller explicitly signaling the call is over (e.g. "no, that's all,
    # thank you") gets a genuine one-time sign-off instead of an infinite
    # "anything else?" loop. Same cycling bookkeeping as escalation above.
    session_closed: bool = False
    closing_followup_count: int = 0
    last_closing_input: Optional[str] = None
    last_closing_reply: Optional[str] = None

    # Supporting state not enumerated in the spec's bullet list but needed to
    # run the demo end to end.
    caller_role: Optional[str] = None  # "self" | "representative"
    caller_claimed_name: Optional[str] = None
    representative_verified: bool = False
    email_sent: bool = False
    history: List[Dict[str, str]] = field(default_factory=list)
    tool_log: List[ToolCallRecord] = field(default_factory=list)
    last_ui_hints: Dict[str, Any] = field(default_factory=dict)

    def add_verified_fields(self, fields_: List[str]) -> None:
        for f in fields_:
            if f in COUNTABLE_VERIFICATION_FIELDS and f not in self.verified_fields:
                self.verified_fields.append(f)

    def as_public_dict(self) -> Dict[str, Any]:
        """A trimmed view safe to return to the frontend / assert on in tests."""
        return {
            "phase": self.phase.value,
            "verified": self.verified,
            "verified_fields": list(self.verified_fields),
            "party_id": self.party_id,
            "memory": dict(self.memory),
            "consent": dict(self.consent),
            "off_topic_count": self.off_topic_count,
            "verification_friction_count": self.verification_friction_count,
            "sentiment_history": list(self.sentiment_history),
            "escalation_offered": self.escalation_offered,
            "escalated": self.escalated,
            "email_sent": self.email_sent,
            "session_closed": self.session_closed,
        }

    def last_tool_call_summary(self) -> Optional[str]:
        """One-line human-readable summary of the most recent tool call, for
        the debug panel — driven by the actual tool_log, never guessed."""
        if not self.tool_log:
            return None
        record = self.tool_log[-1]
        result = record.result if isinstance(record.result, dict) else {}
        if record.tool == "verify_identity":
            return f"verify_identity() -> match_count={result.get('match_count')}"
        if record.tool == "send_email_summary":
            return f"send_email_summary() -> {result.get('status', 'unknown')}"
        if record.tool == "get_claim_detail":
            case_id = result.get("case_id") if result else None
            return f"get_claim_detail() -> {'case_id=' + case_id if case_id else 'not found'}"
        if record.tool == "get_claims_for_party":
            count = len(record.result) if isinstance(record.result, list) else 0
            return f"get_claims_for_party() -> {count} claim(s)"
        if record.tool == "get_required_documents":
            docs = result.get("documents", []) if result else []
            return f"get_required_documents() -> {len(docs)} document(s)"
        return f"{record.tool}() -> {record.result!r}"

    def as_debug_dict(self) -> Dict[str, Any]:
        """Everything the debug/state panel in the UI shows, read straight
        off this object — never separately maintained front-end guesswork."""
        verified_labels = [VERIFICATION_FIELD_SHORT_LABELS.get(f, f) for f in self.verified_fields]
        return {
            "phase": self.phase.value,
            "verified": self.verified,
            "verified_fields": list(self.verified_fields),
            "verified_summary": (
                f"{len(self.verified_fields)}/{VERIFICATION_THRESHOLD}: "
                + (", ".join(verified_labels) if verified_labels else "none yet")
            ),
            "party_id": self.party_id,
            "memory": dict(self.memory),
            "escalated": self.escalated,
            "escalation_offered": self.escalation_offered,
            "off_topic_count": self.off_topic_count,
            "verification_friction_count": self.verification_friction_count,
            "email_sent": self.email_sent,
            "session_closed": self.session_closed,
            "last_tool_call": self.last_tool_call_summary(),
        }
