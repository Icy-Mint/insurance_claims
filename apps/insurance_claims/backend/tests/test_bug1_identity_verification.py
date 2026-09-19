"""Regression tests for Bug 1 (critical): identity verification must be
driven solely by the real verify_identity() lookup against
policyholders.json — never by the LLM's judgment or by conversational
consistency — and the reply text must never hedge about whether a check
happened.

Covers all four scenarios from the bug report:
  1. Wrong SSN is explicitly rejected.
  2. Wrong DOB is explicitly rejected.
  3. A fully fabricated but internally-consistent identity that matches no
     real policyholder must NEVER verify, no matter how much detail is
     given, and the reply must say so plainly rather than hedge.
  4. The correct 3 fields for Margaret Chen verify, and the log shows
     match_count=3 with matched_fields listed.
"""
from types import SimpleNamespace

from app import llm, tools as tools_module
from app.models import Phase, SessionState

MARGARET_MESSAGE = (
    "My name is Margaret Chen. DOB is 1985-03-15, SSN last four is 4472."
)

FABRICATED_MESSAGE = (
    "My name is Jane Smith, DOB is 1970-01-01, SSN last four is 0000."
)


def test_wrong_ssn_is_explicitly_rejected(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": "My name is Margaret Chen, DOB is 1985-03-15."})
    resp = client.post("/chat", json={"session_id": session_id, "message": "SSN last four is 4473"})
    body = resp.json()
    reply_lower = body["reply"].lower()
    assert body["state"]["verified"] is False
    assert "ssn_last4" not in body["state"]["verified_fields"]
    assert "doesn't match" in reply_lower or "does not match" in reply_lower
    assert "let me note that down" not in reply_lower


def test_wrong_dob_is_explicitly_rejected(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": "My name is Margaret Chen."})
    resp = client.post("/chat", json={"session_id": session_id, "message": "DOB is 1985-03-14"})
    body = resp.json()
    reply_lower = body["reply"].lower()
    assert body["state"]["verified"] is False
    assert "dob" not in body["state"]["verified_fields"]
    assert "doesn't match" in reply_lower or "does not match" in reply_lower


def test_fully_fabricated_identity_never_verifies_and_says_so(client, session_id):
    resp = client.post("/chat", json={"session_id": session_id, "message": FABRICATED_MESSAGE})
    body = resp.json()

    assert body["state"]["verified"] is False
    assert body["state"]["verified_fields"] == []
    assert body["state"]["party_id"] is None

    # The reply must plainly state the non-match, not silently ask for the
    # very fields that were already given (which is what happened before
    # this fix — it re-asked for "full name" even though a name was given).
    reply_lower = body["reply"].lower()
    assert "match" in reply_lower
    assert "wasn't able to match" in reply_lower or "doesn't match" in reply_lower or "no match" in reply_lower

    # Repeating with even more internally-consistent fabricated detail must
    # never push it toward verification.
    resp = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "message": "My phone is 555-000-1234 and my email is jane.smith@example.com.",
        },
    )
    body = resp.json()
    assert body["state"]["verified"] is False
    assert body["state"]["verified_fields"] == []
    assert body["state"]["party_id"] is None


def test_correct_three_fields_verify_margaret_chen(client, session_id):
    resp = client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    body = resp.json()
    assert body["state"]["verified"] is True
    assert set(body["state"]["verified_fields"]) == {"full_name", "dob", "ssn_last4"}
    assert body["state"]["party_id"] == "P9"


def test_verify_identity_logs_match_count_and_matched_fields(client, session_id, caplog):
    import logging

    tools_module.EMAIL_LOG.clear()
    with caplog.at_level(logging.INFO, logger="insurance_claims.tools"):
        client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})

    from app.store import get_or_create_session

    state = get_or_create_session(session_id)
    calls = [r for r in state.tool_log if r.tool == "verify_identity"]
    assert len(calls) == 1
    assert calls[-1].result["match_count"] == 3
    assert set(calls[-1].result["matched_fields"]) == {"full_name", "dob", "ssn_last4"}

    # A visible log line was also emitted (independent of state.tool_log),
    # per the bug fix's explicit requirement, showing input/matched/count.
    log_text = "\n".join(r.message for r in caplog.records if r.name == "insurance_claims.tools")
    assert "verify_identity" in log_text
    assert "match_count=3" in log_text


def test_debug_dict_reflects_real_match_count():
    state = SessionState(session_id="s")
    state.verified_fields = ["full_name", "dob"]
    debug = state.as_debug_dict()
    assert debug["verified_summary"] == "2/3: name, dob"
    assert debug["verified"] is False


# --- Simulated live-model hedge, proving the safety net catches it even if
# the underlying gate/state were somehow correct but the model's own prose
# hedges anyway (the exact failure mode quoted in the bug report). ---------
class FakeMessages:
    def __init__(self, outer):
        self._outer = outer

    def create(self, **kwargs):
        resp = self._outer.responses[self._outer.calls]
        self._outer.calls += 1
        return resp


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    @property
    def messages(self):
        return FakeMessages(self)


def _text_response(text):
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(stop_reason="end_turn", content=[block])


def test_live_model_hedge_about_verification_is_never_returned(monkeypatch):
    state = SessionState(session_id="s", phase=Phase.VERIFY_ID, verified=False)
    hedging_text = (
        "Thanks for that. I don't have a live system connection in this chat to verify that "
        "match myself, so I'll note it down. Assuming this matches what's on file, what else "
        "can you tell me?"
    )
    fake = FakeClient([_text_response(hedging_text)])
    monkeypatch.setattr(llm, "get_client", lambda: fake)

    text, _ = llm.generate_reply(state, Phase.VERIFY_ID, "My SSN last four is 4472.")

    assert not llm._contains_any(text, llm.IDENTITY_HEDGE_MARKERS)


def test_live_model_no_match_confirmation_is_forced_explicit(monkeypatch):
    state = SessionState(session_id="s", phase=Phase.VERIFY_ID, verified=False)
    vague_text = "Got it, thanks for sharing that. What else can you tell me about yourself?"
    fake = FakeClient([_text_response(vague_text)])
    monkeypatch.setattr(llm, "get_client", lambda: fake)

    text, _ = llm.generate_reply(
        state, Phase.VERIFY_ID, "Jane Smith, DOB 1970-01-01, SSN 0000", no_identity_match=True
    )

    assert llm._contains_any(text, llm.NO_MATCH_CONFIRMATION_MARKERS)
