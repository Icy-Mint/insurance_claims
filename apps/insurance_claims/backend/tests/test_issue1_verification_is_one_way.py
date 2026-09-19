"""Regression tests for Issue 1: verified state (and each individual
verified field) must be a one-way gate — never re-evaluated or overwritten
once matched, no matter what a later message contains or what the LLM
enrichment pass fabricates.

Root cause found via live-API reproduction: llm_extract_enrich() is a
forced tool call (tool_choice=record_extraction), so on a message with NO
identity content at all, the model still has to call the tool and was
observed fabricating a placeholder value (full_name="<UNKNOWN>") instead of
omitting the field. That value then overwrote the caller's already-matched
full_name in state.pii_candidates, causing a real re-check against
policyholders.json to fail and the agent to falsely claim "that full name
you just gave doesn't match" — an apparent un-verification the caller never
triggered.

Two independent fixes are tested here:
  1. llm.llm_extract_enrich format-validates every field before trusting it
     (rejects "<UNKNOWN>", "N/A", junk that doesn't match the field's shape).
  2. phases._apply_extraction never lets a field already in
     state.verified_fields be overwritten by a later turn's extraction,
     even a well-formatted one — once matched, it's locked in for the
     session.
  3. phases._handle_verify_id is a hard no-op once state.verified is True.
"""
from types import SimpleNamespace

from app import llm
from app.models import Phase, SessionState
from app.phases import _apply_extraction, _handle_verify_id
from app.extraction import ExtractionResult


def test_llm_extract_enrich_rejects_placeholder_full_name(monkeypatch):
    class FakeMessages:
        def create(self, **kwargs):
            block = SimpleNamespace(type="tool_use", input={"full_name": "<UNKNOWN>"})
            return SimpleNamespace(content=[block])

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(llm, "get_client", lambda: FakeClient())

    state = SessionState(session_id="s", phase=Phase.VERIFY_ID)
    result = ExtractionResult()  # regex pass found nothing this turn
    llm.llm_extract_enrich("I'm calling about my denied healthcare claim from January.", state, result)

    assert "full_name" not in result.pii_candidates


def test_llm_extract_enrich_accepts_a_real_looking_name(monkeypatch):
    class FakeMessages:
        def create(self, **kwargs):
            block = SimpleNamespace(type="tool_use", input={"full_name": "Margaret Chen"})
            return SimpleNamespace(content=[block])

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(llm, "get_client", lambda: FakeClient())

    state = SessionState(session_id="s", phase=Phase.VERIFY_ID)
    result = ExtractionResult()
    llm.llm_extract_enrich("that's Margaret Chen", state, result)

    assert result.pii_candidates.get("full_name") == "Margaret Chen"


def test_already_verified_field_cannot_be_overwritten_by_later_extraction():
    state = SessionState(session_id="s", phase=Phase.VERIFY_ID)
    state.pii_candidates["full_name"] = "Margaret Chen"
    state.verified_fields = ["full_name"]

    # A later turn's extraction pass claims a (garbage, or even plausible)
    # new value for a field already locked in as verified.
    bogus_extraction = ExtractionResult(pii_candidates={"full_name": "<UNKNOWN>"})
    _apply_extraction(state, bogus_extraction, "irrelevant message")

    assert state.pii_candidates["full_name"] == "Margaret Chen"


def test_handle_verify_id_is_a_no_op_once_fully_verified():
    state = SessionState(session_id="s", phase=Phase.RESOLVE_INTENT, verified=True)
    state.verified_fields = ["full_name", "dob", "ssn_last4"]
    state.pii_candidates = {"full_name": "Margaret Chen", "dob": "1985-03-15", "ssn_last4": "4472"}
    calls_before = len(state.tool_log)

    extraction = ExtractionResult()
    rejected, no_match = _handle_verify_id(state, extraction)

    assert rejected == []
    assert no_match is False
    assert len(state.tool_log) == calls_before  # verify_identity was never called again


def test_paraphrased_premature_confirmation_is_also_caught(monkeypatch):
    # A different wording of the same false claim, also observed live in
    # the same session as the "<UNKNOWN>" bug — the literal marker list
    # alone didn't catch this phrasing, so the regex pattern layer must.
    state = SessionState(session_id="s", phase=Phase.VERIFY_ID, verified=False)
    state.verified_fields = ["full_name", "ssn_last4"]

    class FakeMessages:
        def create(self, **kwargs):
            text = (
                "Thank you, Margaret — that matches. I now have enough to verify your identity. "
                "Let me pull up your account so I can help you with your claim."
            )
            block = SimpleNamespace(type="text", text=text)
            return SimpleNamespace(stop_reason="end_turn", content=[block])

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(llm, "get_client", lambda: FakeClient())

    text, _ = llm.generate_reply(state, Phase.VERIFY_ID, "SSN last four is 4472")

    assert not llm._matches_any_pattern(text, llm.FALSE_VERIFICATION_CONFIRMATION_PATTERNS)


def test_end_to_end_intent_only_message_after_verification_does_not_unverify(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": "My name is Margaret Chen."})
    client.post("/chat", json={"session_id": session_id, "message": "DOB is 1985-03-15."})
    resp = client.post("/chat", json={"session_id": session_id, "message": "SSN last four is 4472."})
    assert resp.json()["state"]["verified"] is True

    resp = client.post(
        "/chat",
        json={"session_id": session_id, "message": "I'm calling about my denied healthcare claim from January."},
    )
    body = resp.json()
    assert body["state"]["verified"] is True
    assert set(body["state"]["verified_fields"]) == {"full_name", "dob", "ssn_last4"}
    reply_lower = body["reply"].lower()
    assert "doesn't match" not in reply_lower
    assert "does not match" not in reply_lower
    assert "verify your identity" not in reply_lower
