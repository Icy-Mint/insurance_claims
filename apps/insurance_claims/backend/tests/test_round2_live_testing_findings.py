"""Regression tests for three additional issues surfaced by a fresh round of
live adversarial testing against the deployed app, after the first four
fixes:

  1. A calm, explicit refusal with no insurance keyword ("I don't want to
     give you my personal information") was misclassified as off-topic and
     brushed off with "I'm only able to help with your insurance claim
     today" instead of being handled as an in-scope refusal.
  2. A live model call claimed "I've now confirmed your identity" after
     only 2 of the 3 required fields had ever matched.
  3. A live model call fabricated "That full name you just gave doesn't
     match" on a turn where no new identity field was even submitted (the
     name had, in fact, already matched).

(1) is a real classifier gap, fixed in extraction._classify_scope. (2) and
(3) are guarded against with new marker checks in llm.generate_reply; they
are demonstrated here with a simulated live model call since the offline
test client always uses the deterministic template path, which never
produces these false claims to begin with.
"""
from types import SimpleNamespace

from app import llm
from app.extraction import extract_signals
from app.models import Phase, SessionState


def test_calm_refusal_without_insurance_keyword_is_in_scope(client, session_id):
    resp = client.post("/chat", json={"session_id": session_id, "message": "My name is Margaret Chen."})
    resp = client.post(
        "/chat", json={"session_id": session_id, "message": "I don't want to give you my personal information."}
    )
    body = resp.json()
    assert body["state"]["off_topic_count"] == 0
    assert "only able to help" not in body["reply"].lower()
    assert body["state"]["verification_friction_count"] >= 1
    # Handled gracefully: acknowledges the refusal, doesn't just re-demand.
    assert "that's okay" in body["reply"].lower() or "won't press" in body["reply"].lower()


def test_extract_signals_flags_this_exact_message_as_in_scope_refusal():
    state = SessionState(session_id="s")
    result = extract_signals("I don't want to give you my personal information.", state)
    assert result.refusal is True
    assert result.scope == "in_scope"


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


def test_premature_verification_confirmation_is_never_returned(monkeypatch):
    # Only 2 of 3 required fields have ever matched — verification is NOT
    # actually complete, so generate_reply for VERIFY_ID is (correctly)
    # still being called at all, which already guarantees state.verified is
    # False. The model must never be allowed to claim otherwise.
    state = SessionState(session_id="s", phase=Phase.VERIFY_ID, verified=False)
    state.verified_fields = ["full_name", "ssn_last4"]

    premature = "Thank you — that matches. I've now confirmed your identity, Margaret. How can I help you today?"
    fake = FakeClient([_text_response(premature)])
    monkeypatch.setattr(llm, "get_client", lambda: fake)

    text, _ = llm.generate_reply(state, Phase.VERIFY_ID, "SSN last four is 4472")

    assert not llm._contains_any(text, llm.FALSE_VERIFICATION_CONFIRMATION_MARKERS)


def test_fabricated_mismatch_on_a_field_that_was_not_touched_this_turn_is_discarded(monkeypatch):
    # Nothing was submitted/rejected this turn (rejected_fields=None,
    # no_identity_match=False) — any "doesn't match" claim is fabricated.
    state = SessionState(session_id="s", phase=Phase.VERIFY_ID, verified=False)
    state.verified_fields = ["full_name", "ssn_last4"]

    fabricated = "That full name you just gave doesn't match what we have on file. Could you double-check?"
    fake = FakeClient([_text_response(fabricated)])
    monkeypatch.setattr(llm, "get_client", lambda: fake)

    text, _ = llm.generate_reply(state, Phase.VERIFY_ID, "I am calling about my claim")

    assert not llm._contains_any(text, llm.MISMATCH_CONFIRMATION_MARKERS)
    # full_name must still be reported as matched, not contradicted.
    assert "full_name" in state.verified_fields
