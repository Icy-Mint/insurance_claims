"""Regression tests for two issues found in a live transcript:

1. A bare acknowledgment ("ok") mid-VERIFY_ID was misclassified as
   off-topic and got the generic "I'm only able to help with your
   insurance claim today" decline, incrementing off_topic_count for a
   message that was obviously part of the ongoing call.
2. After a correct field brought match_count to 2/3, a live model reply
   ("Let me just make sure I have everything I need to move forward —
   give me just a moment") neither confirmed nor denied progress and
   never asked for the still-missing field, leaving the caller stuck.
   Since it wasn't a specific false claim (no marker phrase matched), the
   existing guards let it through; the fix is a general "a VERIFY_ID reply
   must ask something" invariant, since state.verified is guaranteed False
   whenever this phase's generate_reply runs at all.
"""
from types import SimpleNamespace

from app import llm
from app.extraction import extract_signals
from app.models import Phase, SessionState


def test_bare_ok_is_in_scope_during_verify_id(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": "My name is Margaret Chen."})
    resp = client.post("/chat", json={"session_id": session_id, "message": "ok"})
    debug = resp.json()["debug_state"]
    assert debug["off_topic_count"] == 0
    assert debug["phase"] == "VERIFY_ID"


def test_extract_signals_classifies_ok_as_in_scope():
    state = SessionState(session_id="s")
    result = extract_signals("ok", state)
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


def test_a_non_asking_stall_reply_is_replaced_with_the_deterministic_ask(monkeypatch):
    state = SessionState(session_id="s", phase=Phase.VERIFY_ID, verified=False)
    state.verified_fields = ["full_name", "ssn_last4"]

    stall_text = "Thanks, that one lines up. Let me just make sure I have everything I need to move forward — give me just a moment."
    fake = FakeClient([_text_response(stall_text)])
    monkeypatch.setattr(llm, "get_client", lambda: fake)

    text, _ = llm.generate_reply(state, Phase.VERIFY_ID, "SSN last four is 4472")

    assert "?" in text  # the deterministic fallback always asks for the next field
    assert text != stall_text


def test_a_reply_that_asks_something_is_left_alone(monkeypatch):
    state = SessionState(session_id="s", phase=Phase.VERIFY_ID, verified=False)
    state.verified_fields = ["full_name", "ssn_last4"]

    good_text = "Thanks, that matches. Could you also give me your date of birth?"
    fake = FakeClient([_text_response(good_text)])
    monkeypatch.setattr(llm, "get_client", lambda: fake)

    text, _ = llm.generate_reply(state, Phase.VERIFY_ID, "SSN last four is 4472")

    assert text == good_text
