"""Regression test: once verified, RESOLVE_INTENT must never hedge or cast
doubt on the fact that verification succeeded (e.g. "I don't have
confirmation these details matched a record") — found via self-testing
against a live model, which will hedge unless explicitly guarded against."""
from types import SimpleNamespace

from app import llm
from app.models import Phase, SessionState


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


def test_resolve_intent_never_doubts_a_completed_verification(monkeypatch):
    state = SessionState(session_id="s", phase=Phase.RESOLVE_INTENT, verified=True)
    state.verified_fields = ["full_name", "dob", "ssn_last4"]

    hedging_text = (
        "I don't have any tool results or account data confirming these details actually "
        "match a record, since nothing has come back from the verification system."
    )
    fake = FakeClient([_text_response(hedging_text)])
    monkeypatch.setattr(llm, "get_client", lambda: fake)

    text, tool_calls = llm.generate_reply(state, Phase.RESOLVE_INTENT, "I'm calling about my claim.")

    assert not llm._contains_any(text, llm.VERIFICATION_DOUBT_MARKERS)
    assert tool_calls == []
