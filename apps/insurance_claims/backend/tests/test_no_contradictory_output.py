"""Regression tests for Bug 1: a live model call must never be allowed to
return text that contradicts the phase invariant it was generated under
(e.g. disclosing claim facts while also asking for more verification, or
disclosing facts at all before verification is complete). Since the offline
test environment has no real Anthropic client, these simulate a
misbehaving one directly to prove the post-generation guard catches it."""
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


def _tool_use_response(tool_name, call_id="call1"):
    block = SimpleNamespace(type="tool_use", name=tool_name, input={}, id=call_id)
    return SimpleNamespace(stop_reason="tool_use", content=[block])


def _text_response(text):
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(stop_reason="end_turn", content=[block])


def test_process_case_never_mixes_verification_request_with_disclosure(monkeypatch):
    state = SessionState(session_id="s", phase=Phase.PROCESS_CASE, verified=True)
    state.memory["resolved_case_id"] = "CL-2048"

    contradictory_text = (
        "I still need one more piece of identifying information before I can discuss any "
        "claim details. Your claim CL-2048 was denied because the pathology report and "
        "office note were missing."
    )
    fake = FakeClient([_tool_use_response("get_claim_detail"), _text_response(contradictory_text)])
    monkeypatch.setattr(llm, "get_client", lambda: fake)

    text, tool_calls = llm.generate_reply(state, Phase.PROCESS_CASE, "why was my claim denied?")

    assert not llm._contains_any(text, llm.VERIFICATION_REQUEST_MARKERS)
    assert "pathology report" in text  # the real, grounded fact is preserved
    assert any(t["tool"] == "get_claim_detail" for t in tool_calls)


def test_verify_id_never_discloses_claim_facts_even_if_the_model_tries(monkeypatch):
    state = SessionState(session_id="s", phase=Phase.VERIFY_ID, verified=False)
    hallucinated_text = "Thanks! By the way, your claim CL-2048 was denied due to a missing pathology report."
    fake = FakeClient([_text_response(hallucinated_text)])
    monkeypatch.setattr(llm, "get_client", lambda: fake)

    text, tool_calls = llm.generate_reply(state, Phase.VERIFY_ID, "hello")

    assert not llm._contains_any(text, llm.CLAIM_DISCLOSURE_MARKERS)
    assert tool_calls == []


def test_post_process_is_never_routed_through_the_llm():
    state = SessionState(session_id="s", phase=Phase.POST_PROCESS, verified=True)
    try:
        llm.generate_reply(state, Phase.POST_PROCESS, "yes")
        assert False, "expected an AssertionError guarding against LLM-authored consent handling"
    except AssertionError:
        pass
