"""Unit-level checks on the structural gate itself: the LLM must never be
handed the claim-lookup tool definitions before verification, and the
executor must refuse to run them even if something tried to force it."""
import pytest

from app import llm
from app.models import Phase, SessionState


def test_no_claim_tools_offered_before_verification():
    state = SessionState(session_id="s", phase=Phase.VERIFY_ID, verified=False)
    assert llm.build_tool_defs(state) == []


def test_claim_tools_offered_once_verified():
    state = SessionState(session_id="s", phase=Phase.PROCESS_CASE, verified=True)
    names = {t["name"] for t in llm.build_tool_defs(state)}
    assert names == {"get_claim_detail", "get_required_documents"}


def test_execute_tool_refuses_when_not_verified():
    state = SessionState(session_id="s", phase=Phase.VERIFY_ID, verified=False)
    state.memory["resolved_case_id"] = "CL-2048"
    with pytest.raises(llm.ToolNotAllowedError):
        llm.execute_tool("get_claim_detail", state)


def test_execute_tool_only_returns_the_resolved_case_never_an_arbitrary_one():
    state = SessionState(session_id="s", phase=Phase.PROCESS_CASE, verified=True)
    state.memory["resolved_case_id"] = "CL-2048"
    result = llm.execute_tool("get_claim_detail", state)
    assert result["case_id"] == "CL-2048"
