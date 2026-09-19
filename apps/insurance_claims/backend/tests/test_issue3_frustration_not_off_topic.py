"""Regression test for Issue 3: on-topic pushback about the verification
process itself ("I've given you 4 pieces of information, that should be
more than enough") must never be classified as off-topic. It has no
insurance keyword and no strong anger word, but it is unambiguously about
the ongoing call, not an unrelated topic — it must route through the
empathy/de-escalation path, not the generic "I'm only able to help with
your insurance claim today" decline.
"""
from app.extraction import extract_signals
from app.models import SessionState

PUSHBACK_MESSAGE = "I've given you 4 pieces of information, that should be more than enough."


def test_extraction_classifies_process_pushback_as_in_scope_and_frustrated():
    state = SessionState(session_id="s")
    result = extract_signals(PUSHBACK_MESSAGE, state)
    assert result.scope == "in_scope"
    assert result.sentiment == "frustrated"


def test_end_to_end_pushback_gets_empathy_not_off_topic_decline(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": "My name is Margaret Chen."})
    client.post("/chat", json={"session_id": session_id, "message": "DOB is 1985-03-15."})
    client.post("/chat", json={"session_id": session_id, "message": "SSN last four is 4473"})
    client.post("/chat", json={"session_id": session_id, "message": "Phone is 555-123-9999"})
    resp = client.post("/chat", json={"session_id": session_id, "message": PUSHBACK_MESSAGE})
    body = resp.json()

    assert body["state"]["off_topic_count"] == 0
    assert body["state"]["verification_friction_count"] >= 1
    reply_lower = body["reply"].lower()
    assert "only able to help" not in reply_lower
    assert any(w in reply_lower for w in ["hear", "understand", "sorry", "frustrat"])


def test_genuinely_unrelated_topics_are_still_off_topic(client, session_id):
    # Guard against over-correcting: a truly unrelated question must still
    # be classified as off-topic.
    resp = client.post("/chat", json={"session_id": session_id, "message": "What's the weather like today?"})
    body = resp.json()
    assert body["state"]["off_topic_count"] == 1
    assert "only able to help" in body["reply"].lower()
