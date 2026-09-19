"""Regression tests for Issue 4: no real terminal/closing state — the agent
was looping the identical verbatim reply forever in two situations:
  (a) after escalation, every follow-up got the exact same "please hold"
      transfer sentence repeated, or a live model call risked inventing
      contradictory "in progress" narration instead.
  (b) after a caller said "no, that's all, thank you" in POST_PROCESS,
      every repeat got the identical "anything else?" prompt with no real
      close ever happening.

(a) is covered by test_escalation_intent.py / test_bug3_escalation_state.py.
This file covers (b) and the general "3 different messages -> not all
identical replies" rule for both terminal states in one place.
"""
from app import tools as tools_module
from app.prompts import SESSION_CLOSED_FOLLOWUP_VARIANTS, SESSION_CLOSING_SIGNOFF

MARGARET_MESSAGE = (
    "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about "
    "my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472."
)


def test_closing_signal_gets_a_genuine_signoff_once(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    client.post("/chat", json={"session_id": session_id, "message": "yes"})  # send the email
    resp = client.post("/chat", json={"session_id": session_id, "message": "no, that's all, thank you"})
    body = resp.json()
    assert body["state"]["session_closed"] is True
    assert body["reply"] == SESSION_CLOSING_SIGNOFF


def test_repeated_closing_messages_do_not_loop_the_same_anything_else_prompt(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    client.post("/chat", json={"session_id": session_id, "message": "no thanks"})  # decline email
    resp = client.post("/chat", json={"session_id": session_id, "message": "no, that's all, thank you"})
    assert resp.json()["state"]["session_closed"] is True
    first_signoff = resp.json()["reply"]
    assert first_signoff == SESSION_CLOSING_SIGNOFF

    replies = []
    for msg in ["no, that's all, thank you", "goodbye", "thanks, bye"]:
        resp = client.post("/chat", json={"session_id": session_id, "message": msg})
        body = resp.json()
        assert body["state"]["session_closed"] is True
        assert "anything else i can help with" not in body["reply"].lower()
        replies.append(body["reply"])
    # None of the post-close follow-ups repeat the original "anything
    # else?" loop, and they're never all the same generated-on-the-fly text
    # (they come from the fixed, deterministic variant list).
    for r in replies:
        assert r in SESSION_CLOSED_FOLLOWUP_VARIANTS


def test_no_email_action_possible_after_session_closed(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    client.post("/chat", json={"session_id": session_id, "message": "no thanks"})
    client.post("/chat", json={"session_id": session_id, "message": "no, that's all, thank you"})
    resp = client.post("/chat", json={"session_id": session_id, "message": "yes"})
    body = resp.json()
    # A "yes" after the session is closed must not be reinterpreted as
    # email consent — the session is over.
    assert body["state"]["email_sent"] is False
    assert tools_module.EMAIL_LOG == []
