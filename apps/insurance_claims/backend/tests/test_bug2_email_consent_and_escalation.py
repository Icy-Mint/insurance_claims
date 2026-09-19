"""Regression tests for Bug 2 (critical): email-send consent must be scoped
to the exact pending yes/no question this code just posed, a compound reply
must not silently drop one of its two intents, and the agent must never
narrate a send that didn't actually happen.

Covers all four scenarios from the bug report.
"""
from app import tools as tools_module

MARGARET_MESSAGE = (
    "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about "
    "my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472."
)


def test_1_unrelated_yes_before_the_scoped_offer_never_sends(client, session_id):
    # "Yes" answering something else entirely (here: still mid-verification,
    # nowhere near the email offer) must never be captured as email consent.
    client.post("/chat", json={"session_id": session_id, "message": "My name is Margaret Chen."})
    resp = client.post("/chat", json={"session_id": session_id, "message": "Yes"})
    body = resp.json()
    assert body["state"]["phase"] == "VERIFY_ID"
    assert body["state"]["email_sent"] is False
    assert tools_module.EMAIL_LOG == []


def test_2_compound_yes_plus_human_transfer_at_the_scoped_offer(client, session_id):
    # Reaches the actual scoped email yes/no question, then replies with a
    # compound message. The transfer intent must not be dropped, and the
    # system must not fabricate an "email sent" message.
    client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    resp = client.post(
        "/chat", json={"session_id": session_id, "message": "yes, I would like to talk to a human representative"}
    )
    body = resp.json()

    # Transfer intent handled — not silently ignored.
    assert body["state"]["escalated"] is True
    assert "human representative" in body["reply"].lower() or "transferring" in body["reply"].lower()

    # No fabricated send: never claims "Done" / "I've sent" here.
    assert body["state"]["email_sent"] is False
    assert tools_module.EMAIL_LOG == []
    assert "i've sent" not in body["reply"].lower()
    assert "done —" not in body["reply"].lower()


def test_3_plain_no_at_the_scoped_offer_does_not_send_or_push_back(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    resp = client.post("/chat", json={"session_id": session_id, "message": "no"})
    body = resp.json()
    assert body["state"]["email_sent"] is False
    assert tools_module.EMAIL_LOG == []
    assert body["state"]["consent"]["email"] == "no"

    reply_lower = body["reply"].lower()
    assert "no problem" in reply_lower or "understood" in reply_lower
    # No pushback / no re-asking of the same yes/no question.
    assert "(yes/no)" not in body["reply"]


def test_4_plain_yes_at_the_scoped_offer_actually_calls_send_email_summary(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    resp = client.post("/chat", json={"session_id": session_id, "message": "yes"})
    body = resp.json()

    assert body["state"]["email_sent"] is True
    assert body["state"]["consent"]["email"] == "yes"
    assert len(tools_module.EMAIL_LOG) == 1
    assert tools_module.EMAIL_LOG[0]["session_id"] == session_id
    assert "i've sent" in body["reply"].lower()

    from app.store import get_or_create_session

    state = get_or_create_session(session_id)
    send_calls = [r for r in state.tool_log if r.tool == "send_email_summary"]
    assert len(send_calls) == 1
    assert send_calls[0].result["status"] == "sent"


def test_yes_to_anything_else_after_decline_does_not_retroactively_send(client, session_id):
    # After an explicit decline, the follow-up prompt changes to "is there
    # anything else?" — a "yes" answering THAT must not be reinterpreted as
    # email consent either.
    client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    client.post("/chat", json={"session_id": session_id, "message": "no thanks"})
    resp = client.post("/chat", json={"session_id": session_id, "message": "yes"})
    body = resp.json()
    assert body["state"]["email_sent"] is False
    assert tools_module.EMAIL_LOG == []
