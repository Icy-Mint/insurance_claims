"""Regression tests for Bug 2: email consent must be scoped to the specific
pending question, and a "sent" claim must never appear without an actual
logged send."""
from app import tools as tools_module

MARGARET_MESSAGE = (
    "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about "
    "my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472."
)


def test_unrelated_yes_before_the_offer_does_not_send(client, session_id):
    # An affirmative reply to something else entirely, well before the email
    # offer has even been raised, must never be treated as email consent.
    client.post("/chat", json={"session_id": session_id, "message": "My name is Margaret Chen."})
    resp = client.post("/chat", json={"session_id": session_id, "message": "Yes"})
    body = resp.json()
    assert body["state"]["phase"] == "VERIFY_ID"
    assert body["state"]["email_sent"] is False
    assert tools_module.EMAIL_LOG == []

    # Finish verification + resolve the case in the same follow-up message.
    resp = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "message": "Sorry — DOB is 1985-03-15, SSN last four is 4472, denied healthcare claim from January.",
        },
    )
    body = resp.json()
    assert body["state"]["phase"] == "POST_PROCESS"
    assert body["state"]["email_sent"] is False
    assert tools_module.EMAIL_LOG == []
    assert body["ui_hints"]["quick_replies"] == ["Yes, email me", "No thanks"]

    # Only *now*, answering the actual pending offer, does "yes" send.
    resp = client.post("/chat", json={"session_id": session_id, "message": "yes"})
    body = resp.json()
    assert body["state"]["email_sent"] is True
    assert len(tools_module.EMAIL_LOG) == 1


def test_reply_with_punctuation_still_recognized_as_consent(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    resp = client.post("/chat", json={"session_id": session_id, "message": "Yes, email me"})
    body = resp.json()
    assert body["state"]["email_sent"] is True
    assert len(tools_module.EMAIL_LOG) == 1


def test_ambiguous_reply_while_awaiting_consent_re_asks_and_never_sends(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    resp = client.post(
        "/chat", json={"session_id": session_id, "message": "can you tell me more about the documents first?"}
    )
    body = resp.json()
    assert body["state"]["email_sent"] is False
    assert tools_module.EMAIL_LOG == []
    assert "email" in body["reply"].lower()
    assert body["ui_hints"]["quick_replies"] == ["Yes, email me", "No thanks"]


def test_never_claims_sent_without_a_logged_send(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    resp = client.post("/chat", json={"session_id": session_id, "message": "not right now"})
    body = resp.json()
    assert body["state"]["email_sent"] is False
    assert tools_module.EMAIL_LOG == []
    reply_lower = body["reply"].lower()
    assert "i've sent" not in reply_lower
    assert "i'll have that summary sent" not in reply_lower
