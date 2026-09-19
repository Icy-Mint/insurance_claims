"""Regression tests found via a goal-oriented pass against the bonus spec:
"Recognize frustration, anxiety, anger, confusion, or refusal" and "Know when
to stop persuading and escalate to a human." Both were broken for messages
with no insurance keyword and no explicit "transfer me" phrasing."""
from app.models import FRICTION_HARD_ESCALATION_THRESHOLD


def test_pure_anger_without_insurance_keywords_is_not_treated_as_off_topic(client, session_id):
    resp = client.post(
        "/chat",
        json={"session_id": session_id, "message": "This is absolutely unacceptable, I am furious, your company is a joke."},
    )
    body = resp.json()
    assert body["state"]["off_topic_count"] == 0
    assert "only able to help" not in body["reply"].lower()
    assert body["state"]["sentiment_history"][-1] == "angry"


def test_confusion_without_insurance_keywords_is_not_treated_as_off_topic(client, session_id):
    resp = client.post(
        "/chat", json={"session_id": session_id, "message": "I don't understand what you're asking me for, what do you mean?"}
    )
    body = resp.json()
    assert body["state"]["off_topic_count"] == 0
    assert "only able to help" not in body["reply"].lower()


def test_sustained_friction_auto_escalates_without_ever_leaking(client, session_id):
    frustrated_message = "I already told you who I am. This is ridiculous. Just tell me why my claim was denied."
    body = None
    for _ in range(FRICTION_HARD_ESCALATION_THRESHOLD):
        resp = client.post("/chat", json={"session_id": session_id, "message": frustrated_message})
        body = resp.json()

    assert body["state"]["verification_friction_count"] >= FRICTION_HARD_ESCALATION_THRESHOLD
    assert body["state"]["escalated"] is True
    assert body["state"]["verified"] is False

    leaked = ["CL-2048", "pathology report", "office note", "2026-03-18"]
    assert not any(l.lower() in body["reply"].lower() for l in leaked)

    # And it stays escalated — no further stalling loop.
    resp = client.post("/chat", json={"session_id": session_id, "message": frustrated_message})
    assert resp.json()["state"]["escalated"] is True
