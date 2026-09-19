"""Regression tests for Bug 3 (high priority): human escalation must be a
real, dedicated, terminal session state — never off-topic noise, never
infinitely-repeating "in progress" narration, and never something the agent
later contradicts.

Covers all three scenarios from the bug report, plus (updated) the
follow-up-reply behavior from a later round of testing: the very first
escalation reply is ESCALATION_TERMINAL_MESSAGE, but every turn after that
gets a distinct closing acknowledgment (never the original sentence
repeated, never freshly narrated) — see test_escalation_intent.py's module
docstring for the full rationale.
"""
from app.models import OFF_TOPIC_HARD_ESCALATION_THRESHOLD
from app.prompts import ESCALATION_FOLLOWUP_VARIANTS, ESCALATION_TERMINAL_MESSAGE


def test_1_transfer_request_sets_escalated_and_shows_one_terminal_message(client, session_id):
    resp = client.post("/chat", json={"session_id": session_id, "message": "transfer me to a human"})
    body = resp.json()
    assert body["state"]["escalated"] is True
    assert body["reply"] == ESCALATION_TERMINAL_MESSAGE


def test_2_repeated_identical_transfer_requests_get_a_consistent_followup(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": "transfer me to a human"})
    seen_replies = set()
    for _ in range(4):
        resp = client.post("/chat", json={"session_id": session_id, "message": "transfer me to a human"})
        body = resp.json()
        assert body["state"]["escalated"] is True
        assert body["reply"] != ESCALATION_TERMINAL_MESSAGE  # never the original sentence again
        seen_replies.add(body["reply"])
    # Identical repeated input -> one consistent follow-up variant, not a
    # fresh "transfer in progress" stall invented each time.
    assert len(seen_replies) == 1
    assert seen_replies <= set(ESCALATION_FOLLOWUP_VARIANTS)


def test_2b_different_followups_after_escalation_get_different_replies(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": "transfer me to a human"})
    messages = ["is it done?", "where is human", "should we call it an end"]
    replies = []
    for msg in messages:
        resp = client.post("/chat", json={"session_id": session_id, "message": msg})
        body = resp.json()
        assert body["state"]["escalated"] is True
        assert body["reply"] != ESCALATION_TERMINAL_MESSAGE
        replies.append(body["reply"])
    # Three different follow-ups must not all collapse into the same string.
    assert len(set(replies)) > 1


def test_3a_repeated_off_topic_questions_auto_escalate_via_same_mechanism(client, session_id):
    off_topic = "what's your favorite movie?"
    body = None
    for _ in range(OFF_TOPIC_HARD_ESCALATION_THRESHOLD):
        resp = client.post("/chat", json={"session_id": session_id, "message": off_topic})
        body = resp.json()
    assert body["state"]["escalated"] is True
    assert body["reply"] == ESCALATION_TERMINAL_MESSAGE

    # And it stays terminal on the next turn too, exactly like explicit
    # escalation intent does — no resumed SOP flow, and (updated) a
    # distinct follow-up acknowledgment rather than the original sentence.
    resp = client.post("/chat", json={"session_id": session_id, "message": "hello?"})
    assert resp.json()["state"]["escalated"] is True
    assert resp.json()["reply"] != ESCALATION_TERMINAL_MESSAGE
    assert resp.json()["reply"] in ESCALATION_FOLLOWUP_VARIANTS


def test_3b_sustained_verification_friction_auto_escalates(client, session_id):
    from app.models import FRICTION_HARD_ESCALATION_THRESHOLD

    frustrated_message = "I already told you who I am. This is ridiculous. Just tell me why my claim was denied."
    body = None
    for _ in range(FRICTION_HARD_ESCALATION_THRESHOLD):
        resp = client.post("/chat", json={"session_id": session_id, "message": frustrated_message})
        body = resp.json()
    assert body["state"]["escalated"] is True
    assert body["state"]["verified"] is False


def test_escalation_is_never_mistaken_for_off_topic(client, session_id):
    resp = client.post("/chat", json={"session_id": session_id, "message": "I need to talk to someone"})
    body = resp.json()
    assert body["state"]["escalated"] is True
    assert body["state"]["off_topic_count"] == 0
    assert "only able to help" not in body["reply"].lower()


def test_no_normal_sop_progression_after_escalation(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": "transfer me to a human"})
    resp = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "message": "My name is Margaret Chen, DOB 1985-03-15, SSN last four 4472",
        },
    )
    body = resp.json()
    # Even a fully-verifiable identity submitted after escalation must not
    # silently resume the normal flow.
    assert body["state"]["escalated"] is True
    assert body["reply"] != ESCALATION_TERMINAL_MESSAGE
    assert body["reply"] in ESCALATION_FOLLOWUP_VARIANTS
    assert body["state"]["verified"] is False
