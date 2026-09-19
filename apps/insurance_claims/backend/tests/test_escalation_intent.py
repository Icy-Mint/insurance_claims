"""Regression tests for the escalation-intent bug: "talk to a human" must be
a first-class, terminal state transition — never swallowed by the off-topic
classifier, never swallowed by unrelated consent parsing, and never left to
loop forever on repeated identical requests.

Follow-up-reply behavior (updated): the ESCALATION_TERMINAL_MESSAGE itself
("I'm transferring you now... please hold") is shown exactly once, on the
triggering turn. Every turn after that gets a distinct closing
acknowledgment from prompts.ESCALATION_FOLLOWUP_VARIANTS instead — never
the original transfer sentence again (that reads as a stuck loop), and
never a freshly-generated model narration either (that risks contradicting
itself). A repeated identical follow-up gets the same acknowledgment back;
a different one advances to the next variant."""
from app import tools as tools_module
from app.prompts import ESCALATION_FOLLOWUP_VARIANTS, ESCALATION_TERMINAL_MESSAGE


def test_transfer_request_escalates_exactly_once_and_does_not_loop(client, session_id):
    resp = client.post("/chat", json={"session_id": session_id, "message": "transfer me to a human"})
    body = resp.json()
    assert body["state"]["escalated"] is True
    assert body["reply"] == ESCALATION_TERMINAL_MESSAGE

    # Repeating the identical request must never re-show the original
    # transfer sentence, and must never re-run any SOP logic — but since
    # the input is identical each time, the follow-up acknowledgment stays
    # consistent across repeats too (no fresh phrasing invented for no reason).
    replies = set()
    for _ in range(3):
        resp = client.post("/chat", json={"session_id": session_id, "message": "transfer me to a human"})
        body = resp.json()
        assert body["state"]["escalated"] is True
        assert body["reply"] != ESCALATION_TERMINAL_MESSAGE
        assert body["reply"] in ESCALATION_FOLLOWUP_VARIANTS
        replies.add(body["reply"])
    assert len(replies) == 1


def test_escalation_request_not_swallowed_by_off_topic_classifier(client, session_id):
    # No insurance keyword here at all — this used to be misclassified as
    # generic off-topic and refused outright (conversation 5, failure a).
    resp = client.post("/chat", json={"session_id": session_id, "message": "I need to talk to someone"})
    body = resp.json()
    assert body["state"]["escalated"] is True
    assert body["state"]["off_topic_count"] == 0
    assert "only able to help" not in body["reply"].lower()


def test_escalation_request_not_swallowed_by_consent_parsing(client, session_id):
    # A "yes" that's actually answering an escalation offer, not the email
    # question, must not be misrouted into sending the email (conversation
    # 5, failure a: "yes, i would like to talk to human representative").
    margaret_message = (
        "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about "
        "my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472."
    )
    client.post("/chat", json={"session_id": session_id, "message": margaret_message})
    resp = client.post(
        "/chat", json={"session_id": session_id, "message": "yes, i would like to talk to human representative"}
    )
    body = resp.json()
    assert body["state"]["escalated"] is True
    assert body["state"]["email_sent"] is False
    assert tools_module.EMAIL_LOG == []
