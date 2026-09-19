"""Regression test for the debug/state panel feature: the /chat response
must expose a debug_state field driven straight off the backend
SessionState object (phase, verified fields + count, memory, escalated,
counters, and a summary of the last tool call) so phase-by-phase
enforcement is verifiable in the UI itself, not just inferable from the
chat transcript.
"""
MARGARET_MESSAGE = (
    "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about "
    "my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472."
)


def test_debug_state_present_and_matches_backend_state_verify_id(client, session_id):
    resp = client.post("/chat", json={"session_id": session_id, "message": "My name is Margaret Chen."})
    body = resp.json()
    debug = body["debug_state"]
    assert debug["phase"] == "VERIFY_ID"
    assert debug["verified"] is False
    assert debug["verified_summary"] == "1/3: name"
    assert debug["escalated"] is False
    assert debug["off_topic_count"] == 0
    assert debug["verification_friction_count"] == 0
    assert debug["last_tool_call"].startswith("verify_identity() -> match_count=")


def test_debug_state_reflects_full_verification_and_case_resolution(client, session_id):
    resp = client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    body = resp.json()
    debug = body["debug_state"]

    assert debug["phase"] == "POST_PROCESS"
    assert debug["verified"] is True
    assert debug["verified_summary"] == "3/3: name, dob, ssn"
    assert debug["party_id"] == "P9"
    assert debug["memory"]["resolved_case_id"] == "CL-2048"
    assert debug["escalated"] is False


def test_debug_state_reflects_email_send(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    resp = client.post("/chat", json={"session_id": session_id, "message": "yes"})
    body = resp.json()
    debug = body["debug_state"]
    assert debug["email_sent"] is True
    assert debug["last_tool_call"] == "send_email_summary() -> sent"


def test_debug_state_reflects_escalation(client, session_id):
    resp = client.post("/chat", json={"session_id": session_id, "message": "transfer me to a human"})
    body = resp.json()
    assert body["debug_state"]["escalated"] is True
