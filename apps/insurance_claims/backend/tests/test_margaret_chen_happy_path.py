"""Validates the spec's primary test case end to end, against the
deterministic fallback responder (no ANTHROPIC_API_KEY needed)."""
from app import tools as tools_module

MARGARET_MESSAGE = (
    "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about "
    "my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472."
)

# Facts that belong to OTHER claims and must never leak into the CL-2048 reply.
OTHER_CLAIM_FACTS = ["CL-2011", "CL-1899", "CL-2102", "780.00", "425.00", "3200.00"]


def test_margaret_chen_full_flow(client, session_id):
    resp = client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    assert resp.status_code == 200
    body = resp.json()
    state = body["state"]

    # Verification: exactly the 3 countable fields present in this message
    # (full_name, dob, ssn_last4) get counted; policy_number is a lookup key,
    # not a countable field.
    assert state["verified"] is True
    assert set(state["verified_fields"]) == {"full_name", "dob", "ssn_last4"}
    assert state["party_id"] == "P9"

    # The intent/case hint from "denied healthcare claim from January" was
    # stashed and used to resolve the case without asking again.
    assert state["memory"]["case_type_hint"] == "healthcare"
    assert state["memory"]["status_hint"] == "denied"
    assert state["memory"]["month_hint"] == "january"
    assert state["memory"]["resolved_case_id"] == "CL-2048"

    # By now the cascade has carried us through PROCESS_CASE into POST_PROCESS
    # within this single turn, and the reply narrates CL-2048 using only
    # grounded facts from claims.json / required_document_guideline.json.
    assert state["phase"] == "POST_PROCESS"
    reply = body["reply"]
    assert "pathology report" in reply
    assert "office note" in reply
    assert "2026-03-18" in reply
    for fact in OTHER_CLAIM_FACTS:
        assert fact not in reply

    # It must offer the email summary but not have sent anything yet.
    assert state["email_sent"] is False
    assert tools_module.EMAIL_LOG == []
    assert body["ui_hints"].get("quick_replies") == ["Yes, email me", "No thanks"]


def test_post_process_only_sends_on_explicit_yes(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})

    # A non-yes reply must not trigger a send.
    resp = client.post("/chat", json={"session_id": session_id, "message": "hmm not sure"})
    assert resp.json()["state"]["email_sent"] is False
    assert tools_module.EMAIL_LOG == []

    # Explicit yes sends (logs) exactly one summary, grounded in the claim.
    resp = client.post("/chat", json={"session_id": session_id, "message": "yes"})
    state = resp.json()["state"]
    assert state["email_sent"] is True
    assert state["consent"]["email"] == "yes"
    assert len(tools_module.EMAIL_LOG) == 1
    assert tools_module.EMAIL_LOG[0]["session_id"] == session_id
    assert "CL-2048" in tools_module.EMAIL_LOG[0]["summary"]


def test_phase_stays_verify_id_until_three_fields_matched(client, session_id):
    r1 = client.post("/chat", json={"session_id": session_id, "message": "My name is Margaret Chen."})
    s1 = r1.json()["state"]
    assert s1["phase"] == "VERIFY_ID"
    assert s1["verified"] is False
    assert s1["verified_fields"] == ["full_name"]

    r2 = client.post("/chat", json={"session_id": session_id, "message": "My date of birth is 1985-03-15."})
    s2 = r2.json()["state"]
    assert s2["phase"] == "VERIFY_ID"
    assert s2["verified"] is False
    assert set(s2["verified_fields"]) == {"full_name", "dob"}

    r3 = client.post(
        "/chat", json={"session_id": session_id, "message": "The last four digits of my SSN are 4472."}
    )
    s3 = r3.json()["state"]
    assert s3["verified"] is True
    assert s3["phase"] != "VERIFY_ID"
