"""Regression test: verify_identity() must be a real exact-match lookup
against policyholders.json, and a wrong SSN/DOB must be explicitly rejected
by name — never silently "noted" as if it had been accepted."""


def test_wrong_dob_and_ssn_are_explicitly_rejected(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": "My name is Margaret Chen"})

    # Wrong DOB (real value in policyholders.json is 1985-03-15).
    resp = client.post("/chat", json={"session_id": session_id, "message": "DOB is 1985-03-14"})
    body = resp.json()
    assert body["state"]["verified_fields"] == ["full_name"]
    assert body["state"]["verified"] is False
    reply_lower = body["reply"].lower()
    assert "match" in reply_lower and ("doesn't match" in reply_lower or "does not match" in reply_lower)
    assert "note that down" not in reply_lower
    assert "let me note" not in reply_lower

    # Wrong SSN last-4 (real value is 4472).
    resp = client.post("/chat", json={"session_id": session_id, "message": "SSN last four is 4473"})
    body = resp.json()
    assert body["state"]["verified"] is False
    assert set(body["state"]["verified_fields"]) == {"full_name"}
    reply_lower = body["reply"].lower()
    assert "doesn't match" in reply_lower or "does not match" in reply_lower

    # A record was located to compare against, confirming the lookup ran
    # against real fixture data rather than never actually being called.
    assert body["state"]["party_id"] == "P9"


def test_verify_identity_calls_are_logged_with_input_and_output(client, session_id):
    from app.store import get_or_create_session

    client.post("/chat", json={"session_id": session_id, "message": "My name is Margaret Chen"})
    client.post("/chat", json={"session_id": session_id, "message": "SSN last four is 4473"})  # wrong

    state = get_or_create_session(session_id)
    verify_calls = [r for r in state.tool_log if r.tool == "verify_identity"]
    assert len(verify_calls) == 2

    last_call = verify_calls[-1]
    assert last_call.args["ssn_last4"] == "4473"
    assert "ssn_last4" not in last_call.result["matched_fields"]
    assert last_call.result["party_id"] == "P9"


def test_correct_fields_after_a_mismatch_still_verify(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": "My name is Margaret Chen"})
    client.post("/chat", json={"session_id": session_id, "message": "DOB is 1985-03-14"})  # wrong
    resp = client.post("/chat", json={"session_id": session_id, "message": "Actually, DOB is 1985-03-15"})
    body = resp.json()
    assert set(body["state"]["verified_fields"]) == {"full_name", "dob"}

    resp = client.post("/chat", json={"session_id": session_id, "message": "SSN last four is 4472"})
    body = resp.json()
    assert body["state"]["verified"] is True
    assert set(body["state"]["verified_fields"]) == {"full_name", "dob", "ssn_last4"}
