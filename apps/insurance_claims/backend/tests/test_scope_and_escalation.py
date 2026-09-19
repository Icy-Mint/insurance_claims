OFF_TOPIC = "Off topic, but what is RL?"


def test_out_of_scope_is_redirected_without_disrupting_phase(client, session_id):
    resp = client.post("/chat", json={"session_id": session_id, "message": OFF_TOPIC})
    body = resp.json()
    assert body["state"]["off_topic_count"] == 1
    assert body["state"]["phase"] == "VERIFY_ID"
    assert "insurance claim" in body["reply"].lower()
    assert "quick_replies" not in body["ui_hints"]


def test_repeated_off_topic_offers_human_transfer(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": OFF_TOPIC})
    resp = client.post("/chat", json={"session_id": session_id, "message": "also, who won the game last night?"})
    body = resp.json()
    assert body["state"]["off_topic_count"] == 2
    assert body["state"]["escalation_offered"] is True
    assert "Transfer me to a human" in body["ui_hints"]["quick_replies"]


def test_returning_to_scope_resets_the_counter(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": OFF_TOPIC})
    resp = client.post(
        "/chat", json={"session_id": session_id, "message": "My name is Margaret Chen, my policy is POL-9921."}
    )
    assert resp.json()["state"]["off_topic_count"] == 0


def test_partial_id_answers_and_refusal_do_not_verify(client, session_id):
    resp = client.post("/chat", json={"session_id": session_id, "message": "I'd rather not share my SSN."})
    body = resp.json()
    assert body["state"]["verified"] is False
    assert body["state"]["verification_friction_count"] >= 1
    assert body["state"]["phase"] == "VERIFY_ID"
