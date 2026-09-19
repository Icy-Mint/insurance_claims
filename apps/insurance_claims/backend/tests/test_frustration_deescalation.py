"""Validates the spec's frustration/de-escalation scenario: the agent must
acknowledge feelings, explain the gate, offer alternatives, and never reveal
claim details or skip verification, no matter how insistent the caller is."""

FRUSTRATED_MESSAGE = (
    "I already told you who I am. This is ridiculous. Just tell me why my claim was denied."
)

# Facts that would only be knowable if verification had been bypassed.
LEAKED_FACT_MARKERS = [
    "pathology report",
    "office note",
    "2026-03-18",
    "CL-2048",
    "the review file did not include",
]


def test_frustration_does_not_bypass_verification(client, session_id):
    resp = client.post("/chat", json={"session_id": session_id, "message": FRUSTRATED_MESSAGE})
    body = resp.json()
    state = body["state"]

    assert state["phase"] == "VERIFY_ID"
    assert state["verified"] is False
    assert state["sentiment_history"][-1] in ("frustrated", "angry")
    assert state["verification_friction_count"] >= 1

    reply_lower = body["reply"].lower()
    for marker in LEAKED_FACT_MARKERS:
        assert marker.lower() not in reply_lower

    # De-escalation: acknowledges feelings, explains the gate, offers an
    # alternative before repeating the request.
    assert any(w in reply_lower for w in ["hear you", "sorry", "understand"])
    assert "verify" in reply_lower
    assert any(w in reply_lower for w in ["transfer", "different"])


def test_repeated_friction_offers_human_transfer_without_ever_leaking(client, session_id):
    for _ in range(3):
        resp = client.post("/chat", json={"session_id": session_id, "message": FRUSTRATED_MESSAGE})

    body = resp.json()
    state = body["state"]
    assert state["verified"] is False
    assert state["phase"] == "VERIFY_ID"
    assert state["escalation_offered"] is True
    assert "Transfer me to a human" in body["ui_hints"].get("quick_replies", [])

    reply_lower = body["reply"].lower()
    for marker in LEAKED_FACT_MARKERS:
        assert marker.lower() not in reply_lower
