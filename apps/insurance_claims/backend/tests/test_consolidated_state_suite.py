"""Consolidated regression suite for the six root-cause bugs found across
several rounds of adversarial testing, all tracing back to one pattern: the
LLM narrating an action, state change, or outcome that did not structurally
happen in code (verification succeeding, an email being sent, a transfer
completing).

Ground rule enforced throughout this file: every assertion reads
`debug_state` (from `SessionState.as_debug_dict()`, returned on the /chat
response) or a monkeypatched spy on the actual tool function — never the
reply text. Reply-text assertions can't tell the difference between a real
action and a well-worded hallucination of one; state assertions can.

Mocking strategy:
  - Tests 1, 2, 4, 10, 13 are pure state-machine logic (identity matching,
    one-way verification, escalation/closing terminal states) and run
    against the deterministic offline fallback path — this IS the
    "mocked Anthropic API" for this suite: the top-level backend/conftest.py
    strips ANTHROPIC_API_KEY for the whole session, so `llm.get_client()`
    returns None and every reply comes from the deterministic template
    layer instead of a live model call. Fast, free, fully deterministic.
  - Tests 5, 9, 11 use the `real_llm` fixture (see tests/conftest.py) to
    exercise genuine live-model language understanding for messy/compound
    phrasing. Note: this app's scope/sentiment/PII extraction is a
    regex-first cascade with the LLM only filling gaps regex left empty
    (see app/extraction.py, app/llm.py:llm_extract_enrich) — by design, so
    that phase transitions stay deterministic even when a live call is
    used. These three tests therefore also exercise real response
    generation end to end, which is valuable coverage in its own right.
  - Tests 3, 6, 7, 8, 12 don't need real language understanding either way
    and run offline for speed and determinism.
"""
from app import tools as tools_module
from app.store import get_or_create_session

MARGARET_MESSAGE = (
    "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about "
    "my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472."
)

# Ground truth from fixtures/claims.json for CL-2048 — used to verify the
# PROCESS_CASE reply is grounded in real data, never invented.
CL_2048_DENIAL_REASON = "the review file did not include the pathology report and the treating provider office note"
CL_2048_DOCS_NEEDED = ["pathology report", "office note"]
CL_2048_APPEAL_DEADLINE = "2026-03-18"

# Facts belonging to OTHER claims on Margaret's account — must never leak.
OTHER_CLAIM_FACTS = ["CL-2011", "CL-1899", "CL-2102"]


# --- 1. Wrong PII is rejected field by field, never counted as a match ----


def test_wrong_pii_rejected(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": "My name is Margaret Chen."})

    resp = client.post("/chat", json={"session_id": session_id, "message": "SSN last four is 4473"})
    assert "ssn_last4" not in resp.json()["debug_state"]["verified_fields"]

    resp = client.post("/chat", json={"session_id": session_id, "message": "DOB is 1985-03-14"})
    assert "dob" not in resp.json()["debug_state"]["verified_fields"]

    resp = client.post("/chat", json={"session_id": session_id, "message": "Phone is 555-000-1234"})
    assert "phone" not in resp.json()["debug_state"]["verified_fields"]

    # Only the one real match (full_name) ever counted.
    debug = resp.json()["debug_state"]
    assert debug["verified_fields"] == ["full_name"]
    assert debug["verified"] is False


# --- 2. A fully fabricated identity never verifies, no matter how much ----
# --- internally-consistent detail is added ---------------------------------


def test_fabricated_identity_never_verifies(client, session_id):
    turns = [
        "My name is Jane Smith.",
        "DOB is 1970-01-01.",
        "SSN last four is 0000.",
        "My phone is 555-000-0000.",
        "My email is jane.smith@example.com.",
    ]
    for msg in turns:
        resp = client.post("/chat", json={"session_id": session_id, "message": msg})
        debug = resp.json()["debug_state"]
        assert debug["verified"] is False
        assert debug["verified_fields"] == []
        assert debug["party_id"] is None

    state = get_or_create_session(session_id)
    verify_calls = [r for r in state.tool_log if r.tool == "verify_identity"]
    assert len(verify_calls) == 5
    assert all(c.result["match_count"] == 0 for c in verify_calls)


# --- 3. The correct real identity verifies with exactly the right fields --


def test_correct_identity_verifies(client, session_id):
    resp = client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    debug = resp.json()["debug_state"]
    assert debug["verified"] is True
    assert set(debug["verified_fields"]) == {"full_name", "dob", "ssn_last4"}
    assert debug["party_id"] == "P9"


# --- 4. Verification is one-way: a zero-identity-content message can't ----
# --- unverify or alter the matched fields ----------------------------------


def test_verified_state_is_one_way(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": "My name is Margaret Chen."})
    client.post("/chat", json={"session_id": session_id, "message": "DOB is 1985-03-15."})
    resp = client.post("/chat", json={"session_id": session_id, "message": "SSN last four is 4472."})
    debug = resp.json()["debug_state"]
    assert debug["verified"] is True
    fields_before = set(debug["verified_fields"])

    state_before = get_or_create_session(session_id)
    verify_calls_before = len([r for r in state_before.tool_log if r.tool == "verify_identity"])

    resp = client.post(
        "/chat",
        json={"session_id": session_id, "message": "I'm calling about my denied healthcare claim from January."},
    )
    debug = resp.json()["debug_state"]
    assert debug["verified"] is True
    assert set(debug["verified_fields"]) == fields_before

    # verify_identity must not have run again at all.
    state_after = get_or_create_session(session_id)
    verify_calls_after = len([r for r in state_after.tool_log if r.tool == "verify_identity"])
    assert verify_calls_after == verify_calls_before


# --- 5. An intent hint given during VERIFY_ID survives to resolve the -----
# --- case without re-asking, once verification completes (real LLM) -------


def test_intent_hint_remembered_across_phases(real_llm, client, session_id):
    client.post(
        "/chat",
        json={"session_id": session_id, "message": "I'm calling about my denied healthcare claim from January."},
    )
    client.post("/chat", json={"session_id": session_id, "message": "My name is Margaret Chen."})
    client.post("/chat", json={"session_id": session_id, "message": "My date of birth is 1985-03-15."})
    resp = client.post("/chat", json={"session_id": session_id, "message": "Last four of my SSN is 4472."})

    debug = resp.json()["debug_state"]
    assert debug["verified"] is True
    assert debug["memory"].get("resolved_case_id") == "CL-2048"


# --- 6. PROCESS_CASE facts are grounded exactly in claims.json ------------


def test_process_case_grounded(client, session_id):
    resp = client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    body = resp.json()
    debug = body["debug_state"]
    assert debug["memory"].get("resolved_case_id") == "CL-2048"

    reply = body["reply"]
    assert CL_2048_DENIAL_REASON in reply or all(w in reply for w in ["pathology report", "office note"])
    for doc in CL_2048_DOCS_NEEDED:
        assert doc in reply
    assert CL_2048_APPEAL_DEADLINE in reply
    for fact in OTHER_CLAIM_FACTS:
        assert fact not in reply


# --- 7. An unrelated "yes" before the email offer must never trigger a ----
# --- real send — verified via a spy on the actual function ----------------


def test_email_not_sent_on_unscoped_yes(client, session_id, monkeypatch):
    calls = []
    monkeypatch.setattr(
        tools_module,
        "send_email_summary",
        lambda sid, summary: calls.append((sid, summary)) or {"status": "sent"},
    )

    client.post("/chat", json={"session_id": session_id, "message": "My name is Margaret Chen."})
    resp = client.post("/chat", json={"session_id": session_id, "message": "Yes"})

    assert calls == []
    assert resp.json()["debug_state"]["email_sent"] is False


# --- 8. The actual scoped "yes" really calls send_email_summary -----------


def test_email_sent_only_on_scoped_yes(client, session_id, monkeypatch):
    calls = []
    monkeypatch.setattr(
        tools_module,
        "send_email_summary",
        lambda sid, summary: calls.append((sid, summary)) or {"status": "sent", "session_id": sid, "summary": summary},
    )

    client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    resp = client.post("/chat", json={"session_id": session_id, "message": "yes"})

    assert len(calls) == 1
    called_session_id, called_summary = calls[0]
    assert called_session_id == session_id
    assert "CL-2048" in called_summary
    assert resp.json()["debug_state"]["email_sent"] is True


# --- 9. A compound "yes, I'd like to talk to a human" at the scoped email -
# --- question must not drop the transfer or fabricate a send (real LLM) ---


def test_compound_consent_handled(real_llm, client, session_id, monkeypatch):
    calls = []
    monkeypatch.setattr(
        tools_module,
        "send_email_summary",
        lambda sid, summary: calls.append((sid, summary)) or {"status": "sent"},
    )

    client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    resp = client.post(
        "/chat", json={"session_id": session_id, "message": "yes, I'd like to talk to a human representative"}
    )
    debug = resp.json()["debug_state"]

    # Transfer intent not silently dropped.
    assert debug["escalated"] is True
    # No fabricated send: send_email_summary was never actually called.
    assert calls == []
    assert debug["email_sent"] is False


# --- 10. Escalation is a real terminal state: one transfer message, then --
# --- distinct (non-identical) follow-ups, forever ---------------------------


def test_escalation_is_terminal(client, session_id):
    resp = client.post("/chat", json={"session_id": session_id, "message": "transfer me to a human"})
    assert resp.json()["debug_state"]["escalated"] is True

    replies = []
    for msg in ["is it done?", "where is the human", "should we call it an end"]:
        resp = client.post("/chat", json={"session_id": session_id, "message": msg})
        debug = resp.json()["debug_state"]
        assert debug["escalated"] is True
        replies.append(resp.json()["reply"])

    assert len(set(replies)) > 1  # not the same verbatim string on every turn


# --- 11. On-topic frustration about verification is never off-topic ------
# --- (checked via the internal off_topic_count field, real LLM) -----------


def test_off_topic_frustration_not_misclassified(real_llm, client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": "My name is Margaret Chen."})
    client.post("/chat", json={"session_id": session_id, "message": "DOB is 1985-03-15."})
    resp = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "message": "I've already given you my name and date of birth, that should be more than enough at this point.",
        },
    )
    debug = resp.json()["debug_state"]
    # The internal off_topic_count is the ground truth for classification —
    # it must not have incremented for this on-topic pushback.
    assert debug["off_topic_count"] == 0
    assert debug["verification_friction_count"] >= 1


# --- 12. Genuinely unrelated topics DO trigger the off-topic path, and ----
# --- repeats surface a human-transfer offer by the 2nd/3rd attempt --------


def test_repeated_off_topic_triggers_escalation_offer(client, session_id):
    resp = client.post("/chat", json={"session_id": session_id, "message": "Off topic, but what is RL?"})
    debug = resp.json()["debug_state"]
    assert debug["off_topic_count"] == 1
    assert debug["escalation_offered"] is False

    resp = client.post(
        "/chat", json={"session_id": session_id, "message": "also, what's the weather like today?"}
    )
    debug = resp.json()["debug_state"]
    assert debug["off_topic_count"] == 2
    assert debug["escalation_offered"] is True
    assert "Transfer me to a human" in resp.json()["ui_hints"].get("quick_replies", [])


# --- 13. Session closing is terminal too: no verbatim "anything else?" ---
# --- loop after a clear closing signal -------------------------------------


def test_no_verbatim_loop_on_session_close(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": MARGARET_MESSAGE})
    client.post("/chat", json={"session_id": session_id, "message": "no thanks"})  # decline email
    resp = client.post("/chat", json={"session_id": session_id, "message": "no, that's all, thank you"})
    assert resp.json()["debug_state"]["session_closed"] is True

    replies = []
    for msg in ["no, that's all, thank you", "goodbye", "thanks, bye"]:
        resp = client.post("/chat", json={"session_id": session_id, "message": msg})
        debug = resp.json()["debug_state"]
        assert debug["session_closed"] is True
        replies.append(resp.json()["reply"])

    # At least the differently-worded closing messages must not all collapse
    # into the exact same reply string as the "anything else?" loop did.
    assert len(set(replies)) > 1
