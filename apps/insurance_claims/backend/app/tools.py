"""Fixed, deterministic tool functions over the fixture JSON.

The LLM never generates queries against these files — it only triggers one
of these named functions (or, in the fallback/no-API-key path, the backend
calls them directly). Every fact that ever reaches a caller during
PROCESS_CASE must have come from one of these return values.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"

# Visible logging for every real tool call (bug fix: identity checks and
# email sends must be observable, not just trusted on the LLM's say-so).
# A dedicated handler is attached (rather than relying on root logger
# config) so these lines show up in the console regardless of how the host
# process configures logging.
logger = logging.getLogger("insurance_claims.tools")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(_handler)
# Left as the default (True) so pytest's caplog fixture — which attaches its
# own handler up the logger hierarchy — can also observe these records.


def _load(name: str) -> Any:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


POLICYHOLDERS: List[Dict[str, Any]] = _load("policyholders.json")
CLAIMS: List[Dict[str, Any]] = _load("claims.json")
REPRESENTATIVES: List[Dict[str, Any]] = _load("representatives.json")
CONSENT_SCENARIOS: Dict[str, Any] = _load("consent_scenarios.json")
DOC_GUIDELINE: Dict[str, Any] = _load("required_document_guideline.json")

# Only ever mutated by the deterministic "send email" step, never by the LLM.
EMAIL_LOG: List[Dict[str, str]] = []

COUNTABLE_FIELDS = {"full_name", "dob", "phone", "email", "ssn_last4"}


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _norm_phone(s: Optional[str]) -> str:
    return re.sub(r"\D", "", s or "")[-10:]


def verify_identity(candidate_fields: Dict[str, str]) -> Dict[str, Any]:
    """Match candidate identity fields against policyholders.json.

    Returns the best-matching policyholder along with which countable fields
    matched. policy_number is used only to help locate a record; it never
    counts toward the verification threshold.
    """
    best_party_id: Optional[str] = None
    best_countable: List[str] = []
    best_all: List[str] = []

    for ph in POLICYHOLDERS:
        matched: List[str] = []

        if candidate_fields.get("policy_number") and _norm(candidate_fields["policy_number"]) == _norm(
            ph["policy_number"]
        ):
            matched.append("policy_number")

        if candidate_fields.get("full_name"):
            names = [ph["name"]] + ph.get("name_aliases", [])
            if _norm(candidate_fields["full_name"]) in {_norm(n) for n in names}:
                matched.append("full_name")

        if candidate_fields.get("dob") and candidate_fields["dob"] == ph["dob"]:
            matched.append("dob")

        if candidate_fields.get("phone"):
            phones = [ph["phone"]] + ph.get("phone_aliases", [])
            if _norm_phone(candidate_fields["phone"]) in {_norm_phone(p) for p in phones}:
                matched.append("phone")

        if candidate_fields.get("email"):
            emails = [ph["email"]] + ph.get("email_aliases", [])
            if _norm(candidate_fields["email"]) in {_norm(e) for e in emails}:
                matched.append("email")

        if candidate_fields.get("ssn_last4") and candidate_fields["ssn_last4"] == ph["id_last4"]:
            matched.append("ssn_last4")

        countable = [m for m in matched if m in COUNTABLE_FIELDS]
        if len(countable) > len(best_countable):
            best_party_id = ph["party_id"]
            best_countable = countable
            best_all = matched

    if best_party_id is None or not best_all:
        result = {"match_count": 0, "matched_fields": [], "party_id": None}
    else:
        result = {
            "match_count": len(best_countable),
            "matched_fields": best_countable,
            "party_id": best_party_id,
        }

    # This log line (input fields checked, which matched, resulting count)
    # is the visible audit trail proving this was a real exact-match lookup
    # against policyholders.json, not an LLM guess.
    logger.info(
        "verify_identity(candidate_fields=%s) -> matched_fields=%s match_count=%d party_id=%s",
        candidate_fields,
        result["matched_fields"],
        result["match_count"],
        result["party_id"],
    )
    return result


def check_representative(caller_name: str, party_id: str) -> Optional[Dict[str, Any]]:
    for rep in REPRESENTATIVES:
        if _norm(rep["rep_name"]) == _norm(caller_name) and rep["buyer_party_id"] == party_id:
            return rep
    return None


def get_claims_for_party(party_id: str) -> List[Dict[str, Any]]:
    return [c for c in CLAIMS if c["party_id"] == party_id]


def get_claim_detail(case_id: str) -> Optional[Dict[str, Any]]:
    for c in CLAIMS:
        if c["case_id"] == case_id:
            return c
    return None


def _find_matching_key(doc_name: str, keys: List[str]) -> Optional[str]:
    doc_words = set(_norm(doc_name).split())
    for k in keys:
        k_words = set(_norm(k).split())
        if doc_words.issubset(k_words):
            return k
    return None


def get_required_documents(case_type: str, missing_docs: List[str]) -> Dict[str, Any]:
    guidance = DOC_GUIDELINE
    documents = []
    for doc in missing_docs:
        guidance_key = _find_matching_key(doc, list(guidance["document_guidance"].keys()))
        alt_key = _find_matching_key(doc, list(guidance["document_alternative_guidance"].keys()))
        documents.append(
            {
                "name": doc,
                "guidance": guidance["document_guidance"].get(guidance_key, {}).get("en"),
                "alternative": guidance["document_alternative_guidance"]
                .get(alt_key, guidance["document_alternative_guidance"]["default"])
                .get("en"),
            }
        )
    return {
        "case_type_guidance": guidance["case_type_guidance"].get(case_type, {}).get("en"),
        "default_guidance": guidance["default_guidance"]["en"],
        "documents": documents,
    }


_TOPIC_KEYWORDS = {
    "submission_timing": ["how soon do i need to submit", "how soon do i need to send", "when do i need to submit",
                          "when do i need to send", "when should i submit", "when should i send"],
    "processing_time_after_submission": ["how long", "how soon", "processing time", "review time", "once i submit",
                                         "after i submit", "after i send", "after you receive",
                                         "how long does it take"],
    "submission_method": ["how do i submit", "where do i submit", "where should i send", "how should i send",
                          "portal", "upload link"],
    "file_format_requirements": ["format", "file type", "photo", "scan", "pdf", "image", "clear enough"],
    "receipt_confirmation": ["how do i know you got it", "will i get confirmation", "confirm receipt",
                             "show up in status"],
}


def get_follow_up_template(
    intent: str, case_id: str, documents: Optional[List[str]] = None, user_message: str = ""
) -> str:
    """Pick the most specific matching claim_followup_guidance entry."""
    guidance = DOC_GUIDELINE
    doc_str = ", ".join(documents or [])
    lowered = _norm(user_message)

    best_entry = None
    for entry in guidance["claim_followup_guidance"]:
        if intent not in entry.get("intent_hints", []):
            continue
        match_any = entry.get("match_any")
        if match_any:
            if any(phrase in lowered for phrase in match_any):
                best_entry = entry
                break
        elif best_entry is None:
            best_entry = entry

    if best_entry is None:
        text = guidance["claim_followup_fallback"]["en"]
    else:
        text = best_entry["en"]

    return text.format(
        case_id=case_id,
        documents=doc_str,
        average_processing_time_after_submission=guidance["claim_followup_settings"][
            "average_processing_time_after_submission"
        ]["en"],
    )


def get_consent_status(scenario: str, call_index: int = 0) -> str:
    seq = CONSENT_SCENARIOS.get(scenario, CONSENT_SCENARIOS["default"])["status_sequence"]
    idx = min(call_index, len(seq) - 1)
    return seq[idx]


def send_email_summary(session_id: str, summary_text: str) -> Dict[str, str]:
    """Simulated send: logs it to EMAIL_LOG and to the visible tool-call log.
    Never invoked by LLM choice or narrated speculatively — only called by
    the deterministic POST_PROCESS handler after an explicit, scoped 'yes',
    and the agent may only say a summary was sent after this returns."""
    record = {"session_id": session_id, "summary": summary_text}
    EMAIL_LOG.append(record)
    result = {"status": "sent", **record}
    logger.info("send_email_summary(session_id=%s) -> status=sent", session_id)
    return result
