#!/usr/bin/env python3
"""SKO-026 offline governed directory review-feedback importer.

The module has no remote, write, apply, publication, or Airtable client path.
It validates immutable feedback rows against caller-supplied accepted artifacts,
appends accepted decisions to a ledger, and derives current state from history.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "sko-026-directory-feedback-v1"
FINGERPRINT_VERSION = "airtable-profile-fingerprint-v1"
RELATIONSHIP_TYPES = {"group", "division", "service_line", "brand", "establishment", "operating_unit"}
DECISIONS = {
    "crosswalk": {"approve", "reject", "hold", "revoke"},
    "relationship": {"approve", "reject", "hold", "revoke"},
    "counting_mapping": {"approve", "reject", "revoke"},
    "readiness": {"directory_ready", "not_ready", "research_needed", "excluded", "needs_update", "hold"},
    "new_profile_candidate": {"approve_future_creation", "reject", "hold"},
}
FIELDS = (
    "schema_version", "feedback_id", "feedback_batch_id", "target_type", "target_id",
    "decision", "reviewer", "decision_at", "reason", "source_review_file",
    "source_review_file_hash", "related_batch_id", "related_proposal_id", "crosswalk_id",
    "airtable_record_id", "entity_id", "relationship_type", "counting_entity_id",
    "fingerprint_version", "prior_state_hash", "supersedes_feedback_id",
)
RECORD_ID = re.compile(r"^rec[A-Za-z0-9]+$")
HASH = re.compile(r"^[0-9a-f]{64}$")


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def normalize(row: Mapping[str, Any]) -> dict[str, str]:
    """Return the complete normalized, hash-stable v1 payload."""
    return {field: _clean(row.get(field)) for field in FIELDS}


def logical_target(row: Mapping[str, str]) -> str:
    return f"{row['target_type']}:{row['target_id']}"


def derive_current_state(ledger: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Derive one current decision per logical target from append-only history."""
    rows = [dict(row) for row in ledger]
    by_id = {str(row["feedback_id"]): row for row in rows}
    superseded = {str(row.get("supersedes_feedback_id", "")) for row in rows if row.get("supersedes_feedback_id")}
    current = [row for fid, row in by_id.items() if fid not in superseded]
    return sorted(current, key=lambda row: (str(row["logical_target"]), str(row["feedback_id"])))


def current_state_hash(row: Mapping[str, Any] | None) -> str:
    if not row:
        return content_hash(None)
    governed = {key: row.get(key, "") for key in FIELDS}
    governed["payload_hash"] = row.get("payload_hash", "")
    return content_hash(governed)


def _valid_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return "T" in value and parsed.tzinfo is not None
    except (TypeError, ValueError):
        return False


def _reason(row: Mapping[str, str], context: Mapping[str, Any]) -> str:
    target_type, target_id = row["target_type"], row["target_id"]
    if row["schema_version"] != SCHEMA_VERSION:
        return "unsupported_schema_version"
    if not row["feedback_id"] or not row["feedback_batch_id"] or not target_id:
        return "missing_required_identifier"
    if target_type not in DECISIONS or row["decision"] not in DECISIONS[target_type]:
        return "unsupported_target_or_decision"
    if not row["reviewer"]:
        return "missing_reviewer"
    if not _valid_timestamp(row["decision_at"]):
        return "invalid_decision_timestamp"
    if not HASH.fullmatch(row["source_review_file_hash"]):
        return "invalid_source_review_file_hash"
    if row["relationship_type"] and row["relationship_type"] not in RELATIONSHIP_TYPES:
        return "unsupported_relationship_type"

    targets = context.get("targets", {}).get(target_type, {})
    target = targets.get(target_id)
    if target is None:
        return "unknown_target_id"
    if _clean(target.get("status")) in {"retired", "superseded"}:
        return "retired_or_superseded_target"

    expected_entity = _clean(target.get("entity_id"))
    if row["entity_id"] and row["entity_id"] != expected_entity:
        return "entity_id_mismatch"
    expected_relationship = _clean(target.get("relationship_type"))
    if row["relationship_type"] and row["relationship_type"] != expected_relationship:
        return "relationship_type_mismatch"

    existing_profile = bool(_clean(target.get("airtable_record_id")))
    if existing_profile:
        record_id = row["airtable_record_id"]
        if not RECORD_ID.fullmatch(record_id):
            return "malformed_airtable_record_id"
        if record_id != _clean(target.get("airtable_record_id")):
            return "airtable_record_id_mismatch"
        if row["fingerprint_version"] != FINGERPRINT_VERSION:
            return "wrong_fingerprint_version"
        if _clean(target.get("approved_fingerprint")) != _clean(target.get("current_fingerprint")):
            return "stale_fingerprint"
    elif row["airtable_record_id"]:
        return "fabricated_airtable_record_id"

    if target_type == "crosswalk" and row["crosswalk_id"] != target_id:
        return "crosswalk_id_mismatch"
    if target_type == "new_profile_candidate" and row["decision"] == "approve_future_creation" and row["airtable_record_id"]:
        return "fabricated_airtable_record_id"
    if target_type == "counting_mapping":
        counting = row["counting_entity_id"] or row["entity_id"]
        valid_entities = set(context.get("valid_entity_ids", ()))
        if row["entity_id"] not in valid_entities or counting not in valid_entities:
            return "invalid_counting_entity_id"
        expected = _clean(target.get("counting_entity_id")) or row["entity_id"]
        if counting != expected:
            return "invalid_non_default_counting_mapping"
        if counting != row["entity_id"] and row["decision"] != "approve":
            return "non_default_counting_mapping_not_approved"
    return ""


def import_feedback(
    feedback_rows: Iterable[Mapping[str, Any]], *, context: Mapping[str, Any],
    existing_ledger: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Validate feedback and return append-only history, current state and QA.

    Rows are evaluated in canonical order. Rejected rows never enter the ledger.
    Existing ledger entries are never edited.
    """
    inputs = [normalize(row) for row in feedback_rows]
    ledger = [dict(row) for row in existing_ledger]
    existing_by_id = {str(row["feedback_id"]): row for row in ledger}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    idempotent = 0

    # A batch cannot carry more than one distinct decision for one target.
    batch_claims: dict[tuple[str, str], set[str]] = {}
    for row in inputs:
        batch_claims.setdefault((row["feedback_batch_id"], logical_target(row)), set()).add(row["decision"])
    conflicting = {key for key, values in batch_claims.items() if len(values) > 1}

    for row in sorted(inputs, key=lambda item: (item["feedback_id"], _canonical(item))):
        payload_hash = content_hash(row)
        old = existing_by_id.get(row["feedback_id"])
        if old:
            if old.get("payload_hash") == payload_hash:
                idempotent += 1
            else:
                rejected.append({**row, "rejection_reason": "feedback_id_payload_conflict"})
            continue
        if (row["feedback_batch_id"], logical_target(row)) in conflicting:
            rejected.append({**row, "rejection_reason": "conflicting_same_batch_feedback"})
            continue
        reason = _reason(row, context)
        current_by_target = {item["logical_target"]: item for item in derive_current_state(ledger)}
        prior = current_by_target.get(logical_target(row))
        supersedes = row["supersedes_feedback_id"]
        if not reason and row["prior_state_hash"] and row["prior_state_hash"] != current_state_hash(prior):
            reason = "prior_state_hash_mismatch"
        if not reason and supersedes:
            superseded = existing_by_id.get(supersedes)
            if not superseded:
                reason = "unknown_superseded_feedback_id"
            elif superseded.get("logical_target") != logical_target(row):
                reason = "supersession_target_mismatch"
            elif not prior or prior.get("feedback_id") != supersedes:
                reason = "supersession_not_current"
            elif superseded.get("decision") == row["decision"]:
                reason = "unnecessary_supersession"
        elif not reason and prior:
            if prior.get("decision") != row["decision"]:
                reason = "contradictory_decision_requires_supersession"
            else:
                reason = "current_decision_already_exists"
        if reason:
            rejected.append({**row, "rejection_reason": reason})
            continue
        entry = {**row, "logical_target": logical_target(row), "payload_hash": payload_hash}
        ledger.append(entry)
        existing_by_id[row["feedback_id"]] = entry
        accepted.append(entry)

    ledger = sorted(ledger, key=lambda row: (str(row.get("decision_at", "")), str(row["feedback_id"])))
    rejected = sorted(rejected, key=lambda row: (row["feedback_id"], row["rejection_reason"], _canonical(row)))
    current = derive_current_state(ledger)
    types = sorted(DECISIONS)
    evidence = {
        "feedback_rows_received": len(inputs), "feedback_rows_accepted": len(accepted),
        "feedback_rows_rejected_or_held": len(rejected),
        "decisions_by_type": {name: sum(row["target_type"] == name for row in accepted) for name in types},
        "approved_rejected_crosswalk_decisions": sum(row["target_type"] == "crosswalk" and row["decision"] in {"approve", "reject"} for row in accepted),
        "relationship_decisions": sum(row["target_type"] == "relationship" for row in accepted),
        "counting_decisions": sum(row["target_type"] == "counting_mapping" for row in accepted),
        "readiness_decisions": sum(row["target_type"] == "readiness" for row in accepted),
        "new_profile_decisions": sum(row["target_type"] == "new_profile_candidate" for row in accepted),
        "stale_feedback_blocks": sum(row["rejection_reason"] in {"stale_fingerprint", "wrong_fingerprint_version", "prior_state_hash_mismatch"} for row in rejected),
        "duplicate_conflict_blocks": sum("conflict" in row["rejection_reason"] or row["rejection_reason"] == "current_decision_already_exists" for row in rejected),
        "malformed_target_blocks": sum(row["rejection_reason"] in {"unknown_target_id", "malformed_airtable_record_id", "airtable_record_id_mismatch"} for row in rejected),
        "superseded_decisions": sum(bool(row.get("supersedes_feedback_id")) for row in accepted),
        "current_governed_decisions": len(current), "idempotent_duplicates": idempotent,
        "invalid_counting_mappings": sum("counting" in row["rejection_reason"] for row in rejected),
        "unsupported_relationship_type_blocks": sum(row["rejection_reason"] == "unsupported_relationship_type" for row in rejected),
        "fabricated_record_ids": sum(row["rejection_reason"] == "fabricated_airtable_record_id" for row in rejected),
        "sko024_ineligible_new_profile_candidates": sum(row["target_type"] == "new_profile_candidate" and row["decision"] == "approve_future_creation" for row in accepted),
        "protected_field_mutation_attempts": 0, "publication_mutation_attempts": 0,
        "network_calls": 0, "write_capability": False, "apply_capability": False,
    }
    evidence["governed_state_hash"] = content_hash({"ledger": ledger, "current": current, "rejected": rejected})
    return {"ledger": ledger, "accepted": accepted, "rejected": rejected, "current_state": current, "evidence": evidence}
