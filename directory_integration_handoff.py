#!/usr/bin/env python3
"""SKO-024 deterministic, no-write directory operational handoff gate.

This module packages proposals for review by a future, separately authorised
adapter.  It has no remote client and no application or mutation entry point.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from directory_integration import PROPOSAL_FIELDS, PROTECTED_FIELDS


LIFECYCLE = ("PROPOSAL", "CANDIDATE_BATCH", "REVIEWABLE_BATCH", "APPROVED_BATCH", "HANDOFF_READY")
PUBLICATION_FIELDS = {"publication_status", "published", "publishable"}
APPLICATION_OUTCOMES = {
    "not_attempted", "applied", "already_in_desired_state", "failed", "stale_at_application"
}
BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")


class HandoffGateViolation(RuntimeError):
    """Raised when a no-write handoff safety gate fails."""


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _records(frame_or_records: Any) -> list[dict[str, Any]]:
    if hasattr(frame_or_records, "to_dict"):
        return frame_or_records.to_dict("records")
    return [dict(row) for row in frame_or_records]


def _canonical_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    normalized = [{str(key): _clean(value) for key, value in row.items()} for row in rows]
    return sorted(normalized, key=_canonical)


def _unique_index(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> tuple[dict[tuple[str, ...], dict[str, Any]], set[tuple[str, ...]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(_clean(row.get(field)) for field in fields), []).append(row)
    duplicates = {key for key, values in grouped.items() if len(values) > 1}
    return ({key: values[0] for key, values in grouped.items() if len(values) == 1}, duplicates)


def _valid_timestamp(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return "T" in value
    except (TypeError, ValueError):
        return False


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    disabled = ("network_enabled", "credentials_enabled", "mutation_enabled", "application_enabled")
    if config.get("mode") != "handoff_only" or any(config.get(key) is not False for key in disabled):
        raise HandoffGateViolation("Configuration must disable network, credentials, mutation and application")
    if config.get("allowed_proposal_fields") != PROPOSAL_FIELDS:
        raise HandoffGateViolation("Configured allowlist differs from the accepted eight fields")
    if not config.get("relationship_types") or "APPLIED" in config.get("lifecycle_states", []):
        raise HandoffGateViolation("Relationship types are required and APPLIED is forbidden")
    return config


def _change_set(before: Mapping[str, Any], proposed: Mapping[str, Any]) -> tuple[list[str], dict[str, str], dict[str, str]]:
    before_owned = {field: _clean(before.get(field)) for field in PROPOSAL_FIELDS}
    proposed_owned = {field: _clean(proposed.get(field)) for field in PROPOSAL_FIELDS}
    changed = [field for field in PROPOSAL_FIELDS if before_owned[field] != proposed_owned[field]]
    return changed, before_owned, proposed_owned


def build_candidate_batch(
    proposals: Any,
    integration_references: Any,
    decisions: Any,
    airtable_profiles: Any,
    crosswalks: Any = (),
    canonical_entities: Any = (),
    *,
    batch_id: str,
    created_at: str,
    created_by: str,
    config: Mapping[str, Any] | None = None,
    proposal_input_hash: str | None = None,
    config_version: str = "sko-024-v1",
) -> dict[str, Any]:
    """Bind SKO-022 proposals to accepted records and produce a deterministic manifest."""
    if not BATCH_ID_PATTERN.fullmatch(batch_id):
        raise HandoffGateViolation("Invalid or empty integration batch ID")
    if not _clean(created_by) or not _valid_timestamp(created_at):
        raise HandoffGateViolation("Creation metadata requires an actor and ISO-8601 timestamp")
    relationships = set((config or {}).get("relationship_types", ["group", "division", "service_line"]))
    config_version = _clean((config or {}).get("config_version", config_version))
    proposal_rows = sorted(_records(proposals), key=_canonical)
    references = _records(integration_references)
    decision_rows = _records(decisions)
    profiles = _records(airtable_profiles)
    crosswalk_rows = _records(crosswalks)
    entity_rows = _records(canonical_entities)
    refs_by_entity: dict[str, list[dict[str, Any]]] = {}
    for row in references:
        refs_by_entity.setdefault(_clean(row.get("entity_id")), []).append(row)
    decision_index, duplicate_decisions = _unique_index(decision_rows, ("candidate_id",))
    profile_index, duplicate_profiles = _unique_index(profiles, ("airtable_record_id",))
    crosswalk_index, duplicate_crosswalks = _unique_index(
        [row for row in crosswalk_rows if _clean(row.get("crosswalk_status")) == "approved"],
        ("airtable_record_id",),
    )
    entity_index, duplicate_entities = _unique_index(entity_rows, ("entity_id",))
    binding_queues: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for reference in references:
        record_id = _clean(reference.get("airtable_record_id"))
        crosswalk = crosswalk_index.get((record_id,), {})
        entity_id = _clean(reference.get("entity_id"))
        relationship = _clean(crosswalk.get("profile_relationship_type"))
        if crosswalk and _clean(crosswalk.get("entity_id")) == entity_id:
            binding_queues.setdefault((entity_id, relationship), []).append((reference, crosswalk))
    for queue in binding_queues.values():
        queue.sort(key=lambda pair: (_clean(pair[0].get("airtable_record_id")), _canonical(pair[0])))
    binding_offsets: dict[tuple[str, str], int] = {}

    items: list[dict[str, Any]] = []
    violations: list[str] = []
    for index, proposal in enumerate(proposal_rows):
        fields = set(proposal)
        extra = fields - set(PROPOSAL_FIELDS)
        protected = extra & PROTECTED_FIELDS
        publication = extra & PUBLICATION_FIELDS
        allowlist = extra - protected - publication
        entity_id = _clean(proposal.get("entity_id"))
        relationship = _clean(proposal.get("profile_relationship_type"))
        cross_key = (entity_id, relationship)
        queue = binding_queues.get(cross_key, [])
        offset = binding_offsets.get(cross_key, 0)
        reference, crosswalk = (queue[offset] if offset < len(queue) else (queue[0] if len(queue) == 1 else ({}, {})))
        binding_offsets[cross_key] = offset + 1
        record_id = _clean(crosswalk.get("airtable_record_id"))
        candidates = [r for r in refs_by_entity.get(entity_id, []) if _clean(r.get("airtable_record_id")) == record_id]
        reference = candidates[0] if len(candidates) == 1 else {}
        candidate_id = _clean(reference.get("candidate_id"))
        decision = decision_index.get((candidate_id,), {})
        profile = profile_index.get((record_id,), {})
        canonical = entity_index.get((entity_id,), {})
        source_fp = _clean(profile.get("record_fingerprint"))
        approved_fp = _clean(crosswalk.get("approved_fingerprint"))
        changed, before, proposed_values = _change_set(profile, proposal)
        source_batch_id = proposed_values["integration_batch_id"]
        proposed_values["integration_batch_id"] = batch_id
        changed = [field for field in PROPOSAL_FIELDS if before[field] != proposed_values[field]]
        expected_counting = _clean(canonical.get("counting_entity_id"))
        proposed_counting = _clean(proposal.get("counting_entity_id"))
        gate = {
            "record_id_present": bool(record_id),
            "crosswalk_key_unique": bool(record_id) and (record_id,) not in duplicate_crosswalks,
            "profile_record_id_unique": (record_id,) not in duplicate_profiles,
            "reference_unique": len(candidates) == 1,
            "decision_unique": (candidate_id,) not in duplicate_decisions,
            "canonical_entity_unique": (entity_id,) not in duplicate_entities,
            "fingerprint_current": bool(source_fp) and source_fp == approved_fp,
            "identity": _clean(decision.get("identity_status")) == "approved_crosswalk",
            "classification": _clean(decision.get("classification_status")) == "eligible",
            "readiness": _clean(decision.get("readiness_status")) == "ready",
            "relationship_type": relationship in relationships,
            "counting_entity_id": bool(proposed_counting) and bool(expected_counting) and proposed_counting == expected_counting,
            "allowlist": not allowlist,
            "protected_fields": not protected,
            "publication_fields": not publication,
        }
        reasons = [name for name, passed in gate.items() if not passed]
        item = {
            "proposal_id": "prp_" + content_hash({"batch": batch_id, "record": record_id, "proposal": proposal})[:20],
            "integration_batch_id": batch_id,
            "source_integration_batch_id": source_batch_id,
            "airtable_record_id": record_id,
            "entity_id": entity_id,
            "profile_relationship_type": relationship,
            "counting_entity_id": _clean(proposal.get("counting_entity_id")),
            "expected_counting_entity_id": expected_counting,
            "source_fingerprint": source_fp,
            "approved_fingerprint": approved_fp,
            "current_fingerprint": source_fp,
            "reviewed_before_values": before,
            "proposed_values": proposed_values,
            "changed_fields": changed,
            "gate_outcomes": gate,
            "status": "actionable" if all(gate.values()) and changed else ("noop" if all(gate.values()) else "holdout"),
            "holdout_reason": ",".join(reasons),
        }
        items.append(item)

    # A record/field may occur once only. Identical repeats are duplicates; differing
    # desired values are conflicts. Both are held out fail-closed.
    claims: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for i, item in enumerate(items):
        for field in item["changed_fields"]:
            claims.setdefault((item["airtable_record_id"], field), []).append((i, item["proposed_values"][field]))
    duplicate_count = conflict_count = 0
    for claim_rows in claims.values():
        if len(claim_rows) > 1:
            conflict = len({value for _, value in claim_rows}) > 1
            conflict_count += int(conflict)
            duplicate_count += int(not conflict)
            for i, _ in claim_rows:
                items[i]["status"] = "holdout"
                reason = "conflicting_proposal" if conflict else "duplicate_proposal"
                items[i]["holdout_reason"] = ",".join(filter(None, [items[i]["holdout_reason"], reason]))

    items.sort(key=lambda r: (r["airtable_record_id"], r["proposal_id"]))
    actionable = [r for r in items if r["status"] == "actionable"]
    noops = [r for r in items if r["status"] == "noop"]
    holdouts = [r for r in items if r["status"] == "holdout"]
    immutable_content = {"batch_id": batch_id, "items": items, "config_version": config_version}
    batch_hash = content_hash(immutable_content)
    counts = lambda gate: sum(not r["gate_outcomes"][gate] for r in items)
    manifest = {
        "batch_id": batch_id, "lifecycle_state": "REVIEWABLE_BATCH", "created_at": created_at,
        "created_by": created_by, "proposal_input_hash": proposal_input_hash or content_hash(_canonical_rows(proposal_rows)),
        "batch_content_hash": batch_hash, "config_version": config_version,
        "input_proposal_count": len(items), "actionable_count": len(actionable),
        "noop_count": len(noops), "holdout_count": len(holdouts),
        "unique_airtable_record_ids": len({r["airtable_record_id"] for r in items if r["airtable_record_id"]}),
        "unique_entity_id_count": len({r["entity_id"] for r in items}),
        "unique_counting_entity_id_count": len({r["counting_entity_id"] for r in items if r["counting_entity_id"] and r["gate_outcomes"]["counting_entity_id"]}),
        "counts_by_relationship_type": {key: sum(r["profile_relationship_type"] == key for r in items) for key in sorted({r["profile_relationship_type"] for r in items})},
        "stale_fingerprint_count": counts("fingerprint_current"), "duplicate_count": duplicate_count,
        "conflict_count": conflict_count, "missing_record_id_count": counts("record_id_present"),
        "duplicate_crosswalk_key_count": counts("crosswalk_key_unique"),
        "duplicate_profile_record_id_count": counts("profile_record_id_unique"),
        "identity_gate_blocks": counts("identity"), "classification_gate_blocks": counts("classification"),
        "readiness_gate_blocks": counts("readiness"), "invalid_relationship_type_count": counts("relationship_type"),
        "invalid_counting_entity_id_count": counts("counting_entity_id"),
        "allowlist_violations": counts("allowlist"), "protected_field_mutation_attempts": counts("protected_fields"),
        "publication_mutation_attempts": counts("publication_fields"), "approval_state": "not_approved",
        "approval_hash_match": False, "handoff_ready": False,
    }
    return {"manifest": manifest, "items": items, "actionable": actionable, "noops": noops, "holdouts": holdouts}


def record_approval(batch: Mapping[str, Any], *, decision: str, approver: str, approved_at: str, note: str = "") -> dict[str, str]:
    if decision not in {"approved", "rejected"} or not _clean(approver) or not _valid_timestamp(approved_at):
        raise HandoffGateViolation("Approval requires a valid decision, approver, and ISO-8601 timestamp")
    return {"batch_id": batch["manifest"]["batch_id"], "batch_content_hash": batch["manifest"]["batch_content_hash"],
            "decision": decision, "approver": approver, "approval_timestamp": approved_at, "approval_note": note}


def create_handoff_package(batch: Mapping[str, Any], approval: Mapping[str, Any]) -> dict[str, Any]:
    manifest = deepcopy(batch["manifest"])
    current_hash = content_hash({"batch_id": manifest["batch_id"], "items": batch["items"], "config_version": manifest["config_version"]})
    hash_match = current_hash == manifest["batch_content_hash"] == approval.get("batch_content_hash")
    batch_match = approval.get("batch_id") == manifest["batch_id"]
    canonical_actionable = [deepcopy(row) for row in batch["items"] if row["status"] == "actionable"]
    hard_counts = ["holdout_count", "stale_fingerprint_count", "duplicate_count", "conflict_count",
                   "duplicate_crosswalk_key_count", "duplicate_profile_record_id_count", "missing_record_id_count",
                   "identity_gate_blocks", "classification_gate_blocks", "readiness_gate_blocks", "invalid_counting_entity_id_count",
                   "invalid_relationship_type_count", "allowlist_violations", "protected_field_mutation_attempts", "publication_mutation_attempts"]
    counts_match = (manifest["actionable_count"] == len(canonical_actionable)
                    and manifest["holdout_count"] == sum(row["status"] == "holdout" for row in batch["items"])
                    and manifest["noop_count"] == sum(row["status"] == "noop" for row in batch["items"]))
    ready = bool(canonical_actionable) and not any(manifest[name] for name in hard_counts) and counts_match and approval.get("decision") == "approved" and hash_match and batch_match
    if not ready:
        raise HandoffGateViolation("Batch is not handoff-ready: approval/hash/gates/actionable content invalid")
    manifest.update(lifecycle_state="HANDOFF_READY", approval_state="approved", approval_hash_match=True, handoff_ready=True)
    contract = {
        "contract_version": "sko-024-future-adapter-v1", "write_capability": False,
        "pre_application_obligations": [
            "recheck current fingerprint immediately before each application",
            "capture actual immediate pre-application integration-owned values before mutation",
        ],
        "idempotency_key_fields": ["batch_id", "proposal_id", "airtable_record_id", "batch_content_hash"],
        "allowed_outcomes": sorted(APPLICATION_OUTCOMES),
        "operations": canonical_actionable,
    }
    return {"manifest": manifest, "approval": dict(approval), "application_contract": contract,
            "package_hash": content_hash({"manifest": manifest, "approval": approval, "contract": contract})}


def derive_retry_package(handoff: Mapping[str, Any], outcomes: Iterable[Mapping[str, Any]], current_fingerprints: Mapping[str, str]) -> dict[str, Any]:
    outcome_by_id = {_clean(r.get("proposal_id")): dict(r) for r in outcomes}
    eligible, reconciled, stale = [], [], []
    for op in handoff["application_contract"]["operations"]:
        outcome = _clean(outcome_by_id.get(op["proposal_id"], {}).get("outcome")) or "not_attempted"
        if outcome not in APPLICATION_OUTCOMES:
            raise HandoffGateViolation(f"Unknown application outcome: {outcome}")
        if outcome in {"applied", "already_in_desired_state"}:
            reconciled.append(op["proposal_id"])
        elif outcome == "stale_at_application" or current_fingerprints.get(op["airtable_record_id"]) != op["current_fingerprint"]:
            stale.append(op["proposal_id"])
        elif outcome in {"not_attempted", "failed"}:
            eligible.append(op)
    return {"original_batch_id": handoff["manifest"]["batch_id"], "eligible_operations": eligible,
            "reconciled_proposal_ids": sorted(reconciled), "stale_proposal_ids": sorted(stale),
            "retry_hash": content_hash(eligible)}


def build_rollback_candidate(handoff: Mapping[str, Any], outcomes: Iterable[Mapping[str, Any]], *, application_reference: str) -> dict[str, Any]:
    outcomes_by_id = {_clean(row.get("proposal_id")): dict(row) for row in outcomes}
    operations = []
    for op in handoff["application_contract"]["operations"]:
        outcome = outcomes_by_id.get(op["proposal_id"], {})
        if outcome.get("outcome") == "applied":
            supplied = outcome.get("application_time_before_values")
            application_before = None
            if supplied is not None:
                if set(supplied) - set(PROPOSAL_FIELDS):
                    raise HandoffGateViolation("Application-time state contains non-integration-owned fields")
                application_before = {field: _clean(supplied.get(field)) for field in PROPOSAL_FIELDS}
            operations.append({"original_batch_id": handoff["manifest"]["batch_id"], "proposal_id": op["proposal_id"],
                               "airtable_record_id": op["airtable_record_id"], "application_outcome_reference": application_reference,
                               "reviewed_before_values": op["reviewed_before_values"],
                               "application_time_before_values": application_before,
                               "application_time_state_supplied": application_before is not None,
                               "expected_post_application_values": op["proposed_values"]})
    payload = {"package_type": "rollback_planning_candidate", "lifecycle_state": "CANDIDATE_BATCH", "executable": False,
               "requires_validation": True, "requires_owner_approval": True,
               "original_batch_id": handoff["manifest"]["batch_id"], "operations": operations}
    payload["rollback_proposal_hash"] = content_hash(payload)
    return payload
