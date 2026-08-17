#!/usr/bin/env python3
"""SKO-025 governed, offline validation against a frozen Airtable snapshot.

This module has no Airtable client, credential, network, write, or apply path.
It validates frozen inputs and produces local review proposals and aggregate QA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from directory_integration import PROPOSAL_FIELDS, PROTECTED_FIELDS
from directory_integration_handoff import PUBLICATION_FIELDS


EXPECTED_COLUMNS = [
    "Organisation", "Country", "Website", "Business Summary",
    "Identified Clients", "Sector", "Clients Publicly Referenced (Yes/No)",
    "Social Mission", "Countries served", "Airtable Record ID",
]
RELATIONSHIP_TYPES = {
    "group", "division", "service_line", "brand", "establishment",
    "operating_unit",
}
RECORD_ID_PATTERN = re.compile(r"^rec[A-Za-z0-9]+$")
FINGERPRINT_VERSION = "airtable-profile-fingerprint-v1"
FINGERPRINT_PROJECTION = ["Organisation", "Country", "Website hostname"]


class OperationalValidationError(RuntimeError):
    """Raised when an SKO-025 hard invariant or frozen-input contract fails."""


def _clean(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_snapshot(
    snapshot: pd.DataFrame, *, actual_sha256: str, expected_sha256: str,
) -> dict[str, Any]:
    """Validate the frozen real-export contract without exposing row content."""
    columns = list(snapshot.columns)
    record_ids = snapshot["Airtable Record ID"].map(_clean) if "Airtable Record ID" in snapshot else pd.Series(dtype=str)
    qa = {
        "export_sha256": actual_sha256,
        "checksum_match": actual_sha256 == expected_sha256,
        "row_count": len(snapshot),
        "column_count": len(columns),
        "column_names": columns,
        "exact_schema": columns == EXPECTED_COLUMNS,
        "blank_airtable_record_ids": int(record_ids.eq("").sum()),
        "duplicate_airtable_record_ids": int(record_ids.duplicated().sum()),
        "invalid_airtable_record_ids": int((~record_ids.map(lambda value: bool(RECORD_ID_PATTERN.fullmatch(value)))).sum()),
        "integration_fields_present": sum(field in columns for field in PROPOSAL_FIELDS),
        "integration_fields_expected": len(PROPOSAL_FIELDS),
        "pre_integration_state": True,
        "before_state_interpretation": "integration fields not deployed; no blank values invented",
    }
    failures = [
        not qa["checksum_match"], qa["row_count"] != 321,
        not qa["exact_schema"], bool(qa["blank_airtable_record_ids"]),
        bool(qa["duplicate_airtable_record_ids"]), bool(qa["invalid_airtable_record_ids"]),
        bool(qa["integration_fields_present"]),
    ]
    if any(failures):
        raise OperationalValidationError("frozen Airtable input failed its accepted structural contract")
    return qa


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        return value.to_dict("records")
    return [dict(row) for row in value]


def classify_crosswalks(
    snapshot: pd.DataFrame, crosswalks: Any, *, fingerprint_config: Mapping[str, Any],
    canonical_entities: Any = (),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Classify profiles using only explicit approved exact-Record-ID decisions."""
    from scripts.sko025a_crosswalk import fingerprint

    profiles = { _clean(row["Airtable Record ID"]): row for row in snapshot.to_dict("records") }
    approved = [row for row in _records(crosswalks) if _clean(row.get("crosswalk_status")) == "approved"]
    canonical = {
        _clean(row.get("entity_id")): row for row in _records(canonical_entities)
        if _clean(row.get("entity_id"))
    }
    claims = Counter(_clean(row.get("airtable_record_id")) for row in approved)
    classification: dict[str, str] = {record_id: "unresolved_unlinked_legacy_profile" for record_id in profiles}
    valid: list[dict[str, Any]] = []
    missing = stale = invalid_counting = unsupported_non_default = 0
    for row in approved:
        record_id = _clean(row.get("airtable_record_id"))
        entity_id = _clean(row.get("entity_id"))
        relationship = _clean(row.get("profile_relationship_type"))
        proposed_counting = _clean(row.get("counting_entity_id")) or entity_id
        if record_id not in profiles:
            missing += 1
            continue
        if (claims[record_id] != 1 or not entity_id or relationship not in RELATIONSHIP_TYPES
                or _clean(row.get("fingerprint_version")) != FINGERPRINT_VERSION):
            classification[record_id] = "invalid_conflicting_binding"
            continue
        canonical_row = canonical.get(entity_id, {})
        expected_counting = _clean(canonical_row.get("counting_entity_id"))
        if not expected_counting or proposed_counting != expected_counting:
            classification[record_id] = "invalid_conflicting_binding"
            invalid_counting += 1
            unsupported_non_default += int(proposed_counting != entity_id)
            continue
        try:
            current = fingerprint(profiles[record_id], fingerprint_config)
        except ValueError:
            current = ""
        if not current or current != _clean(row.get("approved_fingerprint")):
            classification[record_id] = "stale_fingerprint"
            stale += 1
            continue
        classification[record_id] = "exact_approved_existing_profile_binding"
        valid.append({**row, "counting_entity_id": proposed_counting, "current_fingerprint": current})
    rows = pd.DataFrame([
        {"airtable_record_id": record_id, "operational_classification": state}
        for record_id, state in sorted(classification.items())
    ])
    counts = Counter(classification.values())
    summary = {
        "approved_existing_profile_bindings": counts["exact_approved_existing_profile_binding"],
        "approved_crosswalk_missing_current_record_id": missing,
        "stale_fingerprints": stale,
        "unresolved_legacy_profiles": counts["unresolved_unlinked_legacy_profile"],
        "invalid_conflicting_bindings": counts["invalid_conflicting_binding"],
        "invalid_counting_mappings": invalid_counting,
        "unsupported_non_default_counting_mappings": unsupported_non_default,
    }
    return pd.DataFrame(valid), {key: int(value) for key, value in summary.items()} | {"classified_profiles": len(rows)}


def build_initial_proposals(
    valid_bindings: pd.DataFrame, readiness: Any, *, batch_id: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build initial values for absent integration fields; never an update diff."""
    ready_by_entity = {_clean(row.get("entity_id")): row for row in _records(readiness)}
    proposals: list[dict[str, str]] = []
    holdouts = 0
    for binding in valid_bindings.to_dict("records"):
        entity_id = _clean(binding.get("entity_id"))
        state = ready_by_entity.get(entity_id, {})
        is_ready = _clean(state.get("readiness_status")) == "ready"
        if not is_ready:
            holdouts += 1
            continue
        proposals.append({
            "entity_id": entity_id,
            "profile_relationship_type": _clean(binding.get("profile_relationship_type")),
            "counting_entity_id": _clean(binding.get("counting_entity_id")) or entity_id,
            "crosswalk_status": "approved",
            "skopia_readiness_status": "ready",
            "skopia_readiness_assessed_at": _clean(state.get("assessed_at")),
            "skopia_readiness_reference": _clean(state.get("evidence_reference")),
            "integration_batch_id": batch_id,
        })
    frame = pd.DataFrame(proposals, columns=PROPOSAL_FIELDS).sort_values(
        ["entity_id", "profile_relationship_type"], ignore_index=True
    ) if proposals else pd.DataFrame(columns=PROPOSAL_FIELDS)
    return frame, {
        "initial_valid_proposals": len(frame), "holdouts": holdouts,
        "unresolved_no_proposal": 0,
    }


def proposal_outcome_qa(crosswalk_qa: Mapping[str, int], proposal_qa: Mapping[str, int]) -> dict[str, int]:
    unresolved = int(crosswalk_qa["unresolved_legacy_profiles"])
    total = sum([
        unresolved, int(crosswalk_qa["stale_fingerprints"]),
        int(crosswalk_qa["invalid_conflicting_bindings"]), int(proposal_qa["holdouts"]),
    ])
    return {
        "unresolved_no_proposal": unresolved,
        "total_no_proposal_profiles": total,
        "approved_binding_noops": 0,
    }


def counting_qa(proposals: pd.DataFrame) -> dict[str, Any]:
    relationships = Counter(proposals.get("profile_relationship_type", pd.Series(dtype=str)))
    by_entity = proposals.groupby("entity_id").size() if not proposals.empty else pd.Series(dtype=int)
    return {
        "profile_count": len(proposals),
        "unique_entity_ids": int(proposals["entity_id"].nunique()) if not proposals.empty else 0,
        "unique_counting_entity_ids": int(proposals["counting_entity_id"].nunique()) if not proposals.empty else 0,
        "one_to_many_entity_count": int((by_entity > 1).sum()),
        "additional_profiles_without_supplier_inflation": len(proposals) - (int(proposals["counting_entity_id"].nunique()) if not proposals.empty else 0),
        "relationship_type_counts": {name: int(relationships[name]) for name in sorted(RELATIONSHIP_TYPES)},
    }


def validate_forward_flow(new_candidates: Any = (), holdouts: Any = ()) -> dict[str, int]:
    new_rows, held_rows = _records(new_candidates), _records(holdouts)
    fabricated = sum(bool(_clean(row.get("airtable_record_id"))) for row in new_rows)
    if fabricated:
        raise OperationalValidationError("new-profile candidate contains a fabricated Airtable Record ID")
    return {"new_profile_candidates": len(new_rows), "new_profile_holdouts": len(held_rows), "fabricated_airtable_record_ids": fabricated}


def derive_hard_invariants(
    proposals: pd.DataFrame, *, new_candidates: Any = (), handoff_batches: Any = (),
) -> dict[str, Any]:
    """Derive safety QA from generated proposal, candidate and handoff artifacts."""
    fields = set(proposals.columns)
    extra = fields - set(PROPOSAL_FIELDS)
    protected = extra & PROTECTED_FIELDS
    publication = extra & PUBLICATION_FIELDS
    allowlist = extra - protected - publication
    candidate_qa = validate_forward_flow(new_candidates)
    batches = _records(handoff_batches)
    return {
        "allowlist_violations": len(allowlist),
        "protected_editorial_mutation_attempts": len(proposals) if protected else 0,
        "publication_mutation_attempts": len(proposals) if publication else 0,
        "fabricated_airtable_record_ids": candidate_qa["fabricated_airtable_record_ids"],
        "handoff_ready_existing_profile_batch_count": sum(
            bool(batch.get("manifest", {}).get("handoff_ready")) for batch in batches
        ),
    }


def validate_config(config: Mapping[str, Any]) -> None:
    disabled = ("network_enabled", "credentials_enabled", "mutation_enabled")
    if config.get("mode") != "proposal_only" or any(config.get(key) is not False for key in disabled):
        raise OperationalValidationError("SKO-025 requires proposal-only, offline, no-mutation configuration")
    if config.get("allowed_proposal_fields") != PROPOSAL_FIELDS:
        raise OperationalValidationError("the accepted eight-field proposal allowlist changed")


def validate_fingerprint_config(config: Mapping[str, Any]) -> None:
    if config.get("fingerprint_version") != FINGERPRINT_VERSION:
        raise OperationalValidationError("the accepted Airtable fingerprint version changed")
    if config.get("projection") != FINGERPRINT_PROJECTION:
        raise OperationalValidationError("the accepted Airtable fingerprint projection changed")


def run(
    *, snapshot_path: Path, crosswalk_path: Path, readiness_path: Path | None,
    config_path: Path, fingerprint_config_path: Path,
    canonical_entities_path: Path | None, expected_sha256: str,
    batch_id: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fp_config = yaml.safe_load(fingerprint_config_path.read_text(encoding="utf-8"))
    validate_config(config)
    validate_fingerprint_config(fp_config)
    snapshot = pd.read_csv(snapshot_path, dtype=str, keep_default_na=False)
    crosswalks = pd.read_csv(crosswalk_path, dtype=str, keep_default_na=False)
    canonical_entities = pd.read_parquet(canonical_entities_path) if canonical_entities_path else []
    readiness = pd.read_csv(readiness_path, dtype=str, keep_default_na=False) if readiness_path else []
    input_qa = validate_snapshot(snapshot, actual_sha256=sha256_file(snapshot_path), expected_sha256=expected_sha256)
    valid, crosswalk_qa = classify_crosswalks(
        snapshot, crosswalks, fingerprint_config=fp_config,
        canonical_entities=canonical_entities,
    )
    proposals, proposal_qa = build_initial_proposals(valid, readiness, batch_id=batch_id)
    proposal_qa.update(proposal_outcome_qa(crosswalk_qa, proposal_qa))
    hard_invariants = derive_hard_invariants(proposals)
    qa = {
        **input_qa, **crosswalk_qa, **proposal_qa, **counting_qa(proposals),
        "proposal_interpretation": "initial proposed values for fields not yet deployed",
        "publication_field_before_after_compared": False,
        "publication_limitation": "publication controls are absent from the frozen export; protection is architectural and aggregate-only",
        **hard_invariants,
        "network_calls": 0,
        "write_apply_capability": False,
    }
    return qa, valid, proposals


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline SKO-025 operational validation")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--readiness", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fingerprint-config", type=Path, required=True)
    parser.add_argument("--canonical-entities", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    qa, bindings, proposals = run(
        snapshot_path=args.snapshot, crosswalk_path=args.crosswalk,
        readiness_path=args.readiness, config_path=args.config,
        fingerprint_config_path=args.fingerprint_config,
        canonical_entities_path=args.canonical_entities,
        expected_sha256=args.expected_sha256, batch_id=args.batch_id,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregate_qa.json").write_text(json.dumps(qa, indent=2, sort_keys=True), encoding="utf-8")
    bindings.to_csv(args.output_dir / "approved_existing_profile_bindings.csv", index=False)
    proposals.to_csv(args.output_dir / "initial_integration_proposals.csv", index=False)
    print(json.dumps(qa, sort_keys=True))


if __name__ == "__main__":
    main()
