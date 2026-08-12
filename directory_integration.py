#!/usr/bin/env python3
"""SKO-022 proposal-only directory integration prototype.

This module deliberately contains no remote client, credential handling, or
mutation implementation. It produces local review proposals and QA data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml


PROPOSAL_FIELDS = [
    "entity_id",
    "profile_relationship_type",
    "counting_entity_id",
    "crosswalk_status",
    "skopia_readiness_status",
    "skopia_readiness_assessed_at",
    "skopia_readiness_reference",
    "integration_batch_id",
]
PROTECTED_FIELDS = {
    "Organisation", "Website", "Business Summary", "Sector",
    "Social Mission", "Countries Served", "Corporate Clients",
    "Client Evidence Notes", "publication_status",
}
TRANSITION_FIELDS = [
    "transition_id", "candidate_id", "batch_id", "actor", "timestamp",
    "reason", "evidence", "from_status", "to_status",
]


class ProposalOnlyViolation(RuntimeError):
    """Raised for every attempted mutation in proposal-only mode."""


def request_mutation(*_args: Any, **_kwargs: Any) -> None:
    """Reject mutation unconditionally; there is no mutation implementation."""
    raise ProposalOnlyViolation("Mutation is forbidden in proposal-only mode")


def clean(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def truthy(value: Any) -> bool:
    return clean(value).casefold() in {"1", "true", "yes", "y"}


def fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixture_pack(fixture_dir: Path) -> dict[str, pd.DataFrame]:
    names = [
        "canonical_entities", "directory_candidates", "airtable_profiles",
        "crosswalks", "classifications", "readiness", "activity_metrics",
        "lifecycle_events", "integration_history", "expected_behaviours",
    ]
    return {
        name: pd.read_csv(fixture_dir / f"{name}.csv", dtype=str,
                          keep_default_na=False)
        for name in names
    }


def _one(frame: pd.DataFrame, column: str, value: str) -> dict[str, str]:
    rows = frame.loc[frame[column].eq(value)]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _transition(candidate_id: str, batch: str, actor: str, timestamp: str,
                reason: str, evidence: str, to_status: str) -> dict[str, str]:
    token = "|".join([candidate_id, batch, reason, to_status])
    return {
        "transition_id": "tr_" + hashlib.sha256(token.encode()).hexdigest()[:16],
        "candidate_id": candidate_id, "batch_id": batch, "actor": actor,
        "timestamp": timestamp, "reason": reason, "evidence": evidence,
        "from_status": "candidate", "to_status": to_status,
    }


def build_proposals(
    tables: dict[str, pd.DataFrame], *, actor: str = "sko-022-test",
    timestamp: str = "2026-08-05T10:00:00Z",
) -> dict[str, pd.DataFrame]:
    """Evaluate four independent gates and return local proposal artefacts."""
    entities = tables["canonical_entities"]
    candidates = tables["directory_candidates"]
    profiles = tables["airtable_profiles"]
    crosswalks = tables["crosswalks"]
    classifications = tables["classifications"]
    readiness = tables["readiness"]

    approved = crosswalks.loc[crosswalks["crosswalk_status"].eq("approved")]
    conflicts = set(
        approved.groupby("airtable_record_id")["entity_id"].nunique()
        .loc[lambda series: series > 1].index
    )
    decisions: list[dict[str, str]] = []
    proposals: list[dict[str, str]] = []
    references: list[dict[str, str]] = []
    transitions: list[dict[str, str]] = []

    for candidate in candidates.sort_values("candidate_id").to_dict("records"):
        candidate_id = clean(candidate["candidate_id"])
        original_entity_id = clean(candidate["entity_id"])
        relationship = clean(candidate["profile_relationship_type"])
        batch = clean(candidate["integration_batch_id"])
        entity = _one(entities, "entity_id", original_entity_id)
        entity_id = clean(entity.get("redirect_to")) or original_entity_id
        canonical = _one(entities, "entity_id", entity_id)
        classification = _one(classifications, "entity_id", entity_id)
        ready = _one(readiness, "entity_id", entity_id)
        cross = approved.loc[
            approved["entity_id"].eq(original_entity_id)
            & approved["profile_relationship_type"].eq(relationship)
        ]
        cross = cross.iloc[0].to_dict() if not cross.empty else {}
        record_id = clean(cross.get("airtable_record_id"))
        profile = _one(profiles, "airtable_record_id", record_id)

        identity_status = "approved_crosswalk" if cross else "new_profile_candidate"
        outcome = identity_status
        reason = "identity evaluated without mutation"
        evidence = record_id or candidate_id
        blocked = False

        if record_id in conflicts:
            identity_status, outcome, blocked = "identity_conflict", "identity_conflict", True
            reason = "multiple canonical entities claim one Airtable record"
        elif clean(entity.get("entity_status")) == "merged":
            identity_status, outcome, blocked = "redirected_review", "merge_review_required", True
            reason = f"canonical merge redirects to {entity_id}; review required"
        elif clean(entity.get("split_status")) == "pending_reassignment":
            identity_status, outcome, blocked = "suspended", "split_suspended", True
            reason = "canonical split requires profile reassignment"
        elif record_id and clean(cross.get("approved_fingerprint")) != clean(profile.get("record_fingerprint")):
            identity_status, outcome, blocked = "stale_fingerprint", "stale_proposal_blocked", True
            reason = "Airtable record fingerprint changed after approval"
        elif clean(profile.get("profile_status")) == "retired":
            outcome, blocked = "retired_history_retained", True
            reason = "retired profile retained without deletion"
        elif not cross:
            similar = profiles.loc[
                profiles["Organisation"].str.casefold().str.contains(
                    clean(canonical.get("canonical_name")).casefold()[:-1], regex=False
                )
            ] if len(clean(canonical.get("canonical_name"))) > 1 else profiles.iloc[0:0]
            if not similar.empty:
                identity_status, outcome, blocked = "possible_duplicate", "possible_duplicate_review", True
                reason = "name-only similarity requires identity review"

        eligible = clean(classification.get("classification_status")) == "eligible"
        enrichment_complete = truthy(ready.get("enrichment_complete"))
        readiness_status = "ready" if eligible and enrichment_complete and clean(ready.get("readiness_status")) == "ready" else "not_ready"
        publication_status = clean(profile.get("publication_status")) or "unpublished"
        publishable = eligible and readiness_status == "ready" and publication_status == "published"

        counting_entity_id = clean(canonical.get("counting_entity_id"))
        if not counting_entity_id:
            raise ValueError(f"Active profile lacks counting_entity_id: {entity_id}")

        if cross and not blocked:
            proposals.append({
                "entity_id": entity_id,
                "profile_relationship_type": relationship,
                "counting_entity_id": counting_entity_id,
                "crosswalk_status": clean(cross.get("crosswalk_status")),
                "skopia_readiness_status": readiness_status,
                "skopia_readiness_assessed_at": clean(ready.get("assessed_at")),
                "skopia_readiness_reference": clean(ready.get("evidence_reference")),
                "integration_batch_id": batch,
            })
            references.append({
                "candidate_id": candidate_id, "entity_id": entity_id,
                "airtable_record_id": record_id, "proposal_operation": "review_update",
            })

        decisions.append({
            "candidate_id": candidate_id, "entity_id": entity_id,
            "airtable_record_id": record_id, "identity_status": identity_status,
            "classification_status": clean(classification.get("classification_status")),
            "readiness_status": readiness_status,
            "publication_status": publication_status,
            "publishable": str(publishable).lower(), "outcome": outcome,
            "proposal_generated": str(bool(cross and not blocked)).lower(),
        })
        transitions.append(_transition(candidate_id, batch, actor, timestamp,
                                       reason, evidence, outcome))

    proposals_frame = pd.DataFrame(proposals, columns=PROPOSAL_FIELDS)
    decision_frame = pd.DataFrame(decisions).sort_values("candidate_id").reset_index(drop=True)
    transition_frame = pd.DataFrame(transitions, columns=TRANSITION_FIELDS)

    activity = tables["activity_metrics"].copy()
    if activity.empty:
        metrics = pd.DataFrame(columns=["counting_entity_id", "supplier_count", "spend_eur", "impact_count"])
    else:
        activity = activity.merge(entities[["entity_id", "counting_entity_id"]], on="entity_id", how="left")
        # Repeated profile/activity observations count once per canonical counting entity.
        metrics = (activity.groupby("counting_entity_id", as_index=False)
                   .agg(supplier_count=("supplier_count", "max"),
                        spend_eur=("spend_eur", "max"),
                        impact_count=("impact_count", "max")))

    history = tables["integration_history"].copy()
    rollback_rows: list[dict[str, str]] = []
    for row in history.loc[history["entity_id"].eq("ent_rollback")].to_dict("records"):
        prior = json.loads(row["prior_integration_values"])
        rollback_rows.append({key: clean(value) for key, value in prior.items() if key in PROPOSAL_FIELDS})
    rollback = pd.DataFrame(rollback_rows)

    return {
        "proposals": proposals_frame.sort_values(PROPOSAL_FIELDS[:2]).reset_index(drop=True),
        "decisions": decision_frame,
        "integration_references": pd.DataFrame(references),
        "transitions": transition_frame.sort_values("candidate_id").reset_index(drop=True),
        "counting_metrics": metrics,
        "rollback_fields": rollback,
        "history": history,
    }


def write_outputs(results: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, str]:
    """Write authoritative Parquet, DuckDB QA, and derived review artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, frame in results.items():
        parquet = output_dir / f"{name}.parquet"
        frame.to_parquet(parquet, index=False)
        frame.to_csv(output_dir / f"{name}.csv", index=False)
        (output_dir / f"{name}.json").write_text(
            frame.to_json(orient="records", indent=2), encoding="utf-8"
        )
        hashes[name] = fingerprint_file(parquet)
    database = output_dir / "directory_integration_qa.duckdb"
    if database.exists():
        database.unlink()
    connection = duckdb.connect(str(database))
    try:
        for name, frame in results.items():
            connection.register(f"source_{name}", frame)
            connection.execute(f"CREATE TABLE {name} AS SELECT * FROM source_{name}")
        connection.execute(
            "CREATE VIEW proposal_field_qa AS SELECT "
            f"{len(PROPOSAL_FIELDS)} AS allowed_field_count, "
            f"{len(set(PROPOSAL_FIELDS) & PROTECTED_FIELDS)} AS protected_field_count"
        )
    finally:
        connection.close()
    (output_dir / "manifest.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8"
    )
    return hashes


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("mode") != "proposal_only" or config.get("mutation_enabled"):
        raise ProposalOnlyViolation("Configuration must remain proposal-only")
    if config.get("network_enabled") or config.get("credentials_enabled"):
        raise ProposalOnlyViolation("Network and credentials must remain disabled")
    if config.get("allowed_proposal_fields") != PROPOSAL_FIELDS:
        raise ValueError("Configured proposal fields differ from the accepted allowlist")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local proposal-only directory artefacts")
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    load_config(args.config)
    results = build_proposals(load_fixture_pack(args.fixture_dir))
    write_outputs(results, args.output_dir)
    print(f"proposal_rows={len(results['proposals'])}")
    print("mutation_capability=false")


if __name__ == "__main__":
    main()
