#!/usr/bin/env python3
"""Build the deterministic SKO-023 production-shaped synthetic fixture pack."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ENTITY_COUNT = 330
BATCH_ID = "batch_sko023_production_shaped"


def entity_id(index: int) -> str:
    return f"prod_ent_{index:03d}"


def record_id(index: int, relationship: str = "group") -> str:
    return f"prod_rec_{index:03d}_{relationship}"


def scenario(index: int) -> str:
    if index < 160 or index >= 310:
        return "clean_existing"
    if index < 185:
        return "new_profile"
    if index < 205:
        return "possible_duplicate"
    if index < 215:
        return "identity_conflict"
    if index < 225:
        return "stale_fingerprint"
    if index < 235:
        return "canonical_merge"
    if index < 245:
        return "split_pending"
    if index < 255:
        return "retired_profile"
    if index < 270:
        return "ineligible"
    if index < 285:
        return "incomplete_readiness"
    if index < 300:
        return "ready_unpublished"
    return "published_not_ready"


def relationships(index: int) -> list[str]:
    result = ["group"]
    if index < 80:
        result.append("division")
    if index < 20:
        result.append("service_line")
    return result


def build_fixture_tables() -> dict[str, pd.DataFrame]:
    """Return a deterministic, entirely synthetic mixed population."""
    entities: list[dict[str, object]] = []
    candidates: list[dict[str, str]] = []
    profiles: list[dict[str, str]] = []
    crosswalks: list[dict[str, str]] = []
    classifications: list[dict[str, str]] = []
    readiness: list[dict[str, str]] = []
    activities: list[dict[str, object]] = []
    events: list[dict[str, str]] = []
    history: list[dict[str, str]] = []
    expected: list[dict[str, str]] = []

    for index in range(ENTITY_COUNT):
        current_scenario = scenario(index)
        current_entity = entity_id(index)
        survivor = entity_id(315 + (index - 225)) if current_scenario == "canonical_merge" else ""
        counting_id = survivor or current_entity
        entity_status = "merged" if current_scenario == "canonical_merge" else "active"
        split_status = "pending_reassignment" if current_scenario == "split_pending" else ""
        name = f"Synthetic Production Organisation {index:03d}"
        if current_scenario == "new_profile":
            name = f"Synthetic Novel Candidate Zeta {index:03d}"
        entities.append({
            "entity_id": current_entity,
            "canonical_name": name,
            "entity_status": entity_status,
            "counting_entity_id": counting_id,
            "redirect_to": survivor,
            "split_status": split_status,
        })

        classification_status = "ineligible" if current_scenario == "ineligible" else "eligible"
        classifications.append({
            "entity_id": current_entity,
            "classification_status": classification_status,
            "evidence_reference": f"synthetic_classification_{index:03d}",
        })
        enrichment_complete = current_scenario not in {"incomplete_readiness", "published_not_ready"}
        readiness.append({
            "entity_id": current_entity,
            "enrichment_complete": str(enrichment_complete).lower(),
            "readiness_status": "ready",
            "assessed_at": "2026-08-12T09:00:00Z",
            "evidence_reference": f"synthetic_readiness_{index:03d}",
        })

        for relationship in relationships(index):
            candidate_id = f"prod_candidate_{index:03d}_{relationship}"
            candidates.append({
                "candidate_id": candidate_id,
                "entity_id": current_entity,
                "profile_relationship_type": relationship,
                "integration_batch_id": BATCH_ID,
            })
            expected.append({
                "test_id": candidate_id,
                "expected_outcome": current_scenario,
            })
            activities.append({
                "activity_id": f"activity_{index:03d}_{relationship}",
                "entity_id": current_entity,
                "supplier_count": 1,
                "spend_eur": (index + 1) * 1000,
                "impact_count": (index % 7) + 1,
            })

            if current_scenario in {"new_profile", "possible_duplicate"}:
                continue

            current_record = record_id(index, relationship)
            if current_scenario == "identity_conflict":
                current_record = f"prod_rec_conflict_{(index - 205) // 2:03d}"
            approved_fingerprint = f"fp_{index:03d}_{relationship}"
            actual_fingerprint = (
                f"changed_{approved_fingerprint}"
                if current_scenario == "stale_fingerprint"
                else approved_fingerprint
            )
            crosswalks.append({
                "entity_id": current_entity,
                "airtable_record_id": current_record,
                "profile_relationship_type": relationship,
                "crosswalk_status": "approved",
                "approved_fingerprint": approved_fingerprint,
            })
            if not any(row["airtable_record_id"] == current_record for row in profiles):
                publication = "published"
                if current_scenario in {"ready_unpublished", "incomplete_readiness"} or (index % 3 and current_scenario != "published_not_ready"):
                    publication = "unpublished"
                profile_status = "retired" if current_scenario == "retired_profile" else "active"
                if profile_status == "retired":
                    publication = "retired"
                profiles.append({
                    "airtable_record_id": current_record,
                    "Organisation": f"{name} {relationship}",
                    "publication_status": publication,
                    "profile_status": profile_status,
                    "record_fingerprint": actual_fingerprint,
                })

        if current_scenario == "possible_duplicate":
            profiles.append({
                "airtable_record_id": f"prod_rec_similar_{index:03d}",
                "Organisation": name[:-1],
                "publication_status": "unpublished",
                "profile_status": "active",
                "record_fingerprint": f"fp_similar_{index:03d}",
            })
        if current_scenario == "canonical_merge":
            events.append({
                "event_id": f"merge_{index:03d}",
                "event_type": "canonical_merge",
                "entity_id": current_entity,
                "target_entity_id": survivor,
                "reason": "synthetic approved merge",
                "evidence_reference": f"synthetic_merge_evidence_{index:03d}",
            })
        elif current_scenario == "split_pending":
            events.append({
                "event_id": f"split_{index:03d}",
                "event_type": "canonical_split",
                "entity_id": current_entity,
                "target_entity_id": "",
                "reason": "synthetic split pending reassignment",
                "evidence_reference": f"synthetic_split_evidence_{index:03d}",
            })
        elif current_scenario == "retired_profile":
            events.append({
                "event_id": f"retire_{index:03d}",
                "event_type": "profile_retired",
                "entity_id": current_entity,
                "target_entity_id": "",
                "reason": "synthetic profile retirement",
                "evidence_reference": f"synthetic_retirement_evidence_{index:03d}",
            })
            history.append({
                "history_id": f"history_{index:03d}",
                "entity_id": current_entity,
                "airtable_record_id": record_id(index),
                "prior_integration_values": '{"crosswalk_status":"approved"}',
                "editorial_fingerprint": f"fp_{index:03d}_group",
                "transition_reason": "synthetic retirement",
                "evidence_reference": f"synthetic_retirement_evidence_{index:03d}",
            })

    columns = {
        "canonical_entities": ["entity_id", "canonical_name", "entity_status", "counting_entity_id", "redirect_to", "split_status"],
        "directory_candidates": ["candidate_id", "entity_id", "profile_relationship_type", "integration_batch_id"],
        "airtable_profiles": ["airtable_record_id", "Organisation", "publication_status", "profile_status", "record_fingerprint"],
        "crosswalks": ["entity_id", "airtable_record_id", "profile_relationship_type", "crosswalk_status", "approved_fingerprint"],
        "classifications": ["entity_id", "classification_status", "evidence_reference"],
        "readiness": ["entity_id", "enrichment_complete", "readiness_status", "assessed_at", "evidence_reference"],
        "activity_metrics": ["activity_id", "entity_id", "supplier_count", "spend_eur", "impact_count"],
        "lifecycle_events": ["event_id", "event_type", "entity_id", "target_entity_id", "reason", "evidence_reference"],
        "integration_history": ["history_id", "entity_id", "airtable_record_id", "prior_integration_values", "editorial_fingerprint", "transition_reason", "evidence_reference"],
        "expected_behaviours": ["test_id", "expected_outcome"],
    }
    records = {
        "canonical_entities": entities,
        "directory_candidates": candidates,
        "airtable_profiles": profiles,
        "crosswalks": crosswalks,
        "classifications": classifications,
        "readiness": readiness,
        "activity_metrics": activities,
        "lifecycle_events": events,
        "integration_history": history,
        "expected_behaviours": expected,
    }
    return {name: pd.DataFrame(records[name], columns=columns[name]) for name in columns}


def write_fixture_pack(output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = build_fixture_tables()
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, lineterminator="\n")
    return {name: len(frame) for name, frame in tables.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name, count in write_fixture_pack(args.output_dir).items():
        print(f"{name}={count}")


if __name__ == "__main__":
    main()
