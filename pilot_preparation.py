#!/usr/bin/env python3
"""SKO-031 governed pilot preparation and contextual review output.

This module validates a caller-selected population and presents already-produced
SKO-030 records.  It does not create entities, classify geography or activity,
join indicators, acquire sources, or infer impact or causality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


SUBJECT_TYPES = {"canonical_entity", "supplier_observation"}
LOCATION_ROLES = {"registered_or_hq", "contracted_activity", "unknown"}
TRACEABILITY_FIELDS = [
    "supplier_display_name", "source_client", "source_supplier_record_key",
    "source_file_reference", "source_row_reference", "selection_basis",
]
SUBJECT_FIELDS = [
    "selected", "subject_type", "subject_id", "entity_id", "country",
    "postcode", "city", "address", "location_role", *TRACEABILITY_FIELDS,
]
CONTEXT_REVIEW_FIELDS = [
    "subject_type", "subject_id", "entity_id", "supplier_display_name",
    "source_client", "source_supplier_record_key", "location_role", "country",
    "postcode", "geography_scheme", "geography_version", "geography_level",
    "geography_code", "geography_name", "mapping_status", "mapping_confidence",
    "indicator_id", "indicator_name", "indicator_theme", "period", "value", "unit",
    "value_status", "source_id", "source_name", "dataset_id", "dataset_version",
    "activity_evidence_count", "activity_authority_counts", "activity_status_counts",
    "activity_has_authoritative_or_strong", "record_type", "missingness_reason",
    "review_required",
]
SKO030_REQUIRED = {
    "record_type", "subject_type", "subject_id", "entity_id", "geography_scheme",
    "geography_version", "geography_level", "geography_code", "geography_name",
    "mapping_status", "indicator_id", "indicator_name", "indicator_theme", "period",
    "value", "unit", "value_status", "activity_evidence_count",
    "activity_authority_counts", "activity_status_counts",
    "activity_has_authoritative_or_strong", "missingness_reason",
}
SKO030_QA_MAP = {
    "subjects_with_resolved_geography": "subjects_with_resolved_geography",
    "subjects_with_unresolved_geography": "subjects_with_unresolved_geography",
    "subjects_with_compatible_indicators": "subjects_with_compatible_indicators",
    "subjects_with_resolved_geography_no_compatible_indicator": "subjects_with_resolved_geography_no_compatible_indicator",
    "blocked_geography_version_mismatch_count": "blocked_geography_version_mismatch_count",
    "subjects_with_any_activity_evidence": "subjects_with_any_activity_evidence",
    "subjects_with_authoritative_or_strong_activity_evidence": "subjects_with_authoritative_or_strong_activity_evidence",
    "subjects_with_contextual_or_weak_activity_evidence_only": "subjects_with_contextual_or_weak_only_activity_evidence",
    "subjects_with_no_activity_evidence": "subjects_with_no_activity_evidence",
}


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    required = {"preparation_version", "allowed_subject_types", "allowed_location_roles", "selected_values", "readiness_statuses", "required_subject_fields", "contextual_claim_boundary"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration missing keys: {sorted(missing)}")
    if {_clean(v) for v in config["allowed_subject_types"]} != SUBJECT_TYPES:
        raise ValueError("allowed_subject_types must match the accepted subject types")
    if {_clean(v) for v in config["allowed_location_roles"]} != LOCATION_ROLES:
        raise ValueError("allowed_location_roles must match the accepted location roles")
    if set(config["readiness_statuses"]) != {"ready_for_sko_032", "ready_with_known_gaps", "blocked"}:
        raise ValueError("Invalid readiness statuses")
    return config


def validate_selected_subjects(rows: list[dict[str, str]], config: Mapping[str, Any], *, allow_identical_duplicates: bool = False) -> tuple[list[dict[str, str]], dict[str, int]]:
    required = set(config["required_subject_fields"])
    columns = set(rows[0]) if rows else set()
    missing = required - columns
    if missing:
        raise ValueError(f"pilot subjects missing columns: {sorted(missing)}")
    selected_values = {_clean(v).casefold() for v in config["selected_values"]}
    accepted: dict[tuple[str, str], dict[str, str]] = {}
    duplicate_keys = 0
    for source in rows:
        if _clean(source.get("selected")).casefold() not in selected_values:
            continue
        row = {field: _clean(source.get(field)) for field in SUBJECT_FIELDS}
        subject_type, subject_id = row["subject_type"], row["subject_id"]
        if subject_type not in SUBJECT_TYPES:
            raise ValueError(f"Unsupported subject_type: {subject_type!r}")
        if not subject_id:
            raise ValueError("Selected pilot subject requires subject_id")
        if row["location_role"] not in LOCATION_ROLES:
            raise ValueError(f"Unsupported location_role: {row['location_role']!r}")
        if subject_type == "canonical_entity" and not row["entity_id"]:
            raise ValueError("canonical_entity subject requires its existing entity_id")
        key = (subject_type, subject_id)
        if key in accepted:
            if accepted[key] != row:
                raise ValueError(f"Conflicting selected subject: {key!r}")
            duplicate_keys += 1
            if not allow_identical_duplicates:
                raise ValueError(f"Duplicate selected subject: {key!r}")
            continue
        accepted[key] = row
    ordered = [accepted[key] for key in sorted(accepted)]
    return ordered, {"duplicate_subject_keys": duplicate_keys, "conflicting_subject_keys": 0}


def _subject_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (_clean(row.get("subject_type")), _clean(row.get("subject_id")))


def build_context_review(subjects: list[dict[str, str]], context_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    columns = set(context_rows[0]) if context_rows else set()
    missing = SKO030_REQUIRED - columns
    if context_rows and missing:
        raise ValueError(f"SKO-030 context input missing columns: {sorted(missing)}")
    subject_map = {_subject_key(row): row for row in subjects}
    seen_context: set[tuple[str, ...]] = set()
    output: list[dict[str, str]] = []
    if not context_rows:
        for subject in subjects:
            row = {field: "" for field in CONTEXT_REVIEW_FIELDS}
            for field in ("subject_type", "subject_id", "entity_id", "supplier_display_name", "source_client", "source_supplier_record_key", "location_role", "country", "postcode"):
                row[field] = subject.get(field, "")
            row.update({
                "record_type": "preparation_only",
                "missingness_reason": "contextual_runtime_not_available",
                "review_required": "false",
            })
            output.append(row)
        return output
    for context in context_rows:
        key = _subject_key(context)
        if key not in subject_map:
            continue
        subject = subject_map[key]
        if _clean(context.get("entity_id")) != subject["entity_id"]:
            raise ValueError(f"SKO-030 entity_id conflicts with pilot subject {key!r}")
        identity = tuple(_clean(context.get(field)) for field in ("context_integration_id", "record_type", "indicator_id", "period", "geography_code"))
        if identity in seen_context:
            raise ValueError(f"Duplicate SKO-030 contextual record for {key!r}")
        seen_context.add(identity)
        row = {field: "" for field in CONTEXT_REVIEW_FIELDS}
        for field in CONTEXT_REVIEW_FIELDS:
            row[field] = _clean(context.get(field))
        for field in ("subject_type", "subject_id", "entity_id", "supplier_display_name", "source_client", "source_supplier_record_key", "location_role", "country", "postcode"):
            row[field] = subject.get(field, "")
        row["mapping_confidence"] = _clean(context.get("mapping_confidence"))
        row["source_id"] = _clean(context.get("source_id"))
        row["source_name"] = _clean(context.get("source_name"))
        row["dataset_id"] = _clean(context.get("dataset_id"))
        row["dataset_version"] = _clean(context.get("dataset_version"))
        counts = json.loads(row["activity_status_counts"] or "{}")
        row["review_required"] = str(bool(row["missingness_reason"] or counts.get("review_required", 0) or counts.get("ambiguous", 0))).lower()
        output.append(row)
    output.sort(key=lambda row: tuple(row[field] for field in ("subject_type", "subject_id", "record_type", "indicator_id", "period", "geography_code")))
    return output


def build_qa(subjects: list[dict[str, str]], validation_qa: Mapping[str, int], sko030_qa: Mapping[str, Any], *, tracked_live_files: int) -> dict[str, Any]:
    keys = {_subject_key(row) for row in subjects}
    entities = {row["entity_id"] for row in subjects if row["entity_id"]}
    qa: dict[str, Any] = {
        "selected_subjects": len(subjects),
        "canonical_entity_subjects": sum(row["subject_type"] == "canonical_entity" for row in subjects),
        "supplier_observation_subjects": sum(row["subject_type"] == "supplier_observation" for row in subjects),
        "distinct_entity_ids": len(entities),
        "duplicate_subject_keys": int(validation_qa.get("duplicate_subject_keys", 0)),
        "conflicting_subject_keys": int(validation_qa.get("conflicting_subject_keys", 0)),
        "canonical_subjects_missing_entity_id": sum(row["subject_type"] == "canonical_entity" and not row["entity_id"] for row in subjects),
        "supplier_observation_entity_id_context_count": sum(row["subject_type"] == "supplier_observation" and bool(row["entity_id"]) for row in subjects),
        "subjects_with_country": sum(bool(row["country"]) for row in subjects),
        "subjects_with_postcode": sum(bool(row["postcode"]) for row in subjects),
        "subjects_with_country_and_postcode": sum(bool(row["country"] and row["postcode"]) for row in subjects),
        "subjects_lacking_sufficient_geography_input": sum(not (row["country"] and row["postcode"]) for row in subjects),
        "location_role_registered_or_hq": sum(row["location_role"] == "registered_or_hq" for row in subjects),
        "location_role_contracted_activity": sum(row["location_role"] == "contracted_activity" for row in subjects),
        "location_role_unknown": sum(row["location_role"] == "unknown" for row in subjects),
        "tracked_live_files": int(tracked_live_files),
    }
    runtime_available = bool(sko030_qa)
    qa.update({
        "sko030_runtime_available": runtime_available,
        "geography_runtime_available": runtime_available,
        "indicator_runtime_available": runtime_available,
        "activity_runtime_available": runtime_available,
    })
    if sko030_qa:
        if int(sko030_qa.get("selected_subjects", -1)) != len(keys):
            raise ValueError("SKO-030 QA selected_subjects does not match pilot population")
        for upstream, local in SKO030_QA_MAP.items():
            if upstream not in sko030_qa:
                raise ValueError(f"SKO-030 QA missing metric: {upstream}")
            qa[local] = int(sko030_qa[upstream])
        qa["subjects_with_activity_review_required"] = int(sko030_qa.get("activity_review_required_rows", 0))
    else:
        for local in SKO030_QA_MAP.values():
            qa[local] = None
        qa["subjects_with_activity_review_required"] = None
    qa["geography_scheme_version_combinations"] = []
    return qa


def readiness_decision(qa: Mapping[str, Any], *, structural_errors: Iterable[str] = (), dependencies_present: bool, context_row_count: int) -> dict[str, Any]:
    blockers = sorted(set(structural_errors))
    if int(qa.get("duplicate_subject_keys", 0)) or int(qa.get("conflicting_subject_keys", 0)) or int(qa.get("canonical_subjects_missing_entity_id", 0)) or int(qa.get("tracked_live_files", 0)):
        blockers.append("structural_or_governance_failure")
    blockers = sorted(set(blockers))
    gaps = []
    if int(qa.get("subjects_lacking_sufficient_geography_input", 0)):
        gaps.append("subjects_lacking_sufficient_geography_input")
    if not qa.get("sko030_runtime_available", False):
        gaps.append("contextual_runtime_not_available")
    if qa.get("subjects_with_unresolved_geography"):
        gaps.append("unresolved_geography")
    if qa.get("subjects_with_resolved_geography_no_compatible_indicator"):
        gaps.append("resolved_geography_no_compatible_indicator")
    if qa.get("subjects_with_no_activity_evidence"):
        gaps.append("no_activity_evidence")
    if not dependencies_present:
        gaps.append("runtime_dependencies_absent")
    if blockers:
        status = "blocked"
    elif gaps or not context_row_count:
        status = "ready_with_known_gaps"
    else:
        status = "ready_for_sko_032"
    return {"status": status, "structural_blockers": blockers, "known_gaps": sorted(set(gaps))}


def _dependency_entries(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(sorted(entry.items())) for entry in sorted(entries, key=_canonical_json)]


def build_manifest(subjects: list[dict[str, str]], *, pilot_id: str, config: Mapping[str, Any], prepared_at: str, repository_commit: str, input_files: Iterable[Mapping[str, Any]], geography_dependencies: Iterable[Mapping[str, Any]] = (), indicator_dependencies: Iterable[Mapping[str, Any]] = (), activity_dependencies: Iterable[Mapping[str, Any]] = (), expected_local_root: str, tracked_live_files: int) -> dict[str, Any]:
    if not all(_clean(v) for v in (pilot_id, prepared_at, repository_commit, expected_local_root)):
        raise ValueError("pilot_id, prepared_at, repository_commit and expected_local_root are required")
    return {
        "pilot_id": pilot_id, "preparation_version": _clean(config["preparation_version"]),
        "prepared_at": prepared_at, "repository_commit": repository_commit,
        "selected_subject_count": len(subjects),
        "canonical_entity_subject_count": sum(row["subject_type"] == "canonical_entity" for row in subjects),
        "supplier_observation_subject_count": sum(row["subject_type"] == "supplier_observation" for row in subjects),
        "distinct_entity_id_count": len({row["entity_id"] for row in subjects if row["entity_id"]}),
        "source_client_counts": dict(sorted(Counter(row["source_client"] for row in subjects if row["source_client"]).items())),
        "location_role_counts": dict(sorted(Counter(row["location_role"] for row in subjects).items())),
        "input_files": _dependency_entries(input_files),
        "geography_dependencies": _dependency_entries(geography_dependencies),
        "indicator_dependencies": _dependency_entries(indicator_dependencies),
        "activity_dependencies": _dependency_entries(activity_dependencies),
        "live_data_safety": {"expected_local_root": expected_local_root, "tracked_live_file_count": int(tracked_live_files)},
    }


def write_csv(rows: Iterable[Mapping[str, Any]], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    raw_subjects = read_csv(args.pilot_subjects)
    subjects, validation = validate_selected_subjects(raw_subjects, config, allow_identical_duplicates=args.allow_identical_duplicates)
    context = read_csv(args.context_integration) if args.context_integration else []
    upstream_qa = read_json(args.context_qa) if args.context_qa else {}
    qa = build_qa(subjects, validation, upstream_qa, tracked_live_files=args.tracked_live_files)
    review = build_context_review(subjects, context)
    qa["geography_scheme_version_combinations"] = sorted({f"{row['geography_scheme']}:{row['geography_version']}" for row in review if row["geography_scheme"] and row["geography_version"]})
    inputs = [{"logical_role": "pilot_subjects", "path_reference": str(args.pilot_subjects), "sha256": file_sha256(args.pilot_subjects), "row_count": len(raw_subjects)}]
    if args.context_integration:
        inputs.append({"logical_role": "sko_030_context", "path_reference": str(args.context_integration), "sha256": file_sha256(args.context_integration), "row_count": len(context)})
    if args.context_qa:
        inputs.append({"logical_role": "sko_030_qa", "path_reference": str(args.context_qa), "sha256": file_sha256(args.context_qa)})
    manifest = build_manifest(subjects, pilot_id=args.pilot_id, config=config, prepared_at=args.prepared_at, repository_commit=args.repository_commit, input_files=inputs, expected_local_root=args.expected_local_root, tracked_live_files=args.tracked_live_files)
    readiness = readiness_decision(qa, dependencies_present=args.dependencies_present, context_row_count=len(review))
    readiness.update({"pilot_id": args.pilot_id, "preparation_version": config["preparation_version"], "prepared_at": args.prepared_at})
    write_csv(subjects, args.prepared_subjects_output, SUBJECT_FIELDS)
    write_json(manifest, args.manifest_output)
    write_json(qa, args.qa_output)
    write_csv(review, args.context_review_output, CONTEXT_REVIEW_FIELDS)
    write_json(readiness, args.readiness_output)
    return {"subjects": subjects, "manifest": manifest, "qa": qa, "review": review, "readiness": readiness}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot-subjects", type=Path, required=True)
    parser.add_argument("--context-integration", type=Path)
    parser.add_argument("--context-qa", type=Path)
    parser.add_argument("--pilot-id", required=True)
    parser.add_argument("--prepared-at", required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--expected-local-root", default="data/pilots/sko-031/")
    parser.add_argument("--tracked-live-files", type=int, default=0)
    parser.add_argument("--dependencies-present", action="store_true")
    parser.add_argument("--allow-identical-duplicates", action="store_true")
    parser.add_argument("--prepared-subjects-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--qa-output", type=Path, required=True)
    parser.add_argument("--context-review-output", type=Path, required=True)
    parser.add_argument("--readiness-output", type=Path, required=True)
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
