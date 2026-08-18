#!/usr/bin/env python3
"""SKO-030 deterministic orchestration of accepted contextual evidence.

This module joins already-produced SKO-027, SKO-028, and SKO-029 records.  It
does not classify geography or activity, transform indicators, infer industry,
convert geography schemes/versions, or assert supplier impact or causality.
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
RESOLVED_STATUS = "resolved"
GEOGRAPHY_FIELDS = (
    "geography_scheme", "geography_version", "geography_level", "geography_code"
)
OUTPUT_FIELDS = [
    "context_integration_id", "record_type", "subject_type", "subject_id", "entity_id",
    "classification_id", "geography_scheme", "geography_version", "geography_level",
    "geography_code", "geography_name", "mapping_status", "observation_id",
    "indicator_id", "indicator_name", "indicator_theme", "period", "period_type",
    "value", "unit", "value_status", "activity_evidence_count",
    "activity_evidence_ids", "activity_authority_counts", "activity_status_counts",
    "activity_has_authoritative_or_strong", "activity_review_required_count",
    "activity_ambiguous_count", "missingness_reason", "integration_version",
    "integrated_at", "record_fingerprint",
]
QA_FIELDS = [
    "selected_subjects", "canonical_selected_subjects", "non_canonical_selected_subjects",
    "geography_classification_rows", "resolved_geography_assertions",
    "unresolved_geography_assertions", "subjects_with_resolved_geography",
    "subjects_with_unresolved_geography", "subjects_with_compatible_indicators",
    "subjects_with_resolved_geography_no_compatible_indicator",
    "subjects_with_any_external_indicator",
    "subjects_with_no_compatible_external_indicator",
    "subjects_with_all_three_layers",
    "subjects_with_geography_and_indicators_only",
    "subjects_with_geography_and_activity_only",
    "subjects_with_activity_only",
    "distinct_indicator_ids", "distinct_geography_schemes",
    "distinct_geography_versions", "blocked_geography_version_mismatch_count",
    "compatible_indicator_observations", "enriched_rows", "staging_rows", "output_rows",
    "subjects_with_any_activity_evidence",
    "subjects_with_authoritative_or_strong_activity_evidence",
    "subjects_with_contextual_or_weak_activity_evidence_only",
    "subjects_with_multiple_activity_evidence_items", "subjects_with_no_activity_evidence",
    "activity_evidence_rows", "activity_review_required_rows", "activity_ambiguous_rows",
    "exact_geography_join_matches", "entity_id_only_join_matches",
    "case_j_multiple_resolved_levels",
]


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _require_columns(rows: list[dict[str, str]], required: set[str], label: str) -> None:
    columns = set(rows[0]) if rows else set()
    missing = required - columns
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    required = {
        "integration_version", "selected_values", "allowed_subject_types",
        "resolved_mapping_status", "strong_activity_authorities",
        "weak_only_activity_authorities", "review_activity_statuses",
        "geography_join_fields", "subject_join_fields",
    }
    if {_clean(v) for v in config["review_activity_statuses"]} != {"review_required", "ambiguous"}:
        raise ValueError("review_activity_statuses must contain review_required and ambiguous")
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration missing keys: {sorted(missing)}")
    if list(config["subject_join_fields"]) != ["subject_type", "subject_id"]:
        raise ValueError("subject_join_fields must be subject_type + subject_id")
    if tuple(config["geography_join_fields"]) != GEOGRAPHY_FIELDS:
        raise ValueError("geography_join_fields must be the exact four-field geography key")
    if {_clean(v) for v in config["allowed_subject_types"]} != SUBJECT_TYPES:
        raise ValueError("allowed_subject_types must contain both accepted subject types")
    if _clean(config["resolved_mapping_status"]) != RESOLVED_STATUS:
        raise ValueError("resolved_mapping_status must be resolved")
    return config


def _selected_subjects(
    rows: list[dict[str, str]], config: Mapping[str, Any]
) -> dict[tuple[str, str], dict[str, str]]:
    _require_columns(rows, {"selected", "subject_type", "subject_id", "entity_id"}, "subjects")
    selected_values = {_clean(v).casefold() for v in config["selected_values"]}
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if _clean(row.get("selected")).casefold() not in selected_values:
            continue
        subject_type = _clean(row.get("subject_type"))
        subject_id = _clean(row.get("subject_id"))
        entity_id = _clean(row.get("entity_id"))
        if subject_type not in SUBJECT_TYPES or not subject_id:
            raise ValueError("Selected subject requires an accepted subject_type and subject_id")
        if subject_type == "canonical_entity" and not entity_id:
            raise ValueError("canonical_entity subject requires entity_id")
        key = (subject_type, subject_id)
        value = {"subject_type": subject_type, "subject_id": subject_id, "entity_id": entity_id}
        if key in selected and selected[key] != value:
            raise ValueError(f"Conflicting selected subject: {key!r}")
        selected[key] = value
    return selected


def _subject_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (_clean(row.get("subject_type")), _clean(row.get("subject_id")))


def _geography_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return tuple(_clean(row.get(field)) for field in GEOGRAPHY_FIELDS)  # type: ignore[return-value]


def _activity_summary(rows: Iterable[Mapping[str, str]], config: Mapping[str, Any]) -> dict[str, str]:
    evidence = sorted(rows, key=lambda row: row["activity_evidence_id"])
    authorities = Counter(row["evidence_authority"] for row in evidence)
    statuses = Counter(row["evidence_status"] for row in evidence)
    strong = {_clean(v) for v in config["strong_activity_authorities"]}
    return {
        "activity_evidence_count": str(len(evidence)),
        "activity_evidence_ids": _canonical_json([row["activity_evidence_id"] for row in evidence]),
        "activity_authority_counts": _canonical_json(dict(sorted(authorities.items()))),
        "activity_status_counts": _canonical_json(dict(sorted(statuses.items()))),
        "activity_has_authoritative_or_strong": str(any(row["evidence_authority"] in strong for row in evidence)).lower(),
        "activity_review_required_count": str(statuses.get("review_required", 0)),
        "activity_ambiguous_count": str(statuses.get("ambiguous", 0)),
    }


def integrate_context(
    selected_rows: list[dict[str, str]], classification_rows: list[dict[str, str]],
    observation_rows: list[dict[str, str]], activity_rows: list[dict[str, str]],
    *, config: Mapping[str, Any], integrated_at: str,
) -> dict[str, Any]:
    """Join accepted records without altering their semantics or granularity."""
    if not _clean(integrated_at):
        raise ValueError("integrated_at must be supplied explicitly")
    subjects = _selected_subjects(selected_rows, config)
    _require_columns(classification_rows, {
        "classification_id", "subject_type", "subject_id", "entity_id", "geography_scheme",
        "geography_version", "geography_level", "geography_code", "geography_name", "mapping_status",
    }, "classifications")
    _require_columns(observation_rows, {
        "observation_id", "indicator_id", "indicator_name", "indicator_theme", "geography_scheme",
        "geography_version", "geography_level", "geography_code", "period", "period_type",
        "value", "unit", "value_status",
    }, "observations")
    _require_columns(activity_rows, {
        "activity_evidence_id", "subject_type", "subject_id", "entity_id",
        "evidence_authority", "evidence_status",
    }, "activity evidence")

    classifications: list[dict[str, str]] = []
    classification_ids: set[str] = set()
    for row in classification_rows:
        key = _subject_key(row)
        if key not in subjects:
            continue
        if _clean(row["entity_id"]) != subjects[key]["entity_id"]:
            raise ValueError(f"Classification entity_id conflicts with selected subject {key!r}")
        classification_id = _clean(row["classification_id"])
        if classification_id in classification_ids:
            raise ValueError(f"Duplicate classification_id: {classification_id}")
        classification_ids.add(classification_id)
        classifications.append(row)
    classified_subjects = {_subject_key(row) for row in classifications}
    missing_classifications = set(subjects) - classified_subjects
    if missing_classifications:
        raise ValueError(f"Selected subjects missing classification rows: {sorted(missing_classifications)}")

    activity_by_subject: dict[tuple[str, str], list[dict[str, str]]] = {key: [] for key in subjects}
    activity_ids: set[str] = set()
    for row in activity_rows:
        key = _subject_key(row)
        if key not in subjects:
            continue
        if _clean(row["entity_id"]) != subjects[key]["entity_id"]:
            raise ValueError(f"Activity entity_id conflicts with selected subject {key!r}")
        evidence_id = _clean(row["activity_evidence_id"])
        if evidence_id in activity_ids:
            raise ValueError(f"Duplicate activity_evidence_id: {evidence_id}")
        activity_ids.add(evidence_id)
        activity_by_subject[key].append(row)

    observations_by_geo: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    observation_ids: set[str] = set()
    for row in observation_rows:
        observation_id = _clean(row["observation_id"])
        if observation_id in observation_ids:
            raise ValueError(f"Duplicate observation_id: {observation_id}")
        observation_ids.add(observation_id)
        observations_by_geo.setdefault(_geography_key(row), []).append(row)
    for values in observations_by_geo.values():
        values.sort(key=lambda row: (
            row["indicator_id"], row["period"], row["period_type"], row["observation_id"]
        ))

    outputs: list[dict[str, str]] = []
    subjects_with_compatible: set[tuple[str, str]] = set()
    resolved_no_indicator: set[tuple[str, str]] = set()
    for classification in sorted(classifications, key=lambda row: (_subject_key(row), row["classification_id"])):
        key = _subject_key(classification)
        subject = subjects[key]
        summary = _activity_summary(activity_by_subject[key], config)
        resolved = _clean(classification["mapping_status"]) == RESOLVED_STATUS
        matches = observations_by_geo.get(_geography_key(classification), []) if resolved else []
        if matches:
            subjects_with_compatible.add(key)
            candidates: list[tuple[str, Mapping[str, str], str]] = [
                ("enriched", observation, "") for observation in matches
            ]
        elif resolved:
            resolved_no_indicator.add(key)
            candidates = [("staging", {}, "no_compatible_indicator")]
        else:
            candidates = [("staging", {}, "unresolved_geography")]
        for record_type, observation, missingness_reason in candidates:
            base = {
                "record_type": record_type, **subject,
                "classification_id": classification["classification_id"],
                "geography_scheme": _clean(classification["geography_scheme"]),
                "geography_version": _clean(classification["geography_version"]),
                "geography_level": _clean(classification["geography_level"]),
                "geography_code": _clean(classification["geography_code"]),
                "geography_name": _clean(classification["geography_name"]),
                "mapping_status": _clean(classification["mapping_status"]),
                "observation_id": _clean(observation.get("observation_id")),
                "indicator_id": _clean(observation.get("indicator_id")),
                "indicator_name": _clean(observation.get("indicator_name")),
                "indicator_theme": _clean(observation.get("indicator_theme")),
                "period": _clean(observation.get("period")),
                "period_type": _clean(observation.get("period_type")),
                "value": _clean(observation.get("value")),
                "unit": _clean(observation.get("unit")),
                "value_status": _clean(observation.get("value_status")),
                **summary, "missingness_reason": missingness_reason,
                "integration_version": _clean(config["integration_version"]),
                "integrated_at": integrated_at,
            }
            identity = {
                "subject_type": base["subject_type"], "subject_id": base["subject_id"],
                "classification_id": base["classification_id"],
                "observation_id": base["observation_id"], "missingness_reason": missingness_reason,
                "integration_version": base["integration_version"],
            }
            base["context_integration_id"] = "ctx_" + _sha256_text(_canonical_json(identity))[:24]
            fingerprint_content = {field: base[field] for field in OUTPUT_FIELDS if field != "record_fingerprint"}
            base["record_fingerprint"] = _sha256_text(_canonical_json(fingerprint_content))
            outputs.append({field: base[field] for field in OUTPUT_FIELDS})

    outputs.sort(key=lambda row: (
        row["subject_type"], row["subject_id"], row["classification_id"],
        row["record_type"], row["indicator_id"], row["period"], row["observation_id"],
    ))
    resolved_rows = [row for row in classifications if row["mapping_status"] == RESOLVED_STATUS]
    unresolved_rows = [row for row in classifications if row["mapping_status"] != RESOLVED_STATUS]
    resolved_subjects = {_subject_key(row) for row in resolved_rows}
    unresolved_subjects = {_subject_key(row) for row in unresolved_rows}
    with_activity = {key for key, values in activity_by_subject.items() if values}
    strong_authorities = {_clean(v) for v in config["strong_activity_authorities"]}
    weak_authorities = {_clean(v) for v in config["weak_only_activity_authorities"]}
    strong_subjects = {
        key for key, values in activity_by_subject.items()
        if any(row["evidence_authority"] in strong_authorities for row in values)
    }
    weak_only = {
        key for key in with_activity
        if all(row["evidence_authority"] in weak_authorities for row in activity_by_subject[key])
    }
    all_three_layers = resolved_subjects & subjects_with_compatible & with_activity
    geography_and_indicators_only = (
        (resolved_subjects & subjects_with_compatible) - with_activity
    )
    geography_and_activity_only = (
        (resolved_subjects & with_activity) - subjects_with_compatible
    )
    activity_only = unresolved_subjects & with_activity
    observation_versions_by_partial_geo: dict[tuple[str, str, str], set[str]] = {}
    for observation in observation_rows:
        partial_key = (
            _clean(observation["geography_scheme"]),
            _clean(observation["geography_level"]),
            _clean(observation["geography_code"]),
        )
        observation_versions_by_partial_geo.setdefault(partial_key, set()).add(
            _clean(observation["geography_version"])
        )
    blocked_version_mismatch_subjects = {
        _subject_key(row)
        for row in resolved_rows
        if _subject_key(row) not in subjects_with_compatible
        and observation_versions_by_partial_geo.get(
            (
                _clean(row["geography_scheme"]),
                _clean(row["geography_level"]),
                _clean(row["geography_code"]),
            ),
            set(),
        ) - {_clean(row["geography_version"])}
    }
    qa = {
        "selected_subjects": len(subjects),
        "canonical_selected_subjects": sum(key[0] == "canonical_entity" for key in subjects),
        "non_canonical_selected_subjects": sum(key[0] == "supplier_observation" for key in subjects),
        "geography_classification_rows": len(classifications),
        "resolved_geography_assertions": len(resolved_rows),
        "unresolved_geography_assertions": len(unresolved_rows),
        "subjects_with_resolved_geography": len(resolved_subjects),
        "subjects_with_unresolved_geography": len(unresolved_subjects),
        "subjects_with_compatible_indicators": len(subjects_with_compatible),
        "subjects_with_resolved_geography_no_compatible_indicator": len(resolved_no_indicator),
        "subjects_with_any_external_indicator": len(subjects_with_compatible),
        "subjects_with_no_compatible_external_indicator": len(
            set(subjects) - subjects_with_compatible
        ),
        "subjects_with_all_three_layers": len(all_three_layers),
        "subjects_with_geography_and_indicators_only": len(geography_and_indicators_only),
        "subjects_with_geography_and_activity_only": len(geography_and_activity_only),
        "subjects_with_activity_only": len(activity_only),
        "distinct_indicator_ids": len({
            row["indicator_id"] for row in outputs if row["record_type"] == "enriched"
        }),
        "distinct_geography_schemes": len({
            _clean(row["geography_scheme"])
            for row in resolved_rows if _clean(row["geography_scheme"])
        }),
        "distinct_geography_versions": len({
            _clean(row["geography_version"])
            for row in resolved_rows if _clean(row["geography_version"])
        }),
        "blocked_geography_version_mismatch_count": len(blocked_version_mismatch_subjects),
        "compatible_indicator_observations": sum(row["record_type"] == "enriched" for row in outputs),
        "enriched_rows": sum(row["record_type"] == "enriched" for row in outputs),
        "staging_rows": sum(row["record_type"] == "staging" for row in outputs),
        "output_rows": len(outputs),
        "subjects_with_any_activity_evidence": len(with_activity),
        "subjects_with_authoritative_or_strong_activity_evidence": len(strong_subjects),
        "subjects_with_contextual_or_weak_activity_evidence_only": len(weak_only),
        "subjects_with_multiple_activity_evidence_items": sum(len(v) > 1 for v in activity_by_subject.values()),
        "subjects_with_no_activity_evidence": len(subjects) - len(with_activity),
        "activity_evidence_rows": sum(len(v) for v in activity_by_subject.values()),
        "activity_review_required_rows": sum(row["evidence_status"] == "review_required" for row in activity_rows if _subject_key(row) in subjects),
        "activity_ambiguous_rows": sum(row["evidence_status"] == "ambiguous" for row in activity_rows if _subject_key(row) in subjects),
        "exact_geography_join_matches": sum(row["record_type"] == "enriched" for row in outputs),
        "entity_id_only_join_matches": 0,
        "case_j_multiple_resolved_levels": 0,
    }
    return {"records": outputs, "qa": {field: qa[field] for field in QA_FIELDS}}


def write_csv(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_qa(qa: Mapping[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(qa), indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selected-subjects", type=Path, required=True)
    parser.add_argument("--geographic-classifications", type=Path, required=True)
    parser.add_argument("--external-indicators", type=Path, required=True)
    parser.add_argument("--activity-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa-output", type=Path, required=True)
    parser.add_argument("--integrated-at", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = integrate_context(
        read_csv(args.selected_subjects), read_csv(args.geographic_classifications),
        read_csv(args.external_indicators), read_csv(args.activity_evidence),
        config=load_config(args.config), integrated_at=args.integrated_at,
    )
    write_csv(result["records"], args.output)
    write_qa(result["qa"], args.qa_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
