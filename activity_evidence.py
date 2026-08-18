#!/usr/bin/env python3
"""SKO-029 governed, selected-only activity/industry evidence capture.

This module preserves configured local source evidence without classifying it,
crosswalking it to NACE, or creating or changing canonical entities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


OUTPUT_FIELDS = [
    "activity_evidence_id", "subject_type", "subject_id", "entity_id",
    "evidence_type", "source_name", "source_record_id", "source_reference",
    "source_code_field", "source_description_field", "activity_scheme",
    "activity_version", "activity_code_raw", "activity_description_raw",
    "is_principal_activity", "evidence_authority", "evidence_status",
    "source_version", "source_file_sha256", "retrieved_at", "extracted_at",
    "evidence_fingerprint", "schema_version",
]
QA_FIELDS = [
    "selected_subjects", "subjects_with_any_activity_evidence",
    "subjects_with_authoritative_or_strong_evidence",
    "subjects_with_contextual_or_weak_evidence_only",
    "subjects_with_multiple_evidence_items", "subjects_with_no_activity_evidence",
    "review_required", "ambiguous",
]
SUBJECT_REQUIRED_FIELDS = {"selected", "subject_type", "subject_id", "entity_id"}
SOURCE_REQUIRED_FIELDS = {"subject_type", "subject_id"}


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _raw(value: Any) -> str:
    return "" if value is None else str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


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
        "schema_version", "capture_method", "selected_values",
        "allowed_subject_types", "allowed_evidence_types", "permitted_schemes",
        "principal_values", "authority_status", "sources",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration missing keys: {sorted(missing)}")
    if config["capture_method"] != "configured_local_activity_evidence":
        raise ValueError("SKO-029 v1 only supports configured_local_activity_evidence")
    subject_types = {_clean(v) for v in config["allowed_subject_types"]}
    if subject_types != {"canonical_entity", "supplier_observation"}:
        raise ValueError("Allowed subject types must be canonical_entity and supplier_observation")
    evidence_types = {_clean(v) for v in config["allowed_evidence_types"]}
    authorities = {"authoritative", "strong", "supporting", "contextual", "weak"}
    statuses = {"usable", "review_required", "ambiguous", "insufficient_evidence", "unsupported"}
    mappings = config["authority_status"]
    if set(mappings) != evidence_types:
        raise ValueError("authority_status must map every allowed evidence type exactly once")
    for evidence_type, rule in mappings.items():
        if _clean(rule.get("authority")) not in authorities or _clean(rule.get("status")) not in statuses:
            raise ValueError(f"Invalid authority/status rule for {evidence_type!r}")
    schemes = {_clean(v) for v in config["permitted_schemes"]}
    if not {"", "CNAE", "CCAE", "NACE", "other"}.issubset(schemes):
        raise ValueError("Permitted schemes must include blank, CNAE, CCAE, NACE, and other")
    principal = config["principal_values"]
    if set(principal) != {"true", "false", "unknown"}:
        raise ValueError("principal_values must define true, false, and unknown")
    if not isinstance(config["sources"], dict) or not config["sources"]:
        raise ValueError("Configuration requires at least one source")
    for source_id, source in config["sources"].items():
        needed = {"file_name", "source_name", "source_version", "source_reference", "record_id_field", "evidence_items"}
        absent = needed - set(source)
        if absent:
            raise ValueError(f"Source {source_id!r} missing keys: {sorted(absent)}")
        if not source["evidence_items"]:
            raise ValueError(f"Source {source_id!r} requires evidence_items")
        for item in source["evidence_items"]:
            if _clean(item.get("evidence_type")) not in evidence_types:
                raise ValueError(f"Source {source_id!r} has unsupported evidence_type")
            if _clean(item.get("scheme", "")) not in schemes:
                raise ValueError(f"Source {source_id!r} has unsupported scheme")
            if not (_clean(item.get("code_field")) or _clean(item.get("description_field"))):
                raise ValueError(f"Source {source_id!r} evidence item has no raw field")
    return config


def _selected_subjects(
    rows: list[dict[str, str]], config: Mapping[str, Any]
) -> dict[tuple[str, str], dict[str, str]]:
    _require_columns(rows, SUBJECT_REQUIRED_FIELDS, "selected-subject input")
    selected_values = {_clean(v).casefold() for v in config["selected_values"]}
    allowed = {_clean(v) for v in config["allowed_subject_types"]}
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if _clean(row.get("selected")).casefold() not in selected_values:
            continue
        subject_type, subject_id = _clean(row.get("subject_type")), _clean(row.get("subject_id"))
        entity_id = _clean(row.get("entity_id"))
        if subject_type not in allowed:
            raise ValueError(f"Unsupported subject_type: {subject_type!r}")
        if not subject_id:
            raise ValueError("Selected subject requires subject_id")
        if subject_type == "canonical_entity" and not entity_id:
            raise ValueError("canonical_entity subject requires its existing entity_id")
        key = (subject_type, subject_id)
        value = {"subject_type": subject_type, "subject_id": subject_id, "entity_id": entity_id}
        if key in selected and selected[key] != value:
            raise ValueError(f"Conflicting selected subject: {key!r}")
        selected[key] = value
    return selected


def _principal_value(raw_value: str, config: Mapping[str, Any]) -> str:
    value = _clean(raw_value).casefold()
    for result, configured in config["principal_values"].items():
        if value in {_clean(v).casefold() for v in configured}:
            return result
    return "unknown"


def _identity_content(row: Mapping[str, str]) -> dict[str, str]:
    fields = [
        "subject_type", "subject_id", "entity_id", "evidence_type", "source_name",
        "source_record_id", "source_reference", "source_code_field",
        "source_description_field", "activity_scheme", "activity_version",
        "activity_code_raw", "activity_description_raw", "is_principal_activity",
        "source_version", "schema_version",
    ]
    return {field: row[field] for field in fields}


def _conflict_key(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in (
        "subject_type", "subject_id", "evidence_type", "source_name",
        "source_record_id", "source_code_field", "source_description_field",
    ))


def capture_activity_evidence(
    selected_rows: list[dict[str, str]], *, config: Mapping[str, Any],
    input_dir: Path, retrieved_at: str, extracted_at: str,
) -> dict[str, Any]:
    """Capture evidence for explicitly selected subjects and compute pilot QA."""
    if not _clean(retrieved_at) or not _clean(extracted_at):
        raise ValueError("retrieved_at and extracted_at must be supplied explicitly")
    subjects = _selected_subjects(selected_rows, config)
    candidates: dict[str, dict[str, str]] = {}
    conflict_members: dict[tuple[str, ...], set[str]] = {}

    for source_id, source in sorted(dict(config["sources"]).items()):
        path = input_dir / _clean(source["file_name"])
        rows = read_csv(path)
        configured_fields = {_clean(source["record_id_field"])}
        for item in source["evidence_items"]:
            configured_fields.update(
                _clean(item.get(field))
                for field in ("code_field", "description_field", "version_field", "principal_field")
            )
        configured_fields.discard("")
        _require_columns(
            rows, SOURCE_REQUIRED_FIELDS | configured_fields, f"source {source_id!r}"
        )
        source_sha = file_sha256(path)
        for source_row in rows:
            key = (_clean(source_row.get("subject_type")), _clean(source_row.get("subject_id")))
            subject = subjects.get(key)
            if subject is None:
                continue
            if not _clean(source_row.get(_clean(source["record_id_field"]))):
                raise ValueError(f"Selected evidence row in source {source_id!r} requires record ID")
            for item in source["evidence_items"]:
                code_field = _clean(item.get("code_field"))
                description_field = _clean(item.get("description_field"))
                code_raw = _raw(source_row.get(code_field)) if code_field else ""
                description_raw = _raw(source_row.get(description_field)) if description_field else ""
                if not code_raw and not description_raw:
                    continue
                evidence_type = _clean(item["evidence_type"])
                governance = config["authority_status"][evidence_type]
                version_field = _clean(item.get("version_field"))
                principal_field = _clean(item.get("principal_field"))
                activity_version = _raw(source_row.get(version_field)) if version_field else _clean(item.get("version", ""))
                status = _clean(governance["status"])
                if code_raw and _clean(item.get("scheme")) and not activity_version:
                    status = _clean(item.get("unresolved_version_status", "review_required"))
                output = {
                    **subject,
                    "evidence_type": evidence_type,
                    "source_name": _clean(source["source_name"]),
                    "source_record_id": _raw(source_row.get(_clean(source["record_id_field"]))),
                    "source_reference": _clean(source["source_reference"]),
                    "source_code_field": code_field,
                    "source_description_field": description_field,
                    "activity_scheme": _clean(item.get("scheme", "")),
                    "activity_version": activity_version,
                    "activity_code_raw": code_raw,
                    "activity_description_raw": description_raw,
                    "is_principal_activity": _principal_value(_raw(source_row.get(principal_field)), config) if principal_field else "unknown",
                    "evidence_authority": _clean(governance["authority"]),
                    "evidence_status": status,
                    "source_version": _clean(source["source_version"]),
                    "source_file_sha256": source_sha,
                    "retrieved_at": retrieved_at,
                    "extracted_at": extracted_at,
                    "schema_version": _clean(config["schema_version"]),
                }
                identity_json = _canonical_json(_identity_content(output))
                evidence_id = "actev_" + _sha256_text(identity_json)[:24]
                output["activity_evidence_id"] = evidence_id
                candidates[evidence_id] = output
                conflict_members.setdefault(_conflict_key(output), set()).add(evidence_id)

    for evidence_ids in conflict_members.values():
        if len(evidence_ids) > 1:
            for evidence_id in evidence_ids:
                candidates[evidence_id]["evidence_status"] = "ambiguous"

    evidence = []
    for row in candidates.values():
        fingerprint_content = {field: row[field] for field in OUTPUT_FIELDS if field != "evidence_fingerprint"}
        row["evidence_fingerprint"] = _sha256_text(_canonical_json(fingerprint_content))
        evidence.append({field: row[field] for field in OUTPUT_FIELDS})
    evidence.sort(key=lambda row: (
        row["subject_type"], row["subject_id"], row["evidence_type"],
        row["source_name"], row["source_record_id"], row["activity_scheme"],
        row["activity_version"], row["activity_code_raw"],
        row["activity_description_raw"], row["is_principal_activity"],
        row["activity_evidence_id"],
    ))
    return {"evidence": evidence, "qa": pilot_qa(subjects, evidence)}


def pilot_qa(
    subjects: Mapping[tuple[str, str], Mapping[str, str]],
    evidence: Iterable[Mapping[str, str]],
) -> dict[str, int]:
    rows = list(evidence)
    by_subject: dict[tuple[str, str], list[Mapping[str, str]]] = {key: [] for key in subjects}
    for row in rows:
        by_subject[(row["subject_type"], row["subject_id"])].append(row)
    with_any = {key for key, values in by_subject.items() if values}
    strong = {
        key for key, values in by_subject.items()
        if any(row["evidence_authority"] in {"authoritative", "strong"} for row in values)
    }
    weak_only = {
        key for key in with_any
        if all(row["evidence_authority"] in {"contextual", "weak"} for row in by_subject[key])
    }
    qa = {
        "selected_subjects": len(subjects),
        "subjects_with_any_activity_evidence": len(with_any),
        "subjects_with_authoritative_or_strong_evidence": len(strong),
        "subjects_with_contextual_or_weak_evidence_only": len(weak_only),
        "subjects_with_multiple_evidence_items": sum(len(values) > 1 for values in by_subject.values()),
        "subjects_with_no_activity_evidence": len(subjects) - len(with_any),
        "review_required": sum(row["evidence_status"] == "review_required" for row in rows),
        "ambiguous": sum(row["evidence_status"] == "ambiguous" for row in rows),
    }
    return {field: qa[field] for field in QA_FIELDS}


def write_csv(rows: Iterable[Mapping[str, Any]], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_qa(qa: Mapping[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(qa), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selected-subjects", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa-output", type=Path, required=True)
    parser.add_argument("--retrieved-at", required=True)
    parser.add_argument("--extracted-at", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = capture_activity_evidence(
        read_csv(args.selected_subjects), config=load_config(args.config),
        input_dir=args.input_dir, retrieved_at=args.retrieved_at,
        extracted_at=args.extracted_at,
    )
    write_csv(result["evidence"], args.output, OUTPUT_FIELDS)
    write_qa(result["qa"], args.qa_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
