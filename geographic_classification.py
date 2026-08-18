#!/usr/bin/env python3
"""SKO-027 governed, selected-only statistical-geography classification.

This module classifies location observations.  It does not create or alter
canonical entities, and deliberately contains no city, address, coordinate,
fuzzy, or directory-derived geographic inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


SUBJECT_TYPES = {"canonical_entity", "supplier_observation"}
LOCATION_ROLES = {"registered_or_hq", "contracted_activity", "unknown"}
OUTPUT_FIELDS = [
    "classification_id", "subject_type", "subject_id", "entity_id",
    "country", "postcode", "city", "address", "location_role",
    "geography_scheme", "geography_version", "geography_level", "geography_code",
    "geography_name", "mapping_status", "mapping_method", "mapping_source",
    "mapping_source_version", "mapping_source_sha256", "mapping_evidence",
    "mapping_confidence", "input_fingerprint", "classifier_version",
    "classified_at",
]


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_postcode(value: Any) -> str:
    """Conservatively normalize a postcode: trim, uppercase, collapse spaces."""
    return " ".join(_clean(value).upper().split())


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


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    required = {
        "classifier_version", "automatic_mapping_method", "selected_values",
        "valid_iso2_country_codes", "country_mapping", "statistical_geography_schemes",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration missing keys: {sorted(missing)}")
    if config["automatic_mapping_method"] != "postcode_correspondence":
        raise ValueError("E4.2 v1 only supports postcode_correspondence")
    return config


def normalize_country(value: Any, config: Mapping[str, Any]) -> tuple[str, bool]:
    """Return (ISO2 country, supported); mappings are explicitly configured."""
    raw = _clean(value)
    valid_iso2 = {_clean(code).upper() for code in config["valid_iso2_country_codes"]}
    if raw.upper() in valid_iso2:
        return raw.upper(), True
    mappings = {
        _clean(key).casefold(): _clean(mapped).upper()
        for key, mapped in dict(config["country_mapping"]).items()
    }
    mapped = mappings.get(raw.casefold(), "")
    if mapped in valid_iso2:
        return mapped, True
    return raw, False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _require_columns(rows: list[dict[str, str]], required: set[str], label: str) -> None:
    columns = set(rows[0]) if rows else set()
    missing = required - columns
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _validate_supplier(row: Mapping[str, str]) -> None:
    subject_type = _clean(row.get("subject_type"))
    role = _clean(row.get("location_role"))
    if subject_type not in SUBJECT_TYPES:
        raise ValueError(f"Unsupported subject_type: {subject_type!r}")
    if role not in LOCATION_ROLES:
        raise ValueError(f"Unsupported location_role: {role!r}")
    if not _clean(row.get("subject_id")):
        raise ValueError("Selected row requires subject_id")
    if subject_type == "canonical_entity" and not _clean(row.get("entity_id")):
        raise ValueError("canonical_entity row requires its existing entity_id")


def _result_key(row: Mapping[str, str]) -> tuple[str, str, str, str, str]:
    return (
        _clean(row.get("geography_scheme")),
        _clean(row.get("geography_version")),
        _clean(row.get("geography_level")),
        _clean(row.get("geography_code")),
        _clean(row.get("geography_name")),
    )


def _build_index(
    correspondence_rows: Iterable[Mapping[str, str]], config: Mapping[str, Any]
) -> dict[tuple[str, str], dict[tuple[str, str, str, str, str], list[dict[str, str]]]]:
    index: dict[tuple[str, str], dict[tuple[str, str, str, str, str], list[dict[str, str]]]] = {}
    governed_schemes = dict(config["statistical_geography_schemes"])
    for source_row in correspondence_rows:
        country, supported_country = normalize_country(source_row.get("country"), config)
        postcode = normalize_postcode(source_row.get("postcode"))
        result = _result_key(source_row)
        scheme, version, level, code, name = result
        governance = governed_schemes.get(scheme, {})
        allowed_countries = {
            _clean(value).upper() for value in governance.get("allowed_countries", [])
        }
        allowed_versions = {_clean(value) for value in governance.get("allowed_versions", [])}
        allowed_levels = {_clean(value) for value in governance.get("allowed_levels", [])}
        if not (
            supported_country and postcode and country in allowed_countries
            and version in allowed_versions and level in allowed_levels
            and code and name
        ):
            continue
        evidence = {
            "country": country, "postcode": postcode,
            "geography_scheme": scheme, "geography_version": version,
            "geography_level": level, "geography_code": code, "geography_name": name,
        }
        bucket = index.setdefault((country, postcode), {}).setdefault(result, [])
        if evidence not in bucket:
            bucket.append(evidence)
    return index


def classify_rows(
    supplier_rows: list[dict[str, str]],
    correspondence_rows: list[dict[str, str]],
    *, config: Mapping[str, Any], classified_at: str,
    mapping_source: str, mapping_source_version: str,
    mapping_source_sha256: str,
) -> list[dict[str, str]]:
    """Classify explicitly selected rows and return stable-sorted results."""
    if not _clean(classified_at):
        raise ValueError("classified_at must be supplied explicitly")
    if not _clean(mapping_source) or not _clean(mapping_source_version):
        raise ValueError("mapping source and source version are required")
    if len(mapping_source_sha256) != 64 or any(c not in "0123456789abcdef" for c in mapping_source_sha256):
        raise ValueError("mapping_source_sha256 must be a lowercase SHA-256")

    _require_columns(supplier_rows, {
        "selected", "subject_type", "subject_id", "entity_id", "country",
        "postcode", "city", "address", "location_role",
    }, "supplier input")
    _require_columns(correspondence_rows, {
        "country", "postcode", "geography_scheme", "geography_version",
        "geography_level", "geography_code", "geography_name",
    }, "correspondence input")
    selected_values = {_clean(v).casefold() for v in config["selected_values"]}
    selected_rows = [row for row in supplier_rows if _clean(row.get("selected")).casefold() in selected_values]
    index = _build_index(correspondence_rows, config)
    outputs: list[dict[str, str]] = []

    for row in selected_rows:
        _validate_supplier(row)
        country, country_supported = normalize_country(row.get("country"), config)
        postcode = normalize_postcode(row.get("postcode"))
        governed_input = {
            "subject_type": _clean(row.get("subject_type")),
            "subject_id": _clean(row.get("subject_id")),
            "entity_id": _clean(row.get("entity_id")),
            "country": country, "postcode": postcode,
            "city": _clean(row.get("city")), "address": _clean(row.get("address")),
            "location_role": _clean(row.get("location_role")),
        }
        input_fingerprint = _sha256_text(_canonical_json(governed_input))
        matches = index.get((country, postcode), {}) if country_supported and postcode else {}
        distinct = sorted(matches)
        if not country_supported:
            status, confidence = "unsupported_country", "none"
        elif len(distinct) == 1:
            status, confidence = "resolved", "strong"
        elif len(distinct) > 1:
            status, confidence = "ambiguous", "none"
        else:
            status, confidence = "insufficient_evidence", "none"

        chosen = distinct[0] if status == "resolved" else ("", "", "", "", "")
        evidence_results = [
            {"geography_scheme": key[0], "geography_version": key[1],
             "geography_level": key[2], "geography_code": key[3], "geography_name": key[4]}
            for key in distinct
        ]
        evidence = {
            "lookup_country": country,
            "lookup_postcode": postcode,
            "distinct_supported_results": evidence_results,
            "distinct_supported_result_count": len(distinct),
        }
        identity = {
            "input_fingerprint": input_fingerprint,
            "classifier_version": _clean(config["classifier_version"]),
            "mapping_source_sha256": mapping_source_sha256,
        }
        output = {
            "classification_id": "geo_" + _sha256_text(_canonical_json(identity))[:24],
            **governed_input,
            "geography_scheme": chosen[0], "geography_version": chosen[1],
            "geography_level": chosen[2], "geography_code": chosen[3], "geography_name": chosen[4],
            "mapping_status": status, "mapping_method": "postcode_correspondence",
            "mapping_source": mapping_source, "mapping_source_version": mapping_source_version,
            "mapping_source_sha256": mapping_source_sha256,
            "mapping_evidence": _canonical_json(evidence), "mapping_confidence": confidence,
            "input_fingerprint": input_fingerprint,
            "classifier_version": _clean(config["classifier_version"]),
            "classified_at": classified_at,
        }
        outputs.append({field: output[field] for field in OUTPUT_FIELDS})

    return sorted(outputs, key=lambda item: (
        item["subject_type"], item["subject_id"], item["entity_id"],
        item["location_role"], item["input_fingerprint"],
    ))


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suppliers", type=Path, required=True)
    parser.add_argument("--correspondence", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classified-at", required=True)
    parser.add_argument("--mapping-source", required=True)
    parser.add_argument("--mapping-source-version", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    correspondence_hash = file_sha256(args.correspondence)
    rows = classify_rows(
        read_csv(args.suppliers), read_csv(args.correspondence),
        config=load_config(args.config), classified_at=args.classified_at,
        mapping_source=args.mapping_source,
        mapping_source_version=args.mapping_source_version,
        mapping_source_sha256=correspondence_hash,
    )
    write_csv(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
