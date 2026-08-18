#!/usr/bin/env python3
"""SKO-028 deterministic ingestion of small local external-indicator extracts.

The normalized observations describe geographic context, not supplier impact or
causality.  This module has no network, entity, supplier, or linkage capability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


OUTPUT_FIELDS = [
    "observation_id", "source_id", "source_name", "dataset_id", "dataset_title",
    "dataset_version", "indicator_id", "indicator_name", "indicator_description",
    "indicator_theme", "geography_scheme", "geography_version", "geography_level",
    "geography_code", "period", "period_type", "value", "unit", "value_status",
    "source_version", "source_reference", "retrieved_at", "transformation_method",
    "transformation_version", "input_file_sha256", "ingestion_version",
    "observation_fingerprint",
]
REJECTION_FIELDS = [
    "extract_id", "input_file", "input_file_sha256", "row_number", "reason",
    "source_row",
]
REQUIRED_MAPPED_FIELDS = {
    "indicator_id", "geography_scheme", "geography_version", "geography_level",
    "geography_code", "period", "period_type", "value", "unit", "value_status",
}


class ConflictingObservationError(ValueError):
    """Raised when one fully versioned observation key has conflicting content."""


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


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


def _normalise_decimal(value: Any) -> str:
    raw = _clean(value)
    try:
        number = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid numeric value: {raw!r}") from exc
    if not number.is_finite():
        raise ValueError(f"Invalid numeric value: {raw!r}")
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", ""} else normalized


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    required = {
        "ingestion_version", "transformation_method", "transformation_version",
        "geography_governance", "extracts",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration missing keys: {sorted(missing)}")
    if config["transformation_method"] != "configured_local_csv_extract":
        raise ValueError("SKO-028 v1 only supports configured_local_csv_extract")
    extracts = config["extracts"]
    if not isinstance(extracts, dict) or not extracts:
        raise ValueError("Configuration requires at least one extract")
    for extract_id, extract in extracts.items():
        required_extract = {
            "file_name", "source_id", "source_name", "dataset_id", "dataset_title",
            "dataset_version", "source_version", "source_reference", "field_map",
            "expected_geography", "indicators",
        }
        missing_extract = required_extract - set(extract)
        if missing_extract:
            raise ValueError(f"Extract {extract_id!r} missing keys: {sorted(missing_extract)}")
        mapped = set(extract["field_map"])
        if REQUIRED_MAPPED_FIELDS - mapped:
            raise ValueError(
                f"Extract {extract_id!r} missing field mappings: "
                f"{sorted(REQUIRED_MAPPED_FIELDS - mapped)}"
            )
        if not isinstance(extract["indicators"], dict) or not extract["indicators"]:
            raise ValueError(f"Extract {extract_id!r} requires configured indicators")
    return config


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _mapped_row(source_row: Mapping[str, str], field_map: Mapping[str, str]) -> dict[str, str]:
    return {target: _clean(source_row.get(source)) for target, source in field_map.items()}


def _geography_supported(
    row: Mapping[str, str], governance: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    scheme = _clean(row["geography_scheme"])
    version = _clean(row["geography_version"])
    level = _clean(row["geography_level"])
    rules = governance.get(scheme, {})
    return bool(
        rules
        and scheme == _clean(expected.get("scheme"))
        and version == _clean(expected.get("version"))
        and level in {_clean(v) for v in expected.get("levels", [])}
        and version in {_clean(v) for v in rules.get("versions", [])}
        and level in {_clean(v) for v in rules.get("levels", [])}
        and _clean(row["geography_code"])
    )


def _observation_key(row: Mapping[str, str]) -> dict[str, str]:
    fields = [
        "source_id", "dataset_id", "dataset_version", "indicator_id",
        "geography_scheme", "geography_version", "geography_level", "geography_code",
        "period", "period_type", "unit", "source_version",
    ]
    return {field: row[field] for field in fields}


def _rejection(
    extract_id: str, path: Path, sha256: str, row_number: int, reason: str,
    source_row: Mapping[str, str],
) -> dict[str, str]:
    return {
        "extract_id": extract_id, "input_file": path.name,
        "input_file_sha256": sha256, "row_number": str(row_number),
        "reason": reason, "source_row": _canonical_json(dict(source_row)),
    }


def ingest_extracts(
    config: Mapping[str, Any], *, input_dir: Path, retrieved_at: str,
) -> dict[str, list[dict[str, str]]]:
    """Ingest configured local extracts into generic observations and rejections."""
    if not _clean(retrieved_at):
        raise ValueError("retrieved_at must be supplied explicitly")
    observations_by_key: dict[str, dict[str, str]] = {}
    rejections: list[dict[str, str]] = []

    for extract_id, extract in sorted(dict(config["extracts"]).items()):
        path = input_dir / _clean(extract["file_name"])
        source_hash = file_sha256(path)
        source_rows = read_csv(path)
        field_map = dict(extract["field_map"])
        missing_columns = set(field_map.values()) - (set(source_rows[0]) if source_rows else set())
        if missing_columns:
            raise ValueError(f"Extract {extract_id!r} missing columns: {sorted(missing_columns)}")

        for row_number, source_row in enumerate(source_rows, start=2):
            mapped = _mapped_row(source_row, field_map)
            indicator_id = mapped["indicator_id"]
            indicator = dict(extract["indicators"]).get(indicator_id)
            if indicator is None:
                rejections.append(_rejection(
                    extract_id, path, source_hash, row_number,
                    "unconfigured_indicator", source_row,
                ))
                continue
            if not _geography_supported(
                mapped, config["geography_governance"], extract["expected_geography"],
            ):
                rejections.append(_rejection(
                    extract_id, path, source_hash, row_number,
                    "unsupported_geography_metadata", source_row,
                ))
                continue
            expected_units = {_clean(unit) for unit in indicator.get("allowed_units", [])}
            if expected_units and mapped["unit"] not in expected_units:
                rejections.append(_rejection(
                    extract_id, path, source_hash, row_number,
                    "unsupported_unit", source_row,
                ))
                continue
            try:
                value = _normalise_decimal(mapped["value"])
            except ValueError:
                rejections.append(_rejection(
                    extract_id, path, source_hash, row_number,
                    "invalid_value", source_row,
                ))
                continue

            output = {
                "source_id": _clean(extract["source_id"]),
                "source_name": _clean(extract["source_name"]),
                "dataset_id": _clean(extract["dataset_id"]),
                "dataset_title": _clean(extract["dataset_title"]),
                "dataset_version": _clean(extract["dataset_version"]),
                "indicator_id": indicator_id,
                "indicator_name": _clean(indicator["name"]),
                "indicator_description": _clean(indicator["description"]),
                "indicator_theme": _clean(indicator["theme"]),
                "geography_scheme": mapped["geography_scheme"],
                "geography_version": mapped["geography_version"],
                "geography_level": mapped["geography_level"],
                "geography_code": mapped["geography_code"],
                "period": mapped["period"], "period_type": mapped["period_type"],
                "value": value, "unit": mapped["unit"],
                "value_status": mapped["value_status"],
                "source_version": _clean(extract["source_version"]),
                "source_reference": _clean(extract["source_reference"]),
                "retrieved_at": retrieved_at,
                "transformation_method": _clean(config["transformation_method"]),
                "transformation_version": _clean(config["transformation_version"]),
                "input_file_sha256": source_hash,
                "ingestion_version": _clean(config["ingestion_version"]),
            }
            key_json = _canonical_json(_observation_key(output))
            observation_id = "obs_" + _sha256_text(key_json)[:24]
            fingerprint_content = {**output, "observation_id": observation_id}
            output["observation_id"] = observation_id
            output["observation_fingerprint"] = _sha256_text(_canonical_json(fingerprint_content))
            output = {field: output[field] for field in OUTPUT_FIELDS}
            existing = observations_by_key.get(key_json)
            if existing is not None and existing != output:
                raise ConflictingObservationError(
                    f"Conflicting duplicate for observation key {key_json}"
                )
            observations_by_key[key_json] = output

    observations = sorted(
        observations_by_key.values(),
        key=lambda row: (
            row["source_id"], row["dataset_id"], row["dataset_version"],
            row["indicator_id"], row["geography_scheme"], row["geography_version"],
            row["geography_level"], row["geography_code"], row["period"],
            row["period_type"], row["unit"], row["source_version"],
        ),
    )
    rejections.sort(key=lambda row: (
        row["extract_id"], row["input_file"], int(row["row_number"]),
        row["reason"], row["source_row"],
    ))
    return {"observations": observations, "rejections": rejections}


def write_csv(rows: Iterable[Mapping[str, str]], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejections-output", type=Path, required=True)
    parser.add_argument("--retrieved-at", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = ingest_extracts(
        load_config(args.config), input_dir=args.input_dir,
        retrieved_at=args.retrieved_at,
    )
    write_csv(result["observations"], args.output, OUTPUT_FIELDS)
    write_csv(result["rejections"], args.rejections_output, REJECTION_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
