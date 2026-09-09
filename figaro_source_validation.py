#!/usr/bin/env python3
"""SKO-039 FIGARO source freeze and normalized-contract validator.

This module does not download or reinterpret FIGARO. It validates a locally frozen,
normalized source package before the SKO-039 indirect engine may consume it.

Expected normalized files:
- figaro_transactions.csv
- figaro_outputs.csv
- figaro_satellites.csv
- figaro_source_manifest.json

The manifest binds source filenames to sha256 checksums and records exact edition,
reference year, table architecture and provenance. Live/client programme data remain
outside version control.
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

TRANSACTION_FIELDS = {
    "origin_country",
    "origin_sector",
    "destination_country",
    "destination_sector",
    "value_million_eur",
}
OUTPUT_FIELDS = {"country", "sector", "output_million_eur"}
SATELLITE_FIELDS = {"country", "sector", "outcome", "value", "unit"}
MANIFEST_REQUIRED = {
    "organisation",
    "product_id",
    "edition",
    "reference_year",
    "table_type",
    "classification",
    "valuation",
    "currency",
    "unit",
    "files",
}
ALLOWED_OUTCOMES = {"GVA", "GHG", "EMPLOYMENT_PERSONS"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    return value


def read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("config must be a YAML mapping")
    return value


def _require_fields(rows: list[dict[str, str]], required: set[str], label: str) -> None:
    if not rows:
        raise ValueError(f"{label} is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{label} missing fields: {sorted(missing)}")


def _number(value: Any, label: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {label}: {value!r}") from exc
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _node(country: str, sector: str) -> tuple[str, str]:
    return country.strip().upper(), sector.strip().upper()


def validate_manifest(manifest: Mapping[str, Any], config: Mapping[str, Any], package_dir: Path) -> dict[str, Any]:
    missing = MANIFEST_REQUIRED - set(manifest)
    if missing:
        raise ValueError(f"manifest missing fields: {sorted(missing)}")

    source = config.get("figaro_source", {})
    comparisons = {
        "organisation": source.get("organisation"),
        "product_id": source.get("product_id"),
        "edition": source.get("edition"),
        "reference_year": source.get("reference_year"),
        "table_type": source.get("table_type"),
        "classification": source.get("classification"),
        "valuation": source.get("valuation"),
        "currency": source.get("currency"),
        "unit": source.get("unit"),
    }
    mismatch = {}
    for key, expected in comparisons.items():
        actual = manifest.get(key)
        if str(actual) != str(expected):
            mismatch[key] = {"expected": expected, "actual": actual}
    if mismatch:
        raise ValueError(f"manifest/config mismatch: {mismatch}")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest files must be a non-empty list")

    seen = set()
    checked = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("manifest file entries must be objects")
        filename = str(item.get("filename", "")).strip()
        expected_sha = str(item.get("sha256", "")).strip().lower()
        role = str(item.get("role", "")).strip()
        if not filename or not expected_sha or not role:
            raise ValueError("manifest file entries require filename, role and sha256")
        if filename in seen:
            raise ValueError(f"duplicate manifest filename: {filename}")
        seen.add(filename)
        path = package_dir / filename
        if not path.is_file():
            raise ValueError(f"manifest source file missing: {filename}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"sha256 mismatch for {filename}: expected {expected_sha}, got {actual_sha}")
        checked.append({"filename": filename, "role": role, "sha256": actual_sha})

    required_roles = {"transactions", "outputs", "satellites"}
    roles = {row["role"] for row in checked}
    if not required_roles.issubset(roles):
        raise ValueError(f"manifest missing required roles: {sorted(required_roles - roles)}")
    return {"manifest_matches_config": True, "checksums_verified": True, "files": checked}


def validate_normalized_sources(
    transactions: list[dict[str, str]],
    outputs: list[dict[str, str]],
    satellites: list[dict[str, str]],
) -> dict[str, Any]:
    _require_fields(transactions, TRANSACTION_FIELDS, "transactions")
    _require_fields(outputs, OUTPUT_FIELDS, "outputs")
    _require_fields(satellites, SATELLITE_FIELDS, "satellites")

    output_nodes: set[tuple[str, str]] = set()
    for row in outputs:
        node = _node(row.get("country", ""), row.get("sector", ""))
        if not all(node):
            raise ValueError("output row has blank country/sector")
        if node in output_nodes:
            raise ValueError(f"duplicate output node: {node}")
        _number(row.get("output_million_eur"), f"output {node}", positive=True)
        output_nodes.add(node)

    transaction_pairs = Counter()
    for row in transactions:
        origin = _node(row.get("origin_country", ""), row.get("origin_sector", ""))
        destination = _node(row.get("destination_country", ""), row.get("destination_sector", ""))
        if origin not in output_nodes or destination not in output_nodes:
            raise ValueError(f"transaction references unknown node: {origin} -> {destination}")
        _number(row.get("value_million_eur"), f"transaction {origin}->{destination}", nonnegative=True)
        transaction_pairs[(origin, destination)] += 1
    duplicate_transactions = [pair for pair, count in transaction_pairs.items() if count > 1]
    if duplicate_transactions:
        raise ValueError(f"duplicate normalized transaction pairs: {duplicate_transactions[:5]}")

    satellite_cells = set()
    outcome_counts = Counter()
    satellite_nodes_by_outcome: dict[str, set[tuple[str, str]]] = {key: set() for key in ALLOWED_OUTCOMES}
    for row in satellites:
        node = _node(row.get("country", ""), row.get("sector", ""))
        outcome = str(row.get("outcome", "")).strip().upper()
        unit = str(row.get("unit", "")).strip()
        if node not in output_nodes:
            raise ValueError(f"satellite references unknown node: {node}")
        if outcome not in ALLOWED_OUTCOMES:
            raise ValueError(f"unsupported satellite outcome: {outcome}")
        if not unit:
            raise ValueError(f"satellite unit missing for {outcome} {node}")
        _number(row.get("value"), f"satellite {outcome} {node}")
        cell = (outcome, node)
        if cell in satellite_cells:
            raise ValueError(f"duplicate satellite cell: {cell}")
        satellite_cells.add(cell)
        outcome_counts[outcome] += 1
        satellite_nodes_by_outcome[outcome].add(node)

    missing_outcomes = sorted(ALLOWED_OUTCOMES - set(outcome_counts))
    if missing_outcomes:
        raise ValueError(f"normalized package missing satellite outcomes: {missing_outcomes}")

    coverage = {
        outcome: {
            "covered_nodes": len(nodes),
            "total_nodes": len(output_nodes),
            "coverage_pct": (100.0 * len(nodes) / len(output_nodes)) if output_nodes else 0.0,
            "missing_nodes": sorted(f"{c}-{s}" for c, s in output_nodes - nodes),
        }
        for outcome, nodes in sorted(satellite_nodes_by_outcome.items())
    }
    return {
        "output_nodes": len(output_nodes),
        "transactions": len(transactions),
        "satellite_rows": len(satellites),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "satellite_coverage": coverage,
        "normalized_contract_valid": True,
    }


def _file_for_role(manifest: Mapping[str, Any], role: str) -> str:
    matches = [str(item["filename"]) for item in manifest["files"] if item.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"manifest must contain exactly one {role} file; found {len(matches)}")
    return matches[0]


def validate_package(package_dir: Path, config_path: Path) -> dict[str, Any]:
    config = read_config(config_path)
    manifest_path = package_dir / "figaro_source_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("figaro_source_manifest.json is missing")
    manifest = read_json(manifest_path)
    manifest_result = validate_manifest(manifest, config, package_dir)

    transactions = read_csv(package_dir / _file_for_role(manifest, "transactions"))
    outputs = read_csv(package_dir / _file_for_role(manifest, "outputs"))
    satellites = read_csv(package_dir / _file_for_role(manifest, "satellites"))
    contract_result = validate_normalized_sources(transactions, outputs, satellites)

    fingerprint_source = json.dumps(
        {
            "manifest": manifest,
            "contract": contract_result,
            "file_checksums": manifest_result["files"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "status": "source_package_validated",
        "manifest": manifest_result,
        "contract": contract_result,
        "package_fingerprint": hashlib.sha256(fingerprint_source).hexdigest(),
        "pilot_execution_permitted": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a frozen normalized FIGARO source package")
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--config", default=Path("config/figaro_indirect_attribution.yaml"), type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    result = validate_package(args.package_dir, args.config)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
