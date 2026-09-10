#!/usr/bin/env python3
"""SKO-039 FIGARO source freeze and normalized-contract validator.

This module does not download or reinterpret FIGARO. It validates a locally frozen,
normalized source package before the SKO-039 indirect engine may consume it.

Supported live package shapes:
- legacy normalized CSV package: transactions + outputs + satellites;
- compact live package: compact FIGARO model + outputs + one or more satellites.

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

import numpy as np
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
DEFAULT_OUTCOMES = {"GVA", "GHG", "EMPLOYMENT_PERSONS"}
COMPACT_MODEL_ROLES = {"compact_model", "model"}
COMPACT_MODEL_SCHEMA = "sko-039-figaro-compact-model-v1"


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
    if not np.isfinite(result):
        raise ValueError(f"non-finite numeric value for {label}: {value!r}")
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _node(country: str, sector: str) -> tuple[str, str]:
    return country.strip().upper(), sector.strip().upper()


def _outcome_policy(config: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    outcomes = config.get("outcomes")
    if not isinstance(outcomes, Mapping) or not outcomes:
        return set(DEFAULT_OUTCOMES), set(DEFAULT_OUTCOMES)
    allowed = {str(key).strip().upper() for key in outcomes if str(key).strip()}
    required = {
        str(key).strip().upper()
        for key, spec in outcomes.items()
        if isinstance(spec, Mapping) and str(spec.get("status", "")).strip().lower() == "primary"
    }
    if not required:
        required = set(allowed)
    return allowed, required


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

    roles = {row["role"] for row in checked}
    missing_roles = {"outputs", "satellites"} - roles
    if missing_roles:
        raise ValueError(f"manifest missing required roles: {sorted(missing_roles)}")
    if "transactions" not in roles and not (roles & COMPACT_MODEL_ROLES):
        raise ValueError("manifest requires transactions or compact_model role")
    return {"manifest_matches_config": True, "checksums_verified": True, "files": checked}


def validate_compact_model(path: Path, outputs: list[dict[str, str]]) -> dict[str, Any]:
    output_by_node = {
        _node(row.get("country", ""), row.get("sector", "")): _number(
            row.get("output_million_eur"), "compact model output", positive=True
        )
        for row in outputs
    }
    with np.load(path, allow_pickle=False) as package:
        required = {
            "schema_version",
            "source_filename",
            "source_sha256",
            "countries",
            "sectors",
            "z_million_eur",
            "output_million_eur",
        }
        missing = required - set(package.files)
        if missing:
            raise ValueError(f"compact FIGARO package missing arrays: {sorted(missing)}")
        schema = str(np.asarray(package["schema_version"]).reshape(-1)[0])
        if schema != COMPACT_MODEL_SCHEMA:
            raise ValueError(f"unsupported compact FIGARO schema: {schema}")
        countries = np.asarray(package["countries"]).astype(str)
        sectors = np.asarray(package["sectors"]).astype(str)
        z = np.asarray(package["z_million_eur"], dtype=float)
        x = np.asarray(package["output_million_eur"], dtype=float)
        source_filename = str(np.asarray(package["source_filename"]).reshape(-1)[0]).strip()
        source_sha = str(np.asarray(package["source_sha256"]).reshape(-1)[0]).strip().lower()

    if countries.ndim != 1 or countries.shape != sectors.shape:
        raise ValueError("compact FIGARO country/sector arrays must be aligned 1-D vectors")
    nodes = [_node(c, s) for c, s in zip(countries.tolist(), sectors.tolist())]
    n = len(nodes)
    if len(set(nodes)) != n:
        raise ValueError("duplicate compact FIGARO model node")
    if z.shape != (n, n):
        raise ValueError(f"compact FIGARO Z matrix shape {z.shape} does not match {n} nodes")
    if x.shape != (n,):
        raise ValueError(f"compact FIGARO output vector shape {x.shape} does not match {n} nodes")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(x)):
        raise ValueError("compact FIGARO model contains non-finite values")
    if np.any(z < 0):
        raise ValueError("compact FIGARO model contains negative intermediate transactions")
    if np.any(x <= 0):
        raise ValueError("compact FIGARO output must be positive")
    if len(output_by_node) != len(outputs):
        raise ValueError("duplicate output node in normalized outputs")
    if set(nodes) != set(output_by_node):
        raise ValueError("compact FIGARO nodes do not match normalized outputs")
    expected_x = np.asarray([output_by_node[node] for node in nodes], dtype=float)
    if not np.allclose(x, expected_x, rtol=1e-12, atol=1e-9):
        raise ValueError("compact FIGARO output vector does not match normalized outputs")
    if not source_filename:
        raise ValueError("compact FIGARO source filename is blank")
    if len(source_sha) != 64 or any(ch not in "0123456789abcdef" for ch in source_sha):
        raise ValueError("compact FIGARO source sha256 is invalid")
    return {
        "schema_version": schema,
        "nodes": n,
        "nonzero_transactions": int(np.count_nonzero(z)),
        "source_filename": source_filename,
        "source_sha256": source_sha,
        "compact_model_valid": True,
    }


def validate_normalized_sources(
    transactions: list[dict[str, str]] | None,
    outputs: list[dict[str, str]],
    satellites: list[dict[str, str]],
    *,
    allowed_outcomes: set[str] | None = None,
    required_outcomes: set[str] | None = None,
) -> dict[str, Any]:
    if transactions is not None:
        _require_fields(transactions, TRANSACTION_FIELDS, "transactions")
    _require_fields(outputs, OUTPUT_FIELDS, "outputs")
    _require_fields(satellites, SATELLITE_FIELDS, "satellites")

    allowed = set(allowed_outcomes or DEFAULT_OUTCOMES)
    required = set(required_outcomes or allowed)

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
    if transactions is not None:
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
    satellite_nodes_by_outcome: dict[str, set[tuple[str, str]]] = {key: set() for key in allowed}
    for row in satellites:
        node = _node(row.get("country", ""), row.get("sector", ""))
        outcome = str(row.get("outcome", "")).strip().upper()
        unit = str(row.get("unit", "")).strip()
        if node not in output_nodes:
            raise ValueError(f"satellite references unknown node: {node}")
        if outcome not in allowed:
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

    missing_required = sorted(required - set(outcome_counts))
    if missing_required:
        raise ValueError(f"normalized package missing required satellite outcomes: {missing_required}")

    coverage = {
        outcome: {
            "supplied": outcome in outcome_counts,
            "required": outcome in required,
            "covered_nodes": len(nodes),
            "total_nodes": len(output_nodes),
            "coverage_pct": (100.0 * len(nodes) / len(output_nodes)) if output_nodes else 0.0,
            "missing_nodes": sorted(f"{c}-{s}" for c, s in output_nodes - nodes),
        }
        for outcome, nodes in sorted(satellite_nodes_by_outcome.items())
    }
    return {
        "output_nodes": len(output_nodes),
        "transactions": len(transactions) if transactions is not None else None,
        "transaction_contract_checked": transactions is not None,
        "satellite_rows": len(satellites),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "required_outcomes": sorted(required),
        "satellite_coverage": coverage,
        "normalized_contract_valid": True,
    }


def _files_for_role(manifest: Mapping[str, Any], role: str) -> list[str]:
    return [str(item["filename"]) for item in manifest["files"] if item.get("role") == role]


def _single_file_for_roles(manifest: Mapping[str, Any], roles: Iterable[str]) -> str | None:
    matches = [str(item["filename"]) for item in manifest["files"] if item.get("role") in set(roles)]
    if len(matches) > 1:
        raise ValueError(f"manifest contains multiple files for roles {sorted(set(roles))}: {matches}")
    return matches[0] if matches else None


def validate_package(package_dir: Path, config_path: Path) -> dict[str, Any]:
    config = read_config(config_path)
    manifest_path = package_dir / "figaro_source_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("figaro_source_manifest.json is missing")
    manifest = read_json(manifest_path)
    manifest_result = validate_manifest(manifest, config, package_dir)

    output_files = _files_for_role(manifest, "outputs")
    if len(output_files) != 1:
        raise ValueError(f"manifest must contain exactly one outputs file; found {len(output_files)}")
    satellite_files = _files_for_role(manifest, "satellites")
    if not satellite_files:
        raise ValueError("manifest must contain at least one satellites file")

    outputs = read_csv(package_dir / output_files[0])
    satellites = []
    for filename in satellite_files:
        satellites.extend(read_csv(package_dir / filename))

    transaction_files = _files_for_role(manifest, "transactions")
    if len(transaction_files) > 1:
        raise ValueError(f"manifest must contain at most one transactions file; found {len(transaction_files)}")
    transactions = read_csv(package_dir / transaction_files[0]) if transaction_files else None

    compact_filename = _single_file_for_roles(manifest, COMPACT_MODEL_ROLES)
    compact_result = validate_compact_model(package_dir / compact_filename, outputs) if compact_filename else None

    allowed_outcomes, required_outcomes = _outcome_policy(config)
    contract_result = validate_normalized_sources(
        transactions,
        outputs,
        satellites,
        allowed_outcomes=allowed_outcomes,
        required_outcomes=required_outcomes,
    )

    fingerprint_source = json.dumps(
        {
            "manifest": manifest,
            "contract": contract_result,
            "compact_model": compact_result,
            "file_checksums": manifest_result["files"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "status": "source_package_validated",
        "manifest": manifest_result,
        "contract": contract_result,
        "compact_model": compact_result,
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
