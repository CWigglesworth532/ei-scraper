#!/usr/bin/env python3
"""SKO-037 governed direct environmental coefficient materialisation.

Builds direct, residence-bound environmental intensity coefficients from normalized
source rows. V1 supports headline direct GHG only. Supplier/social-economy status is
not part of this layer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

SOURCE_FIELDS = [
    "source_record_id", "source_family", "source_organisation", "source_dataset_id",
    "source_release_version", "source_release_date", "retrieved_at", "source_url",
    "licence", "country", "source_classification", "source_classification_version",
    "source_sector_code", "source_sector_label", "model_classification",
    "model_classification_version", "model_sector_code", "model_sector_label",
    "reference_year", "concept_code", "concept_label", "value", "normalized_unit",
    "currency", "price_basis", "status_flag", "source_fingerprint", "schema_version",
]

COEFFICIENT_FIELDS = [
    "coefficient_id", "coefficient_schema_version", "coefficient_release_version",
    "country", "model_classification", "model_classification_version",
    "coefficient_source_sector_code", "coefficient_source_sector_label", "reference_year",
    "outcome_code", "outcome_label", "numerator_concept_code", "numerator_value",
    "numerator_unit", "denominator_concept_code", "denominator_value", "denominator_unit",
    "denominator_currency", "price_basis", "coefficient_value", "coefficient_unit",
    "denominator_route", "denominator_method", "denominator_compatibility_class",
    "source_family", "source_organisation", "source_dataset_ids", "source_release_versions",
    "source_release_dates", "source_retrieved_at", "source_urls", "licences",
    "source_status_flags", "boundary_scope", "transformation_method", "qa_status",
    "qa_reason", "source_record_ids", "source_fingerprints", "generated_at",
    "record_fingerprint",
]

COVERAGE_FIELDS = [
    "country", "coefficient_source_sector_code", "coefficient_source_sector_label",
    "reference_year", "outcome_code", "outcome_label", "denominator_route",
    "denominator_concept_code", "denominator_currency", "availability_status",
    "availability_reason", "coefficient_value", "coefficient_unit", "coefficient_id",
    "source_family", "source_dataset_ids", "source_release_versions", "source_release_dates",
]


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def canon(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canon(value).encode("utf-8")).hexdigest()


def dec(value: Any, label: str) -> Decimal:
    try:
        result = Decimal(clean(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid decimal for {label}: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"Non-finite decimal for {label}")
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(rows: Iterable[Mapping[str, Any]], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _require_columns(rows: list[dict[str, str]], fields: set[str], label: str) -> None:
    available = set(rows[0]) if rows else set()
    missing = fields - available
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "schema_version", "coefficient_schema_version", "coefficient_release_version",
        "model_classification", "model_classification_version", "boundary",
        "reference_year_policy", "source_status_policy", "denominator_routes", "outcomes",
        "granularity_fallback_policy", "currency_normalisation",
    }
    if not isinstance(config, dict) or required - set(config):
        raise ValueError("Environmental configuration contract incomplete")
    status_policy = config["source_status_policy"]
    if status_policy.get("mode") != "explicit_usable_flags":
        raise ValueError("SKO-037 source status policy must use explicit_usable_flags")
    if not isinstance(status_policy.get("usable_flags"), list):
        raise ValueError("source_status_policy.usable_flags must be a list")
    routes = config["denominator_routes"]
    if "default" not in routes:
        raise ValueError("Default denominator route required")
    for route_name, route in routes.items():
        needed = {
            "source_family", "denominator_concept_code", "method",
            "compatibility_class", "required_denominator_unit",
        }
        if needed - set(route):
            raise ValueError(f"Denominator route {route_name} incomplete")
    for outcome_code, outcome in config["outcomes"].items():
        needed = {
            "label", "numerator_source_family", "numerator_concept_code",
            "required_numerator_unit", "coefficient_unit", "scale",
        }
        if needed - set(outcome):
            raise ValueError(f"Outcome {outcome_code} incomplete")
        dec(outcome["scale"], f"outcome {outcome_code} scale")
    policy = config["reference_year_policy"]
    if policy.get("mode") != "fixed_release_year":
        raise ValueError("SKO-037 v1 requires fixed_release_year")
    if policy.get("automatic_year_fallback_enabled"):
        raise ValueError("Automatic year fallback must be disabled")
    return config


def _route_for_sector(sector_code: str, config: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    routes = config["denominator_routes"]
    for route_name, route in routes.items():
        if route_name == "default":
            continue
        if sector_code in {clean(x) for x in route.get("model_sector_codes", [])}:
            return route_name, route
    return "default", routes["default"]


def normalize_source_rows(
    rows: list[dict[str, str]], *, config: Mapping[str, Any]
) -> tuple[list[dict[str, str]], dict[str, int]]:
    required = {
        "source_family", "source_organisation", "source_dataset_id", "source_release_version",
        "source_release_date", "retrieved_at", "source_url", "licence", "country",
        "source_classification", "source_classification_version", "source_sector_code",
        "source_sector_label", "model_classification", "model_classification_version",
        "model_sector_code", "model_sector_label", "reference_year", "concept_code",
        "concept_label", "value", "normalized_unit", "currency", "price_basis", "status_flag",
    }
    _require_columns(rows, required, "source input")
    qa = {
        "source_rows": len(rows), "unique_source_rows": 0, "duplicate_source_rows": 0,
        "conflicting_source_rows": 0, "groups": 0, "coefficient_records": 0,
        "calculated_coefficients": 0, "held_out_coefficients": 0,
        "missing_denominator_records": 0, "missing_numerator_records": 0,
        "source_status_holdouts": 0, "unit_mismatch_holdouts": 0,
        "trade_route_groups": 0, "output_route_groups": 0,
    }
    by_key: dict[tuple[str, ...], dict[str, str]] = {}
    for raw in rows:
        row = {key: clean(raw.get(key)) for key in required}
        if row["model_classification"] != config["model_classification"]:
            raise ValueError("Unexpected model classification")
        if row["model_classification_version"] != config["model_classification_version"]:
            raise ValueError("Unexpected model classification version")
        if not row["country"] or not row["model_sector_code"] or not row["reference_year"] or not row["concept_code"]:
            raise ValueError("country, model sector, reference year and concept are required")
        dec(row["value"], "source value")
        key = (
            row["source_family"], row["source_organisation"], row["source_dataset_id"],
            row["source_release_version"], row["country"], row["model_sector_code"],
            row["reference_year"], row["concept_code"],
        )
        fingerprint = digest({k: row[k] for k in sorted(row)})
        normalized = {
            "source_record_id": "envs_" + fingerprint[:24], **row,
            "source_fingerprint": fingerprint, "schema_version": config["schema_version"],
        }
        if key in by_key:
            if by_key[key] == normalized:
                qa["duplicate_source_rows"] += 1
                continue
            qa["conflicting_source_rows"] += 1
            raise ValueError(f"Conflicting source observation for key {key}")
        by_key[key] = normalized
    normalized_rows = sorted(by_key.values(), key=lambda r: r["source_record_id"])
    qa["unique_source_rows"] = len(normalized_rows)
    return normalized_rows, qa


def _select(rows: list[dict[str, str]], concept_code: str) -> dict[str, str] | None:
    matches = [row for row in rows if row["concept_code"] == concept_code]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple rows for concept {concept_code} in coefficient group")
    return matches[0]


def _join_distinct(rows: list[dict[str, str]], field: str) -> str:
    return "|".join(sorted({row[field] for row in rows if row.get(field)}))


def _status_usable(row: dict[str, str], config: Mapping[str, Any]) -> bool:
    usable = {clean(flag).lower() for flag in config["source_status_policy"]["usable_flags"]}
    return clean(row.get("status_flag")).lower() in usable


def build_coefficients(
    source_rows: list[dict[str, str]], *, config: Mapping[str, Any], generated_at: str
) -> dict[str, Any]:
    if not clean(generated_at):
        raise ValueError("generated_at required")
    normalized_rows, qa = normalize_source_rows(source_rows, config=config)
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    all_keys: set[tuple[str, str, str]] = set()
    for row in normalized_rows:
        grouped[(row["source_family"], row["country"], row["model_sector_code"], row["reference_year"])].append(row)
        all_keys.add((row["country"], row["model_sector_code"], row["reference_year"]))

    qa["groups"] = len(all_keys)
    coefficients: list[dict[str, str]] = []
    coverage: list[dict[str, str]] = []
    boundary_scope = clean(config.get("boundary", {}).get("scope"))

    for country, sector_code, year in sorted(all_keys):
        route_name, route = _route_for_sector(sector_code, config)
        qa["trade_route_groups" if route_name == "trade_turnover" else "output_route_groups"] += 1
        denominator_rows = grouped.get((route["source_family"], country, sector_code, year), [])
        denominator = _select(denominator_rows, route["denominator_concept_code"])

        label_row = denominator
        if label_row is None:
            for rows in grouped.values():
                if rows and rows[0]["country"] == country and rows[0]["model_sector_code"] == sector_code and rows[0]["reference_year"] == year:
                    label_row = rows[0]
                    break
        sector_label = label_row["model_sector_label"] if label_row else ""

        for outcome_code, outcome in sorted(config["outcomes"].items()):
            numerator_rows = grouped.get((outcome["numerator_source_family"], country, sector_code, year), [])
            numerator = _select(numerator_rows, outcome["numerator_concept_code"])
            qa_status = "calculated"
            qa_reason = ""

            if denominator is None:
                qa_status, qa_reason = "held_out", "missing_required_denominator"
                qa["missing_denominator_records"] += 1
            elif numerator is None:
                qa_status, qa_reason = "held_out", "missing_required_numerator"
                qa["missing_numerator_records"] += 1
            elif not _status_usable(denominator, config) or not _status_usable(numerator, config):
                qa_status, qa_reason = "held_out", "source_status_not_usable"
                qa["source_status_holdouts"] += 1
            elif denominator["normalized_unit"] != route["required_denominator_unit"]:
                qa_status, qa_reason = "held_out", "denominator_unit_mismatch"
                qa["unit_mismatch_holdouts"] += 1
            elif numerator["normalized_unit"] != outcome["required_numerator_unit"]:
                qa_status, qa_reason = "held_out", "numerator_unit_mismatch"
                qa["unit_mismatch_holdouts"] += 1
            elif dec(denominator["value"], "denominator") <= 0:
                qa_status, qa_reason = "held_out", "nonpositive_denominator"

            source_used = [r for r in (denominator, numerator) if r is not None]
            coefficient_value = ""
            if qa_status == "calculated":
                coefficient = (
                    dec(numerator["value"], "numerator")
                    / dec(denominator["value"], "denominator")
                    * dec(outcome["scale"], "scale")
                )
                coefficient_value = format(coefficient.normalize(), "f")

            core = {
                "coefficient_schema_version": config["coefficient_schema_version"],
                "coefficient_release_version": config["coefficient_release_version"],
                "country": country,
                "model_classification": config["model_classification"],
                "model_classification_version": config["model_classification_version"],
                "coefficient_source_sector_code": sector_code,
                "coefficient_source_sector_label": sector_label,
                "reference_year": year,
                "outcome_code": outcome_code,
                "outcome_label": outcome["label"],
                "numerator_concept_code": outcome["numerator_concept_code"],
                "numerator_value": numerator["value"] if numerator else "",
                "numerator_unit": numerator["normalized_unit"] if numerator else "",
                "denominator_concept_code": route["denominator_concept_code"],
                "denominator_value": denominator["value"] if denominator else "",
                "denominator_unit": denominator["normalized_unit"] if denominator else "",
                "denominator_currency": denominator["currency"] if denominator else "",
                "price_basis": denominator["price_basis"] if denominator else "",
                "coefficient_value": coefficient_value,
                "coefficient_unit": outcome["coefficient_unit"],
                "denominator_route": route_name,
                "denominator_method": route["method"],
                "denominator_compatibility_class": route["compatibility_class"],
                "source_family": _join_distinct(source_used, "source_family"),
                "source_organisation": _join_distinct(source_used, "source_organisation"),
                "source_dataset_ids": _join_distinct(source_used, "source_dataset_id"),
                "source_release_versions": _join_distinct(source_used, "source_release_version"),
                "source_release_dates": _join_distinct(source_used, "source_release_date"),
                "source_retrieved_at": _join_distinct(source_used, "retrieved_at"),
                "source_urls": _join_distinct(source_used, "source_url"),
                "licences": _join_distinct(source_used, "licence"),
                "source_status_flags": _join_distinct(source_used, "status_flag"),
                "boundary_scope": boundary_scope,
                "transformation_method": f"{outcome['numerator_concept_code']}/{route['denominator_concept_code']}*{outcome['scale']}",
                "qa_status": qa_status,
                "qa_reason": qa_reason,
                "source_record_ids": _join_distinct(source_used, "source_record_id"),
                "source_fingerprints": _join_distinct(source_used, "source_fingerprint"),
                "generated_at": generated_at,
            }
            fp = digest(core)
            record = {"coefficient_id": "envc_" + fp[:24], **core, "record_fingerprint": fp}
            coefficients.append(record)
            coverage.append({
                "country": country,
                "coefficient_source_sector_code": sector_code,
                "coefficient_source_sector_label": sector_label,
                "reference_year": year,
                "outcome_code": outcome_code,
                "outcome_label": outcome["label"],
                "denominator_route": route_name,
                "denominator_concept_code": route["denominator_concept_code"],
                "denominator_currency": denominator["currency"] if denominator else "",
                "availability_status": "available" if qa_status == "calculated" else "held_out",
                "availability_reason": qa_reason,
                "coefficient_value": coefficient_value,
                "coefficient_unit": outcome["coefficient_unit"],
                "coefficient_id": record["coefficient_id"],
                "source_family": record["source_family"],
                "source_dataset_ids": record["source_dataset_ids"],
                "source_release_versions": record["source_release_versions"],
                "source_release_dates": record["source_release_dates"],
            })
            qa["coefficient_records"] += 1
            qa["calculated_coefficients" if qa_status == "calculated" else "held_out_coefficients"] += 1

    coefficients.sort(key=lambda r: (r["country"], r["coefficient_source_sector_code"], int(r["reference_year"]), r["outcome_code"]))
    coverage.sort(key=lambda r: (r["country"], r["coefficient_source_sector_code"], int(r["reference_year"]), r["outcome_code"]))
    return {"source_rows": normalized_rows, "coefficients": coefficients, "coverage_matrix": coverage, "qa": qa}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-input", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--coefficient-output", type=Path, required=True)
    parser.add_argument("--coverage-output", type=Path, required=True)
    parser.add_argument("--qa-output", type=Path, required=True)
    parser.add_argument("--generated-at", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    result = build_coefficients(read_csv(args.source_input), config=config, generated_at=args.generated_at)
    write_csv(result["source_rows"], args.source_output, SOURCE_FIELDS)
    write_csv(result["coefficients"], args.coefficient_output, COEFFICIENT_FIELDS)
    write_csv(result["coverage_matrix"], args.coverage_output, COVERAGE_FIELDS)
    args.qa_output.parent.mkdir(parents=True, exist_ok=True)
    args.qa_output.write_text(json.dumps(result["qa"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
