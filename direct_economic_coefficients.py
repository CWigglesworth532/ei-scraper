#!/usr/bin/env python3
"""SKO-036 versioned direct economic coefficient materialisation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

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
    "model_sector_code", "model_sector_label", "reference_year", "outcome_code",
    "outcome_label", "numerator_concept_code", "numerator_value", "numerator_unit",
    "denominator_concept_code", "denominator_value", "denominator_unit",
    "denominator_currency", "price_basis", "coefficient_value", "coefficient_unit",
    "denominator_method", "denominator_compatibility_class", "source_family",
    "source_organisation", "source_dataset_ids", "source_release_versions",
    "source_release_dates", "source_retrieved_at", "source_urls", "licences",
    "source_status_flags", "transformation_method", "qa_status", "qa_reason",
    "source_record_ids", "source_fingerprints", "generated_at", "record_fingerprint",
]

COVERAGE_FIELDS = [
    "country", "model_sector_code", "model_sector_label", "reference_year",
    "denominator_route", "source_family", "denominator_concept_code",
    "denominator_available", "outcomes_requested", "outcomes_calculated",
    "outcomes_held_out", "outcomes_not_applicable", "coverage_status", "coverage_reason",
]

QA_FIELDS = [
    "source_rows", "unique_source_rows", "duplicate_source_rows",
    "conflicting_source_rows", "groups", "coefficient_records",
    "calculated_coefficients", "held_out_coefficients", "not_applicable_coefficients",
    "missing_denominator_groups", "nonpositive_denominator_groups",
    "missing_numerator_records", "derived_numerator_records",
    "accounting_identity_checks", "accounting_identity_failures",
    "trade_route_groups", "output_route_groups",
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


def write_csv(rows: list[dict[str, Any]], path: Path, fields: list[str]) -> None:
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
        "model_classification", "model_classification_version", "denominator_routes",
        "outcomes", "accounting_identity",
    }
    if not isinstance(config, dict) or required - set(config):
        raise ValueError("Configuration contract incomplete")
    routes = config["denominator_routes"]
    if "default" not in routes:
        raise ValueError("Default denominator route required")
    for route_name, route in routes.items():
        needed = {"source_family", "denominator_concept_code", "method", "compatibility_class"}
        if needed - set(route):
            raise ValueError(f"Denominator route {route_name} incomplete")
    for outcome_code, outcome in config["outcomes"].items():
        needed = {"label", "numerator_concept_code", "coefficient_unit", "scale"}
        if needed - set(outcome):
            raise ValueError(f"Outcome {outcome_code} incomplete")
        dec(outcome["scale"], f"outcome {outcome_code} scale")
        derivation = outcome.get("derivation")
        if derivation:
            if derivation.get("operation") != "sum" or not derivation.get("source_concept_codes"):
                raise ValueError(f"Unsupported derivation for outcome {outcome_code}")
    return config


def _route_for_sector(sector_code: str, config: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    routes = config["denominator_routes"]
    for route_name, route in routes.items():
        if route_name == "default":
            continue
        if sector_code in {clean(x) for x in route.get("model_sector_codes", [])}:
            return route_name, route
    return "default", routes["default"]


def normalize_source_rows(rows: list[dict[str, str]], *, config: Mapping[str, Any]) -> tuple[list[dict[str, str]], dict[str, int]]:
    required = {
        "source_family", "source_organisation", "source_dataset_id", "source_release_version",
        "source_release_date", "retrieved_at", "source_url", "licence", "country",
        "source_classification", "source_classification_version", "source_sector_code",
        "source_sector_label", "model_classification", "model_classification_version",
        "model_sector_code", "model_sector_label", "reference_year", "concept_code",
        "concept_label", "value", "normalized_unit", "currency", "price_basis", "status_flag",
    }
    _require_columns(rows, required, "source input")
    qa = {k: 0 for k in QA_FIELDS}
    qa["source_rows"] = len(rows)
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
        core = {k: row[k] for k in sorted(row)}
        fingerprint = digest(core)
        normalized = {
            "source_record_id": "decs_" + fingerprint[:24],
            **row,
            "source_fingerprint": fingerprint,
            "schema_version": config["schema_version"],
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


def _select_concept(rows: list[dict[str, str]], concept_code: str) -> dict[str, str] | None:
    matches = [row for row in rows if row["concept_code"] == concept_code]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple rows for concept {concept_code} in coefficient group")
    return matches[0]


def _join_distinct(rows: list[dict[str, str]], field: str) -> str:
    return "|".join(sorted({row[field] for row in rows if row[field]}))


def _numerator(rows: list[dict[str, str]], outcome: Mapping[str, Any]) -> tuple[Decimal | None, str, list[dict[str, str]], str]:
    direct = _select_concept(rows, outcome["numerator_concept_code"])
    if direct is not None:
        return dec(direct["value"], "numerator"), direct["normalized_unit"], [direct], outcome["numerator_concept_code"]
    derivation = outcome.get("derivation")
    if not derivation:
        return None, "", [], outcome["numerator_concept_code"]
    components = []
    for concept_code in derivation["source_concept_codes"]:
        row = _select_concept(rows, concept_code)
        if row is None:
            return None, "", [], outcome["numerator_concept_code"]
        components.append(row)
    units = {r["normalized_unit"] for r in components}
    if len(units) != 1:
        return None, "", components, outcome["numerator_concept_code"]
    value = sum((dec(r["value"], "derived numerator") for r in components), Decimal("0"))
    expression = "+".join(derivation["source_concept_codes"])
    return value, components[0]["normalized_unit"], components, expression


def _coefficient_record(*, group_key, rows, route, outcome_code, outcome, denominator,
                        numerator_value, numerator_unit, numerator_rows, numerator_expression,
                        config, generated_at, qa_status, qa_reason):
    country, sector_code, year = group_key
    source_rows = [r for r in ([denominator] if denominator else []) + list(numerator_rows) if r is not None]
    label_row = source_rows[0] if source_rows else rows[0]
    coefficient_value = ""
    if qa_status == "calculated":
        result = numerator_value / dec(denominator["value"], "denominator")
        result *= dec(outcome["scale"], "scale")
        coefficient_value = format(result.normalize(), "f")
    transform_numerator = f"({numerator_expression})" if "+" in numerator_expression else numerator_expression
    core = {
        "coefficient_schema_version": config["coefficient_schema_version"],
        "coefficient_release_version": config["coefficient_release_version"],
        "country": country,
        "model_classification": config["model_classification"],
        "model_classification_version": config["model_classification_version"],
        "model_sector_code": sector_code,
        "model_sector_label": label_row["model_sector_label"],
        "reference_year": year,
        "outcome_code": outcome_code,
        "outcome_label": outcome["label"],
        "numerator_concept_code": outcome["numerator_concept_code"],
        "numerator_value": format(numerator_value, "f") if numerator_value is not None else "",
        "numerator_unit": numerator_unit,
        "denominator_concept_code": route["denominator_concept_code"],
        "denominator_value": denominator["value"] if denominator else "",
        "denominator_unit": denominator["normalized_unit"] if denominator else "",
        "denominator_currency": denominator["currency"] if denominator else "",
        "price_basis": denominator["price_basis"] if denominator else "",
        "coefficient_value": coefficient_value,
        "coefficient_unit": outcome["coefficient_unit"],
        "denominator_method": route["method"],
        "denominator_compatibility_class": route["compatibility_class"],
        "source_family": route["source_family"],
        "source_organisation": _join_distinct(source_rows, "source_organisation"),
        "source_dataset_ids": _join_distinct(source_rows, "source_dataset_id"),
        "source_release_versions": _join_distinct(source_rows, "source_release_version"),
        "source_release_dates": _join_distinct(source_rows, "source_release_date"),
        "source_retrieved_at": _join_distinct(source_rows, "retrieved_at"),
        "source_urls": _join_distinct(source_rows, "source_url"),
        "licences": _join_distinct(source_rows, "licence"),
        "source_status_flags": _join_distinct(source_rows, "status_flag"),
        "transformation_method": f"{transform_numerator}/{route['denominator_concept_code']}*{outcome['scale']}",
        "qa_status": qa_status,
        "qa_reason": qa_reason,
        "source_record_ids": _join_distinct(source_rows, "source_record_id"),
        "source_fingerprints": _join_distinct(source_rows, "source_fingerprint"),
        "generated_at": generated_at,
    }
    fp = digest(core)
    return {"coefficient_id": "decc_" + fp[:24], **core, "record_fingerprint": fp}


def build_coefficients(source_rows: list[dict[str, str]], *, config: Mapping[str, Any], generated_at: str) -> dict[str, Any]:
    if not clean(generated_at):
        raise ValueError("generated_at required")
    normalized_rows, qa = normalize_source_rows(source_rows, config=config)
    grouped = defaultdict(list)
    all_grouped = defaultdict(list)
    for row in normalized_rows:
        grouped[(row["source_family"], row["country"], row["model_sector_code"], row["reference_year"])].append(row)
        all_grouped[(row["country"], row["model_sector_code"], row["reference_year"])].append(row)

    calculation_keys = sorted(all_grouped)
    qa["groups"] = len(calculation_keys)
    coefficient_rows = []
    coverage_rows = []
    identity = config["accounting_identity"]

    for group_key in calculation_keys:
        country, sector_code, year = group_key
        route_name, route = _route_for_sector(sector_code, config)
        qa["output_route_groups" if route_name == "default" else "trade_route_groups"] += 1
        family_rows = grouped.get((route["source_family"], country, sector_code, year), [])
        fallback_rows = all_grouped[group_key]
        denominator = _select_concept(family_rows, route["denominator_concept_code"]) if family_rows else None
        denominator_problem = ""
        if denominator is None:
            qa["missing_denominator_groups"] += 1
            denominator_problem = "denominator_missing"
        elif dec(denominator["value"], "denominator") <= 0:
            qa["nonpositive_denominator_groups"] += 1
            denominator_problem = "denominator_nonpositive"

        identity_rows = grouped.get((identity["source_family"], country, sector_code, year), [])
        left = _select_concept(identity_rows, identity["left_concept_code"]) if identity_rows else None
        right_a = _select_concept(identity_rows, identity["right_concept_code_a"]) if identity_rows else None
        right_b = _select_concept(identity_rows, identity["right_concept_code_b"]) if identity_rows else None
        if left and right_a and right_b:
            qa["accounting_identity_checks"] += 1
            diff = abs(dec(left["value"], "identity left") - (dec(right_a["value"], "identity a") + dec(right_b["value"], "identity b")))
            if diff > dec(identity["absolute_tolerance"], "identity tolerance"):
                qa["accounting_identity_failures"] += 1

        calc_count = held_count = na_count = 0
        for outcome_code, outcome in config["outcomes"].items():
            applicable_routes = outcome.get("applicable_routes")
            if applicable_routes and route_name not in applicable_routes:
                status, reason = "not_applicable", "outcome_not_applicable_to_denominator_route"
                numerator_value, numerator_unit, numerator_rows, numerator_expression = None, "", [], outcome["numerator_concept_code"]
            else:
                numerator_value, numerator_unit, numerator_rows, numerator_expression = _numerator(family_rows, outcome)
                status, reason = "calculated", ""
                if denominator_problem:
                    status, reason = "held_out", denominator_problem
                elif numerator_value is None:
                    status, reason = "held_out", "numerator_missing"
                    qa["missing_numerator_records"] += 1
                elif outcome.get("required_numerator_unit") and numerator_unit != outcome["required_numerator_unit"]:
                    status, reason = "held_out", "numerator_unit_incompatible"
                elif route.get("required_denominator_unit") and denominator and denominator["normalized_unit"] != route["required_denominator_unit"]:
                    status, reason = "held_out", "denominator_unit_incompatible"
                elif len(numerator_rows) > 1:
                    qa["derived_numerator_records"] += 1
            if status == "calculated":
                calc_count += 1
                qa["calculated_coefficients"] += 1
            elif status == "not_applicable":
                na_count += 1
                qa["not_applicable_coefficients"] += 1
            else:
                held_count += 1
                qa["held_out_coefficients"] += 1
            coefficient_rows.append(_coefficient_record(
                group_key=group_key, rows=family_rows or fallback_rows, route=route,
                outcome_code=outcome_code, outcome=outcome, denominator=denominator,
                numerator_value=numerator_value, numerator_unit=numerator_unit,
                numerator_rows=numerator_rows, numerator_expression=numerator_expression,
                config=config, generated_at=generated_at, qa_status=status, qa_reason=reason,
            ))

        applicable_count = len(config["outcomes"]) - na_count
        if applicable_count and calc_count == applicable_count:
            coverage_status, coverage_reason = "complete", ""
        elif calc_count > 0:
            coverage_status, coverage_reason = "partial", "one_or_more_applicable_outcomes_unavailable"
        else:
            coverage_status, coverage_reason = "unmodelled", denominator_problem or "all_applicable_numerators_missing"
        coverage_rows.append({
            "country": country,
            "model_sector_code": sector_code,
            "model_sector_label": (family_rows[0] if family_rows else fallback_rows[0])["model_sector_label"],
            "reference_year": year,
            "denominator_route": route_name,
            "source_family": route["source_family"],
            "denominator_concept_code": route["denominator_concept_code"],
            "denominator_available": str(denominator is not None and not denominator_problem).lower(),
            "outcomes_requested": str(len(config["outcomes"])),
            "outcomes_calculated": str(calc_count),
            "outcomes_held_out": str(held_count),
            "outcomes_not_applicable": str(na_count),
            "coverage_status": coverage_status,
            "coverage_reason": coverage_reason,
        })

    coefficient_rows.sort(key=lambda row: (row["country"], row["model_sector_code"], row["reference_year"], row["outcome_code"]))
    coverage_rows.sort(key=lambda row: (row["country"], row["model_sector_code"], row["reference_year"]))
    qa["coefficient_records"] = len(coefficient_rows)
    return {"sources": normalized_rows, "coefficients": coefficient_rows, "coverage": coverage_rows, "qa": qa}


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
    write_csv(result["sources"], args.source_output, SOURCE_FIELDS)
    write_csv(result["coefficients"], args.coefficient_output, COEFFICIENT_FIELDS)
    write_csv(result["coverage"], args.coverage_output, COVERAGE_FIELDS)
    args.qa_output.parent.mkdir(parents=True, exist_ok=True)
    args.qa_output.write_text(json.dumps(result["qa"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
