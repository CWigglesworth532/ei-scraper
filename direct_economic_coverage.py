#!/usr/bin/env python3
"""SKO-036 governed coefficient coverage matrix and year-availability diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import direct_economic_coefficients as de

COVERAGE_MATRIX_FIELDS = [
    "country", "model_sector_code", "model_sector_label", "reference_year",
    "denominator_route", "denominator_concept_code", "outcome_code", "outcome_label",
    "numerator_concept_code", "availability_status", "availability_reason",
    "coefficient_value", "coefficient_unit", "coefficient_id", "source_family",
    "source_dataset_ids", "source_release_versions", "source_release_dates",
]

YEAR_DIAGNOSTIC_FIELDS = [
    "country", "model_sector_code", "model_sector_label", "outcome_code", "outcome_label",
    "available_years", "earliest_available_year", "latest_available_year",
    "available_year_count", "year_span", "internal_gap_years", "internal_gap_count",
    "fallback_policy_status",
]


def _route_name(sector_code: str, config: Mapping[str, Any]) -> str:
    name, _ = de._route_for_sector(sector_code, config)
    return name


def _build_coefficients_fast(source_rows: list[dict[str, str]], *, config: Mapping[str, Any], generated_at: str) -> dict[str, Any]:
    """Equivalent SKO-036 materialisation with pre-indexed group lookups.

    The original builder is intentionally simple but performed a full normalized-row
    scan once per country/sector/year group for fallback label rows. On live Eurostat
    extracts this becomes quadratic enough to be operationally unusable. This version
    preserves the same governed calculation contract while indexing both source-family
    and all-family group rows once up front.
    """
    if not de.clean(generated_at):
        raise ValueError("generated_at required")

    normalized_rows, qa = de.normalize_source_rows(source_rows, config=config)
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    grouped_all: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in normalized_rows:
        grouped[(row["source_family"], row["country"], row["model_sector_code"], row["reference_year"])].append(row)
        grouped_all[(row["country"], row["model_sector_code"], row["reference_year"])].append(row)

    calculation_keys = sorted(grouped_all)
    qa["groups"] = len(calculation_keys)
    coefficient_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, str]] = []
    identity = config["accounting_identity"]

    for group_key in calculation_keys:
        country, sector_code, year = group_key
        route_name, route = de._route_for_sector(sector_code, config)
        qa["output_route_groups" if route_name == "default" else "trade_route_groups"] += 1
        family_rows = grouped.get((route["source_family"], country, sector_code, year), [])
        fallback_rows = grouped_all[group_key]

        denominator = de._select_concept(family_rows, route["denominator_concept_code"]) if family_rows else None
        denominator_problem = ""
        if denominator is None:
            qa["missing_denominator_groups"] += 1
            denominator_problem = "denominator_missing"
        elif de.dec(denominator["value"], "denominator") <= 0:
            qa["nonpositive_denominator_groups"] += 1
            denominator_problem = "denominator_nonpositive"

        identity_rows = grouped.get((identity["source_family"], country, sector_code, year), [])
        left = de._select_concept(identity_rows, identity["left_concept_code"]) if identity_rows else None
        right_a = de._select_concept(identity_rows, identity["right_concept_code_a"]) if identity_rows else None
        right_b = de._select_concept(identity_rows, identity["right_concept_code_b"]) if identity_rows else None
        if left and right_a and right_b:
            qa["accounting_identity_checks"] += 1
            diff = abs(
                de.dec(left["value"], "identity left")
                - (de.dec(right_a["value"], "identity a") + de.dec(right_b["value"], "identity b"))
            )
            if diff > de.dec(identity["absolute_tolerance"], "identity tolerance"):
                qa["accounting_identity_failures"] += 1

        calc_count = 0
        held_count = 0
        for outcome_code, outcome in config["outcomes"].items():
            numerator = de._select_concept(family_rows, outcome["numerator_concept_code"]) if family_rows else None
            status, reason = "calculated", ""
            if denominator_problem:
                status, reason = "held_out", denominator_problem
            elif numerator is None:
                status, reason = "held_out", "numerator_missing"
                qa["missing_numerator_records"] += 1
            elif outcome.get("required_numerator_unit") and numerator["normalized_unit"] != outcome["required_numerator_unit"]:
                status, reason = "held_out", "numerator_unit_incompatible"
            elif route.get("required_denominator_unit") and denominator and denominator["normalized_unit"] != route["required_denominator_unit"]:
                status, reason = "held_out", "denominator_unit_incompatible"

            if status == "calculated":
                calc_count += 1
                qa["calculated_coefficients"] += 1
            else:
                held_count += 1
                qa["held_out_coefficients"] += 1

            coefficient_rows.append(de._coefficient_record(
                group_key=group_key,
                rows=family_rows or fallback_rows,
                route=route,
                outcome_code=outcome_code,
                outcome=outcome,
                denominator=denominator,
                numerator=numerator,
                config=config,
                generated_at=generated_at,
                qa_status=status,
                qa_reason=reason,
            ))

        if calc_count == len(config["outcomes"]):
            coverage_status, coverage_reason = "complete", ""
        elif calc_count > 0:
            coverage_status, coverage_reason = "partial", "one_or_more_outcomes_unavailable"
        else:
            coverage_status, coverage_reason = "unmodelled", denominator_problem or "all_numerators_missing"

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
            "coverage_status": coverage_status,
            "coverage_reason": coverage_reason,
        })

    coefficient_rows.sort(key=lambda row: (row["country"], row["model_sector_code"], row["reference_year"], row["outcome_code"]))
    coverage_rows.sort(key=lambda row: (row["country"], row["model_sector_code"], row["reference_year"]))
    qa["coefficient_records"] = len(coefficient_rows)
    return {"sources": normalized_rows, "coefficients": coefficient_rows, "coverage": coverage_rows, "qa": qa}


def build_coverage(source_rows: list[dict[str, str]], *, config: Mapping[str, Any], generated_at: str) -> dict[str, Any]:
    result = _build_coefficients_fast(source_rows, config=config, generated_at=generated_at)
    matrix: list[dict[str, str]] = []
    by_key: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)

    for row in result["coefficients"]:
        status = "available" if row["qa_status"] == "calculated" else "held_out"
        rec = {
            "country": row["country"],
            "model_sector_code": row["model_sector_code"],
            "model_sector_label": row["model_sector_label"],
            "reference_year": row["reference_year"],
            "denominator_route": _route_name(row["model_sector_code"], config),
            "denominator_concept_code": row["denominator_concept_code"],
            "outcome_code": row["outcome_code"],
            "outcome_label": row["outcome_label"],
            "numerator_concept_code": row["numerator_concept_code"],
            "availability_status": status,
            "availability_reason": row["qa_reason"],
            "coefficient_value": row["coefficient_value"],
            "coefficient_unit": row["coefficient_unit"],
            "coefficient_id": row["coefficient_id"],
            "source_family": row["source_family"],
            "source_dataset_ids": row["source_dataset_ids"],
            "source_release_versions": row["source_release_versions"],
            "source_release_dates": row["source_release_dates"],
        }
        matrix.append(rec)
        by_key[(rec["country"], rec["model_sector_code"], rec["outcome_code"])].append(rec)

    matrix.sort(key=lambda r: (r["country"], r["model_sector_code"], int(r["reference_year"]), r["outcome_code"]))

    diagnostics: list[dict[str, str]] = []
    fallback_status = config.get("year_fallback_policy", {}).get("status", "not_configured")
    for (country, sector, outcome), rows in sorted(by_key.items()):
        usable = sorted({int(r["reference_year"]) for r in rows if r["availability_status"] == "available"})
        gaps: list[int] = []
        if len(usable) >= 2:
            present = set(usable)
            gaps = [year for year in range(usable[0], usable[-1] + 1) if year not in present]
        diagnostics.append({
            "country": country,
            "model_sector_code": sector,
            "model_sector_label": rows[0]["model_sector_label"],
            "outcome_code": outcome,
            "outcome_label": rows[0]["outcome_label"],
            "available_years": "|".join(map(str, usable)),
            "earliest_available_year": str(usable[0]) if usable else "",
            "latest_available_year": str(usable[-1]) if usable else "",
            "available_year_count": str(len(usable)),
            "year_span": str(usable[-1] - usable[0] + 1) if usable else "0",
            "internal_gap_years": "|".join(map(str, gaps)),
            "internal_gap_count": str(len(gaps)),
            "fallback_policy_status": fallback_status,
        })

    summary = {
        "matrix_records": len(matrix),
        "available_records": sum(r["availability_status"] == "available" for r in matrix),
        "held_out_records": sum(r["availability_status"] == "held_out" for r in matrix),
        "country_count": len({r["country"] for r in matrix}),
        "sector_count": len({(r["country"], r["model_sector_code"]) for r in matrix}),
        "outcome_count": len({r["outcome_code"] for r in matrix}),
        "diagnostic_rows": len(diagnostics),
        "fallback_policy_status": fallback_status,
        "coefficient_qa": result["qa"],
    }
    return {"coverage_matrix": matrix, "year_diagnostics": diagnostics, "summary": summary}


def write_csv(rows: Iterable[Mapping[str, Any]], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-input", type=Path, required=True)
    parser.add_argument("--coverage-output", type=Path, required=True)
    parser.add_argument("--year-diagnostics-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--generated-at", required=True)
    args = parser.parse_args()
    config = de.load_config(args.config)
    result = build_coverage(de.read_csv(args.source_input), config=config, generated_at=args.generated_at)
    write_csv(result["coverage_matrix"], args.coverage_output, COVERAGE_MATRIX_FIELDS)
    write_csv(result["year_diagnostics"], args.year_diagnostics_output, YEAR_DIAGNOSTIC_FIELDS)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
