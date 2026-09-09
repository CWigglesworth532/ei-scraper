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


def build_coverage(source_rows: list[dict[str, str]], *, config: Mapping[str, Any], generated_at: str) -> dict[str, Any]:
    result = de.build_coefficients(source_rows, config=config, generated_at=generated_at)
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
