#!/usr/bin/env python3
"""SKO-036 fixed-reference-year direct economic attribution for procurement spend.

Applies the governed country x A*64 x outcome coefficient layer to supplier-agnostic
procurement observations. The v1 reference year is read from configuration (2023).
No social-economy status is required or used.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

import direct_economic_coefficients as de

OBSERVATION_FIELDS = [
    "selection_id", "supplier", "client", "country", "spend_eur", "spend_year",
    "proposed_nace_rev2_code", "nace_level", "nace_description", "treatment",
    "model_sector_code", "model_sector_label", "model_mapping_status", "model_mapping_reason",
    "coefficient_reference_year", "coefficient_country_supported",
    "applicable_outcomes", "available_outcomes", "held_out_outcomes", "not_applicable_outcomes",
    "any_outcome_available", "observation_status",
]

OUTCOME_FIELDS = [
    "selection_id", "supplier", "client", "country", "spend_eur", "spend_year",
    "proposed_nace_rev2_code", "nace_level", "treatment", "model_sector_code",
    "model_sector_label", "coefficient_reference_year", "outcome_code", "outcome_label",
    "denominator_route", "attribution_status", "attribution_reason", "coefficient_value",
    "coefficient_unit", "coefficient_id", "modelled_value", "modelled_unit",
    "source_family", "source_dataset_ids", "source_release_versions", "source_release_dates",
]

_REQUIRED_COHORT_FIELDS = {
    "selection_id", "supplier", "client", "country", "spend_eur",
    "proposed_nace_rev2_code", "nace_level", "nace_description", "treatment",
}

_REQUIRED_MATRIX_FIELDS = {
    "country", "model_sector_code", "model_sector_label", "reference_year",
    "denominator_route", "outcome_code", "outcome_label", "availability_status",
    "availability_reason", "coefficient_value", "coefficient_unit", "coefficient_id",
    "source_family", "source_dataset_ids", "source_release_versions", "source_release_dates",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(rows: Iterable[Mapping[str, Any]], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _require_fields(rows: list[dict[str, str]], required: set[str], label: str) -> None:
    available = set(rows[0]) if rows else set()
    missing = required - available
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _decimal(value: Any, label: str) -> Decimal:
    try:
        d = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid decimal for {label}: {value!r}") from exc
    if not d.is_finite():
        raise ValueError(f"Non-finite decimal for {label}")
    return d


def _division_from_nace(code: str) -> int | None:
    text = (code or "").strip().upper()
    if not text:
        return None
    # Accept standard numeric class/group/division forms (94.11, 72.1, 72)
    m = re.search(r"(?<!\d)(\d{2})(?:\.|$)", text)
    if m:
        return int(m.group(1))
    # Also accept already model-like codes such as M72 or C10-C12.
    m = re.search(r"[A-U](\d{2})", text)
    return int(m.group(1)) if m else None


def _sector_division_range(model_sector_code: str) -> tuple[int, int] | None:
    text = (model_sector_code or "").strip().upper()
    nums = [int(x) for x in re.findall(r"\d{2}", text)]
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0], nums[0]
    return min(nums[0], nums[1]), max(nums[0], nums[1])


def map_nace_to_model_sector(nace_code: str, model_sectors: Mapping[str, str]) -> tuple[str, str, str, str]:
    """Map a NACE code to the most specific compatible sector in the model vocabulary.

    The Eurostat coefficient extract retains both detailed A*64 rows and some broader
    published aggregates. When more than one model-sector span contains a NACE division,
    the narrower span is the governed choice. A broader aggregate must not make a valid
    detailed/division-level mapping look ambiguous. If two or more equally specific
    candidates remain, the mapping is held out rather than guessed.
    """
    text = (nace_code or "").strip().upper()
    if not text:
        return "", "", "unresolved", "nace_code_missing"
    if text in model_sectors:
        return text, model_sectors[text], "mapped", "exact_model_sector_code"
    division = _division_from_nace(text)
    if division is None:
        return "", "", "unresolved", "nace_division_not_parseable"

    candidates: list[tuple[int, str, str]] = []
    for sector, label in model_sectors.items():
        span = _sector_division_range(sector)
        if span and span[0] <= division <= span[1]:
            width = span[1] - span[0]
            candidates.append((width, sector, label))

    if not candidates:
        return "", "", "unresolved", "no_a64_sector_for_division"

    min_width = min(width for width, _, _ in candidates)
    most_specific = sorted({(sector, label) for width, sector, label in candidates if width == min_width})
    if len(most_specific) == 1:
        sector, label = most_specific[0]
        reason = "division_to_a64_range" if len(candidates) == 1 else "division_to_most_specific_a64_sector"
        return sector, label, "mapped", reason

    return "", "", "unresolved", "ambiguous_equally_specific_a64_sectors"


def _modelled_value(spend_eur: Decimal, coefficient: Decimal, coefficient_unit: str) -> tuple[str, str]:
    unit = (coefficient_unit or "").strip()
    if unit.startswith("persons_per_million_currency_denominator"):
        value = spend_eur / Decimal("1000000") * coefficient
        return format(value.normalize(), "f"), "persons"
    if unit.startswith("hours_per_million_currency_denominator"):
        value = spend_eur / Decimal("1000000") * coefficient
        return format(value.normalize(), "f"), "hours"
    if unit.startswith("currency_"):
        value = spend_eur * coefficient
        return format(value.normalize(), "f"), "EUR"
    raise ValueError(f"Unsupported coefficient unit: {coefficient_unit!r}")


def apply_coefficients(
    cohort_rows: list[dict[str, str]],
    matrix_rows: list[dict[str, str]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    _require_fields(cohort_rows, _REQUIRED_COHORT_FIELDS, "cohort")
    _require_fields(matrix_rows, _REQUIRED_MATRIX_FIELDS, "coverage matrix")

    policy = config.get("reference_year_policy", {})
    if policy.get("mode") != "fixed_release_year":
        raise ValueError("SKO-036 pilot requires fixed_release_year policy")
    ref_year = str(policy.get("primary_reference_year", "")).strip()
    if not ref_year:
        raise ValueError("primary_reference_year required")
    if policy.get("automatic_year_fallback_enabled"):
        raise ValueError("Automatic year fallback must be disabled for fixed-year v1")

    ref_rows = [r for r in matrix_rows if str(r.get("reference_year", "")).strip() == ref_year]
    if not ref_rows:
        raise ValueError(f"Coverage matrix contains no rows for reference year {ref_year}")

    outcomes = sorted({(r["outcome_code"], r["outcome_label"]) for r in ref_rows})
    global_sectors: dict[str, str] = {}
    countries = set()
    cells: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in ref_rows:
        country = row["country"].strip().upper()
        sector = row["model_sector_code"].strip()
        countries.add(country)
        global_sectors.setdefault(sector, row["model_sector_label"])
        key = (country, sector, row["outcome_code"])
        if key in cells:
            raise ValueError(f"Duplicate coefficient matrix cell: {key}")
        cells[key] = row

    observation_rows: list[dict[str, str]] = []
    outcome_rows: list[dict[str, str]] = []
    outcome_stats: dict[str, dict[str, Any]] = {
        code: {
            "outcome_label": label,
            "observations": 0,
            "available_observations": 0,
            "applicable_observations": 0,
            "available_spend_eur": Decimal("0"),
            "applicable_spend_eur": Decimal("0"),
            "modelled_total": Decimal("0"),
            "modelled_unit": "",
        }
        for code, label in outcomes
    }

    total_spend = Decimal("0")
    observations_with_any_available = 0
    spend_with_any_available = Decimal("0")
    mapping_reasons = Counter()
    observation_statuses = Counter()

    for raw in cohort_rows:
        spend = _decimal(raw.get("spend_eur", ""), f"spend {raw.get('selection_id', '')}")
        if spend < 0:
            raise ValueError("Negative procurement spend is not supported in SKO-036 pilot")
        total_spend += spend
        country = (raw.get("country") or "").strip().upper()
        sector, sector_label, mapping_status, mapping_reason = map_nace_to_model_sector(
            raw.get("proposed_nace_rev2_code", ""), global_sectors
        )
        mapping_reasons[mapping_reason] += 1
        available_count = held_count = na_count = applicable_count = 0

        for outcome_code, outcome_label in outcomes:
            status = "held_out"
            reason = ""
            cell: dict[str, str] | None = None
            coeff_value = coeff_unit = coeff_id = route = ""
            modelled_value = modelled_unit = ""
            source_family = source_ids = source_versions = source_dates = ""

            if mapping_status != "mapped":
                reason = mapping_reason
            elif country not in countries:
                reason = "country_not_in_coefficient_layer"
            else:
                cell = cells.get((country, sector, outcome_code))
                if cell is None:
                    reason = "coefficient_cell_missing"
                else:
                    matrix_status = cell["availability_status"]
                    if matrix_status == "not_applicable":
                        status = "not_applicable"
                        reason = cell["availability_reason"] or "outcome_not_applicable"
                    elif matrix_status == "held_out":
                        reason = cell["availability_reason"] or "coefficient_held_out"
                    elif matrix_status == "available":
                        status = "available"
                        coeff_value = cell["coefficient_value"]
                        coeff_unit = cell["coefficient_unit"]
                        coeff_id = cell["coefficient_id"]
                        route = cell["denominator_route"]
                        source_family = cell["source_family"]
                        source_ids = cell["source_dataset_ids"]
                        source_versions = cell["source_release_versions"]
                        source_dates = cell["source_release_dates"]
                        modelled_value, modelled_unit = _modelled_value(
                            spend, _decimal(coeff_value, "coefficient"), coeff_unit
                        )
                    else:
                        raise ValueError(f"Unexpected matrix availability status: {matrix_status!r}")

            stats = outcome_stats[outcome_code]
            stats["observations"] += 1
            if status != "not_applicable":
                applicable_count += 1
                stats["applicable_observations"] += 1
                stats["applicable_spend_eur"] += spend
            if status == "available":
                available_count += 1
                stats["available_observations"] += 1
                stats["available_spend_eur"] += spend
                stats["modelled_total"] += _decimal(modelled_value, "modelled value")
                stats["modelled_unit"] = modelled_unit
            elif status == "held_out":
                held_count += 1
            else:
                na_count += 1

            outcome_rows.append({
                "selection_id": raw.get("selection_id", ""),
                "supplier": raw.get("supplier", ""),
                "client": raw.get("client", ""),
                "country": country,
                "spend_eur": format(spend, "f"),
                "spend_year": raw.get("spend_year", ""),
                "proposed_nace_rev2_code": raw.get("proposed_nace_rev2_code", ""),
                "nace_level": raw.get("nace_level", ""),
                "treatment": raw.get("treatment", ""),
                "model_sector_code": sector,
                "model_sector_label": sector_label,
                "coefficient_reference_year": ref_year,
                "outcome_code": outcome_code,
                "outcome_label": outcome_label,
                "denominator_route": route or (cell["denominator_route"] if cell else ""),
                "attribution_status": status,
                "attribution_reason": reason,
                "coefficient_value": coeff_value,
                "coefficient_unit": coeff_unit,
                "coefficient_id": coeff_id,
                "modelled_value": modelled_value,
                "modelled_unit": modelled_unit,
                "source_family": source_family,
                "source_dataset_ids": source_ids,
                "source_release_versions": source_versions,
                "source_release_dates": source_dates,
            })

        any_available = available_count > 0
        if any_available:
            observations_with_any_available += 1
            spend_with_any_available += spend
        if mapping_status != "mapped":
            obs_status = "held_out_nace_unresolved"
        elif country not in countries:
            obs_status = "held_out_country_source_missing"
        elif any_available:
            obs_status = "modelled_partial" if held_count else "modelled"
        else:
            obs_status = "held_out_no_available_outcomes"
        observation_statuses[obs_status] += 1

        observation_rows.append({
            "selection_id": raw.get("selection_id", ""),
            "supplier": raw.get("supplier", ""),
            "client": raw.get("client", ""),
            "country": country,
            "spend_eur": format(spend, "f"),
            "spend_year": raw.get("spend_year", ""),
            "proposed_nace_rev2_code": raw.get("proposed_nace_rev2_code", ""),
            "nace_level": raw.get("nace_level", ""),
            "nace_description": raw.get("nace_description", ""),
            "treatment": raw.get("treatment", ""),
            "model_sector_code": sector,
            "model_sector_label": sector_label,
            "model_mapping_status": mapping_status,
            "model_mapping_reason": mapping_reason,
            "coefficient_reference_year": ref_year,
            "coefficient_country_supported": str(country in countries).lower(),
            "applicable_outcomes": str(applicable_count),
            "available_outcomes": str(available_count),
            "held_out_outcomes": str(held_count),
            "not_applicable_outcomes": str(na_count),
            "any_outcome_available": str(any_available).lower(),
            "observation_status": obs_status,
        })

    outcome_summary = {}
    for code, stats in outcome_stats.items():
        applicable_spend = stats["applicable_spend_eur"]
        outcome_summary[code] = {
            "outcome_label": stats["outcome_label"],
            "applicable_observations": stats["applicable_observations"],
            "available_observations": stats["available_observations"],
            "observation_availability_rate": (
                stats["available_observations"] / stats["applicable_observations"]
                if stats["applicable_observations"] else 0
            ),
            "applicable_spend_eur": format(applicable_spend, "f"),
            "available_spend_eur": format(stats["available_spend_eur"], "f"),
            "spend_availability_rate": (
                float(stats["available_spend_eur"] / applicable_spend) if applicable_spend else 0
            ),
            "modelled_total": format(stats["modelled_total"].normalize(), "f") if stats["modelled_total"] else "0",
            "modelled_unit": stats["modelled_unit"],
        }

    summary = {
        "coefficient_reference_year": ref_year,
        "reference_year_policy_status": policy.get("status", ""),
        "automatic_year_fallback_enabled": bool(policy.get("automatic_year_fallback_enabled")),
        "observations": len(cohort_rows),
        "total_spend_eur": format(total_spend, "f"),
        "observations_with_any_available_outcome": observations_with_any_available,
        "observation_any_outcome_coverage_rate": observations_with_any_available / len(cohort_rows) if cohort_rows else 0,
        "spend_with_any_available_outcome_eur": format(spend_with_any_available, "f"),
        "spend_any_outcome_coverage_rate": float(spend_with_any_available / total_spend) if total_spend else 0,
        "spend_year_present_observations": sum(bool((r.get("spend_year") or "").strip()) for r in cohort_rows),
        "model_mapping_reasons": dict(sorted(mapping_reasons.items())),
        "observation_statuses": dict(sorted(observation_statuses.items())),
        "outcomes": outcome_summary,
    }
    return {"observations": observation_rows, "outcomes": outcome_rows, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohort-input", type=Path, required=True)
    parser.add_argument("--coverage-matrix", type=Path, required=True)
    parser.add_argument("--observation-output", type=Path, required=True)
    parser.add_argument("--outcome-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    config = de.load_config(args.config)
    result = apply_coefficients(
        _read_csv(args.cohort_input),
        _read_csv(args.coverage_matrix),
        config=config,
    )
    _write_csv(result["observations"], args.observation_output, OBSERVATION_FIELDS)
    _write_csv(result["outcomes"], args.outcome_output, OUTCOME_FIELDS)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
