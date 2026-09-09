#!/usr/bin/env python3
"""SKO-037 fixed-reference-year direct environmental attribution for procurement spend.

Applies governed country x sector direct environmental coefficients to supplier-agnostic
procurement observations. Social-economy status is intentionally ignored.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

import apply_direct_economic_coefficients as econ_apply
import direct_environmental_coefficients as env

OBSERVATION_FIELDS = [
    "selection_id", "supplier", "client", "country", "spend_eur", "spend_year",
    "proposed_nace_rev2_code", "nace_level", "nace_description", "treatment",
    "model_sector_code", "model_sector_label", "model_mapping_status", "model_mapping_reason",
    "coefficient_reference_year", "coefficient_country_supported", "coefficient_source_sector_code",
    "coefficient_specificity_status", "coefficient_specificity_reason", "available_outcomes",
    "held_out_outcomes", "any_outcome_available", "observation_status",
]

OUTCOME_FIELDS = [
    "selection_id", "supplier", "client", "country", "spend_eur", "spend_year", "spend_currency",
    "proposed_nace_rev2_code", "nace_level", "treatment", "model_sector_code", "model_sector_label",
    "coefficient_reference_year", "coefficient_source_sector_code", "coefficient_source_sector_label",
    "coefficient_specificity_status", "coefficient_specificity_reason", "outcome_code", "outcome_label",
    "denominator_route", "denominator_currency", "attribution_status", "attribution_reason",
    "coefficient_value", "coefficient_unit", "coefficient_id", "modelled_value", "modelled_unit",
    "fx_status", "fx_rate", "fx_reference_year", "fx_source_organisation", "fx_source_series_key",
    "source_family", "source_dataset_ids", "source_release_versions", "source_release_dates",
]

_REQUIRED_COHORT_FIELDS = {
    "selection_id", "supplier", "client", "country", "spend_eur",
    "proposed_nace_rev2_code", "nace_level", "nace_description", "treatment",
}

_REQUIRED_MATRIX_FIELDS = {
    "country", "coefficient_source_sector_code", "coefficient_source_sector_label",
    "reference_year", "outcome_code", "outcome_label", "denominator_route",
    "denominator_currency", "availability_status", "availability_reason", "coefficient_value",
    "coefficient_unit", "coefficient_id", "source_family", "source_dataset_ids",
    "source_release_versions", "source_release_dates",
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


def _fallback_map(config: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, str]]:
    policy = config.get("granularity_fallback_policy", {})
    if policy.get("mode") != "explicit_approved_mapping_only":
        raise ValueError("SKO-037 v1 requires explicit approved granularity mappings")
    result: dict[tuple[str, str], dict[str, str]] = {}
    for raw in policy.get("approved_mappings", []):
        country = str(raw.get("country", "")).strip().upper()
        model_sector = str(raw.get("model_sector_code", "")).strip()
        source_sector = str(raw.get("coefficient_source_sector_code", "")).strip()
        trigger = str(raw.get("trigger", "")).strip()
        reason = str(raw.get("reason", "")).strip()
        if not country or not model_sector or not source_sector or not trigger or not reason:
            raise ValueError("Incomplete approved granularity fallback mapping")
        key = (country, model_sector)
        if key in result:
            raise ValueError(f"Duplicate granularity fallback mapping: {key}")
        model_route, _ = env._route_for_sector(model_sector, config)
        source_route, _ = env._route_for_sector(source_sector, config)
        if model_route != source_route:
            raise ValueError(f"Granularity fallback changes denominator route: {key} -> {source_sector}")
        result[key] = {
            "coefficient_source_sector_code": source_sector,
            "trigger": trigger,
            "reason": reason,
        }
    return result


def _fx_details(config: Mapping[str, Any], denominator_currency: str) -> dict[str, str]:
    policy = config.get("currency_normalisation", {})
    spend_currency = str(policy.get("spend_currency", "EUR")).strip().upper()
    denom_currency = (denominator_currency or "").strip().upper()
    base = {
        "spend_currency": spend_currency,
        "denominator_currency": denom_currency,
        "fx_status": "",
        "fx_rate": "",
        "fx_reference_year": str(policy.get("reference_year", "")),
        "fx_source_organisation": "",
        "fx_source_series_key": "",
    }
    if not denom_currency:
        base["fx_status"] = "missing_denominator_currency"
        return base
    rate_info = policy.get("rates", {}).get(denom_currency)
    if not rate_info:
        base["fx_status"] = "missing_fx_rate"
        return base
    rate = _decimal(rate_info.get("rate", ""), f"FX rate {denom_currency}")
    if rate <= 0:
        raise ValueError(f"FX rate for {denom_currency} must be positive")
    base.update({
        "fx_status": "same_currency" if denom_currency == spend_currency else "converted_annual_average",
        "fx_rate": format(rate, "f"),
        "fx_source_organisation": str(rate_info.get("source_organisation", "")),
        "fx_source_series_key": str(rate_info.get("source_series_key", "")),
    })
    return base


def _modelled_value(
    spend_eur: Decimal, coefficient: Decimal, coefficient_unit: str,
    denominator_currency: str, config: Mapping[str, Any],
) -> tuple[str, str, dict[str, str]]:
    unit = (coefficient_unit or "").strip()
    if unit != "tco2e_per_million_currency_denominator":
        raise ValueError(f"Unsupported environmental coefficient unit: {coefficient_unit!r}")
    fx = _fx_details(config, denominator_currency)
    if fx["fx_status"] in {"missing_denominator_currency", "missing_fx_rate"}:
        return "", "", fx
    spend_in_denominator_currency = spend_eur * _decimal(fx["fx_rate"], "FX rate")
    value = spend_in_denominator_currency / Decimal("1000000") * coefficient
    return format(value.normalize(), "f"), "tCO2e", fx


def apply_coefficients(
    cohort_rows: list[dict[str, str]], matrix_rows: list[dict[str, str]], *, config: Mapping[str, Any]
) -> dict[str, Any]:
    _require_fields(cohort_rows, _REQUIRED_COHORT_FIELDS, "cohort")
    _require_fields(matrix_rows, _REQUIRED_MATRIX_FIELDS, "coverage matrix")

    policy = config.get("reference_year_policy", {})
    if policy.get("mode") != "fixed_release_year":
        raise ValueError("SKO-037 v1 requires fixed_release_year policy")
    ref_year = str(policy.get("primary_reference_year", "")).strip()
    if not ref_year:
        raise ValueError("primary_reference_year required")
    if policy.get("automatic_year_fallback_enabled"):
        raise ValueError("Automatic year fallback must be disabled")

    ref_rows = [r for r in matrix_rows if str(r.get("reference_year", "")).strip() == ref_year]
    if not ref_rows:
        raise ValueError(f"Coverage matrix contains no rows for reference year {ref_year}")

    outcomes = sorted({(r["outcome_code"], r["outcome_label"]) for r in ref_rows})
    countries = {r["country"].strip().upper() for r in ref_rows}
    source_sectors: dict[str, str] = {}
    cells: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in ref_rows:
        country = row["country"].strip().upper()
        sector = row["coefficient_source_sector_code"].strip()
        source_sectors.setdefault(sector, row["coefficient_source_sector_label"])
        key = (country, sector, row["outcome_code"])
        if key in cells:
            raise ValueError(f"Duplicate environmental coefficient matrix cell: {key}")
        cells[key] = row

    # Model sectors include source sectors plus explicitly approved narrower sectors.
    model_sectors = dict(source_sectors)
    fallback_map = _fallback_map(config)
    for (_, model_sector), mapping in fallback_map.items():
        model_sectors.setdefault(model_sector, model_sector)

    observations: list[dict[str, str]] = []
    outcome_rows: list[dict[str, str]] = []
    total_spend = Decimal("0")
    spend_with_any = Decimal("0")
    observations_with_any = 0
    fallback_observations = 0
    status_counts = Counter()
    mapping_reasons = Counter()
    spend_currency = str(config.get("currency_normalisation", {}).get("spend_currency", "EUR")).strip().upper()

    outcome_stats: dict[str, dict[str, Any]] = {
        code: {
            "outcome_label": label,
            "available_observations": 0,
            "available_spend_eur": Decimal("0"),
            "modelled_total": Decimal("0"),
            "modelled_unit": "",
        }
        for code, label in outcomes
    }

    for raw in cohort_rows:
        spend = _decimal(raw.get("spend_eur", ""), f"spend {raw.get('selection_id', '')}")
        if spend < 0:
            raise ValueError("Negative procurement spend is not supported in SKO-037 v1")
        total_spend += spend
        country = (raw.get("country") or "").strip().upper()
        model_sector, model_label, mapping_status, mapping_reason = econ_apply.map_nace_to_model_sector(
            raw.get("proposed_nace_rev2_code", ""), model_sectors
        )
        mapping_reasons[mapping_reason] += 1

        specificity_status = ""
        specificity_reason = ""
        coefficient_source_sector = model_sector
        if mapping_status == "mapped" and model_sector:
            approved = fallback_map.get((country, model_sector))
            # Fallback is considered only when no exact available cell exists for any configured outcome.
            exact_available = any(
                cells.get((country, model_sector, code), {}).get("availability_status") == "available"
                for code, _ in outcomes
            )
            if approved and not exact_available:
                coefficient_source_sector = approved["coefficient_source_sector_code"]
                specificity_status = "approved_parent_fallback"
                specificity_reason = approved["reason"]
                fallback_observations += 1
            else:
                specificity_status = "exact"
                specificity_reason = "exact_model_sector_coefficient"

        available_codes: list[str] = []
        held_out_codes: list[str] = []
        country_supported = country in countries

        for outcome_code, outcome_label in outcomes:
            cell = cells.get((country, coefficient_source_sector, outcome_code)) if coefficient_source_sector else None
            attribution_status = "held_out"
            attribution_reason = ""
            modelled_value = ""
            modelled_unit = ""
            fx = {
                "spend_currency": spend_currency, "denominator_currency": "", "fx_status": "",
                "fx_rate": "", "fx_reference_year": "", "fx_source_organisation": "",
                "fx_source_series_key": "",
            }

            if mapping_status != "mapped":
                attribution_reason = f"model_mapping_{mapping_reason}"
            elif not country_supported:
                attribution_reason = "coefficient_country_not_supported"
            elif cell is None:
                attribution_reason = "coefficient_cell_missing"
            elif cell.get("availability_status") != "available":
                attribution_reason = cell.get("availability_reason") or "coefficient_held_out"
            else:
                coefficient = _decimal(cell["coefficient_value"], "coefficient")
                modelled_value, modelled_unit, fx = _modelled_value(
                    spend, coefficient, cell["coefficient_unit"], cell["denominator_currency"], config
                )
                if fx["fx_status"] in {"missing_denominator_currency", "missing_fx_rate"}:
                    attribution_reason = fx["fx_status"]
                else:
                    attribution_status = "available"
                    attribution_reason = "coefficient_applied"

            if attribution_status == "available":
                available_codes.append(outcome_code)
                stat = outcome_stats[outcome_code]
                stat["available_observations"] += 1
                stat["available_spend_eur"] += spend
                stat["modelled_total"] += _decimal(modelled_value, "modelled value")
                stat["modelled_unit"] = modelled_unit
            else:
                held_out_codes.append(outcome_code)

            outcome_rows.append({
                "selection_id": raw.get("selection_id", ""),
                "supplier": raw.get("supplier", ""),
                "client": raw.get("client", ""),
                "country": country,
                "spend_eur": raw.get("spend_eur", ""),
                "spend_year": raw.get("spend_year", ""),
                "spend_currency": spend_currency,
                "proposed_nace_rev2_code": raw.get("proposed_nace_rev2_code", ""),
                "nace_level": raw.get("nace_level", ""),
                "treatment": raw.get("treatment", ""),
                "model_sector_code": model_sector,
                "model_sector_label": model_label,
                "coefficient_reference_year": ref_year,
                "coefficient_source_sector_code": coefficient_source_sector,
                "coefficient_source_sector_label": source_sectors.get(coefficient_source_sector, ""),
                "coefficient_specificity_status": specificity_status,
                "coefficient_specificity_reason": specificity_reason,
                "outcome_code": outcome_code,
                "outcome_label": outcome_label,
                "denominator_route": cell.get("denominator_route", "") if cell else "",
                "denominator_currency": cell.get("denominator_currency", "") if cell else "",
                "attribution_status": attribution_status,
                "attribution_reason": attribution_reason,
                "coefficient_value": cell.get("coefficient_value", "") if cell else "",
                "coefficient_unit": cell.get("coefficient_unit", "") if cell else "",
                "coefficient_id": cell.get("coefficient_id", "") if cell else "",
                "modelled_value": modelled_value,
                "modelled_unit": modelled_unit,
                "fx_status": fx["fx_status"],
                "fx_rate": fx["fx_rate"],
                "fx_reference_year": fx["fx_reference_year"],
                "fx_source_organisation": fx["fx_source_organisation"],
                "fx_source_series_key": fx["fx_source_series_key"],
                "source_family": cell.get("source_family", "") if cell else "",
                "source_dataset_ids": cell.get("source_dataset_ids", "") if cell else "",
                "source_release_versions": cell.get("source_release_versions", "") if cell else "",
                "source_release_dates": cell.get("source_release_dates", "") if cell else "",
            })

        any_available = bool(available_codes)
        if any_available:
            observations_with_any += 1
            spend_with_any += spend
            observation_status = "modelled"
        elif mapping_status != "mapped":
            observation_status = "held_out_unresolved_activity"
        else:
            observation_status = "held_out_no_coefficient"
        status_counts[observation_status] += 1

        observations.append({
            "selection_id": raw.get("selection_id", ""),
            "supplier": raw.get("supplier", ""),
            "client": raw.get("client", ""),
            "country": country,
            "spend_eur": raw.get("spend_eur", ""),
            "spend_year": raw.get("spend_year", ""),
            "proposed_nace_rev2_code": raw.get("proposed_nace_rev2_code", ""),
            "nace_level": raw.get("nace_level", ""),
            "nace_description": raw.get("nace_description", ""),
            "treatment": raw.get("treatment", ""),
            "model_sector_code": model_sector,
            "model_sector_label": model_label,
            "model_mapping_status": mapping_status,
            "model_mapping_reason": mapping_reason,
            "coefficient_reference_year": ref_year,
            "coefficient_country_supported": "true" if country_supported else "false",
            "coefficient_source_sector_code": coefficient_source_sector,
            "coefficient_specificity_status": specificity_status,
            "coefficient_specificity_reason": specificity_reason,
            "available_outcomes": "|".join(available_codes),
            "held_out_outcomes": "|".join(held_out_codes),
            "any_outcome_available": "true" if any_available else "false",
            "observation_status": observation_status,
        })

    summary = {
        "reference_year": ref_year,
        "observations": len(cohort_rows),
        "total_spend_eur": float(total_spend),
        "observations_with_any_available": observations_with_any,
        "spend_with_any_available_eur": float(spend_with_any),
        "observation_coverage_rate": observations_with_any / len(cohort_rows) if cohort_rows else 0,
        "spend_coverage_rate": float(spend_with_any / total_spend) if total_spend else 0,
        "approved_parent_fallback_observations": fallback_observations,
        "observation_statuses": dict(sorted(status_counts.items())),
        "model_mapping_reasons": dict(sorted(mapping_reasons.items())),
        "outcomes": {
            code: {
                "outcome_label": stat["outcome_label"],
                "available_observations": stat["available_observations"],
                "available_spend_eur": float(stat["available_spend_eur"]),
                "modelled_total": float(stat["modelled_total"]),
                "modelled_unit": stat["modelled_unit"],
            }
            for code, stat in sorted(outcome_stats.items())
        },
    }
    return {"observations": observations, "outcomes": outcome_rows, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohort-input", type=Path, required=True)
    parser.add_argument("--coverage-matrix", type=Path, required=True)
    parser.add_argument("--observation-output", type=Path, required=True)
    parser.add_argument("--outcome-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    config = env.load_config(args.config)
    result = apply_coefficients(_read_csv(args.cohort_input), _read_csv(args.coverage_matrix), config=config)
    _write_csv(result["observations"], args.observation_output, OBSERVATION_FIELDS)
    _write_csv(result["outcomes"], args.outcome_output, OUTCOME_FIELDS)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
