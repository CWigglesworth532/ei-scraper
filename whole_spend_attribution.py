#!/usr/bin/env python3
"""SKO-038 governed whole-spend direct attribution composer.

Composes the accepted SKO-036 direct-economic and SKO-037 direct-GHG
application layers for supplier-agnostic procurement spend. This module does
not construct, recalculate, reinterpret, or override coefficient methodology.
It consumes the accepted application functions and normalises their results
into one deterministic observation/outcome/portfolio/QA contract.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

import apply_direct_economic_coefficients as econ_apply
import apply_direct_environmental_coefficients as env_apply
import direct_economic_coefficients as econ_coeff
import direct_environmental_coefficients as env_coeff


OBSERVATION_FIELDS = [
    "selection_id", "supplier", "client", "country", "spend", "spend_currency", "spend_year",
    "proposed_nace_rev2_code", "nace_level", "nace_description", "treatment",
    "model_sector_code", "model_sector_label", "mapping_status", "mapping_reason",
    "coefficient_reference_year", "economic_status", "economic_available_outcomes",
    "economic_held_out_outcomes", "economic_not_applicable_outcomes", "environmental_status",
    "environmental_available_outcomes", "environmental_held_out_outcomes", "any_outcome_modelled",
    "observation_status", "holdout_reason",
]

OUTCOME_FIELDS = [
    "selection_id", "supplier", "client", "country", "spend", "spend_currency", "spend_year",
    "proposed_nace_rev2_code", "nace_level", "nace_description", "treatment",
    "model_sector_code", "model_sector_label", "mapping_status", "mapping_reason",
    "outcome_domain", "outcome_code", "outcome_label", "coefficient_reference_year", "coefficient_id",
    "coefficient_source_sector_code", "coefficient_source_sector_label", "coefficient_specificity_status",
    "coefficient_specificity_reason", "denominator_route", "denominator_currency", "coefficient_value",
    "coefficient_unit", "fx_status", "fx_rate", "fx_reference_year", "fx_source_organisation",
    "fx_source_series_key", "attribution_status", "attribution_reason", "modelled_value", "modelled_unit",
    "source_family", "source_dataset_ids", "source_release_versions", "source_release_dates",
]

PORTFOLIO_FIELDS = [
    "outcome_domain", "outcome_code", "outcome_label", "total_observations", "modelled_observations",
    "held_out_observations", "not_applicable_observations", "total_spend", "modelled_spend",
    "spend_coverage_pct", "modelled_outcome_total", "modelled_unit",
]

QA_BREAKDOWN_FIELDS = [
    "breakdown_dimension", "breakdown_value", "outcome_domain", "outcome_code", "outcome_label",
    "total_observations", "modelled_observations", "held_out_observations", "not_applicable_observations",
    "total_spend", "modelled_spend", "spend_coverage_pct", "modelled_outcome_total", "modelled_unit",
]

_REQUIRED_COHORT_FIELDS = {
    "selection_id", "supplier", "client", "country", "spend_eur", "proposed_nace_rev2_code",
    "nace_level", "nace_description", "treatment",
}

_TRADE_SECTORS = {"G45", "G46", "G47"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(rows: Iterable[Mapping[str, Any]], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _decimal(value: Any, label: str) -> Decimal:
    try:
        result = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid decimal for {label}: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"Non-finite decimal for {label}")
    return result


def _format_decimal(value: Decimal) -> str:
    return format(value.normalize(), "f") if value else "0"


def _require_cohort(cohort_rows: list[dict[str, str]]) -> None:
    if not cohort_rows:
        raise ValueError("cohort is empty")
    missing = _REQUIRED_COHORT_FIELDS - set(cohort_rows[0])
    if missing:
        raise ValueError(f"cohort missing columns: {sorted(missing)}")
    ids = [(row.get("selection_id") or "").strip() for row in cohort_rows]
    if any(not value for value in ids):
        raise ValueError("selection_id must be non-blank")
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate selection_id values: {duplicates}")


def _observation_index(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = (row.get("selection_id") or "").strip()
        if not key:
            raise ValueError(f"{label} contains blank selection_id")
        if key in result:
            raise ValueError(f"{label} contains duplicate selection_id: {key}")
        result[key] = row
    return result


def _normalise_outcome_rows(
    domain: str,
    result: Mapping[str, Any],
    cohort_by_id: Mapping[str, dict[str, str]],
) -> list[dict[str, str]]:
    observations = _observation_index(list(result.get("observations", [])), f"{domain} observations")
    normalised: list[dict[str, str]] = []
    for raw in result.get("outcomes", []):
        selection_id = (raw.get("selection_id") or "").strip()
        if selection_id not in cohort_by_id:
            raise ValueError(f"{domain} outcome references unknown selection_id: {selection_id}")
        obs = observations.get(selection_id)
        if obs is None:
            raise ValueError(f"{domain} outcome has no observation row: {selection_id}")
        cohort = cohort_by_id[selection_id]
        model_sector = (raw.get("model_sector_code") or obs.get("model_sector_code") or "").strip()
        model_label = (raw.get("model_sector_label") or obs.get("model_sector_label") or "").strip()
        mapping_status = (obs.get("model_mapping_status") or "").strip()
        mapping_reason = (obs.get("model_mapping_reason") or "").strip()

        if domain == "environmental":
            source_sector = (raw.get("coefficient_source_sector_code") or model_sector).strip()
            source_label = (raw.get("coefficient_source_sector_label") or model_label).strip()
            specificity_status = (raw.get("coefficient_specificity_status") or "").strip()
            specificity_reason = (raw.get("coefficient_specificity_reason") or "").strip()
        else:
            source_sector = model_sector
            source_label = model_label
            specificity_status = "exact" if model_sector else ""
            specificity_reason = "exact_model_sector_coefficient" if model_sector else ""

        normalised.append({
            "selection_id": selection_id,
            "supplier": cohort.get("supplier", ""),
            "client": cohort.get("client", ""),
            "country": (cohort.get("country") or "").strip().upper(),
            "spend": raw.get("spend_eur", cohort.get("spend_eur", "")),
            "spend_currency": raw.get("spend_currency", "EUR"),
            "spend_year": raw.get("spend_year", cohort.get("spend_year", "")),
            "proposed_nace_rev2_code": cohort.get("proposed_nace_rev2_code", ""),
            "nace_level": cohort.get("nace_level", ""),
            "nace_description": cohort.get("nace_description", ""),
            "treatment": cohort.get("treatment", ""),
            "model_sector_code": model_sector,
            "model_sector_label": model_label,
            "mapping_status": mapping_status,
            "mapping_reason": mapping_reason,
            "outcome_domain": domain,
            "outcome_code": raw.get("outcome_code", ""),
            "outcome_label": raw.get("outcome_label", ""),
            "coefficient_reference_year": raw.get("coefficient_reference_year", ""),
            "coefficient_id": raw.get("coefficient_id", ""),
            "coefficient_source_sector_code": source_sector,
            "coefficient_source_sector_label": source_label,
            "coefficient_specificity_status": specificity_status,
            "coefficient_specificity_reason": specificity_reason,
            "denominator_route": raw.get("denominator_route", ""),
            "denominator_currency": raw.get("denominator_currency", ""),
            "coefficient_value": raw.get("coefficient_value", ""),
            "coefficient_unit": raw.get("coefficient_unit", ""),
            "fx_status": raw.get("fx_status", ""),
            "fx_rate": raw.get("fx_rate", ""),
            "fx_reference_year": raw.get("fx_reference_year", ""),
            "fx_source_organisation": raw.get("fx_source_organisation", ""),
            "fx_source_series_key": raw.get("fx_source_series_key", ""),
            "attribution_status": raw.get("attribution_status", ""),
            "attribution_reason": raw.get("attribution_reason", ""),
            "modelled_value": raw.get("modelled_value", ""),
            "modelled_unit": raw.get("modelled_unit", ""),
            "source_family": raw.get("source_family", ""),
            "source_dataset_ids": raw.get("source_dataset_ids", ""),
            "source_release_versions": raw.get("source_release_versions", ""),
            "source_release_dates": raw.get("source_release_dates", ""),
        })
    return normalised


def _headline_holdout_reason(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "no_outcome_rows"
    mapping_reasons = [row["mapping_reason"] for row in rows if row["mapping_status"] != "mapped" and row["mapping_reason"]]
    if mapping_reasons:
        return sorted(set(mapping_reasons))[0]
    held = [row["attribution_reason"] for row in rows if row["attribution_status"] == "held_out" and row["attribution_reason"]]
    return sorted(set(held))[0] if held else ""


def _build_observations(
    cohort_rows: list[dict[str, str]],
    economic: Mapping[str, Any],
    environmental: Mapping[str, Any],
    ledger: list[dict[str, str]],
) -> list[dict[str, str]]:
    econ_obs = _observation_index(list(economic.get("observations", [])), "economic observations")
    env_obs = _observation_index(list(environmental.get("observations", [])), "environmental observations")
    by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ledger:
        by_id[row["selection_id"]].append(row)

    output: list[dict[str, str]] = []
    for cohort in cohort_rows:
        selection_id = cohort["selection_id"].strip()
        eo = econ_obs.get(selection_id, {})
        vo = env_obs.get(selection_id, {})
        rows = by_id.get(selection_id, [])
        available = [row for row in rows if row["attribution_status"] == "available"]
        held = [row for row in rows if row["attribution_status"] == "held_out"]
        econ_rows = [row for row in rows if row["outcome_domain"] == "economic"]
        env_rows = [row for row in rows if row["outcome_domain"] == "environmental"]

        model_sector = (eo.get("model_sector_code") or vo.get("model_sector_code") or "").strip()
        model_label = (eo.get("model_sector_label") or vo.get("model_sector_label") or "").strip()
        mapping_status = (eo.get("model_mapping_status") or vo.get("model_mapping_status") or "").strip()
        mapping_reason = (eo.get("model_mapping_reason") or vo.get("model_mapping_reason") or "").strip()
        years = sorted({row["coefficient_reference_year"] for row in rows if row["coefficient_reference_year"]})
        if len(years) > 1:
            raise ValueError(f"coefficient reference-year mismatch for {selection_id}: {years}")

        if not available:
            status = "held_out"
        elif held:
            status = "partially_modelled"
        else:
            status = "modelled"

        output.append({
            "selection_id": selection_id,
            "supplier": cohort.get("supplier", ""),
            "client": cohort.get("client", ""),
            "country": (cohort.get("country") or "").strip().upper(),
            "spend": cohort.get("spend_eur", ""),
            "spend_currency": "EUR",
            "spend_year": cohort.get("spend_year", ""),
            "proposed_nace_rev2_code": cohort.get("proposed_nace_rev2_code", ""),
            "nace_level": cohort.get("nace_level", ""),
            "nace_description": cohort.get("nace_description", ""),
            "treatment": cohort.get("treatment", ""),
            "model_sector_code": model_sector,
            "model_sector_label": model_label,
            "mapping_status": mapping_status,
            "mapping_reason": mapping_reason,
            "coefficient_reference_year": years[0] if years else "",
            "economic_status": eo.get("observation_status", ""),
            "economic_available_outcomes": str(sum(row["attribution_status"] == "available" for row in econ_rows)),
            "economic_held_out_outcomes": str(sum(row["attribution_status"] == "held_out" for row in econ_rows)),
            "economic_not_applicable_outcomes": str(sum(row["attribution_status"] == "not_applicable" for row in econ_rows)),
            "environmental_status": vo.get("observation_status", ""),
            "environmental_available_outcomes": str(sum(row["attribution_status"] == "available" for row in env_rows)),
            "environmental_held_out_outcomes": str(sum(row["attribution_status"] == "held_out" for row in env_rows)),
            "any_outcome_modelled": str(bool(available)).lower(),
            "observation_status": status,
            "holdout_reason": _headline_holdout_reason(rows) if status != "modelled" else "",
        })
    return output


def _aggregate_rows(
    rows: list[dict[str, str]],
    *,
    total_observations: int,
    total_spend: Decimal,
) -> dict[str, str]:
    ids = {row["selection_id"] for row in rows}
    modelled_rows = [row for row in rows if row["attribution_status"] == "available"]
    held_rows = [row for row in rows if row["attribution_status"] == "held_out"]
    na_rows = [row for row in rows if row["attribution_status"] == "not_applicable"]
    modelled_ids = {row["selection_id"] for row in modelled_rows}
    spend_by_id: dict[str, Decimal] = {}
    for row in modelled_rows:
        spend_by_id.setdefault(row["selection_id"], _decimal(row["spend"], f"spend {row['selection_id']}"))
    modelled_spend = sum(spend_by_id.values(), Decimal("0"))
    modelled_total = sum((_decimal(row["modelled_value"], "modelled value") for row in modelled_rows), Decimal("0"))
    units = sorted({row["modelled_unit"] for row in modelled_rows if row["modelled_unit"]})
    if len(units) > 1:
        raise ValueError(f"mixed modelled units in aggregation: {units}")
    return {
        "total_observations": str(total_observations),
        "modelled_observations": str(len(modelled_ids)),
        "held_out_observations": str(len({row["selection_id"] for row in held_rows})),
        "not_applicable_observations": str(len({row["selection_id"] for row in na_rows})),
        "total_spend": format(total_spend, "f"),
        "modelled_spend": format(modelled_spend, "f"),
        "spend_coverage_pct": format((modelled_spend / total_spend * Decimal("100")) if total_spend else Decimal("0"), "f"),
        "modelled_outcome_total": _format_decimal(modelled_total),
        "modelled_unit": units[0] if units else "",
        "_row_observations": str(len(ids)),
    }


def _build_portfolio(ledger: list[dict[str, str]], cohort_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    total_spend = sum((_decimal(row["spend_eur"], f"spend {row['selection_id']}") for row in cohort_rows), Decimal("0"))
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in ledger:
        grouped[(row["outcome_domain"], row["outcome_code"], row["outcome_label"])].append(row)
    output = []
    for (domain, code, label), rows in sorted(grouped.items()):
        aggregate = _aggregate_rows(rows, total_observations=len(cohort_rows), total_spend=total_spend)
        aggregate.pop("_row_observations", None)
        output.append({"outcome_domain": domain, "outcome_code": code, "outcome_label": label, **aggregate})
    return output


def _build_qa_breakdowns(ledger: list[dict[str, str]]) -> list[dict[str, str]]:
    dimensions = {
        "country": lambda row: row["country"],
        "model_sector": lambda row: row["model_sector_code"] or "UNMAPPED",
        "denominator_route": lambda row: row["denominator_route"] or "UNSPECIFIED",
        "coefficient_specificity": lambda row: row["coefficient_specificity_status"] or "UNSPECIFIED",
        "holdout_reason": lambda row: row["attribution_reason"] if row["attribution_status"] == "held_out" else "NOT_HELD_OUT",
    }
    output: list[dict[str, str]] = []
    for dimension, extractor in dimensions.items():
        groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in ledger:
            key = (extractor(row), row["outcome_domain"], row["outcome_code"], row["outcome_label"])
            groups[key].append(row)
        for (value, domain, code, label), rows in sorted(groups.items()):
            ids = {row["selection_id"] for row in rows}
            spend_by_id = {row["selection_id"]: _decimal(row["spend"], f"spend {row['selection_id']}") for row in rows}
            aggregate = _aggregate_rows(
                rows,
                total_observations=len(ids),
                total_spend=sum(spend_by_id.values(), Decimal("0")),
            )
            aggregate.pop("_row_observations", None)
            output.append({
                "breakdown_dimension": dimension,
                "breakdown_value": value,
                "outcome_domain": domain,
                "outcome_code": code,
                "outcome_label": label,
                **aggregate,
            })
    return output


def _validate_ledger(ledger: list[dict[str, str]]) -> dict[str, Any]:
    duplicate_keys = Counter((r["selection_id"], r["outcome_domain"], r["outcome_code"]) for r in ledger)
    duplicates = sorted(key for key, count in duplicate_keys.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate observation/outcome rows: {duplicates}")

    lineage_missing = []
    trade_violations = []
    fallback_violations = []
    for row in ledger:
        if row["attribution_status"] == "available":
            if not row["coefficient_id"] or not row["source_dataset_ids"]:
                lineage_missing.append((row["selection_id"], row["outcome_domain"], row["outcome_code"]))
            if row["model_sector_code"] in _TRADE_SECTORS and row["denominator_route"] != "trade_turnover":
                trade_violations.append((row["selection_id"], row["outcome_domain"], row["outcome_code"], row["denominator_route"]))
        if row["coefficient_source_sector_code"] and row["model_sector_code"] and row["coefficient_source_sector_code"] != row["model_sector_code"]:
            if row["coefficient_specificity_status"] != "approved_parent_fallback":
                fallback_violations.append((row["selection_id"], row["outcome_domain"], row["model_sector_code"], row["coefficient_source_sector_code"]))

    if lineage_missing:
        raise ValueError(f"available outcomes missing coefficient/source lineage: {lineage_missing}")
    if trade_violations:
        raise ValueError(f"trade outcomes using non-turnover route: {trade_violations}")
    if fallback_violations:
        raise ValueError(f"unapproved coefficient-sector fallback: {fallback_violations}")

    return {
        "trade_route_invariant_passed": True,
        "fallback_invariant_passed": True,
        "coefficient_lineage_complete": True,
        "duplicate_observation_outcome_rows": 0,
    }


def _reconcile_portfolio(ledger: list[dict[str, str]], portfolio: list[dict[str, str]]) -> bool:
    expected: dict[tuple[str, str], Decimal] = defaultdict(lambda: Decimal("0"))
    for row in ledger:
        if row["attribution_status"] == "available":
            expected[(row["outcome_domain"], row["outcome_code"])] += _decimal(row["modelled_value"], "modelled value")
    actual = {
        (row["outcome_domain"], row["outcome_code"]): _decimal(row["modelled_outcome_total"], "portfolio total")
        for row in portfolio
    }
    return expected == actual


def compose_attribution(
    cohort_rows: list[dict[str, str]],
    economic_matrix_rows: list[dict[str, str]],
    environmental_matrix_rows: list[dict[str, str]],
    *,
    economic_config: Mapping[str, Any],
    environmental_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose accepted SKO-036/SKO-037 applications without recomputing coefficients."""
    _require_cohort(cohort_rows)
    cohort_by_id = {row["selection_id"].strip(): row for row in cohort_rows}

    economic = econ_apply.apply_coefficients(cohort_rows, economic_matrix_rows, config=economic_config)
    environmental = env_apply.apply_coefficients(cohort_rows, environmental_matrix_rows, config=environmental_config)

    econ_year = str(economic.get("summary", {}).get("coefficient_reference_year", ""))
    env_year = str(environmental.get("summary", {}).get("coefficient_reference_year", ""))
    if econ_year and env_year and econ_year != env_year:
        raise ValueError(f"accepted coefficient release mismatch: economic={econ_year}, environmental={env_year}")

    ledger = _normalise_outcome_rows("economic", economic, cohort_by_id)
    ledger.extend(_normalise_outcome_rows("environmental", environmental, cohort_by_id))
    ledger.sort(key=lambda row: (row["selection_id"], row["outcome_domain"], row["outcome_code"]))

    invariants = _validate_ledger(ledger)
    observations = _build_observations(cohort_rows, economic, environmental, ledger)
    observations.sort(key=lambda row: row["selection_id"])
    portfolio = _build_portfolio(ledger, cohort_rows)
    qa_breakdowns = _build_qa_breakdowns(ledger)

    portfolio_ok = _reconcile_portfolio(ledger, portfolio)
    if not portfolio_ok:
        raise ValueError("portfolio totals do not reconcile to observation-level outcome ledger")

    total_spend = sum((_decimal(row["spend_eur"], f"spend {row['selection_id']}") for row in cohort_rows), Decimal("0"))
    modelled_ids = {row["selection_id"] for row in ledger if row["attribution_status"] == "available"}
    modelled_spend = sum((_decimal(cohort_by_id[key]["spend_eur"], f"spend {key}") for key in modelled_ids), Decimal("0"))
    holdout_reasons = Counter(row["holdout_reason"] for row in observations if row["observation_status"] == "held_out")

    summary = {
        "coefficient_reference_year": econ_year or env_year,
        "observations": len(cohort_rows),
        "total_spend": format(total_spend, "f"),
        "spend_currency": "EUR",
        "observations_with_any_modelled_outcome": len(modelled_ids),
        "held_out_observations": len(cohort_rows) - len(modelled_ids),
        "modelled_spend": format(modelled_spend, "f"),
        "spend_coverage_rate": float(modelled_spend / total_spend) if total_spend else 0,
        "holdout_reasons": dict(sorted(holdout_reasons.items())),
        "outcome_rows": len(ledger),
        "portfolio_outcomes": len(portfolio),
        "observation_to_outcome_reconciliation_passed": len({row["selection_id"] for row in ledger}) == len(cohort_rows),
        "portfolio_total_reconciliation_passed": portfolio_ok,
        **invariants,
        "economic_application_summary": economic.get("summary", {}),
        "environmental_application_summary": environmental.get("summary", {}),
    }
    if not summary["observation_to_outcome_reconciliation_passed"]:
        raise ValueError("not every cohort observation survives into the outcome ledger")

    return {
        "observations": observations,
        "outcomes": ledger,
        "portfolio": portfolio,
        "qa_breakdowns": qa_breakdowns,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--economic-config", type=Path, required=True)
    parser.add_argument("--environmental-config", type=Path, required=True)
    parser.add_argument("--cohort-input", type=Path, required=True)
    parser.add_argument("--economic-coverage-matrix", type=Path, required=True)
    parser.add_argument("--environmental-coverage-matrix", type=Path, required=True)
    parser.add_argument("--observation-output", type=Path, required=True)
    parser.add_argument("--outcome-output", type=Path, required=True)
    parser.add_argument("--portfolio-output", type=Path, required=True)
    parser.add_argument("--qa-breakdown-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    result = compose_attribution(
        _read_csv(args.cohort_input),
        _read_csv(args.economic_coverage_matrix),
        _read_csv(args.environmental_coverage_matrix),
        economic_config=econ_coeff.load_config(args.economic_config),
        environmental_config=env_coeff.load_config(args.environmental_config),
    )
    _write_csv(result["observations"], args.observation_output, OBSERVATION_FIELDS)
    _write_csv(result["outcomes"], args.outcome_output, OUTCOME_FIELDS)
    _write_csv(result["portfolio"], args.portfolio_output, PORTFOLIO_FIELDS)
    _write_csv(result["qa_breakdowns"], args.qa_breakdown_output, QA_BREAKDOWN_FIELDS)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
