#!/usr/bin/env python3
"""SKO-036 live Eurostat extraction and normalisation for direct economic coefficients."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import urllib.parse
import urllib.request
from itertools import product
from pathlib import Path
from typing import Any, Iterable

API_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
MODEL_CLASSIFICATION = "NACE"
MODEL_CLASSIFICATION_VERSION = "Rev. 2 A*64"
DEFAULT_COUNTRIES = ["AT", "BE", "CH", "DE", "ES", "FI", "FR", "IE", "IT"]
GOVERNED_TRADE_DIVISIONS = {"G45", "G46", "G47"}

SOURCE_FIELDS = [
    "source_family", "source_organisation", "source_dataset_id", "source_release_version",
    "source_release_date", "retrieved_at", "source_url", "licence", "country",
    "source_classification", "source_classification_version", "source_sector_code",
    "source_sector_label", "model_classification", "model_classification_version",
    "model_sector_code", "model_sector_label", "reference_year", "concept_code",
    "concept_label", "value", "normalized_unit", "currency", "price_basis", "status_flag",
]


def _ordered_categories(payload: dict[str, Any], dim: str) -> list[str]:
    category = payload["dimension"][dim]["category"]
    index = category.get("index", {})
    if isinstance(index, list):
        return list(index)
    return [key for key, _ in sorted(index.items(), key=lambda kv: kv[1])]


def _labels(payload: dict[str, Any], dim: str) -> dict[str, str]:
    return payload["dimension"][dim]["category"].get("label", {}) or {}


def flatten_jsonstat(payload: dict[str, Any]) -> list[dict[str, Any]]:
    dims = list(payload["id"])
    sizes = list(payload["size"])
    cats = [_ordered_categories(payload, dim) for dim in dims]
    values = payload.get("value", {})
    statuses = payload.get("status", {}) or {}
    rows: list[dict[str, Any]] = []
    for coords in product(*[range(size) for size in sizes]):
        flat = 0
        stride = 1
        for pos in range(len(sizes) - 1, -1, -1):
            flat += coords[pos] * stride
            stride *= sizes[pos]
        key = str(flat)
        if key not in values and flat not in values:
            continue
        val = values.get(key, values.get(flat))
        if val is None:
            continue
        row = {dim: cats[i][coords[i]] for i, dim in enumerate(dims)}
        row["value"] = val
        row["status_flag"] = statuses.get(key, statuses.get(flat, ""))
        rows.append(row)
    return rows


def fetch_json(dataset: str, params: dict[str, str], *, timeout: int = 90) -> tuple[dict[str, Any], str]:
    query = urllib.parse.urlencode({"lang": "en", **params}, doseq=False)
    url = f"{API_BASE}/{dataset}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "skopia-sko036/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.load(response)
    if "error" in payload:
        raise RuntimeError(f"Eurostat API error for {dataset}: {payload['error']}")
    return payload, url


def _release_metadata(payload: dict[str, Any]) -> tuple[str, str]:
    updated = str(payload.get("updated", ""))
    if updated:
        try:
            parsed = dt.datetime.fromisoformat(updated.replace("Z", "+00:00"))
            return updated, parsed.date().isoformat()
        except ValueError:
            pass
    return updated or "live", ""


def _base_record(*, dataset: str, payload: dict[str, Any], url: str, country: str,
                 sector: str, sector_label: str, year: str, concept_code: str,
                 concept_label: str, value: Any, normalized_unit: str, currency: str,
                 price_basis: str, status_flag: str, retrieved_at: str,
                 source_family: str) -> dict[str, str]:
    release_version, release_date = _release_metadata(payload)
    return {
        "source_family": source_family,
        "source_organisation": "Eurostat",
        "source_dataset_id": dataset,
        "source_release_version": release_version,
        "source_release_date": release_date,
        "retrieved_at": retrieved_at,
        "source_url": url,
        "licence": "Eurostat reuse policy with source acknowledgement",
        "country": country,
        "source_classification": "NACE",
        "source_classification_version": "Rev. 2",
        "source_sector_code": sector,
        "source_sector_label": sector_label,
        "model_classification": MODEL_CLASSIFICATION,
        "model_classification_version": MODEL_CLASSIFICATION_VERSION,
        "model_sector_code": sector,
        "model_sector_label": sector_label,
        "reference_year": year,
        "concept_code": concept_code,
        "concept_label": concept_label,
        "value": str(value),
        "normalized_unit": normalized_unit,
        "currency": currency,
        "price_basis": price_basis,
        "status_flag": status_flag,
    }


def extract_national_accounts(country: str, start_year: int, end_year: int, retrieved_at: str) -> list[dict[str, str]]:
    dataset = "nama_10_a64"
    concepts = {
        "P1": ("Output", "basic_prices"),
        "P2": ("Intermediate consumption", "purchasers_prices"),
        "B1G": ("Gross value added", "basic_prices"),
        "D1": ("Compensation of employees", "income_account"),
        "D29X39": ("Other taxes less subsidies on production", "income_account"),
        "B2A3N": ("Net operating surplus and mixed income", "income_account"),
        "P51C": ("Consumption of fixed capital", "current_prices"),
    }
    rows: list[dict[str, str]] = []
    for concept, (label, price_basis) in concepts.items():
        payload, url = fetch_json(dataset, {
            "geo": country, "na_item": concept, "unit": "CP_MEUR",
            "sinceTimePeriod": str(start_year), "untilTimePeriod": str(end_year),
        })
        sector_labels = _labels(payload, "nace_r2")
        for raw in flatten_jsonstat(payload):
            sector = raw.get("nace_r2", "")
            year = raw.get("time", raw.get("TIME_PERIOD", ""))
            if not sector or not year:
                continue
            rows.append(_base_record(
                dataset=dataset, payload=payload, url=url, country=country,
                sector=sector, sector_label=sector_labels.get(sector, sector), year=str(year),
                concept_code=concept, concept_label=label, value=raw["value"],
                normalized_unit="million_currency", currency="EUR", price_basis=price_basis,
                status_flag=str(raw.get("status_flag", "")), retrieved_at=retrieved_at,
                source_family="national_accounts",
            ))
    return rows


def extract_employment(country: str, start_year: int, end_year: int, retrieved_at: str) -> list[dict[str, str]]:
    dataset = "nama_10_a64_e"
    unit_map = {
        "THS_PER": ("EMP_PERSONS", "Persons employed", "persons", 1000),
        "THS_HW": ("EMP_HOURS", "Hours worked", "hours", 1000),
    }
    rows: list[dict[str, str]] = []
    for unit, (concept, label, normalized_unit, scale) in unit_map.items():
        payload, url = fetch_json(dataset, {
            "geo": country, "na_item": "EMP_DC", "unit": unit,
            "sinceTimePeriod": str(start_year), "untilTimePeriod": str(end_year),
        })
        sector_labels = _labels(payload, "nace_r2")
        for raw in flatten_jsonstat(payload):
            sector = raw.get("nace_r2", "")
            year = raw.get("time", raw.get("TIME_PERIOD", ""))
            if not sector or not year:
                continue
            rows.append(_base_record(
                dataset=dataset, payload=payload, url=url, country=country,
                sector=sector, sector_label=sector_labels.get(sector, sector), year=str(year),
                concept_code=concept, concept_label=label, value=float(raw["value"]) * scale,
                normalized_unit=normalized_unit, currency="", price_basis="not_applicable",
                status_flag=str(raw.get("status_flag", "")), retrieved_at=retrieved_at,
                source_family="national_accounts",
            ))
    return rows


def extract_capital(country: str, start_year: int, end_year: int, retrieved_at: str) -> list[dict[str, str]]:
    dataset = "nama_10_a64_p5"
    payload, url = fetch_json(dataset, {
        "geo": country, "na_item": "P51G", "unit": "CP_MEUR", "asset10": "N11G",
        "sinceTimePeriod": str(start_year), "untilTimePeriod": str(end_year),
    })
    sector_labels = _labels(payload, "nace_r2")
    rows: list[dict[str, str]] = []
    for raw in flatten_jsonstat(payload):
        sector = raw.get("nace_r2", "")
        year = raw.get("time", raw.get("TIME_PERIOD", ""))
        if not sector or not year or raw.get("asset10", "N11G") != "N11G":
            continue
        rows.append(_base_record(
            dataset=dataset, payload=payload, url=url, country=country,
            sector=sector, sector_label=sector_labels.get(sector, sector), year=str(year),
            concept_code="P51G", concept_label="Gross fixed capital formation", value=raw["value"],
            normalized_unit="million_currency", currency="EUR", price_basis="current_prices",
            status_flag=str(raw.get("status_flag", "")), retrieved_at=retrieved_at,
            source_family="national_accounts",
        ))
    return rows


def extract_sbs(country: str, start_year: int, end_year: int, retrieved_at: str) -> list[dict[str, str]]:
    dataset = "sbs_ovw_act"
    mapping = {
        "NETTUR_MEUR": ("TURNOVER", "Net turnover", "million_currency", "EUR", "transaction_basis"),
        "AV_MEUR": ("B1G", "Value added", "million_currency", "EUR", "business_statistics"),
        "EXPN_SAL_BEN_MEUR": ("LABOUR_COSTS", "Employee benefits expense", "million_currency", "EUR", "business_statistics"),
        "EMP_NR": ("EMP_PERSONS", "Persons employed", "persons", "", "not_applicable"),
    }
    rows: list[dict[str, str]] = []
    for indicator, (concept, label, unit, currency, price_basis) in mapping.items():
        payload, url = fetch_json(dataset, {
            "geo": country, "indic_sbs": indicator,
            "sinceTimePeriod": str(start_year), "untilTimePeriod": str(end_year),
        })
        sector_labels = _labels(payload, "nace_r2")
        for raw in flatten_jsonstat(payload):
            sector = raw.get("nace_r2", "")
            year = raw.get("time", raw.get("TIME_PERIOD", ""))
            if not sector or not year:
                continue
            rows.append(_base_record(
                dataset=dataset, payload=payload, url=url, country=country,
                sector=sector, sector_label=sector_labels.get(sector, sector), year=str(year),
                concept_code=concept, concept_label=label, value=raw["value"],
                normalized_unit=unit, currency=currency, price_basis=price_basis,
                status_flag=str(raw.get("status_flag", "")), retrieved_at=retrieved_at,
                source_family="business_statistics",
            ))
    return rows


def constrain_to_a64_model(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int, int]:
    """Retain A*64 SBS rows plus governed trade-division overrides.

    Ordinary SBS rows are retained only when their NACE code is present in the
    country-specific nama_10_a64 vocabulary, preventing detailed SBS classes/groups
    from becoming pseudo-model sectors. The accepted hybrid denominator architecture
    is an explicit exception: G45/G46/G47 require turnover-compatible SBS denominators,
    so those division rows are retained even where nama_10_a64 exposes only a broader
    trade aggregate.
    """
    a64_by_country: dict[str, set[str]] = {}
    for row in rows:
        if row["source_dataset_id"] == "nama_10_a64":
            a64_by_country.setdefault(row["country"], set()).add(row["model_sector_code"])
    kept: list[dict[str, str]] = []
    dropped = 0
    retained_trade_overrides = 0
    for row in rows:
        if row["source_dataset_id"] != "sbs_ovw_act":
            kept.append(row)
            continue
        sector = row["model_sector_code"]
        if sector in a64_by_country.get(row["country"], set()):
            kept.append(row)
        elif sector in GOVERNED_TRADE_DIVISIONS:
            kept.append(row)
            retained_trade_overrides += 1
        else:
            dropped += 1
    return kept, dropped, retained_trade_overrides


def write_csv(rows: Iterable[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    retrieved_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    rows: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for country in args.countries:
        for name, fn in [
            ("nama_10_a64", extract_national_accounts),
            ("nama_10_a64_e", extract_employment),
            ("nama_10_a64_p5", extract_capital),
            ("sbs_ovw_act", extract_sbs),
        ]:
            try:
                rows.extend(fn(country, args.start_year, args.end_year, retrieved_at))
            except Exception as exc:
                errors.append({"country": country, "dataset": name, "error": f"{type(exc).__name__}: {exc}"})
    raw_rows = len(rows)
    rows, sbs_rows_dropped_non_a64, sbs_trade_override_rows_retained = constrain_to_a64_model(rows)
    rows.sort(key=lambda r: (r["country"], r["source_dataset_id"], r["model_sector_code"], r["reference_year"], r["concept_code"]))
    write_csv(rows, args.output)
    summary = {
        "retrieved_at": retrieved_at,
        "countries_requested": args.countries,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "raw_rows_retrieved": raw_rows,
        "rows_written": len(rows),
        "sbs_rows_dropped_non_a64": sbs_rows_dropped_non_a64,
        "sbs_trade_override_rows_retained": sbs_trade_override_rows_retained,
        "model_sector_pairs": len({(r["country"], r["model_sector_code"]) for r in rows}),
        "countries_with_rows": sorted({r["country"] for r in rows}),
        "datasets_with_rows": sorted({r["source_dataset_id"] for r in rows}),
        "concepts_with_rows": sorted({r["concept_code"] for r in rows}),
        "errors": errors,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
