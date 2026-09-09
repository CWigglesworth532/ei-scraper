#!/usr/bin/env python3
"""SKO-037 live-source extraction for the governed direct GHG pilot.

This module materialises only the environmental coefficient source cells required by
an input procurement cohort. It reuses accepted SKO-036 denominator source rows,
retrieves 2023 Eurostat Air Emissions Accounts GHG numerators, and reads the governed
ONS SIC 72 GHG cell from a caller-provided workbook.

The ONS workbook is intentionally supplied as a local file: ONS may reject automated
Python downloads. No client data are written outside caller-selected local outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

import apply_direct_economic_coefficients as econ_apply
import direct_environmental_coefficients as env
import extract_eurostat_direct_economic as eurostat

SOURCE_INPUT_FIELDS = [
    "source_family", "source_organisation", "source_dataset_id", "source_release_version",
    "source_release_date", "retrieved_at", "source_url", "licence", "country",
    "source_classification", "source_classification_version", "source_sector_code",
    "source_sector_label", "model_classification", "model_classification_version",
    "model_sector_code", "model_sector_label", "reference_year", "concept_code",
    "concept_label", "value", "normalized_unit", "currency", "price_basis", "status_flag",
]

ONS_DATASET_ID = "uk-environmental-accounts-atmospheric-emissions-greenhouse-gases-by-industry-and-gas"
ONS_RELEASE_DATE = "2026-06-05"
ONS_SOURCE_URL = (
    "https://www.ons.gov.uk/economy/environmentalaccounts/datasets/"
    "ukenvironmentalaccountsatmosphericemissionsgreenhousegasemissionsbyeconomicsectorandgasunitedkingdom/current"
)
ONS_GHG_SHEET = "GHG total"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(rows: Iterable[Mapping[str, Any]], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def approved_fallbacks(config: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for row in config.get("granularity_fallback_policy", {}).get("approved_mappings", []):
        key = (str(row["country"]).strip().upper(), str(row["model_sector_code"]).strip())
        result[key] = str(row["coefficient_source_sector_code"]).strip()
    return result


def target_cells(
    cohort_rows: list[dict[str, str]], economic_source_rows: list[dict[str, str]], *, config: Mapping[str, Any]
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Resolve cohort observations to governed coefficient source cells."""
    year = str(config["reference_year_policy"]["primary_reference_year"])
    model_sectors: dict[str, str] = {}
    for row in economic_source_rows:
        if str(row.get("reference_year", "")).strip() != year:
            continue
        code = str(row.get("model_sector_code", "")).strip()
        if code:
            model_sectors.setdefault(code, str(row.get("model_sector_label", "")).strip() or code)

    fallbacks = approved_fallbacks(config)
    for (_, narrower), _parent in fallbacks.items():
        model_sectors.setdefault(narrower, narrower)

    resolved: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in cohort_rows:
        country = str(row.get("country", "")).strip().upper()
        model_sector, model_label, status, reason = econ_apply.map_nace_to_model_sector(
            str(row.get("proposed_nace_rev2_code", "")), model_sectors
        )
        if status != "mapped":
            continue
        source_sector = fallbacks.get((country, model_sector), model_sector)
        key = (country, source_sector)
        if key in seen:
            continue
        seen.add(key)
        resolved.append({
            "country": country,
            "model_sector_code": model_sector,
            "model_sector_label": model_label,
            "coefficient_source_sector_code": source_sector,
            "fallback": "true" if source_sector != model_sector else "false",
            "mapping_reason": reason,
        })
    resolved.sort(key=lambda r: (r["country"], r["coefficient_source_sector_code"], r["model_sector_code"]))
    return resolved, model_sectors


def select_denominators(
    economic_source_rows: list[dict[str, str]], targets: list[dict[str, str]], *, config: Mapping[str, Any]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    year = str(config["reference_year_policy"]["primary_reference_year"])
    by_key: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in economic_source_rows:
        key = (
            str(row.get("source_family", "")).strip(),
            str(row.get("country", "")).strip().upper(),
            str(row.get("model_sector_code", "")).strip(),
            str(row.get("reference_year", "")).strip(),
            str(row.get("concept_code", "")).strip(),
        )
        by_key[key].append(row)

    selected: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for target in targets:
        country = target["country"]
        sector = target["coefficient_source_sector_code"]
        route_name, route = env._route_for_sector(sector, config)
        key = (route["source_family"], country, sector, year, route["denominator_concept_code"])
        matches = by_key.get(key, [])
        if len(matches) == 1:
            selected.append({field: str(matches[0].get(field, "")) for field in SOURCE_INPUT_FIELDS})
        elif not matches:
            missing.append({"country": country, "sector": sector, "reason": "missing_required_denominator", "route": route_name})
        else:
            raise ValueError(f"Multiple accepted SKO-036 denominator rows for {key}")
    return selected, missing


def normalize_eurostat_ghg(
    *, payload: dict[str, Any], url: str, country: str, required_sectors: set[str],
    year: str, retrieved_at: str,
) -> list[dict[str, str]]:
    labels = eurostat._labels(payload, "nace_r2")
    release_version, release_date = eurostat._release_metadata(payload)
    rows: list[dict[str, str]] = []
    for raw in eurostat.flatten_jsonstat(payload):
        sector = str(raw.get("nace_r2", "")).strip()
        row_year = str(raw.get("time", raw.get("TIME_PERIOD", ""))).strip()
        if sector not in required_sectors or row_year != year:
            continue
        # env_ac_ainah_r2 THS_T is thousand tonnes. Normalize before coefficient construction.
        value_tonnes = Decimal(str(raw["value"])) * Decimal("1000")
        rows.append({
            "source_family": "environmental_accounts",
            "source_organisation": "Eurostat",
            "source_dataset_id": "env_ac_ainah_r2",
            "source_release_version": release_version,
            "source_release_date": release_date,
            "retrieved_at": retrieved_at,
            "source_url": url,
            "licence": "Eurostat reuse policy with source acknowledgement",
            "country": country,
            "source_classification": "NACE",
            "source_classification_version": "Rev. 2",
            "source_sector_code": sector,
            "source_sector_label": labels.get(sector, sector),
            "model_classification": "NACE",
            "model_classification_version": "Rev. 2 A*64",
            "model_sector_code": sector,
            "model_sector_label": labels.get(sector, sector),
            "reference_year": year,
            "concept_code": "GHG",
            "concept_label": "Direct greenhouse gas emissions",
            "value": format(value_tonnes.normalize(), "f"),
            "normalized_unit": "tonne_co2e",
            "currency": "",
            "price_basis": "not_applicable",
            "status_flag": str(raw.get("status_flag", "")),
        })
    return rows


def extract_eurostat_targets(
    targets: list[dict[str, str]], *, year: str, retrieved_at: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    sectors_by_country: dict[str, set[str]] = defaultdict(set)
    for target in targets:
        if target["country"] != "UK":
            sectors_by_country[target["country"]].add(target["coefficient_source_sector_code"])

    rows: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for country, required in sorted(sectors_by_country.items()):
        payload, url = eurostat.fetch_json("env_ac_ainah_r2", {
            "geo": country,
            "airpol": "GHG",
            "unit": "THS_T",
            "sinceTimePeriod": year,
            "untilTimePeriod": year,
        })
        extracted = normalize_eurostat_ghg(
            payload=payload, url=url, country=country, required_sectors=required,
            year=year, retrieved_at=retrieved_at,
        )
        rows.extend(extracted)
        found = {r["model_sector_code"] for r in extracted}
        for sector in sorted(required - found):
            missing.append({"country": country, "sector": sector, "reason": "exact_2023_ghg_cell_not_published"})
    return rows, missing


def _xlsx_sheet_cells(path: Path, sheet_name: str) -> dict[str, str]:
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    docrel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{{{main}}}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{main}}}t")))
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        relroot = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rels = {r.attrib["Id"]: r.attrib["Target"] for r in relroot}
        target = None
        for sheet in wb.find(f"{{{main}}}sheets"):
            if sheet.attrib["name"] == sheet_name:
                target = rels[sheet.attrib[f"{{{docrel}}}id"]]
                break
        if target is None:
            raise ValueError(f"ONS workbook sheet not found: {sheet_name}")
        target = target.lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        root = ET.fromstring(z.read(target))
        cells: dict[str, str] = {}
        for cell in root.findall(f".//{{{main}}}c"):
            ref = cell.attrib["r"]
            typ = cell.attrib.get("t")
            v = cell.find(f"{{{main}}}v")
            value = ""
            if v is not None:
                value = shared[int(v.text)] if typ == "s" else str(v.text or "")
            else:
                inline = cell.find(f"{{{main}}}is")
                if inline is not None:
                    value = "".join(t.text or "" for t in inline.iter(f"{{{main}}}t"))
            cells[ref] = value
        return cells


def extract_ons_sic72_from_cells(cells: Mapping[str, str], *, retrieved_at: str, workbook_hash: str) -> dict[str, str]:
    title = str(cells.get("A1", ""))
    boundary = str(cells.get("A4", ""))
    if "thousand tonnes of carbon dioxide equivalent" not in title.lower():
        raise ValueError("ONS GHG total unit contract not confirmed in A1")
    if "uk residence basis" not in boundary.lower():
        raise ValueError("ONS GHG total residence-basis contract not confirmed in A4")

    sic72_col = None
    for ref, value in cells.items():
        match = re.fullmatch(r"([A-Z]+)7", ref)
        if match and str(value).strip() == "72":
            col = match.group(1)
            if str(cells.get(f"{col}8", "")).strip().lower().startswith("scientific research and development"):
                sic72_col = col
                break
    if sic72_col is None:
        raise ValueError("ONS SIC 72 column not found")

    row_2023 = None
    for row in range(1, 500):
        if str(cells.get(f"A{row}", "")).strip() == "2023":
            row_2023 = row
            break
    if row_2023 is None:
        raise ValueError("ONS 2023 row not found")
    raw = str(cells.get(f"{sic72_col}{row_2023}", "")).strip()
    if not raw or raw.startswith("["):
        raise ValueError(f"ONS SIC 72 2023 value is unavailable or suppressed: {raw!r}")
    tonnes = Decimal(raw) * Decimal("1000")
    return {
        "source_family": "environmental_accounts",
        "source_organisation": "Office for National Statistics",
        "source_dataset_id": ONS_DATASET_ID,
        "source_release_version": f"sha256:{workbook_hash}",
        "source_release_date": ONS_RELEASE_DATE,
        "retrieved_at": retrieved_at,
        "source_url": ONS_SOURCE_URL,
        "licence": "Open Government Licence",
        "country": "UK",
        "source_classification": "UK SIC 2007",
        "source_classification_version": "SIC 2007",
        "source_sector_code": "72",
        "source_sector_label": str(cells.get(f"{sic72_col}8", "Scientific research and development services")),
        "model_classification": "NACE",
        "model_classification_version": "Rev. 2 A*64",
        "model_sector_code": "M72",
        "model_sector_label": "Scientific research and development services",
        "reference_year": "2023",
        "concept_code": "GHG",
        "concept_label": "Direct greenhouse gas emissions",
        "value": format(tonnes.normalize(), "f"),
        "normalized_unit": "tonne_co2e",
        "currency": "",
        "price_basis": "not_applicable",
        "status_flag": "published",
    }


def extract_ons_targets(
    targets: list[dict[str, str]], *, workbook: Path, retrieved_at: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    uk = {t["coefficient_source_sector_code"] for t in targets if t["country"] == "UK"}
    if not uk:
        return [], []
    unsupported = sorted(uk - {"M72"})
    if unsupported:
        return [], [{"country": "UK", "sector": s, "reason": "ons_v1_source_contract_not_verified"} for s in unsupported]
    cells = _xlsx_sheet_cells(workbook, ONS_GHG_SHEET)
    row = extract_ons_sic72_from_cells(cells, retrieved_at=retrieved_at, workbook_hash=sha256_file(workbook))
    return [row], []


def build_live_source_input(
    cohort_rows: list[dict[str, str]], economic_source_rows: list[dict[str, str]], *,
    config: Mapping[str, Any], ons_workbook: Path, retrieved_at: str,
) -> dict[str, Any]:
    year = str(config["reference_year_policy"]["primary_reference_year"])
    targets, _model_sectors = target_cells(cohort_rows, economic_source_rows, config=config)
    denominators, missing_denominators = select_denominators(economic_source_rows, targets, config=config)
    eu_rows, missing_eu = extract_eurostat_targets(targets, year=year, retrieved_at=retrieved_at)
    ons_rows, missing_ons = extract_ons_targets(targets, workbook=ons_workbook, retrieved_at=retrieved_at)
    combined = denominators + eu_rows + ons_rows
    combined.sort(key=lambda r: (
        r["country"], r["model_sector_code"], r["reference_year"], r["source_family"], r["concept_code"]
    ))
    return {
        "source_rows": combined,
        "diagnostics": {
            "reference_year": year,
            "target_cells": targets,
            "target_cell_count": len(targets),
            "denominator_rows": len(denominators),
            "environmental_rows": len(eu_rows) + len(ons_rows),
            "missing_denominators": missing_denominators,
            "missing_environmental_cells": missing_eu + missing_ons,
            "ons_workbook_sha256": sha256_file(ons_workbook) if any(t["country"] == "UK" for t in targets) else "",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohort-input", type=Path, required=True)
    parser.add_argument("--economic-source-input", type=Path, required=True)
    parser.add_argument("--ons-ghg-workbook", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--diagnostics-output", type=Path, required=True)
    parser.add_argument("--retrieved-at", required=True)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    # Reuse the coefficient config validator so extraction cannot bypass governance.
    env.load_config(args.config)
    result = build_live_source_input(
        read_csv(args.cohort_input), read_csv(args.economic_source_input),
        config=config, ons_workbook=args.ons_ghg_workbook, retrieved_at=args.retrieved_at,
    )
    write_csv(result["source_rows"], args.source_output, SOURCE_INPUT_FIELDS)
    args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_output.write_text(json.dumps(result["diagnostics"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
