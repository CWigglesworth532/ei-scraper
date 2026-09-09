#!/usr/bin/env python3
"""SKO-036 targeted ONS 2023 extraction for UK SIC 72 / model sector M72.

Reads locally downloaded ONS Supply and Use Tables and BRES workbooks using only
Python's standard library. The initial governed scope is deliberately narrow:
UK SIC 72 / M72 for the 2023 pilot. SUT supplies P1, P2 and B1G; BRES supplies
EMP_PERSONS as separate business-statistics evidence. The coefficient engine
continues to use the governed national-accounts denominator route, so BRES
employment is preserved in the source layer without being silently treated as
national-accounts employment.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

MODEL_CLASSIFICATION = "NACE"
MODEL_CLASSIFICATION_VERSION = "Rev. 2 A*64"
MODEL_SECTOR_CODE = "M72"
MODEL_SECTOR_LABEL = "Scientific research and development"
REFERENCE_YEAR = "2023"

SOURCE_FIELDS = [
    "source_family", "source_organisation", "source_dataset_id", "source_release_version",
    "source_release_date", "retrieved_at", "source_url", "licence", "country",
    "source_classification", "source_classification_version", "source_sector_code",
    "source_sector_label", "model_classification", "model_classification_version",
    "model_sector_code", "model_sector_label", "reference_year", "concept_code",
    "concept_label", "value", "normalized_unit", "currency", "price_basis", "status_flag",
]

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"


def _shared_strings(z: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    return ["".join(t.text or "" for t in si.iter(f"{{{NS}}}t")) for si in root.findall(f"{{{NS}}}si")]


def _sheet_path(z: ZipFile, wanted: str) -> str:
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    relmap = {r.attrib["Id"]: r.attrib["Target"] for r in rels.findall(f"{{{PKG}}}Relationship")}
    sheets = wb.find(f"{{{NS}}}sheets")
    if sheets is None:
        raise ValueError("Workbook contains no sheets")
    for sheet in sheets:
        if sheet.attrib.get("name") == wanted:
            target = relmap[sheet.attrib[f"{{{REL}}}id"]]
            return target if target.startswith("xl/") else "xl/" + target.lstrip("/")
    raise ValueError(f"Workbook sheet not found: {wanted}")


def _colnum(ref: str) -> int:
    m = re.match(r"([A-Z]+)", ref or "")
    if not m:
        return 0
    n = 0
    for ch in m.group(1):
        n = n * 26 + ord(ch) - 64
    return n


def _rows(z: ZipFile, sheet_path: str, shared: list[str]):
    root = ET.fromstring(z.read(sheet_path))
    for row in root.iter(f"{{{NS}}}row"):
        values: dict[int, str] = {}
        for cell in row.findall(f"{{{NS}}}c"):
            ref = cell.attrib.get("r", "")
            ctype = cell.attrib.get("t")
            value_node = cell.find(f"{{{NS}}}v")
            if ctype == "inlineStr":
                value = "".join(x.text or "" for x in cell.iter(f"{{{NS}}}t"))
            elif value_node is None:
                value = ""
            elif ctype == "s":
                value = shared[int(value_node.text or "0")]
            else:
                value = value_node.text or ""
            values[_colnum(ref)] = value
        yield int(row.attrib.get("r", "0")), values


def extract_sut_m72(path: Path, retrieved_at: str) -> list[dict[str, str]]:
    with ZipFile(path) as z:
        shared = _shared_strings(z)
        rows = list(_rows(z, _sheet_path(z, "Table 2"), shared))
    header = next((vals for rn, vals in rows if rn == 3), None)
    if header is None:
        raise ValueError("ONS SUT Table 2 header row 3 missing")
    year_col = next((col for col, value in header.items() if str(value).strip() == REFERENCE_YEAR), None)
    if year_col is None:
        raise ValueError("ONS SUT Table 2 has no 2023 column")

    wanted = {
        "P1": ("Domestic output", "basic_prices"),
        "P2": ("Intermediate consumption", "purchasers_prices"),
        "GVA": ("Gross value added", "basic_prices"),
    }
    found: dict[str, str] = {}
    industry_label = ""
    for _, vals in rows:
        price = str(vals.get(1, "")).strip()
        component = str(vals.get(2, "")).strip()
        sic = str(vals.get(3, "")).strip()
        if price == "CP" and sic == MODEL_SECTOR_CODE and component in wanted:
            found[component] = str(vals.get(year_col, "")).strip()
            industry_label = str(vals.get(4, "")).strip()
    missing = set(wanted) - set(found)
    if missing:
        raise ValueError(f"ONS SUT M72 missing 2023 components: {sorted(missing)}")

    # SUT Table 2 should satisfy P1 = P2 + GVA exactly at published £m precision here.
    if float(found["P1"]) != float(found["P2"]) + float(found["GVA"]):
        raise ValueError("ONS SUT M72 accounting identity P1=P2+GVA failed")

    result = []
    for raw_component, (label, price_basis) in wanted.items():
        concept = "B1G" if raw_component == "GVA" else raw_component
        result.append({
            "source_family": "national_accounts",
            "source_organisation": "Office for National Statistics",
            "source_dataset_id": "input-output-supply-and-use-tables",
            "source_release_version": path.name,
            "source_release_date": "",
            "retrieved_at": retrieved_at,
            "source_url": str(path),
            "licence": "Open Government Licence",
            "country": "UK",
            "source_classification": "UK SIC 2007",
            "source_classification_version": "SIC 2007",
            "source_sector_code": MODEL_SECTOR_CODE,
            "source_sector_label": industry_label or MODEL_SECTOR_LABEL,
            "model_classification": MODEL_CLASSIFICATION,
            "model_classification_version": MODEL_CLASSIFICATION_VERSION,
            "model_sector_code": MODEL_SECTOR_CODE,
            "model_sector_label": MODEL_SECTOR_LABEL,
            "reference_year": REFERENCE_YEAR,
            "concept_code": concept,
            "concept_label": label,
            "value": found[raw_component],
            "normalized_unit": "million_currency",
            "currency": "GBP",
            "price_basis": price_basis,
            "status_flag": "",
        })
    return result


def extract_bres_m72(path: Path, retrieved_at: str) -> list[dict[str, str]]:
    with ZipFile(path) as z:
        shared = _shared_strings(z)
        rows = list(_rows(z, _sheet_path(z, "Table 2b UK"), shared))
    target = None
    for _, vals in rows:
        sic2 = str(vals.get(1, "")).strip()
        sic3 = str(vals.get(2, "")).strip()
        if sic2 == "72" and not sic3:
            target = vals
            break
    if target is None:
        raise ValueError("BRES Table 2b UK has no SIC 72 total row")
    employment_thousand = str(target.get(14, "")).strip()
    if not employment_thousand:
        raise ValueError("BRES SIC 72 total employment missing")
    persons = float(employment_thousand) * 1000
    if persons <= 0:
        raise ValueError("BRES SIC 72 total employment non-positive")
    return [{
        "source_family": "business_statistics",
        "source_organisation": "Office for National Statistics",
        "source_dataset_id": "business-register-and-employment-survey",
        "source_release_version": path.name,
        "source_release_date": "",
        "retrieved_at": retrieved_at,
        "source_url": str(path),
        "licence": "Open Government Licence",
        "country": "UK",
        "source_classification": "UK SIC 2007",
        "source_classification_version": "SIC 2007",
        "source_sector_code": "72",
        "source_sector_label": MODEL_SECTOR_LABEL,
        "model_classification": MODEL_CLASSIFICATION,
        "model_classification_version": MODEL_CLASSIFICATION_VERSION,
        "model_sector_code": MODEL_SECTOR_CODE,
        "model_sector_label": MODEL_SECTOR_LABEL,
        "reference_year": REFERENCE_YEAR,
        "concept_code": "EMP_PERSONS",
        "concept_label": "Total employment",
        "value": format(persons, ".15g"),
        "normalized_unit": "persons",
        "currency": "",
        "price_basis": "not_applicable",
        "status_flag": "",
    }]


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sut-input", type=Path, required=True)
    parser.add_argument("--bres-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    retrieved_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    rows = extract_sut_m72(args.sut_input, retrieved_at) + extract_bres_m72(args.bres_input, retrieved_at)
    rows.sort(key=lambda r: (r["source_family"], r["concept_code"]))
    write_csv(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
