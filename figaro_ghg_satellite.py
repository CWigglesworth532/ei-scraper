#!/usr/bin/env python3
"""Build a governed SKO-039 GHG satellite from Eurostat env_ac_ghgfp.

Accepts Eurostat code-form or Data Browser label-form long CSV exports.
For each 2023 origin-country x NACE activity, selects total destination and
national-accounts dimensions, converts thousand tonnes to tonnes CO2e, and
aligns the environmental Rest-of-World geography to FIGARO 2026.

Eurostat env_ac_ghgfp has fewer explicit origin geographies than FIGARO 2026.
For GHG only, Albania (AL), Montenegro (ME), North Macedonia (MK) and Serbia
(RS) are members of the environmental WRL_REST geography. For each sector,
the WRL_REST emissions total is allocated across FIGW1 + AL + ME + MK + RS in
proportion to their FIGARO output. This preserves the source emissions total
and gives every member of the source geography a common sector intensity.
Missing source geographies outside this governed bridge remain missing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

YEAR = "2023"
REST_CODE_ALIASES = {"WRL_REST", "WLR_REST", "REST_WORLD", "ROW"}
ROW_GROUP_COUNTRIES = ("FIGW1", "AL", "ME", "MK", "RS")
EXCLUDED_NACE_CODES = {"TOTAL", "TOTAL_HH", "HH", "G-U_X_H"}
DEST_TOTAL_VALUES = {"WORLD", "ALL COUNTRIES OF THE WORLD"}
NA_TOTAL_VALUES = {"TOTAL"}
FREQ_VALUES = {"A", "ANNUAL"}
UNIT_VALUES = {"THS_T", "THOUSAND TONNES"}

ORIGIN_LABEL_TO_CODE = {'Argentina': 'AR', 'Austria': 'AT', 'Australia': 'AU', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Brazil': 'BR', 'Canada': 'CA', 'Switzerland': 'CH', 'China': 'CN', 'Cyprus': 'CY', 'Czechia': 'CZ', 'Germany': 'DE', 'Denmark': 'DK', 'Estonia': 'EE', 'Greece': 'GR', 'Spain': 'ES', 'Finland': 'FI', 'France': 'FR', 'Croatia': 'HR', 'Hungary': 'HU', 'Indonesia': 'ID', 'Ireland': 'IE', 'India': 'IN', 'Italy': 'IT', 'Japan': 'JP', 'South Korea': 'KR', 'Lithuania': 'LT', 'Luxembourg': 'LU', 'Latvia': 'LV', 'Malta': 'MT', 'Mexico': 'MX', 'Netherlands': 'NL', 'Norway': 'NO', 'Poland': 'PL', 'Portugal': 'PT', 'Romania': 'RO', 'Russia': 'RU', 'Saudi Arabia': 'SA', 'Sweden': 'SE', 'Slovenia': 'SI', 'Slovakia': 'SK', 'Türkiye': 'TR', 'United Kingdom': 'GB', 'United States': 'US', 'Rest of the world': 'FIGW1', 'South Africa': 'ZA', 'European Union - 27 countries (from 2020)': None, 'Extra-EU27 (from 2020)': None, 'All countries of the world': None}
NACE_LABEL_TO_CODE = {'Agriculture, forestry and fishing': None, 'Crop and animal production, hunting and related service activities': 'A01', 'Forestry and logging': 'A02', 'Fishing and aquaculture': 'A03', 'Mining and quarrying': 'B', 'Manufacturing': None, 'Manufacture of food products; beverages and tobacco products': 'C10T12', 'Manufacture of textiles, wearing apparel, leather and related products': 'C13T15', 'Manufacture of wood and of products of wood and cork, except furniture; manufacture of articles of straw and plaiting materials': 'C16', 'Manufacture of paper and paper products': 'C17', 'Printing and reproduction of recorded media': 'C18', 'Manufacture of coke and refined petroleum products': 'C19', 'Manufacture of chemicals and chemical products': 'C20', 'Manufacture of basic pharmaceutical products and pharmaceutical preparations': 'C21', 'Manufacture of rubber and plastic products': 'C22', 'Manufacture of other non-metallic mineral products': 'C23', 'Manufacture of basic metals': 'C24', 'Manufacture of fabricated metal products, except machinery and equipment': 'C25', 'Manufacture of computer, electronic and optical products': 'C26', 'Manufacture of electrical equipment': 'C27', 'Manufacture of machinery and equipment n.e.c.': 'C28', 'Manufacture of motor vehicles, trailers and semi-trailers': 'C29', 'Manufacture of other transport equipment': 'C30', 'Manufacture of furniture; other manufacturing': 'C31_32', 'Repair and installation of machinery and equipment': 'C33', 'Electricity, gas, steam and air conditioning supply': 'D35', 'Water supply; sewerage, waste management and remediation activities': None, 'Water collection, treatment and supply': 'E36', 'Sewerage, waste management, remediation activities': 'E37T39', 'Construction': 'F', 'Wholesale and retail trade; repair of motor vehicles and motorcycles': None, 'Services (except transportation and storage)': None, 'Wholesale and retail trade and repair of motor vehicles and motorcycles': 'G45', 'Wholesale trade, except of motor vehicles and motorcycles': 'G46', 'Retail trade, except of motor vehicles and motorcycles': 'G47', 'Transportation and storage': None, 'Land transport and transport via pipelines': 'H49', 'Water transport': 'H50', 'Air transport': 'H51', 'Warehousing and support activities for transportation': 'H52', 'Postal and courier activities': 'H53', 'Total activities by households': None, 'Accommodation and food service activities': 'I', 'Information and communication': None, 'Publishing activities': 'J58', 'Motion picture, video, television programme production; programming and broadcasting activities': 'J59_60', 'Telecommunications': 'J61', 'Computer programming, consultancy, and information service activities': 'J62_63', 'Financial and insurance activities': None, 'Financial service activities, except insurance and pension funding': 'K64', 'Insurance, reinsurance and pension funding, except compulsory social security': 'K65', 'Activities auxiliary to financial services and insurance activities': 'K66', 'Real estate activities': 'L', 'Professional, scientific and technical activities': None, 'Legal and accounting activities; activities of head offices; management consultancy activities': 'M69_70', 'Architectural and engineering activities; technical testing and analysis': 'M71', 'Scientific research and development': 'M72', 'Advertising and market research': 'M73', 'Other professional, scientific and technical activities; veterinary activities': 'M74_75', 'Administrative and support service activities': None, 'Rental and leasing activities': 'N77', 'Employment activities': 'N78', 'Travel agency, tour operator and other reservation service and related activities': 'N79', 'Security and investigation, service and landscape, office administrative and support activities': 'N80T82', 'Public administration and defence; compulsory social security': 'O84', 'Education': 'P85', 'Human health and social work activities': None, 'Human health activities': 'Q86', 'Residential care activities and social work activities without accommodation': 'Q87_88', 'Arts, entertainment and recreation': None, 'Creative, arts and entertainment activities; libraries, archives, museums and other cultural activities; gambling and betting activities': 'R90T92', 'Sports activities and amusement and recreation activities': 'R93', 'Other service activities': None, 'Activities of membership organisations': 'S94', 'Repair of computers and personal and household goods': 'S95', 'Other personal service activities': 'S96', 'Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use': 'T', 'Total - all NACE activities': None, 'All NACE activities plus households': None, 'Activities of extraterritorial organisations and bodies': 'U'}

ALIASES = {
    "c_orig": {"c_orig", "origin", "origin_country", "geo_orig"},
    "nace_r2": {"nace_r2", "nace", "activity"},
    "c_dest": {"c_dest", "destination", "destination_country", "geo_dest"},
    "na_item": {"na_item", "final_demand", "national_accounts_item"},
    "freq": {"freq", "frequency"},
    "unit": {"unit"},
    "time_period": {"time_period", "time", "year"},
    "obs_value": {"obs_value", "value", "obsvalue"},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canon_header(fieldnames: Iterable[str]) -> dict[str, str]:
    normalized = {str(name).strip().lower(): str(name) for name in fieldnames if name is not None}
    result = {}
    for canonical, aliases in ALIASES.items():
        matches = [normalized[a] for a in aliases if a in normalized]
        if not matches:
            raise ValueError(f"missing required env_ac_ghgfp field: {canonical}")
        result[canonical] = matches[0]
    return result


def _f(value: str, context: str) -> float:
    text = (value or "").strip()
    if text == "":
        raise ValueError(f"blank numeric value at {context}")
    try:
        x = float(text)
    except ValueError as exc:
        raise ValueError(f"invalid numeric value at {context}: {value!r}") from exc
    if not math.isfinite(x):
        raise ValueError(f"non-finite numeric value at {context}")
    return x


def _norm(value: str) -> str:
    return " ".join((value or "").strip().split()).upper()


def map_origin(value: str) -> tuple[str | None, str]:
    raw = " ".join((value or "").strip().split())
    upper = raw.upper()
    if upper in REST_CODE_ALIASES:
        return "FIGW1", "code_alias"
    if raw in ORIGIN_LABEL_TO_CODE:
        code = ORIGIN_LABEL_TO_CODE[raw]
        return code, "label_map" if code else "aggregate_excluded"
    if upper == "FIGW1" or (len(upper) == 2 and upper.isalpha()):
        return upper, "code"
    raise ValueError(f"unmapped Eurostat origin label/code: {value!r}")


def map_nace(value: str) -> tuple[str | None, str]:
    raw = " ".join((value or "").strip().split())
    upper = raw.upper()
    if upper in EXCLUDED_NACE_CODES:
        return None, "aggregate_excluded"
    if raw in NACE_LABEL_TO_CODE:
        code = NACE_LABEL_TO_CODE[raw]
        return code, "label_map" if code else "aggregate_excluded"
    if upper and upper[0].isalpha():
        return upper, "code"
    raise ValueError(f"unmapped Eurostat NACE label/code: {value!r}")


def load_figaro_outputs(outputs_path: Path) -> dict[tuple[str, str], float]:
    with outputs_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"country", "sector", "output_million_eur"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("FIGARO outputs file missing required columns")
        outputs: dict[tuple[str, str], float] = {}
        for row in reader:
            node = ((row["country"] or "").strip().upper(), (row["sector"] or "").strip().upper())
            if not all(node):
                raise ValueError("blank FIGARO output node")
            if node in outputs:
                raise ValueError(f"duplicate FIGARO output node: {node}")
            value = _f(row["output_million_eur"], f"FIGARO output {node}")
            if value <= 0:
                raise ValueError(f"FIGARO output must be positive for {node}")
            outputs[node] = value
    if not outputs:
        raise ValueError("FIGARO outputs file is empty")
    return outputs


def load_figaro_nodes(outputs_path: Path) -> list[tuple[str, str]]:
    return list(load_figaro_outputs(outputs_path))


def align_rest_of_world(
    selected: dict[tuple[str, str], float],
    figaro_outputs: dict[tuple[str, str], float],
) -> dict:
    """Align env_ac_ghgfp WRL_REST to FIGARO 2026 geography by sector output."""
    sectors = sorted(sector for country, sector in selected if country == "FIGW1")
    sector_records = []
    max_abs_difference = 0.0
    source_total_all = 0.0
    allocated_total_all = 0.0

    for sector in sectors:
        source_key = ("FIGW1", sector)
        source_total = selected[source_key]
        members = [(country, sector) for country in ROW_GROUP_COUNTRIES if (country, sector) in figaro_outputs]
        if not members:
            continue
        output_total = sum(figaro_outputs[node] for node in members)
        if output_total <= 0:
            raise ValueError(f"non-positive FIGARO ROW-group output for sector {sector}")

        allocations = {}
        for node in members:
            allocations[node] = source_total * figaro_outputs[node] / output_total

        allocated_total = sum(allocations.values())
        difference = allocated_total - source_total
        max_abs_difference = max(max_abs_difference, abs(difference))
        tolerance = max(1e-9, abs(source_total) * 1e-12)
        if abs(difference) > tolerance:
            raise ValueError(f"ROW geography allocation failed emissions conservation for {sector}: {difference}")

        del selected[source_key]
        for node, value in allocations.items():
            selected[node] = value

        source_total_all += source_total
        allocated_total_all += allocated_total
        sector_records.append({
            "sector": sector,
            "source_emissions_tco2e": source_total,
            "allocated_emissions_tco2e": allocated_total,
            "allocation_difference_tco2e": difference,
            "member_nodes": [f"{c}_{s}" for c, s in members],
            "combined_output_million_eur": output_total,
            "common_intensity_tco2e_per_million_eur": source_total / output_total,
        })

    return {
        "mode": "WRL_REST_to_FIGARO2026_output_proportional_by_sector",
        "source_geography": "env_ac_ghgfp_WRL_REST",
        "figaro_member_countries": list(ROW_GROUP_COUNTRIES),
        "aligned_sectors": len(sector_records),
        "source_emissions_tco2e": source_total_all,
        "allocated_emissions_tco2e": allocated_total_all,
        "allocation_difference_tco2e": allocated_total_all - source_total_all,
        "max_abs_sector_allocation_difference_tco2e": max_abs_difference,
        "sector_records": sector_records,
        "interpretation": "source geography alignment, not country-specific emissions imputation",
    }


def build_ghg_satellite(source_path: Path, outputs_path: Path, satellite_output: Path, diagnostics_output: Path | None = None, *, year: str = YEAR) -> dict:
    figaro_outputs = load_figaro_outputs(outputs_path)
    figaro_nodes = list(figaro_outputs)
    figaro_set = set(figaro_nodes)
    selected: dict[tuple[str, str], float] = {}
    source_rows = selected_rows = excluded_origin_rows = excluded_nace_rows = 0
    origin_modes, nace_modes = Counter(), Counter()
    excluded_origin_labels, excluded_nace_labels = Counter(), Counter()

    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("env_ac_ghgfp source has no header")
        h = _canon_header(reader.fieldnames)
        for line_no, row in enumerate(reader, start=2):
            source_rows += 1
            if str(row[h["time_period"]]).strip() != str(year):
                continue
            if _norm(row[h["c_dest"]]) not in DEST_TOTAL_VALUES:
                continue
            if _norm(row[h["na_item"]]) not in NA_TOTAL_VALUES:
                continue
            if _norm(row[h["freq"]]) not in FREQ_VALUES:
                continue
            if _norm(row[h["unit"]]) not in UNIT_VALUES:
                continue

            origin, origin_mode = map_origin(row[h["c_orig"]])
            origin_modes[origin_mode] += 1
            if origin is None:
                excluded_origin_rows += 1
                excluded_origin_labels[" ".join((row[h["c_orig"]] or "").strip().split())] += 1
                continue

            nace, nace_mode = map_nace(row[h["nace_r2"]])
            nace_modes[nace_mode] += 1
            if nace is None:
                excluded_nace_rows += 1
                excluded_nace_labels[" ".join((row[h["nace_r2"]] or "").strip().split())] += 1
                continue

            key = (origin, nace)
            value_thousand_tonnes = _f(row[h["obs_value"]], f"line {line_no}")
            if value_thousand_tonnes < 0:
                raise ValueError(f"negative GHG value at line {line_no}")
            if key in selected:
                raise ValueError(f"duplicate selected env_ac_ghgfp cell: {key}")
            selected[key] = value_thousand_tonnes * 1000.0
            selected_rows += 1

    if not selected:
        raise ValueError("no env_ac_ghgfp rows matched governed 2023 total-selection rule")

    geography_alignment = align_rest_of_world(selected, figaro_outputs)

    covered = sorted(node for node in figaro_nodes if node in selected)
    missing = sorted(node for node in figaro_nodes if node not in selected)
    source_not_in_figaro = sorted(node for node in selected if node not in figaro_set)

    satellite_output.parent.mkdir(parents=True, exist_ok=True)
    with satellite_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["country", "sector", "outcome", "value", "unit"])
        for country, sector in covered:
            writer.writerow([country, sector, "GHG", format(selected[(country, sector)], ".15g"), "tCO2e"])

    diagnostics = {
        "status": "ghg_satellite_built",
        "source": {
            "dataset": "env_ac_ghgfp",
            "filename": source_path.name,
            "sha256": sha256_file(source_path),
            "reconstruction_rule": "2023,total destination,total national-accounts item,annual,thousand tonnes by origin x NACE",
            "native_unit": "thousand_tonnes_co2e",
            "normalized_unit": "tCO2e",
            "scale_factor": 1000,
            "accepted_export_forms": ["Eurostat_dimension_codes", "Eurostat_Data_Browser_labels"],
        },
        "coverage": {
            "figaro_nodes": len(figaro_nodes),
            "covered_nodes": len(covered),
            "missing_nodes": len(missing),
            "coverage_pct": 100.0 * len(covered) / len(figaro_nodes),
            "missing_node_labels": [f"{c}_{s}" for c, s in missing],
            "source_cells_not_in_figaro": [f"{c}_{s}" for c, s in source_not_in_figaro],
        },
        "rows": {
            "source_rows": source_rows,
            "selected_rows": selected_rows,
            "excluded_aggregate_origin_rows": excluded_origin_rows,
            "excluded_aggregate_nace_rows": excluded_nace_rows,
        },
        "translation": {
            "origin_modes": dict(sorted(origin_modes.items())),
            "nace_modes": dict(sorted(nace_modes.items())),
            "excluded_origin_labels": dict(sorted(excluded_origin_labels.items())),
            "excluded_nace_labels": dict(sorted(excluded_nace_labels.items())),
            "rest_of_world_figaro_code": "FIGW1",
        },
        "source_geography_alignment": geography_alignment,
        "missing_value_policy": "preserve_missing_not_zero_outside_governed_source_geography_alignment",
        "pilot_use_rule": "downstream outcome must hold out if any materially active upstream FIGARO node lacks GHG intensity after governed source-geography alignment",
    }
    if diagnostics_output:
        diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_output.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--figaro-outputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--diagnostics-output", type=Path)
    parser.add_argument("--year", default=YEAR)
    args = parser.parse_args()
    result = build_ghg_satellite(args.input, args.figaro_outputs, args.output, args.diagnostics_output, year=args.year)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
