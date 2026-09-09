#!/usr/bin/env python3
"""Build a governed SKO-039 GHG satellite from Eurostat env_ac_ghgfp.

Eurostat metadata states that the direct emissions input used in the footprint
model can be reconstructed by selecting, for each year, country of origin and
NACE activity, totals for the remaining dimensions. This module implements that
rule for a long-form export of env_ac_ghgfp and maps Eurostat rest-of-world to
FIGARO's FIGW1 node.

Expected long-form columns (case-insensitive aliases accepted):
  c_orig, nace_r2, c_dest, na_item, freq, unit, time_period, obs_value

Primary v1 selection:
  time_period = 2023
  c_dest = WORLD
  na_item = TOTAL
  freq = A
  unit = THS_T

Output contract:
  country, sector, outcome, value, unit
where outcome=GHG, value is tonnes CO2e, and unit=tCO2e.

Missing FIGARO country x sector cells are reported, never imputed as zero.
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
DEST_TOTAL = "WORLD"
NA_TOTAL = "TOTAL"
FREQ = "A"
UNIT = "THS_T"
REST_ALIASES = {"WRL_REST", "WLR_REST", "REST_WORLD", "ROW"}
EXCLUDED_NACE = {"TOTAL", "TOTAL_HH", "HH"}

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


def map_origin(code: str) -> str:
    value = (code or "").strip().upper()
    if value in REST_ALIASES:
        return "FIGW1"
    return value


def load_figaro_nodes(outputs_path: Path) -> list[tuple[str, str]]:
    with outputs_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"country", "sector", "output_million_eur"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("FIGARO outputs file missing required columns")
        nodes = []
        seen = set()
        for row in reader:
            node = ((row["country"] or "").strip().upper(), (row["sector"] or "").strip().upper())
            if not all(node):
                raise ValueError("blank FIGARO output node")
            if node in seen:
                raise ValueError(f"duplicate FIGARO output node: {node}")
            seen.add(node)
            nodes.append(node)
    if not nodes:
        raise ValueError("FIGARO outputs file is empty")
    return nodes


def build_ghg_satellite(
    source_path: Path,
    outputs_path: Path,
    satellite_output: Path,
    diagnostics_output: Path | None = None,
    *,
    year: str = YEAR,
) -> dict:
    figaro_nodes = load_figaro_nodes(outputs_path)
    figaro_set = set(figaro_nodes)

    selected: dict[tuple[str, str], float] = {}
    source_origin_codes = Counter()
    source_rows = 0
    selected_rows = 0

    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("env_ac_ghgfp source has no header")
        h = _canon_header(reader.fieldnames)
        for line_no, row in enumerate(reader, start=2):
            source_rows += 1
            if (row[h["time_period"]] or "").strip() != str(year):
                continue
            if (row[h["c_dest"]] or "").strip().upper() != DEST_TOTAL:
                continue
            if (row[h["na_item"]] or "").strip().upper() != NA_TOTAL:
                continue
            if (row[h["freq"]] or "").strip().upper() != FREQ:
                continue
            if (row[h["unit"]] or "").strip().upper() != UNIT:
                continue

            nace = (row[h["nace_r2"]] or "").strip().upper()
            if not nace or nace in EXCLUDED_NACE:
                continue
            origin_raw = (row[h["c_orig"]] or "").strip().upper()
            origin = map_origin(origin_raw)
            source_origin_codes[origin_raw] += 1
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
            "reconstruction_rule": "TIME=2023,C_DEST=WORLD,NA_ITEM=TOTAL,FREQ=A,UNIT=THS_T by C_ORIG x NACE_R2",
            "native_unit": "thousand_tonnes_co2e",
            "normalized_unit": "tCO2e",
            "scale_factor": 1000,
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
        },
        "mapping": {
            "rest_of_world_aliases": sorted(REST_ALIASES),
            "rest_of_world_figaro_code": "FIGW1",
            "source_origin_codes": sorted(source_origin_codes),
        },
        "missing_value_policy": "preserve_missing_not_zero",
        "pilot_use_rule": "downstream outcome must hold out if any materially active upstream FIGARO node lacks GHG intensity",
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
    result = build_ghg_satellite(
        args.input,
        args.figaro_outputs,
        args.output,
        args.diagnostics_output,
        year=args.year,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
