#!/usr/bin/env python3
"""Combine normalized SKO-036 direct-economic source CSVs without altering rows."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import extract_eurostat_direct_economic as euro


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [dict(r) for r in csv.DictReader(handle)]
    if rows and set(rows[0]) != set(euro.SOURCE_FIELDS):
        raise ValueError(f"Unexpected source schema in {path}")
    return rows


def combine(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(read_csv(path))
    rows.sort(key=lambda r: (
        r["country"], r["source_family"], r["source_dataset_id"],
        r["model_sector_code"], r["reference_year"], r["concept_code"],
    ))
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=euro.SOURCE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_csv(combine(args.input), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
