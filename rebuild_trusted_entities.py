#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rebuild a clean trusted roster CSV (UTF-8, QUOTE_ALL) from an existing trusted roster,
optionally merging in your current Squarespace directory CSV.

Why:
- Fixes encoding / delimiter / quoting issues that break downstream parsing
- Produces a deterministic "clean" file for resolver + pipelines
- Avoids dependence on the old trusted_entities_seed.csv

Inputs:
- --trusted : your current trusted roster CSV (required)
- --directory : optional V22 directory CSV to merge in
- --out : output path (default: trusted_entities_clean.csv)

Behavior:
- If --directory is provided, it is appended and then deduped by a canonical name key.
- Dedupe preference: TRUSTED roster rows win over DIRECTORY rows.
- Within same source, keep the row with the most filled enrichment fields.
- Special-case: "Grupo Ilunion" and "Ilunion" treated as same key.

Run examples:
  python rebuild_trusted_entities.py --trusted trusted_entities.csv
  python rebuild_trusted_entities.py --trusted trusted_entities.csv --directory V22_With_...csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Robust reader for messy CSVs (encoding + delimiter fallback)."""
    encodings = ["utf-8-sig", "utf-16", "cp1252", "latin1"]
    seps = [",", ";", "\t", "|"]
    last_err = None

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    path,
                    encoding=enc,
                    sep=sep,
                    engine="python",
                    dtype=str,
                    on_bad_lines="warn",
                )
                # accept if we got more than 1 column
                if df.shape[1] > 1:
                    return df
            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(f"Could not read CSV: {path}. Last error: {last_err}")


def norm_key(name: str) -> str:
    """Canonical key for dedupe."""
    s = (name or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s&-]", "", s)

    # treat "grupo/group ilunion" as "ilunion"
    if re.fullmatch(r"(grupo|group)\s+ilunion", s):
        return "ilunion"
    return s


def filled_count(row: pd.Series, cols: list[str]) -> int:
    return sum(1 for c in cols if str(row.get(c, "") or "").strip() != "")


def main() -> None:
    base = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--trusted", required=True, help="Path to your current trusted roster CSV")
    ap.add_argument("--directory", default=None, help="Optional V22 directory CSV to merge in")
    ap.add_argument("--out", default=str(base / "trusted_entities_clean.csv"))
    args = ap.parse_args()

    trusted_path = Path(args.trusted)
    if not trusted_path.exists():
        raise FileNotFoundError(f"Trusted roster not found: {trusted_path}")

    print("Reading trusted roster:", trusted_path)
    trusted = read_csv_robust(trusted_path)

    # Normalize expected columns for trusted roster
    # If you use different headers, map them here
    col_map = {
        "Organisation": "raw_brand_name",
        "organization": "raw_brand_name",
        "organisation": "raw_brand_name",
        "name": "raw_brand_name",
        "Country": "country_hint",
        "Website": "website_hint",
    }
    trusted_cols_lower = {c.lower(): c for c in trusted.columns}
    for src, dst in col_map.items():
        # map case-insensitively
        src_actual = trusted_cols_lower.get(src.lower())
        if src_actual and dst not in trusted.columns:
            trusted[dst] = trusted[src_actual]

    # Ensure required base columns exist
    for c in ["raw_brand_name", "country_hint", "website_hint"]:
        if c not in trusted.columns:
            trusted[c] = ""

    # Mark source priority: TRUSTED wins
    trusted["source"] = trusted.get("source", "TRUSTED_ROSTER")
    trusted["__source_class"] = "TRUSTED"

    # Optionally merge in directory
    if args.directory:
        directory_path = Path(args.directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory file not found: {directory_path}")

        print("Reading directory:", directory_path)
        directory = read_csv_robust(directory_path)

        # Build rows from directory in your standard schema
        required_dir_cols = [
            "Organisation", "Country", "Website", "Business Summary",
            "Identified Clients", "Sector", "Clients Publicly Referenced (Yes/No)"
        ]
        missing = [c for c in required_dir_cols if c not in directory.columns]
        if missing:
            raise RuntimeError(f"Directory CSV missing columns: {missing}")

        dir_rows = pd.DataFrame({
            "raw_brand_name": directory["Organisation"].astype(str).fillna("").str.strip(),
            "country_hint": directory["Country"].astype(str).fillna("").str.strip(),
            "website_hint": directory["Website"].astype(str).fillna("").str.strip(),
            "trusted_reason": "Existing directory entry (V22 blog import)",
            "publish_status": "published",
            "source": "DIRECTORY_V22",
            "business_summary": directory["Business Summary"].astype(str).fillna(""),
            "identified_clients": directory["Identified Clients"].astype(str).fillna(""),
            "sector": directory["Sector"].astype(str).fillna(""),
            "clients_publicly_referenced": directory["Clients Publicly Referenced (Yes/No)"].astype(str).fillna(""),
        })
        dir_rows = dir_rows[dir_rows["raw_brand_name"].str.len() > 0].copy()
        dir_rows["__source_class"] = "DIRECTORY"

        # Align schemas (add missing columns to each)
        for col in dir_rows.columns:
            if col not in trusted.columns:
                trusted[col] = ""
        for col in trusted.columns:
            if col not in dir_rows.columns:
                dir_rows[col] = ""

        combined = pd.concat([trusted, dir_rows[trusted.columns]], ignore_index=True)
    else:
        combined = trusted.copy()

    # Dedupe preference:
    # - TRUSTED beats DIRECTORY
    # - within same source, keep most filled enrichment fields
    enrichment_cols = [
        "website_hint", "business_summary", "identified_clients", "sector",
        "clients_publicly_referenced", "trusted_reason"
    ]
    for c in enrichment_cols:
        if c not in combined.columns:
            combined[c] = ""

    combined["__key"] = combined["raw_brand_name"].astype(str).map(norm_key)
    combined["__pri"] = combined["__source_class"].map(lambda x: 2 if x == "TRUSTED" else 1)
    combined["__filled"] = combined.apply(lambda r: filled_count(r, enrichment_cols), axis=1)

    combined = combined.sort_values(
        by=["__key", "__pri", "__filled"],
        ascending=[True, False, False],
    ).drop_duplicates(subset="__key", keep="first")

    combined = combined.drop(columns=["__key", "__pri", "__filled"], errors="ignore")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write clean CSV (always parseable)
    combined.to_csv(out_path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    print("Wrote:", out_path)
    print("Rows:", len(combined))


if __name__ == "__main__":
    main()
