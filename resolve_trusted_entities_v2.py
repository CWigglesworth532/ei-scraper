#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Resolve trusted roster entries to a stable entity_id in ei_registers_normalized.

This version is tailored to your ei_registers_normalized.csv, which currently has:
- country
- entity_name / entity_name_norm
- tax_id / tax_id_root
- ei_register_name + ei_registration_number
(and typically no precomputed entity_id, and no official website field)

What it does
------------
1) Creates a deterministic entity_id for each master row (if missing), using:
   - country + tax_id_root (preferred)
   - else country + (ei_register_name + ei_registration_number)
   - else country + entity_name_norm
   (hashed to md5 hex for compactness)

2) Resolves each trusted row (raw_brand_name + country_hint) to the best master row using
   a fast blocking strategy:
   - blocks candidates by a 4-char alnum prefix of the normalized name
   - if too small, also blocks by one or two "salient tokens" in the name
   - then applies RapidFuzz WRatio within the candidate set

Outputs
-------
- trusted_entities_resolved.csv
    Adds: entity_id, match_method, match_score, candidate_count, needs_review
- trusted_entities_review_queue.csv
    Rows needing human review, with top candidates.

Run
---
python resolve_trusted_entities_v2.py \
  --trusted trusted_entities_prioritized.csv \
  --master ei_registers_normalized.csv \
  --out trusted_entities_resolved.csv \
  --review trusted_entities_review_queue.csv

Tips
----
- If you get lots of 'too_many_candidates', add website_hint / tax_id hints into the trusted roster,
  or increase blocking strictness (see flags below).
"""

import argparse
import hashlib
import re
import csv

import pandas as pd

try:
    from rapidfuzz import fuzz, process
except Exception as e:
    raise SystemExit(
        "Missing dependency 'rapidfuzz'. Install with: pip install rapidfuzz\n"
        f"Original error: {e}"
    )

STOP_TOKENS = {
    "grupo","group","groupe","company","cooperative","coop","foundation","association",
    "societe","société","services","service","hotel","hotels","the",
    "sociedad","fundacion","fundación","associacion","asociación",
}

def read_csv_robust(path: str, required_cols=None) -> pd.DataFrame:
    """
    Robust CSV reader that:
    - tries common encodings
    - tries common separators deterministically
    - only accepts a parse that contains required columns (if provided)
    """
    if required_cols is None:
        required_cols = []

    encodings = ["utf-8-sig", "utf-16", "cp1252", "latin1"]
    seps = [";", ",", "\t", "|"]

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

                cols_lower = {c.lower(): c for c in df.columns}

                # If required cols specified, validate
                if required_cols:
                    if all(rc.lower() in cols_lower for rc in required_cols):
                        return df
                    else:
                        # Wrong separator/parse; try next
                        continue

                # If no required cols, accept any multi-column parse
                if df.shape[1] > 1:
                    return df

            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(f"Could not read CSV robustly: {path}. Last error: {last_err}")

def normalize_name(s: str) -> str:
    s = str(s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s&-]", "", s)
    return s

def alnum_prefix(s: str, n: int = 4) -> str:
    s = re.sub(r"[^a-z0-9]", "", str(s or "").lower())
    return s[:n]

def choose_tokens(name: str, k: int = 2):
    toks = re.findall(r"[a-z0-9]+", name.lower())
    toks = [t for t in toks if len(t) >= 4 and t not in STOP_TOKENS]
    toks = sorted(set(toks), key=lambda x: (-len(x), x))
    return toks[:k]

def make_entity_id(country: str, tax_id_root: str, tax_id: str, reg_name: str, reg_num: str, name_norm: str) -> str:
    c = str(country or "").upper().strip()
    tax = str(tax_id_root or tax_id or "").strip()
    regn = str(reg_name or "").strip()
    reg = str(reg_num or "").strip()
    nm = str(name_norm or "").strip()
    if tax and tax.lower() != "nan":
        base = f"{c}|TAX|{tax}"
    elif reg and reg.lower() != "nan":
        base = f"{c}|REG|{regn}|{reg}"
    else:
        base = f"{c}|NAME|{nm}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trusted", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--out", default="trusted_entities_resolved.csv")
    ap.add_argument("--review", default="trusted_entities_review_queue.csv")
    ap.add_argument("--fuzzy-threshold", type=int, default=92)
    ap.add_argument("--prefix-len", type=int, default=4)
    ap.add_argument("--max-candidates", type=int, default=200000)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--token-k", type=int, default=2)

    args = ap.parse_args()

    # READ FILES
    trusted = read_csv_robust(args.trusted, required_cols=["raw_brand_name", "country_hint"])
    master = pd.read_csv(args.master, low_memory=False)

    # Master name column preference
    name_col = "entity_name_norm" if "entity_name_norm" in master.columns else "entity_name"
    if name_col not in master.columns:
        raise SystemExit("Master missing entity_name_norm/entity_name columns.")

    # Ensure required cols exist
    for c in ["country","tax_id_root","tax_id","ei_register_name","ei_registration_number"]:
        if c not in master.columns:
            master[c] = ""

    # Create / ensure entity_id
    if "entity_id" not in master.columns:
        master["entity_id"] = ""
    # Fill missing entity_id values deterministically
    mask_missing = master["entity_id"].astype(str).str.strip().eq("") | master["entity_id"].isna()
    if mask_missing.any():
        m = master.loc[mask_missing, ["country","tax_id_root","tax_id","ei_register_name","ei_registration_number", name_col]].copy()
        master.loc[mask_missing, "entity_id"] = [
            make_entity_id(r["country"], r["tax_id_root"], r["tax_id"], r["ei_register_name"], r["ei_registration_number"], r[name_col])
            for _, r in m.iterrows()
        ]

    master["__mname"] = master[name_col].astype(str).fillna("").map(normalize_name)
    master["__mprefix"] = master["__mname"].map(lambda s: alnum_prefix(s, args.prefix_len))

    # Trusted normalization
    for c in ["raw_brand_name","country_hint"]:
        if c not in trusted.columns:
            raise SystemExit("Trusted roster must include raw_brand_name and country_hint.")
    trusted["__tname"] = trusted["raw_brand_name"].astype(str).fillna("").map(normalize_name)
    trusted["__tcountry"] = trusted["country_hint"].astype(str).fillna("").str.upper().str.strip()
    trusted["__tprefix"] = trusted["__tname"].map(lambda s: alnum_prefix(s, args.prefix_len))

    # Restrict master to countries present in trusted roster for speed
    countries = sorted(set(trusted["__tcountry"]) - {""})
    master = master[master["country"].astype(str).str.upper().isin(countries)].copy()

    # Pre-split by country
    master_by_country = {cty: df for cty, df in master.groupby(master["country"].astype(str).str.upper())}

    out = trusted.copy()
    out["entity_id"] = ""
    out["match_method"] = ""
    out["match_score"] = ""
    out["candidate_count"] = ""
    out["needs_review"] = True

    review_rows = []

    for i, row in out.iterrows():
        cty = row["__tcountry"]
        tname = row["__tname"]
        tprefix = row["__tprefix"]
        if not cty or not tname:
            out.at[i, "match_method"] = "missing_country_or_name"
            continue

        mcty = master_by_country.get(cty)
        if mcty is None or mcty.empty:
            out.at[i, "match_method"] = "no_master_for_country"
            continue

        # Primary block: prefix
        cand = mcty[mcty["__mprefix"] == tprefix] if tprefix else mcty.head(0)

        # If too small, broaden with salient tokens
        if len(cand) < 50:
            toks = choose_tokens(tname, k=args.token_k)
            c2 = mcty
            for tok in toks:
                c2 = c2[c2["__mname"].str.contains(re.escape(tok), na=False)]
                if len(c2) < args.max_candidates:
                    break
            cand = pd.concat([cand, c2]).drop_duplicates()

        cand_count = len(cand)
        out.at[i, "candidate_count"] = cand_count

        if cand_count == 0:
            out.at[i, "match_method"] = "no_candidates"
            continue

        if cand_count > args.max_candidates:
            out.at[i, "match_method"] = "too_many_candidates"
            review_rows.append({
                "raw_brand_name": row["raw_brand_name"],
                "country_hint": cty,
                "notes": f"Too many candidates ({cand_count}). Consider adding tax_id/registration hints or more specific tokens.",
            })
            continue

        names = cand["__mname"].tolist()
        best = process.extractOne(tname, names, scorer=fuzz.WRatio)
        if not best:
            out.at[i, "match_method"] = "no_match"
            continue

        _, score, idx = best
        out.at[i, "match_score"] = score

        if score >= args.fuzzy_threshold:
            out.at[i, "entity_id"] = str(cand.iloc[idx]["entity_id"])
            out.at[i, "match_method"] = "fuzzy_blocked"
            out.at[i, "needs_review"] = False
        else:
            out.at[i, "match_method"] = "below_threshold"
            top = process.extract(tname, names, scorer=fuzz.WRatio, limit=args.topk)
            cand_str = []
            for _, score2, idx2 in top:
                r2 = cand.iloc[idx2]
                cand_str.append(f"{r2['entity_id']}|{r2.get('entity_name','')}|{score2}")
            review_rows.append({
                "raw_brand_name": row["raw_brand_name"],
                "country_hint": cty,
                "best_score": score,
                "top_candidates": "; ".join(cand_str),
            })

    out = out.drop(columns=["__tname","__tcountry","__tprefix"], errors="ignore")
    out.to_csv(args.out, index=False)
    pd.DataFrame(review_rows).to_csv(args.review, index=False)

    resolved = (out["needs_review"] == False).sum()
    total = len(out)
    print(f"Resolved {resolved}/{total} trusted rows ({resolved/total:.1%}).")
    print("Wrote:", args.out)
    print("Review queue:", args.review)

if __name__ == "__main__":
    main()
