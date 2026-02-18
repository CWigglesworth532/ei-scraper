#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Resolve trusted entities to entity_id against your canonical master register.

- Input trusted roster: CSV (e.g., trusted_entities_prioritized.csv)
- Input master: CSV or Parquet (e.g., ei_registers_normalized)

Resolution strategy (precision-first):
1) Website domain exact match within country (best)
2) Fuzzy name match within country (RapidFuzz WRatio)

Outputs:
- trusted_entities_resolved.csv  (adds entity_id + match metadata)
- trusted_entities_review_queue.csv (rows needing human review with top candidates)

Example:
  python resolve_trusted_entities.py --trusted trusted_entities_prioritized.csv --master ei_registers_normalized.csv
"""

import argparse
import re
from urllib.parse import urlparse

import pandas as pd

try:
    from rapidfuzz import fuzz, process
except Exception as e:
    raise SystemExit(
        "Missing dependency 'rapidfuzz'. Install with: pip install rapidfuzz\n"
        f"Original error: {e}"
    )

COMMON_NAME_COLS = ["canonical_name", "name", "organisation", "organisation_name", "entity_name", "legal_name"]
COMMON_COUNTRY_COLS = ["country", "country_code", "jurisdiction", "iso_country", "country_hint"]
COMMON_WEBSITE_COLS = ["website", "url", "homepage", "web", "website_hint"]
COMMON_ID_COLS = ["entity_id", "id", "uuid", "org_id"]

def normalize_name(s: str) -> str:
    s = str(s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s&-]", "", s)
    return s

def extract_domain(url: str) -> str:
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    try:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        host = host.split("@")[ -1 ]
        host = host.split(":")[0]
        host = host.lstrip("www.")
        return host
    except Exception:
        return ""

def autodetect_col(df: pd.DataFrame, candidates) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def read_table(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.lower().endswith(".parquet") else pd.read_csv(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trusted", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--out", default="trusted_entities_resolved.csv")
    ap.add_argument("--review", default="trusted_entities_review_queue.csv")

    ap.add_argument("--master-name-col", default=None)
    ap.add_argument("--master-country-col", default=None)
    ap.add_argument("--master-id-col", default=None)
    ap.add_argument("--master-website-col", default=None)

    ap.add_argument("--trusted-name-col", default="raw_brand_name")
    ap.add_argument("--trusted-country-col", default="country_hint")
    ap.add_argument("--trusted-website-col", default="website_hint")

    ap.add_argument("--domain-max-candidates", type=int, default=25)
    ap.add_argument("--fuzzy-threshold", type=int, default=92)
    ap.add_argument("--topk", type=int, default=5)

    args = ap.parse_args()

    trusted = pd.read_csv(args.trusted)
    master = read_table(args.master)

    name_col = args.master_name_col or autodetect_col(master, COMMON_NAME_COLS)
    country_col = args.master_country_col or autodetect_col(master, COMMON_COUNTRY_COLS)
    id_col = args.master_id_col or autodetect_col(master, COMMON_ID_COLS)
    website_col = args.master_website_col or autodetect_col(master, COMMON_WEBSITE_COLS)

    if not name_col or not country_col or not id_col:
        raise SystemExit(
            "Could not autodetect required master columns.\n"
            f"Detected name={name_col}, country={country_col}, id={id_col}, website={website_col}\n"
            "Provide overrides: --master-name-col --master-country-col --master-id-col (and optionally --master-website-col).\n"
        )

    for c in [args.trusted_name_col, args.trusted_country_col, args.trusted_website_col]:
        if c not in trusted.columns:
            trusted[c] = ""

    trusted["__tname"] = trusted[args.trusted_name_col].astype(str).fillna("").map(normalize_name)
    trusted["__tcountry"] = trusted[args.trusted_country_col].astype(str).fillna("").str.upper().str.strip()
    trusted["__tdomain"] = trusted[args.trusted_website_col].astype(str).fillna("").map(extract_domain)

    m = master.copy()
    m[name_col] = m[name_col].astype(str).fillna("")
    m[country_col] = m[country_col].astype(str).fillna("").str.upper().str.strip()
    m["__nname"] = m[name_col].map(normalize_name)
    if website_col and website_col in m.columns:
        m["__domain"] = m[website_col].astype(str).fillna("").map(extract_domain)
    else:
        m["__domain"] = ""
    m = m[m[id_col].notna()].copy()

    domain_groups = {}
    if website_col:
        m_dom = m[m["__domain"] != ""]
        if not m_dom.empty:
            for (cty, dom), grp in m_dom.groupby([country_col, "__domain"]):
                domain_groups[(str(cty), str(dom))] = grp

    fuzzy_index = {}
    for cty, grp in m.groupby(country_col):
        names = grp["__nname"].tolist()
        rows = grp.reset_index(drop=True)
        fuzzy_index[str(cty)] = (names, rows)

    trusted["entity_id"] = ""
    trusted["match_method"] = ""
    trusted["match_score"] = ""
    trusted["match_candidate_count"] = ""
    trusted["needs_review"] = ""

    review_rows = []

    for i, row in trusted.iterrows():
        tname = row["__tname"]
        cty = row["__tcountry"]
        dom = row["__tdomain"]

        if not tname and not dom:
            trusted.at[i, "needs_review"] = True
            trusted.at[i, "match_method"] = "empty"
            continue

        # 1) Domain exact match (within country)
        if dom and (cty, dom) in domain_groups:
            cand = domain_groups[(cty, dom)]
            cand_count = len(cand)
            trusted.at[i, "match_candidate_count"] = cand_count

            if cand_count == 1:
                trusted.at[i, "entity_id"] = str(cand.iloc[0][id_col])
                trusted.at[i, "match_method"] = "domain_exact"
                trusted.at[i, "match_score"] = 100
                trusted.at[i, "needs_review"] = False
                continue

            if cand_count <= args.domain_max_candidates:
                best_row = None
                best_score = -1
                for _, r2 in cand.iterrows():
                    score = fuzz.WRatio(tname, r2["__nname"])
                    if score > best_score:
                        best_score = score
                        best_row = r2
                if best_row is not None and best_score >= args.fuzzy_threshold:
                    trusted.at[i, "entity_id"] = str(best_row[id_col])
                    trusted.at[i, "match_method"] = "domain+fuzzy"
                    trusted.at[i, "match_score"] = best_score
                    trusted.at[i, "needs_review"] = False
                    continue

                top = cand.copy()
                top["__score"] = top["__nname"].map(lambda n: fuzz.WRatio(tname, n))
                top = top.sort_values("__score", ascending=False).head(args.topk)
                review_rows.append({
                    "raw_brand_name": row.get(args.trusted_name_col, ""),
                    "country_hint": cty,
                    "website_hint": row.get(args.trusted_website_col, ""),
                    "match_method": "domain_ambiguous",
                    "notes": f"Domain matched {cand_count} candidates; none cleared threshold {args.fuzzy_threshold}",
                    "candidates": "; ".join([f"{r[id_col]}|{r[name_col]}|{r['__score']}" for _, r in top.iterrows()])
                })
                trusted.at[i, "needs_review"] = True
                trusted.at[i, "match_method"] = "domain_ambiguous"
                trusted.at[i, "match_score"] = best_score if best_score >= 0 else ""
                continue

        # 2) Fuzzy name within country
        if cty in fuzzy_index and tname:
            names, rows = fuzzy_index[cty]
            best = process.extractOne(tname, names, scorer=fuzz.WRatio)
            if best:
                _, score, idx = best
                trusted.at[i, "match_candidate_count"] = len(names)
                trusted.at[i, "match_score"] = score

                if score >= args.fuzzy_threshold:
                    trusted.at[i, "entity_id"] = str(rows.iloc[idx][id_col])
                    trusted.at[i, "match_method"] = "fuzzy_country"
                    trusted.at[i, "needs_review"] = False
                else:
                    top = process.extract(tname, names, scorer=fuzz.WRatio, limit=args.topk)
                    cand_str = []
                    for _, score2, idx2 in top:
                        cand_str.append(f"{rows.iloc[idx2][id_col]}|{rows.iloc[idx2][name_col]}|{score2}")
                    review_rows.append({
                        "raw_brand_name": row.get(args.trusted_name_col, ""),
                        "country_hint": cty,
                        "website_hint": row.get(args.trusted_website_col, ""),
                        "match_method": "fuzzy_below_threshold",
                        "notes": f"Best score {score} below threshold {args.fuzzy_threshold}",
                        "candidates": "; ".join(cand_str)
                    })
                    trusted.at[i, "needs_review"] = True
                    trusted.at[i, "match_method"] = "fuzzy_below_threshold"
            else:
                trusted.at[i, "needs_review"] = True
                trusted.at[i, "match_method"] = "no_candidates"
        else:
            trusted.at[i, "needs_review"] = True
            trusted.at[i, "match_method"] = "no_country_index"

    trusted_out = trusted.drop(columns=["__tname", "__tcountry", "__tdomain"], errors="ignore")
    trusted_out.to_csv(args.out, index=False)
    pd.DataFrame(review_rows).to_csv(args.review, index=False)

    resolved = (trusted_out["entity_id"].astype(str).str.strip() != "").sum()
    total = len(trusted_out)
    print(f"Resolved {resolved}/{total} trusted rows ({resolved/total:.1%}).")
    print("Wrote:", args.out)
    print("Review queue:", args.review)

if __name__ == "__main__":
    main()
