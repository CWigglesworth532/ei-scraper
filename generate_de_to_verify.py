#!/usr/bin/env python3
import re
import argparse
import pandas as pd

# Try to reuse your pipeline's normalization for perfect consistency
try:
    from match_suppliers import normalize_name
except Exception:
    # fallback minimal normalization if import fails
    def normalize_name(s: str) -> str:
        s = (s or "").lower()
        s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
        s = re.sub(r"\s+", " ", s).strip()
        return s

LEGAL_SUFFIX_RE = re.compile(
    r"\b("
    r"gmbh|ggmbh|ug|ag|kg|kgaa|ohg|gbr|"
    r"e\.?g\.?|eg|genossenschaft|"
    r"e\.?v\.?|ev|verein|"
    r"stiftung|"
    r"gag|mbh"
    r")\b",
    re.IGNORECASE,
)

GENERIC_WORDS = {
    "deutsch", "deutsche", "gesellschaft", "verein", "verband", "stiftung",
    "gmbh", "ggmbh", "eg", "genossenschaft", "ev", "e", "v",
    "zentrum", "klinik", "krankenhaus", "gruppe", "service", "services",
}

def clean_query(raw_name: str) -> str:
    """Make a good search string for Handelsregister: remove common legal suffixes, keep distinguishing bits."""
    s = (raw_name or "").strip()
    s = LEGAL_SUFFIX_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def priority_score(reason: str, trigger: str, raw_name: str) -> int:
    """Higher = review first. Keep it simple and explainable."""
    reason = (reason or "").strip().lower()
    trigger = (trigger or "").strip().lower()
    raw = (raw_name or "").strip().lower()

    score = 0

    # Strongest indicators first
    if "ggmbh" in trigger or "gmbh" in trigger and "gemeinn" in raw:
        score += 90
    if trigger in {"eg", "e.g", "e g", "e.g.", "e. g.", "eG".lower()} or "eg" in trigger:
        score += 80
    if "genossenschaft" in raw:
        score += 80

    # By standardized reason bucket
    if reason == "social enterprise":
        score += 60
    elif reason == "coop":
        score += 45
    elif reason == "not for profit":
        score += 40

    # Boost if name contains explicit charity marker
    if re.search(r"\bgemeinn", raw, re.IGNORECASE):
        score += 25

    # Penalize overly generic short names
    norm = normalize_name(raw_name)
    toks = [t for t in norm.split() if len(t) >= 4 and t not in GENERIC_WORDS]
    if len(toks) <= 1:
        score -= 40
    elif len(toks) == 2:
        score -= 10

    # Penalize if name is extremely short overall
    if len(norm) < 10:
        score -= 20

    return score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input matched suppliers CSV (e.g. suppliers_matched_2026_v6.csv)")
    ap.add_argument("--out", dest="out", default="de_to_verify.csv", help="Output CSV (default: de_to_verify.csv)")
    args = ap.parse_args()

    df = pd.read_csv(args.inp, dtype=str).fillna("")

    # Column picks (robust across your variants)
    col_id = pick_col(df, ["supplier_id", "id", "0"])
    col_raw = pick_col(df, ["_supplier_raw_name", "supplier_name", "1"])
    col_cty = pick_col(df, ["supplier_country", "country", "2"])
    col_city = pick_col(df, ["supplier_city", "city", "4"])
    col_post = pick_col(df, ["postcode", "zip", "6"])
    col_reason = pick_col(df, ["name_candidate_reason"])
    col_trigger = pick_col(df, ["name_candidate_trigger"])
    col_candidate = pick_col(df, ["social_economy_name_candidate"])
    col_match_type = pick_col(df, ["match_type"])

    required = [col_raw, col_cty, col_candidate, col_match_type, col_reason, col_trigger]
    if any(c is None for c in required):
        missing = [x for x in ["_supplier_raw_name/supplier_name", "supplier_country", "social_economy_name_candidate", "match_type", "name_candidate_reason", "name_candidate_trigger"] if pick_col(df, [x]) is None]
        raise SystemExit(f"Missing expected columns. Present columns: {df.columns.tolist()}")

    # Filter: Germany only, heuristic YES, not already matched
    de = df[
        (df[col_cty].str.upper() == "DE") &
        (df[col_candidate].str.upper() == "YES") &
        (df[col_match_type].str.strip() == "")
    ].copy()

    # Build helper fields
    de["norm_name"] = de[col_raw].map(normalize_name)
    de["search_query"] = de[col_raw].map(clean_query)
    de["priority"] = [
        priority_score(r, t, n) for r, t, n in zip(de[col_reason], de[col_trigger], de[col_raw])
    ]

    # Dedupe key: normalized name + city/postcode if available
    city = de[col_city] if col_city else ""
    post = de[col_post] if col_post else ""
    de["dedupe_key"] = (
        de["norm_name"].astype(str).str.strip() + "||" +
        pd.Series(city).astype(str).str.upper().str.strip() + "||" +
        pd.Series(post).astype(str).str.strip()
    )

    # Keep the highest priority row per dedupe_key
    de = de.sort_values(["priority"], ascending=False).drop_duplicates("dedupe_key", keep="first")

    # Final columns for reviewers
    out_cols = {
        "supplier_id": de[col_id] if col_id else "",
        "raw_name": de[col_raw],
        "country": de[col_cty].str.upper(),
        "city": de[col_city] if col_city else "",
        "postcode": de[col_post] if col_post else "",
        "reason": de[col_reason],
        "trigger": de[col_trigger],
        "priority": de["priority"],
        "search_query": de["search_query"],
        # reviewer fills these:
        "verified": "",
        "verified_name": "",
        "register_type": "",
        "register_number": "",
        "register_court": "",
        "status": "",
        "source": "handelsregister.de",
        "notes": "",
    }

    out = pd.DataFrame(out_cols)

    # Sort by priority descending for review
    out = out.sort_values(["priority", "raw_name"], ascending=[False, True])

    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows -> {args.out}")

if __name__ == "__main__":
    main()
