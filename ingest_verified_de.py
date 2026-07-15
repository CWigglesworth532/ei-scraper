#!/usr/bin/env python3
import argparse
import re
import pandas as pd

# reuse your pipeline normalization for consistent dedupe / matching
try:
    from match_suppliers import normalize_name
except Exception:
    def normalize_name(s: str) -> str:
        s = (s or "").lower()
        s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
        s = re.sub(r"\s+", " ", s).strip()
        return s

REGNUM_RE = re.compile(r"^\s*([A-ZÄÖÜ]{1,4})\s*[- ]?\s*([0-9]+)\s*$", re.IGNORECASE)

def norm_register_id(register_type: str, register_number: str) -> str:
    rt = (register_type or "").strip().upper().replace(".", "")
    rn = (register_number or "").strip()

    # accept "HRB 29536" in either field
    combo = f"{rt} {rn}".strip()
    m = REGNUM_RE.match(combo)
    if m:
        return f"{m.group(1).upper()}{m.group(2)}"

    # fallback: strip spaces/punct
    rt = re.sub(r"\s+", "", rt)
    rn = re.sub(r"\D+", "", rn)
    if rt and rn:
        return f"{rt}{rn}"
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Reviewed DE queue CSV (de_to_verify.csv)")
    ap.add_argument("--out", dest="out", default="ei_registers_verified_de.csv", help="Verified overlay output CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.inp, dtype=str).fillna("")

    required_cols = [
        "raw_name", "country", "verified",
        "verified_name", "register_type", "register_number",
        "register_court", "status", "source",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in input: {missing}\nFound: {df.columns.tolist()}")

    # keep verified rows only
    v = df[df["verified"].str.strip().str.upper().isin(["Y", "YES", "TRUE", "1"])].copy()

    if len(v) == 0:
        print("No verified rows found (verified != Y). Nothing to do.")
        return

    # normalize register id
    v["register_id"] = [
        norm_register_id(rt, rn) for rt, rn in zip(v["register_type"], v["register_number"])
    ]

    bad = v[v["register_id"] == ""]
    if len(bad):
        print("ERROR: Some verified rows are missing/invalid register id. Fix these and rerun:")
        print(bad[["raw_name","verified_name","register_type","register_number","register_id"]].head(50).to_string(index=False))
        raise SystemExit(2)

    # build overlay rows in the same schema as your headered master
    out = pd.DataFrame({
        "country": v["country"].str.strip().str.upper(),
        "ei_register_name": v["source"].str.strip() + " (" + v["register_court"].str.strip() + ")",
        "entity_name": v["verified_name"].where(v["verified_name"].str.strip() != "", v["raw_name"]),
        # Put register_id in tax_id field so your pipeline can use it as a stable ID
        "tax_id": v["register_id"],
        # extra provenance fields (optional) — keep them if you want
        "verified_source": v["source"].str.strip(),
        "register_court": v["register_court"].str.strip(),
        "register_type": v["register_type"].str.strip(),
        "register_number": v["register_number"].str.strip(),
        "status": v["status"].str.strip(),
        "raw_name": v["raw_name"].str.strip(),
        "norm_name": v["verified_name"].where(v["verified_name"].str.strip() != "", v["raw_name"]).map(normalize_name),
    })

    # load existing overlay if present, else create new
    try:
        existing = pd.read_csv(args.out, dtype=str).fillna("")
        had_existing = True
    except FileNotFoundError:
        existing = pd.DataFrame(columns=out.columns)
        had_existing = False

    combined = pd.concat([existing, out], ignore_index=True)

    # Deduplicate by (country, tax_id/register_id)
    before = len(combined)
    combined = combined.sort_values(["country","tax_id","entity_name"]).drop_duplicates(["country","tax_id"], keep="first")
    after = len(combined)

    combined.to_csv(args.out, index=False)

    added = after - (len(existing.drop_duplicates(["country","tax_id"])) if had_existing else 0)

    print(f"Wrote overlay: {args.out}")
    print(f"Verified rows in input: {len(v)}")
    print(f"New unique entities added: {max(0, added)}")
    print(f"Total entities in overlay: {len(combined)}")

if __name__ == "__main__":
    main()
