#!/usr/bin/env python3
import argparse
import re
import pandas as pd

try:
    from match_suppliers import normalize_name
except Exception:
    def normalize_name(s: str) -> str:
        s = (s or "").lower()
        s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
        s = re.sub(r"\s+", " ", s).strip()
        return s

def norm_id(id_type: str, id_value: str) -> str:
    t = (id_type or "").strip().upper().replace(".", "")
    v = (id_value or "").strip()
    if not t or not v:
        return ""
    # Normalize common patterns: "HRB 29536" in either field, strip spaces
    combo = f"{t} {v}".strip()
    combo = re.sub(r"\s+", " ", combo)
    m = re.match(r"^([A-Z]{1,6})\s*[- ]?\s*([0-9A-Z\-]+)$", combo)
    if m:
        return f"{m.group(1)}:{m.group(2).replace(' ','')}"
    return f"{t}:{v.replace(' ','')}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Reviewed verify_queue.csv")
    ap.add_argument("--out", default="ei_registers_verified_overlay.csv", help="Overlay master output")
    args = ap.parse_args()

    df = pd.read_csv(args.inp, dtype=str).fillna("")

    need = ["country","raw_name","verified","verified_name","verified_id_type","verified_id","verified_source","verified_url"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing columns {miss}. Found: {df.columns.tolist()}")

    v = df[df["verified"].str.strip().str.upper().isin(["Y","YES","TRUE","1"])].copy()
    if len(v) == 0:
        print("No verified rows (verified != Y). Nothing to ingest.")
        return

    v["country"] = v["country"].str.strip().str.upper()
    v["entity_name"] = v["verified_name"].where(v["verified_name"].str.strip() != "", v["raw_name"])
    v["verified_key"] = [norm_id(t, x) for t, x in zip(v["verified_id_type"], v["verified_id"])]

    bad = v[v["verified_key"] == ""]
    if len(bad):
        print("ERROR: verified rows missing verified_id_type/verified_id. Fix and rerun.")
        print(bad[["country","raw_name","verified_name","verified_id_type","verified_id"]].head(50).to_string(index=False))
        raise SystemExit(2)

    out = pd.DataFrame({
        "country": v["country"],
        "ei_register_name": v["verified_source"].str.strip(),
        "entity_name": v["entity_name"].str.strip(),
        # put stable verified key into tax_id so your matcher can use it as an ID
        "tax_id": v["verified_key"],
        "verified_source": v["verified_source"].str.strip(),
        "verified_url": v["verified_url"].str.strip(),
        "verified_id_type": v["verified_id_type"].str.strip(),
        "verified_id": v["verified_id"].str.strip(),
        "raw_name": v["raw_name"].str.strip(),
        "norm_name": v["entity_name"].map(normalize_name),
        "notes": df.get("notes", pd.Series([""]*len(df))).iloc[v.index].fillna("").astype(str),
    })

    try:
        existing = pd.read_csv(args.out, dtype=str).fillna("")
    except FileNotFoundError:
        existing = pd.DataFrame(columns=out.columns)

    combined = pd.concat([existing, out], ignore_index=True)

    # Deduplicate by (country, tax_id) so verified IDs win and stay stable
    combined = combined.sort_values(["country","tax_id","entity_name"]).drop_duplicates(["country","tax_id"], keep="first")

    combined.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} verified rows into {args.out} (total overlay rows: {len(combined)})")

if __name__ == "__main__":
    main()
