#!/usr/bin/env python3
import argparse
import re
import pandas as pd

# Reuse your normalization if available (keeps dedupe consistent with pipeline)
try:
    from match_suppliers import normalize_name
except Exception:
    def normalize_name(s: str) -> str:
        s = (s or "").lower()
        s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
        s = re.sub(r"\s+", " ", s).strip()
        return s

def priority(reason: str, iso2: str, name: str) -> int:
    """
    Higher = review first. Tuned for precision-first verification ROI.
    """
    r = (reason or "").strip().lower()
    n = (name or "").lower()
    score = 0

    # Base by bucket (you standardized these already)
    if r == "social enterprise":
        score += 80
    elif r == "coop":
        score += 70
    elif r == "not for profit":
        score += 60
    else:
        score += 40

    # Strong legal-form cues (safe boosts)
    if re.search(r"\b(gGmbH|ggmbh)\b", name or "", re.I):
        score += 25
    if re.search(r"\b(e\.?\s*g\.?)\b", name or "", re.I):
        score += 20
    if re.search(r"\b(vzw|asbl|aisbl)\b", name or "", re.I):  # BE
        score += 20
    if re.search(r"\b(associat(?:ion|ione)|association|verein|stichting|vereniging)\b", n):
        score += 10
    if re.search(r"\b(genossenschaft|cooperat(?:ive|iva)|coop[ée]rative)\b", n):
        score += 10

    # Slight boost for countries where verification is fast/clear
    iso2 = (iso2 or "").upper().strip()
    if iso2 in {"DE","BE","FR","IT","ES","NL","CH","AT"}:
        score += 5

    return score

def suggested_source(iso2: str, reason: str) -> str:
    """
    Human hint: which official DB to check first (no deep links, just the target).
    """
    iso2 = (iso2 or "").upper().strip()
    if iso2 == "DE":
        return "Handelsregister (HRB/GnR/VR) via handelsregister.de"
    if iso2 == "BE":
        return "KBO/CBE (enterprise number) via kbopub.economie.fgov.be"
    if iso2 == "FR":
        return "INSEE Sirene (SIREN) via sirene.fr"
    if iso2 == "IT":
        return "RUNTS (ETS) + Registro Imprese (CF/P.IVA)"
    if iso2 == "ES":
        return "Registro Mercantil / regional CEE + NIF/CIF"
    if iso2 == "NL":
        return "KvK Handelsregister"
    if iso2 == "CH":
        return "ZEFIX (UID)"
    if iso2 == "AT":
        return "Firmenbuch (FN)"
    return "Official national register (manual)"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input probables XLSX")
    ap.add_argument("--sheet", default=None, help="Sheet name (default first)")
    ap.add_argument("--out", default="verify_queue.csv")
    args = ap.parse_args()

    # Load Excel safely (handle multi-sheet files)
    if args.sheet:
        df = pd.read_excel(args.inp, sheet_name=args.sheet, dtype=str)
    else:
        tmp = pd.read_excel(args.inp, sheet_name=None, dtype=str)
        # take the first sheet automatically
        first_sheet = list(tmp.keys())[0]
        df = tmp[first_sheet]

    df = df.fillna("")


    # Map your columns from the uploaded file
    col_name = "Column2"
    col_iso2 = "Joint vendor country"
    col_city = "City"
    col_post = "Postal code"
    col_reason = "name_candidate_reason"
    col_status = "STATUS"
    for c in [col_name, col_iso2, col_reason]:
        if c not in df.columns:
            raise SystemExit(f"Missing expected column {c}. Found: {df.columns.tolist()}")

    q = df.copy()
    q["country"] = q[col_iso2].str.upper().str.strip()
    q["raw_name"] = q[col_name].str.strip()
    q["city"] = q[col_city].str.strip() if col_city in q.columns else ""
    q["postcode"] = q[col_post].str.strip() if col_post in q.columns else ""
    q["reason"] = q[col_reason].str.strip()
    q["status"] = q[col_status].str.strip() if col_status in q.columns else ""

    q["norm_name"] = q["raw_name"].map(normalize_name)
    q["priority"] = [priority(r, c, n) for r, c, n in zip(q["reason"], q["country"], q["raw_name"])]
    q["suggested_source"] = [suggested_source(c, r) for c, r in zip(q["country"], q["reason"])]

    # Dedupe: norm_name + (country) + (city/postcode if present)
    q["dedupe_key"] = (
        q["country"] + "||" +
        q["norm_name"] + "||" +
        q["city"].str.upper() + "||" +
        q["postcode"]
    )

    q = q.sort_values(["priority"], ascending=False).drop_duplicates("dedupe_key", keep="first")

    out = pd.DataFrame({
        "country": q["country"],
        "raw_name": q["raw_name"],
        "city": q["city"],
        "postcode": q["postcode"],
        "reason": q["reason"],
        "priority": q["priority"],
        "suggested_source": q["suggested_source"],
        "client_status": q["status"],

        # reviewer fills these:
        "verified": "",
        "verified_name": "",
        "verified_id_type": "",   # e.g. HRB, GnR, VR, SIREN, KBO, UID, NIF, CF, VAT
        "verified_id": "",        # e.g. 29536, 123456789, CHE-..., B123...
        "verified_source": "",    # e.g. handelsregister.de, KBO, Sirene, ZEFIX, RUNTS
        "verified_url": "",
        "notes": "",
    }).sort_values(["priority","country","raw_name"], ascending=[False, True, True])

    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows -> {args.out}")

if __name__ == "__main__":
    main()
