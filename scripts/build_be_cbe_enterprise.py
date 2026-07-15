#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

ROOT = Path(__file__).resolve().parents[1]

ENTERPRISE = ROOT / "data/be/cbe/raw/enterprise.csv"
DENOM = ROOT / "data/be/cbe/raw/denomination.csv"
ADDRESS = ROOT / "data/be/cbe/raw/address.csv"
OUT = ROOT / "data/be/cbe/be_cbe_enterprise_latest.csv"

DIGITS = re.compile(r"\D+")

def norm_be_number(x: str) -> str:
    """Normalize BE enterprise/entity number to 10 digits."""
    if x is None:
        return ""
    s = DIGITS.sub("", x)
    if not s:
        return ""
    return s.zfill(10)

def score_denom(lang: str, typ: str, denom: str) -> Tuple[int, int, int]:
    """
    Lower is better. Prefer:
      - official denomination type (commonly '001') if present
      - NL over FR over other languages
      - longer denomination (more informative) as tie-breaker
    """
    lang = (lang or "").strip().lower()
    typ = (typ or "").strip()
    denom = (denom or "").strip()

    # TypeOfDenomination: keep conservative preference; many exports use '001' for official.
    type_rank = 0 if typ == "001" else 5

    lang_rank = 0
    if lang == "nl":
        lang_rank = 0
    elif lang == "fr":
        lang_rank = 1
    elif lang == "de":
        lang_rank = 2
    else:
        lang_rank = 9

    # Prefer longer (invert length into rank bucket)
    len_rank = -len(denom)

    return (type_rank, lang_rank, len_rank)

def score_address(type_of_address: str, striking_off: str) -> Tuple[int, int]:
    """
    Lower is better. Prefer:
      - non-struck-off addresses (empty DateStrikingOff)
      - address types that likely represent registered seat / head office
    """
    t = (type_of_address or "").strip().upper()
    struck = 0 if (striking_off or "").strip() == "" else 1

    # Conservative: prefer types that look like registered/head office.
    # If your export uses different codes, it will fall back gracefully.
    preferred = {"REGO", "REG", "HEAD", "SEAT", "HQ", "REGISTERED", "REGISTERED_OFFICE"}
    type_rank = 0 if t in preferred else 5

    return (struck, type_rank)

def pick_best_denom() -> Dict[str, str]:
    best: Dict[str, Tuple[Tuple[int,int,int], str]] = {}
    with DENOM.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ent = norm_be_number(row.get("EntityNumber", ""))
            if not ent:
                continue
            lang = row.get("Language", "")
            typ = row.get("TypeOfDenomination", "")
            denom = (row.get("Denomination", "") or "").strip()
            if not denom:
                continue
            sc = score_denom(lang, typ, denom)
            prev = best.get(ent)
            if prev is None or sc < prev[0]:
                best[ent] = (sc, denom)
    return {k: v[1] for k, v in best.items()}

def pick_best_address() -> Dict[str, Dict[str, str]]:
    best: Dict[str, Tuple[Tuple[int,int], Dict[str, str]]] = {}
    with ADDRESS.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ent = norm_be_number(row.get("EntityNumber", ""))
            if not ent:
                continue
            sc = score_address(row.get("TypeOfAddress", ""), row.get("DateStrikingOff", ""))

            addr = {
                "zipcode": (row.get("Zipcode", "") or "").strip(),
                "municipality_nl": (row.get("MunicipalityNL", "") or "").strip(),
                "municipality_fr": (row.get("MunicipalityFR", "") or "").strip(),
                "street_nl": (row.get("StreetNL", "") or "").strip(),
                "street_fr": (row.get("StreetFR", "") or "").strip(),
                "house_number": (row.get("HouseNumber", "") or "").strip(),
                "box": (row.get("Box", "") or "").strip(),
                "extra": (row.get("ExtraAddressInfo", "") or "").strip(),
                "country_nl": (row.get("CountryNL", "") or "").strip(),
                "country_fr": (row.get("CountryFR", "") or "").strip(),
                "type": (row.get("TypeOfAddress", "") or "").strip(),
            }

            prev = best.get(ent)
            if prev is None or sc < prev[0]:
                best[ent] = (sc, addr)

    return {k: v[1] for k, v in best.items()}

def compose_address(a: Dict[str, str]) -> Tuple[str, str, str, str]:
    """
    Returns (address_line, postcode, city, country)
    Prefer NL fields; fall back to FR.
    """
    street = a.get("street_nl") or a.get("street_fr") or ""
    city = a.get("municipality_nl") or a.get("municipality_fr") or ""
    postcode = a.get("zipcode") or ""
    country = a.get("country_nl") or a.get("country_fr") or ""

    hn = a.get("house_number") or ""
    box = a.get("box") or ""
    extra = a.get("extra") or ""

    parts = [p for p in [street, hn] if p]
    line = " ".join(parts).strip()
    if box:
        line = (line + f", Box {box}").strip()
    if extra:
        line = (line + f", {extra}").strip()
    return line, postcode, city, country

def main() -> None:
    if not ENTERPRISE.exists():
        raise SystemExit(f"Missing: {ENTERPRISE}")
    if not DENOM.exists():
        raise SystemExit(f"Missing: {DENOM}")
    if not ADDRESS.exists():
        raise SystemExit(f"Missing: {ADDRESS}")

    print("Loading best denominations...")
    denom = pick_best_denom()
    print(f"  denom map size: {len(denom):,}")

    print("Loading best addresses...")
    addr = pick_best_address()
    print(f"  address map size: {len(addr):,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    out_fields = [
        "tax_id",
        "entity_name",
        "legal_form_local",
        "status",
        "start_date",
        "address",
        "postcode",
        "city",
        "country",
        "source",
    ]

    n_in = 0
    n_out = 0

    with ENTERPRISE.open("r", encoding="utf-8", newline="") as f_in, OUT.open("w", encoding="utf-8", newline="") as f_out:
        r = csv.DictReader(f_in)
        w = csv.DictWriter(f_out, fieldnames=out_fields)
        w.writeheader()

        for row in r:
            n_in += 1
            ent = norm_be_number(row.get("EnterpriseNumber", ""))
            if not ent:
                continue

            status = (row.get("Status", "") or "").strip()
            legal_form = (row.get("JuridicalForm", "") or "").strip()
            start_date = (row.get("StartDate", "") or "").strip()

            name = (denom.get(ent) or "").strip()

            a = addr.get(ent)
            if a:
                address_line, postcode, city, country = compose_address(a)
            else:
                address_line, postcode, city, country = "", "", "", ""

            # Precision-first: require at least a name OR a legal form (name usually present if denom file is good)
            if not name and not legal_form:
                continue

            w.writerow({
                "tax_id": ent,
                "entity_name": name,
                "legal_form_local": legal_form,
                "status": status,
                "start_date": start_date,
                "address": address_line,
                "postcode": postcode,
                "city": city,
                "country": country,
                "source": "CBE/KBO open data (enterprise-level build)",
            })
            n_out += 1

    print(f"Wrote: {OUT}")
    print(f"Enterprise rows read: {n_in:,}")
    print(f"Rows written:       {n_out:,}")

if __name__ == "__main__":
    main()
