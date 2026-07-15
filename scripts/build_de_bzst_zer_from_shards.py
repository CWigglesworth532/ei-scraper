from __future__ import annotations

import csv
import json
import re
import string
import zlib
from pathlib import Path

IN_DIR = Path("data/de/bzst/zuwendung/raw_gz")
OUT = Path("data/de/bzst/zuwendung/de_bzst_zer_latest.csv")

RE_INT = re.compile(rb"\d{1,3}")
RE_BYTE_CSV_PREFIX = re.compile(rb"^\s*\d{1,3}\s*,\s*\d{1,3}\s*,")

_ALPH36 = {ch: i for i, ch in enumerate(string.digits + string.ascii_lowercase)}
_ALPH62 = {ch: i for i, ch in enumerate(string.digits + string.ascii_lowercase + string.ascii_uppercase)}

def load_shard(path: Path) -> dict:
    b = path.read_bytes()
    if RE_BYTE_CSV_PREFIX.match(b[:60]):
        b = bytes(int(m.group(0)) for m in RE_INT.finditer(b))
    txt = zlib.decompress(b).decode("utf-8", errors="replace")
    obj = json.loads(txt)
    if not isinstance(obj, dict) or "_" not in obj:
        raise ValueError(f"unexpected shard schema: {path.name}")
    return obj

def decode_token_to_int(tok: str) -> int | None:
    if tok is None:
        return None
    s = str(tok).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)

    s_low = s.lower()
    if all(ch in _ALPH36 for ch in s_low):
        n = 0
        for ch in s_low:
            n = n * 36 + _ALPH36[ch]
        return n

    if all(ch in _ALPH62 for ch in s):
        n = 0
        for ch in s:
            n = n * 62 + _ALPH62[ch]
        return n

    return None

def decode_prefixed_palette(P: list[str], ref) -> str:
    if ref is None:
        return ""
    s = str(ref).strip()
    if not s:
        return ""
    if s.startswith("p:"):
        idx = decode_token_to_int(s[2:])
        if idx is None or idx < 0 or idx >= len(P):
            return ""
        return (P[idx] or "").strip()
    # already decoded
    return s

def decode_palette_list(P: list[str], refs) -> str:
    if not isinstance(refs, list):
        return ""
    out = []
    for r in refs:
        s = decode_prefixed_palette(P, r)
        if s:
            out.append(s)
    return " | ".join(out)

# Load shards
id_sh = load_shard(IN_DIR / "id.compressed.json.gz")
org_sh = load_shard(IN_DIR / "org.compressed.json.gz")
sitz_sh = load_shard(IN_DIR / "sitz.compressed.json.gz")
zwecke_sh = load_shard(IN_DIR / "zwecke.compressed.json.gz")
fin_sh = load_shard(IN_DIR / "finanzamt.compressed.json.gz")

ids = id_sh["_"]
org_refs = org_sh["_"]
sitz_refs = sitz_sh["_"]
zwecke_refs = zwecke_sh["_"]
fin_refs = fin_sh["_"]

N = min(len(ids), len(org_refs), len(sitz_refs), len(zwecke_refs), len(fin_refs))
print("Row count N:", N)

P_org = org_sh.get("P", [])
P_sitz = sitz_sh.get("P", [])
P_zwecke = zwecke_sh.get("P", [])
P_fin = fin_sh.get("P", [])

# After decoding sitz, try to split into components
RE_PLZ = re.compile(r"\b(\d{5})\b")
def split_sitz(s: str):
    s = (s or "").strip()
    if not s:
        return ("", "", "")
    m = RE_PLZ.search(s)
    if not m:
        return ("", "", s)
    plz = m.group(1)
    rest = (s[:m.start()] + " " + s[m.end():]).strip(" ,;")
    parts = [p.strip() for p in re.split(r"\s*,\s*", rest) if p.strip()]
    if parts:
        city = parts[0]
        addr = ", ".join(parts[1:]) if len(parts) > 1 else ""
    else:
        city, addr = "", ""
    if not addr:
        addr = rest
    return (plz, city, addr)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "country","ccaa","ei_register_name","ei_registration_number",
        "entity_name","tax_id","address","postcode","city","province",
        "legal_form_local","base_legal_form_code","base_legal_form_family",
        "se_recognition_type","se_recognition_name","se_recognition_evidence",
        "source_url","source_type","retrieved_at",
        "de_finanzamt","de_zwecke"
    ]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()

    wrote = 0
    for i in range(N):
        tax = (ids[i] or "").strip()
        name = decode_prefixed_palette(P_org, org_refs[i])
        if not tax or not name:
            continue

        sitz = decode_prefixed_palette(P_sitz, sitz_refs[i])
        plz, city, addr = split_sitz(sitz)

        finanzamt = decode_prefixed_palette(P_fin, fin_refs[i])
        zwecke = decode_palette_list(P_zwecke, zwecke_refs[i])

        w.writerow({
            "country": "DE",
            "ccaa": "Germany (National)",
            "ei_register_name": "BZSt — Zuwendungsempfängerregister",
            "ei_registration_number": tax,
            "entity_name": name,
            "tax_id": tax,

            "address": addr,
            "postcode": plz,
            "city": city,
            "province": "",

            "legal_form_local": "",
            "base_legal_form_code": "",
            "base_legal_form_family": "",

            "se_recognition_type": "tax_designation",
            "se_recognition_name": "BZSt Zuwendungsempfängerregister",
            "se_recognition_evidence": "Dictionary-encoded shards: id/org/sitz/zwecke/finanzamt",

            "source_url": "https://zer.bzst.de/",
            "source_type": "CSV(local_build)",
            "retrieved_at": "",

            "de_finanzamt": finanzamt,
            "de_zwecke": zwecke,
        })
        wrote += 1

print("Wrote:", OUT)
print("Rows written:", wrote)
