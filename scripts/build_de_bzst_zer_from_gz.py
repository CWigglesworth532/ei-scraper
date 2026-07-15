from __future__ import annotations

import csv
import json
import re
import zlib
from pathlib import Path
from typing import Any

IN_DIR = Path("data/de/bzst/zuwendung/raw_gz")
OUT = Path("data/de/bzst/zuwendung/de_bzst_zer_latest.csv")

RE_BYTE_CSV_PREFIX = re.compile(rb"^\s*\d{1,3}\s*,\s*\d{1,3}\s*,")
RE_INT = re.compile(rb"\d{1,3}")

def bytes_from_comma_ints(b: bytes) -> bytes:
    nums = [int(m.group(0)) for m in RE_INT.finditer(b)]
    if not nums:
        raise ValueError("no_ints_found")
    return bytes(nums)

def load_shard(path: Path) -> Any:
    raw = path.read_bytes()
    # Convert "120,156,..." ASCII into real bytes
    if RE_BYTE_CSV_PREFIX.match(raw[:60]):
        raw = bytes_from_comma_ints(raw)

    # zlib wrapper (0x78 0x9C etc.)
    if len(raw) >= 2 and raw[0] == 0x78:
        txt = zlib.decompress(raw).decode("utf-8", errors="replace")
        return json.loads(txt)

    # Fallback: try direct JSON (rare)
    txt = raw.decode("utf-8", errors="replace").lstrip()
    return json.loads(txt)

def find_list_of_dicts(obj: Any, path: str = "") -> list[tuple[str, list[dict]]]:
    found: list[tuple[str, list[dict]]] = []

    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        found.append((path or "$", obj))
        return found

    if isinstance(obj, dict):
        for k, v in obj.items():
            child_path = f"{path}.{k}" if path else str(k)
            found.extend(find_list_of_dicts(v, child_path))

    if isinstance(obj, list):
        # list of non-dicts could still contain dicts deeper
        for i, v in enumerate(obj[:200]):  # cap traversal
            found.extend(find_list_of_dicts(v, f"{path}[{i}]"))

    return found

def pick_best(found: list[tuple[str, list[dict]]], want_keys: set[str]) -> tuple[str, list[dict]] | tuple[str, list]:
    best_path = ""
    best_list: list[dict] = []
    best_len = 0

    for pth, lst in found:
        if not lst:
            continue
        keys = set(lst[0].keys())
        if not (keys & want_keys):
            continue
        if len(lst) > best_len:
            best_len = len(lst)
            best_path = pth
            best_list = lst
    return best_path, best_list

base_records: list[dict] = []
blob_records: list[dict] = []
debug_hits = []

for f in sorted(IN_DIR.glob("*.gz")):
    obj = load_shard(f)
    found = find_list_of_dicts(obj)

    # Choose best candidates from this shard
    base_path, base = pick_best(found, {"name", "organisation"})
    blob_path, blob = pick_best(found, {"stNr", "stnr"})

    if base:
        base_records.extend(base)
        debug_hits.append((f.name, "base", base_path, len(base)))
    if blob:
        blob_records.extend(blob)
        debug_hits.append((f.name, "blob", blob_path, len(blob)))

print("Base records:", len(base_records))
print("Blob records:", len(blob_records))
print("Top shard hits (first 20):")
for row in debug_hits[:20]:
    print(" -", row)

# Index blobs by stNr if present
blob_by = {}
for b in blob_records:
    st = b.get("stNr") or b.get("stnr")
    if st:
        blob_by[str(st)] = b

def norm(rec: dict) -> dict:
    st = rec.get("stNr") or rec.get("stnr")
    if st and str(st) in blob_by:
        merged = dict(blob_by[str(st)])
        merged.update(rec)
        rec = merged

    return {
        "country": "DE",
        "ccaa": "Germany (National)",
        "ei_register_name": "BZSt — Zuwendungsempfängerregister",
        "ei_registration_number": st or "",
        "entity_name": rec.get("name") or rec.get("organisation") or "",
        "tax_id": st or "",

        "address": rec.get("anschrift") or rec.get("adresse") or "",
        "postcode": rec.get("plz") or rec.get("postleitzahl") or "",
        "city": rec.get("ort") or rec.get("sitz") or "",
        "province": rec.get("bundesland") or "",

        "legal_form_local": "",
        "base_legal_form_code": "",
        "base_legal_form_family": "",

        "se_recognition_type": "tax_designation",
        "se_recognition_name": "BZSt Zuwendungsempfängerregister",
        "se_recognition_evidence": "Eligible to issue donation receipts (Spendenquittungen)",

        "source_url": "https://zer.bzst.de/",
        "source_type": "CSV(local_build)",
        "retrieved_at": "",
    }

rows = [norm(r) for r in base_records if isinstance(r, dict) and (r.get("name") or r.get("organisation"))]

OUT.parent.mkdir(parents=True, exist_ok=True)
cols = list(rows[0].keys()) if rows else []
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

print("Wrote:", OUT)
print("Rows:", len(rows))
