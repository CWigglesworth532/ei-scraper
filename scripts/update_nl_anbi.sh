#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/data/nl/anbi"
mkdir -p "$DIR"
cd "$DIR"

curl -L -o anbi.zip "https://download.belastingdienst.nl/data/anbi/anbi.zip"
unzip -o anbi.zip

python - <<'PY'
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

xml_path = Path("anbi.xml")
out_path = Path("anbi_latest.csv")

tree = ET.parse(xml_path)
root = tree.getroot()

def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag

children = list(root)
from collections import Counter
tags = [strip_ns(ch.tag) for ch in children]
record_tag, _ = Counter(tags).most_common(1)[0]
records = [ch for ch in children if strip_ns(ch.tag) == record_tag]

def iter_leaves(node, prefix=""):
    for ch in list(node):
        t = strip_ns(ch.tag)
        path = f"{prefix}.{t}" if prefix else t
        if list(ch):
            yield from iter_leaves(ch, path)
        else:
            yield path, (ch.text or "").strip()

cols = ["naam","aliasNaam","fiscaalNummer","vestigingsPlaats","dossierNummer","ingangsDatum","ingangsDatumCultuur","webSite"]

with out_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in records:
        row = {c:"" for c in cols}
        for p,v in iter_leaves(r):
            if p in row and v:
                row[p]=v
        if not row["naam"] or not row["fiscaalNummer"]:
            continue
        w.writerow(row)

print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")
PY
