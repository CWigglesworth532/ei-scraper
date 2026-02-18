#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List

import pdfplumber


def clean(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


ROW_START_RE = re.compile(r"^(\d+)\s+(.*)$")

# Detect CC.AA at the end of a row (allow VALENCIANA as split artifact)
CCAA_RE = re.compile(
    r"(PA[IÍ]S VASCO|ANDALUC[IÍ]A|MURCIA|COMUNIDAD VALENCIANA|VALENCIANA|CATALU[NÑ]A|ARAG[OÓ]N|"
    r"GALICIA|MADRID|CASTILLA.*|EXTREMADURA|NAVARRA|LA RIOJA|CANTABRIA|ASTURIAS|"
    r"BALEARES|CANARIAS|CEUTA|MELILLA)$",
    re.IGNORECASE,
)

SIZE_RE = re.compile(
    r"(GRAN EMPRESA|MEDIANA EMPRESA|PEQUE[NÑ]A EMPRESA)$",
    re.IGNORECASE,
)

# Lines that appear as headers/labels and should not be appended to entity names
CONTINUATION_NOISE = {
    "COOPERATIVAS",
    "SOCIEDADES LABORALES",
    "COMUNIDAD",
    "REGIÓN DE",
    "REGION DE",
    "AGRICULTURA Y",
    "(MILLONES €) EMPRESA",
    "FACTURACIÓN TAMAÑO",
    "FACTURACION TAMAÑO",
}

KNOWN_SECTORS = [
    "MULTISECTORIAL",
    "SERVICIOS",
    "PESCA",
    "AGROALIMENTARIO",
    "INDUSTRIA",
    "FINANZAS",
    "ENERGIA",
    "SALUD",
    "EDUCACION",
    "CONSTRUCCION",
    "VIVIENDA",
    "TRANSPORTE",
    "COMERCIO",
]


def is_noise_line(l: str) -> bool:
    u = clean(l).upper()
    return (
        u in CONTINUATION_NOISE
        or u.startswith("Nº NOMBRE")
        or u.startswith("N° NOMBRE")
        or u.startswith("Nº NOMBRE DE")
        or u.startswith("N° NOMBRE DE")
    )


def strip_numeric_tail(s: str) -> str:
    """
    Remove trailing numeric columns (empleo, facturación etc.) that sometimes leak into the name tail.
    We cut at the first occurrence of a 'big' numeric block.
    """
    s = clean(s)
    if not s:
        return s

    # Cut at patterns like "21.869" or "11.213,00" or "4.707,28 €"
    parts = re.split(r"\s+\d{1,3}(?:\.\d{3})+(?:,\d+)?(?:\s*€)?", s, maxsplit=1)
    s2 = parts[0].strip() if parts else s

    # Also cut at " 123,45 €" / " 123,45"
    parts2 = re.split(r"\s+\d+(?:[.,]\d+)?(?:\s*€)?", s2, maxsplit=1)
    s3 = parts2[0].strip() if parts2 else s2

    return clean(s3)


def parse_row_tail(tail: str) -> Dict[str, str]:
    """
    Given a single numbered row line without the leading rank, extract:
      - entity_name
      - sector (if detectable)
      - region (CCAA, if detectable)
    """
    line = clean(tail)

    # Remove "size" label at end if present
    m = SIZE_RE.search(line)
    if m:
        line = clean(line[: m.start()])

    # Pull region at end if present
    region = ""
    m = CCAA_RE.search(line)
    if m:
        region = clean(m.group(1)).upper()
        line = clean(line[: m.start()])

    if region == "VALENCIANA":
        region = "COMUNIDAD VALENCIANA"

    # Strip numeric tail (empleo/facturación) if still present
    line = strip_numeric_tail(line)

    # Pull known sector at end if present
    sector = ""
    for ks in KNOWN_SECTORS:
        if re.search(rf"\b{re.escape(ks)}\b$", line, flags=re.IGNORECASE):
            sector = ks
            line = clean(re.sub(rf"\b{re.escape(ks)}\b$", "", line, flags=re.IGNORECASE))
            break

    entity_name = clean(line)

    return {
        "entity_name": entity_name,
        "region": region,
        "sector": sector,
    }


def extract(pdf_path: str, start_page_1idx: int = 35) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    start_i = max(0, start_page_1idx - 1)

    with pdfplumber.open(pdf_path) as pdf:
        current: Dict[str, str] | None = None

        for page_idx in range(start_i, len(pdf.pages)):
            text = pdf.pages[page_idx].extract_text() or ""
            lines = [clean(l) for l in text.split("\n") if clean(l)]

            for l in lines:
                if is_noise_line(l):
                    continue

                m = ROW_START_RE.match(l)
                if m:
                    # Flush previous
                    if current and current.get("entity_name"):
                        rows.append(current)

                    tail = m.group(2)
                    parsed = parse_row_tail(tail)

                    current = {
                        "entity_name": parsed["entity_name"],
                        "region": parsed["region"],
                        "sector": parsed["sector"],
                        "source_page": str(page_idx + 1),
                    }
                else:
                    # Continuation line: may contain region fragments or wrapped name pieces
                    if not current:
                        continue

                    u = clean(l).upper()

                    # Handle split "COMUNIDAD" + "VALENCIANA"
                    if current.get("region", "") == "" and u in {"VALENCIANA", "COMUNIDAD VALENCIANA"}:
                        current["region"] = "COMUNIDAD VALENCIANA"
                        continue

                    # Otherwise append to name (long names wrap)
                    current["entity_name"] = clean(current["entity_name"] + " " + l)

        # Flush last
        if current and current.get("entity_name"):
            rows.append(current)

    # Final cleanup / fixes
    out: List[Dict[str, str]] = []
    for r in rows:
        name = clean(r.get("entity_name", ""))
        if not name or len(name) < 3:
            continue

        # Remove any lingering numeric column leaks
        name = strip_numeric_tail(name)

        # Remove dangling noise words at end
        name = re.sub(r"\b(COMUNIDAD|REGI[ÓO]N DE|AGRICULTURA Y)\b$", "", name, flags=re.IGNORECASE).strip()

        # If sector is still empty but stuck at end of name, split it out (fallback)
        if not r.get("sector"):
            m = re.search(
                r"\b(" + "|".join(map(re.escape, KNOWN_SECTORS)) + r")$",
                name,
                flags=re.IGNORECASE,
            )
            if m:
                r["sector"] = m.group(1).upper()
                name = re.sub(r"\s+" + re.escape(m.group(1)) + r"$", "", name, flags=re.IGNORECASE).strip()

        r["entity_name"] = name

        # Normalize region one more time
        if clean(r.get("region", "")).upper() == "VALENCIANA":
            r["region"] = "COMUNIDAD VALENCIANA"

        out.append(r)

    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-page", type=int, default=35)
    ap.add_argument("--dedupe", action="store_true")
    args = ap.parse_args()

    rows = extract(args.pdf, start_page_1idx=args.start_page)

    if args.dedupe:
        seen = set()
        deduped = []
        for r in rows:
            k = (r["entity_name"], r.get("region", ""), r.get("sector", ""))
            if k in seen:
                continue
            seen.add(k)
            deduped.append(r)
        rows = deduped

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["entity_name", "region", "sector", "source_page"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ["entity_name", "region", "sector", "source_page"]})

    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
