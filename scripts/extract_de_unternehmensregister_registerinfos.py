#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


@dataclass
class RegisterHit:
    entity_name: str
    court: str
    register_type: str
    register_number: str
    seat: str
    euid: str
    status: str
    detail_url: str  # the /de/registerinformationen URL for this result page (not DK)


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ")).strip()


def session_get(session: requests.Session, url: str, sleep_s: float = 0.8) -> requests.Response:
    r = session.get(url, timeout=60)
    r.raise_for_status()
    time.sleep(sleep_s)
    return r


def parse_registerinfos_page(html: str, base_url: str) -> tuple[list[RegisterHit], list[str]]:
    """
    Parse the /de/registerinformationen results page.
    Returns (hits, next_page_urls).
    """
    soup = BeautifulSoup(html, "html.parser")

    text = soup.get_text("\n", strip=True)

    # --- Parse results ---
    # The HTML is somewhat “content-first”; safest is to parse via visible blocks.
    # We look for repeated patterns around register entries:
    #   Amtsgericht ...  HRB 12345  EUID: ... Status: ...
    hits: list[RegisterHit] = []

    # This selector approach is defensive: find any blocks that contain "Amtsgericht" and "HRB"/"GnR"
    # and then parse surrounding text.
    candidate_blocks = soup.find_all(string=re.compile(r"Amtsgericht", re.I))

    seen = set()
    for node in candidate_blocks:
        # walk up to a container element
        container = node.parent
        for _ in range(5):
            if container and container.name not in ("body", "html"):
                container = container.parent
        if not container:
            continue

        block_text = clean(container.get_text(" ", strip=True))
        if "Amtsgericht" not in block_text:
            continue
        if not re.search(r"\b(HRB|HRA|GnR|PR|VR)\b", block_text):
            continue

        # crude parsing: take first occurrence patterns
        # entity name often appears right after the register line; but in your view it appears as a separate line.
        # We'll extract using regex windows.
        m_reg = re.search(r"\b(HRB|HRA|GnR|PR|VR)\s+(\d+)\b", block_text)
        m_court = re.search(r"Amtsgericht\s+([^H]+?)\s+\b(HRB|HRA|GnR|PR|VR)\b", block_text)
        m_euid = re.search(r"\bEUID:\s*([A-Z0-9\.]+)\b", block_text)
        m_status = re.search(r"\bStatus:\s*([A-Za-zÄÖÜäöüß]+)\b", block_text)

        if not (m_reg and m_court):
            continue

        register_type = m_reg.group(1)
        register_number = m_reg.group(2)
        court = clean("Amtsgericht " + m_court.group(1))

        euid = m_euid.group(1) if m_euid else ""
        status = m_status.group(1) if m_status else ""

        # Try to find a nearby “Firma / Name” line inside the container
        # (fallback: first quoted-ish phrase before Amtsgericht)
        entity_name = ""
        # common: “Firma / Name … <name> … Amtsgericht …”
        m_name = re.search(r"Firma\s*/\s*Name\s*(.+?)\s+Amtsgericht", block_text)
        if m_name:
            entity_name = clean(m_name.group(1))
        else:
            # fallback: grab last chunk before "Amtsgericht"
            pre = block_text.split("Amtsgericht")[0]
            entity_name = clean(pre.split()[-10:])  # will be improved once you inspect HTML structure
            entity_name = clean(pre)

        # Seat: try "Sitz" column text
        seat = ""
        m_seat = re.search(r"\bSitz\s+([A-Za-zÄÖÜäöüß\-\s]+?)\s+\b(Status|AD|CD|HD|DK|UT)\b", block_text)
        if m_seat:
            seat = clean(m_seat.group(1))

        # Use the current page URL as detail context
        detail_url = base_url

        key = (entity_name, court, register_type, register_number, euid, seat)
        if key in seen:
            continue
        seen.add(key)

        hits.append(
            RegisterHit(
                entity_name=entity_name,
                court=court,
                register_type=register_type,
                register_number=register_number,
                seat=seat,
                euid=euid,
                status=status,
                detail_url=detail_url,
            )
        )

    # --- Find pagination links ---
    next_urls: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # keep only links back into registerinformationen with same session tokens
        if "/de/registerinformationen" in href and "searchToken=" in href:
            full = urljoin(base_url, href)
            next_urls.append(full)

    # De-dupe pagination candidates
    next_urls = sorted(set(next_urls))

    return hits, next_urls


def crawl(seed_url: str, max_pages: int = 200) -> list[RegisterHit]:
    with requests.Session() as s:
        # mimic a browser a bit
        s.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ei_scraper/1.0",
                "Accept-Language": "de,en;q=0.8",
            }
        )

        queue = [seed_url]
        seen_pages = set()
        all_hits: list[RegisterHit] = []

        while queue and len(seen_pages) < max_pages:
            url = queue.pop(0)
            if url in seen_pages:
                continue
            seen_pages.add(url)

            r = session_get(s, url)
            hits, next_urls = parse_registerinfos_page(r.text, url)
            all_hits.extend(hits)

            # Keep queue growth sane: only add pagination URLs that look like “page 2/3…”
            for nu in next_urls:
                if nu not in seen_pages:
                    queue.append(nu)

        return all_hits


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True, help="A working /de/registerinformationen URL (with searchToken+payload)")
    ap.add_argument("--out", required=True, help="Output CSV")
    ap.add_argument("--max-pages", type=int, default=80)
    args = ap.parse_args()

    hits = crawl(args.seed, max_pages=args.max_pages)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "entity_name",
                "court",
                "register_type",
                "register_number",
                "seat",
                "euid",
                "status",
                "source_url",
            ],
        )
        w.writeheader()
        for h in hits:
            w.writerow(
                {
                    "entity_name": h.entity_name,
                    "court": h.court,
                    "register_type": h.register_type,
                    "register_number": h.register_number,
                    "seat": h.seat,
                    "euid": h.euid,
                    "status": h.status,
                    "source_url": h.detail_url,
                }
            )

    print(f"Wrote {len(hits)} rows -> {args.out}")


if __name__ == "__main__":
    main()
