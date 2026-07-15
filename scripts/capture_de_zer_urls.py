from __future__ import annotations

import re
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

OUT = Path("data/de/bzst/zuwendung/zer_requested_urls.txt")
OUT_ALL = Path("data/de/bzst/zuwendung/zer_requested_urls_all.txt")
OUT.parent.mkdir(parents=True, exist_ok=True)

URL = "https://www.bzst.de/DE/Unternehmen/Gemeinnuetzigkeit/Zuwendungsempfaengerregister/Zuwendungsempfaengerregister_node.html"

# Capture anything that looks like the register payload/chunks
want = re.compile(
    r"(compressed\.json(\.gz)?|data/blobs/|zer-statisches-register-app/data/|\.gz(\?|$))",
    re.I,
)

seen: set[str] = set()
seen_all: set[str] = set()

def try_click_consent(page) -> None:
    # Common German consent buttons (best-effort)
    candidates = [
        "Alle akzeptieren", "Akzeptieren", "Einverstanden",
        "Zustimmen", "OK", "Alles akzeptieren"
    ]
    for text in candidates:
        try:
            loc = page.get_by_role("button", name=re.compile(rf"^{re.escape(text)}$", re.I))
            if loc.count() > 0:
                loc.first.click(timeout=1500)
                page.wait_for_timeout(500)
                return
        except Exception:
            pass

def try_trigger_search(page) -> None:
    # Try to find a search input and type something to trigger loading
    # (many SPAs lazy-load data on first search)
    selectors = [
        'input[type="search"]',
        'input[placeholder*="Suche" i]',
        'input[aria-label*="Suche" i]',
        'input[name*="search" i]',
        'input',
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if loc.count() == 0:
                continue
            el = loc.first
            el.click(timeout=1500)
            el.fill("a", timeout=1500)
            el.press("Enter", timeout=1500)
            page.wait_for_timeout(1500)
            return
        except Exception:
            continue

    # Try a "Suchen" button if no input worked
    try:
        btn = page.get_by_role("button", name=re.compile(r"Suchen|Suche", re.I))
        if btn.count() > 0:
            btn.first.click(timeout=1500)
            page.wait_for_timeout(1500)
    except Exception:
        pass

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    def on_request(req):
        u = req.url
        seen_all.add(u)
        if want.search(u):
            seen.add(u)

    page.on("request", on_request)

    # Use DOMContentLoaded (networkidle can fire too early on SPAs)
    page.goto(URL, wait_until="domcontentloaded", timeout=120_000)

    # Let scripts initialize
    page.wait_for_timeout(3_000)

    # Try to clear cookie/consent blockers
    try_click_consent(page)

    # Trigger data loading
    try_trigger_search(page)

    # Wait for background fetches
    page.wait_for_timeout(15_000)

    # One more try: scroll (some apps load on scroll)
    try:
        page.mouse.wheel(0, 3000)
        page.wait_for_timeout(5_000)
    except Exception:
        pass

    browser.close()

OUT.write_text("\n".join(sorted(seen)) + ("\n" if seen else ""), encoding="utf-8")
OUT_ALL.write_text("\n".join(sorted(seen_all)) + ("\n" if seen_all else ""), encoding="utf-8")

print("Captured payload-like URLs:", len(seen))
print("Captured ALL URLs:", len(seen_all))
print("Saved:", OUT)
print("Saved:", OUT_ALL)
