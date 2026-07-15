from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

BASE_URL = "https://zer.bzst.de/"   # use the dedicated ZER host
OUT_DIR = Path("data/de/bzst/zuwendung/raw_gz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# shard patterns observed in your JS bundle
want = re.compile(r"compressed\.json\.gz(\?|$)", re.I)

def safe_filename(url: str) -> str:
    p = urlparse(url)
    name = Path(p.path).name
    return name or "download.gz"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # avoid headless gating
    context = browser.new_context(
        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        viewport={"width": 1280, "height": 720},
    )
    page = context.new_page()

    seen: set[str] = set()

    def on_request(req):
        u = req.url
        if want.search(u):
            seen.add(u)

    page.on("request", on_request)

    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_timeout(3_000)

    # Try to trigger loading: type a single letter into any search-like input and press Enter
    # (best-effort; if no input is found, we still might get shard requests on load)
    try:
        loc = page.locator('input[type="search"], input[placeholder*="Suche" i], input[aria-label*="Suche" i], input')
        if loc.count() > 0:
            loc.first.click(timeout=2000)
            loc.first.fill("a", timeout=2000)
            loc.first.press("Enter", timeout=2000)
    except Exception:
        pass

    # wait for background shard fetches
    page.wait_for_timeout(15_000)

    print("Shard URLs captured:", len(seen))
    if not seen:
        page.screenshot(path=str(OUT_DIR.parent / "zer_debug.png"), full_page=True)
        print("No shards captured. Saved screenshot:", OUT_DIR.parent / "zer_debug.png")
        browser.close()
        raise SystemExit(2)

    # Download shards using the browser context request (inherits cookies/session)
    # NOTE: do not use curl here; let Playwright fetch.
    n_ok = 0
    for u in sorted(seen):
        fn = safe_filename(u)
        dest = OUT_DIR / fn
        if dest.exists() and dest.stat().st_size > 0:
            continue
        resp = context.request.get(u, timeout=120_000)
        if not resp.ok:
            print("WARN:", resp.status, u)
            continue
        dest.write_bytes(resp.body())
        n_ok += 1

    print("Downloaded shard files:", n_ok, "->", OUT_DIR)

    browser.close()
