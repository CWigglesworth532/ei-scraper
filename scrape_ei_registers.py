import re
import io
import yaml
import requests
import os
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date

# ---------------------------
# Helpers: normalization
# ---------------------------

def clean_text(x):
    if x is None:
        return None
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s if s and s.lower() != "nan" else None

import re

_ES_TAX_ID_RE = re.compile(
    r"\b("
    r"\d{8}[A-Z]"                       # NIF: 12345678Z
    r"|[XYZ]\d{7}[A-Z]"                 # NIE: X1234567L
    r"|[ABCDEFGHJNPQRSUVW]\d{7}[0-9A-J]"# CIF: B12345678 / P1234567H
    r")\b",
    flags=re.IGNORECASE
)

def normalize_fr_siren(value):
    """
    Normalize SIREN to 9-digit string.
    Handles Excel floats and NaN safely.
    """
    if value is None:
        return None

    s = str(value).strip()

    if not s or s.lower() == "nan":
        return None

    # Remove .0 from Excel floats
    if s.endswith(".0"):
        s = s[:-2]

    # Keep digits only
    s = re.sub(r"\D", "", s)

    if len(s) == 9:
        return s.zfill(9)

    return None

def extract_tax_id_es(value: object) -> str:
    """Extract Spanish NIF/NIE/CIF from a noisy string. Returns '' if none."""
    if value is None:
        return ""

    s = str(value).upper().strip()
    if not s or s == "NAN":
        return ""

    # Remove common separators/spaces
    s = s.replace(" ", "").replace("-", "").replace(".", "").replace(":", "")

    m = _ES_TAX_ID_RE.search(s)
    return m.group(1).upper() if m else ""

def expand_urls(url: str, pagination: dict | None) -> list[str]:
    """
    Expand a base URL into multiple URLs when pagination is configured.
    Supports a WordPress-style /page/N/ pattern.
    """
    if not pagination:
        return [url]

    kind = str(pagination.get("kind", "")).strip().lower()
    start = int(pagination.get("start", 1))
    end = int(pagination.get("end", start))

    if kind == "wp_page":
        # Page 1 is the base URL, others are /page/N/
        urls = [url]
        base = url.rstrip("/") + "/"
        for n in range(max(2, start), end + 1):
            urls.append(f"{base}page/{n}/")
        return urls

    # Unknown pagination kind → just return base
    return [url]

def find_pdf_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if ".pdf" in href.lower():
            links.append(requests.compat.urljoin(base_url, href))
    # de-dupe preserving order
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def extract_tax_id(text):
    """
    Extract CIF/NIF from any input (safe for floats/NaN).
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return None
    m = re.search(r"\b([A-Z]\d{7}[A-Z0-9]|\d{8}[A-Z])\b", s.upper())
    return m.group(1) if m else None

def extract_tax_id_fr(text):
    if not text:
        return None
    s = str(text).strip()

    # SIRET (14 digits) preferred
    m = re.search(r"\b(\d{14})\b", s)
    if m:
        return m.group(1)

    # SIREN (9 digits)
    m = re.search(r"\b(\d{9})\b", s)
    if m:
        return m.group(1)

    return None

def derive_tax_id_root(country, tax_id):
    """
    Derive a stable 'root' identifier for linkage.

    - FR: SIRET (14) -> SIREN (9), SIREN stays SIREN
    - Other countries: return cleaned tax_id as-is
    """
    if tax_id is None:
        return None

    s = str(tax_id).strip()
    if not s or s.lower() == "nan":
        return None

    cc = str(country or "").upper()

    if cc == "FR":
        digits = re.sub(r"\D", "", s)

        if len(digits) == 14:
            return digits[:9]

        if len(digits) == 9:
            return digits

        return None

    return s

def siren_of(tax_id):
    """
    Extract SIREN (9-digit legal entity ID) from SIREN or SIRET.

    Returns:
        - 9-digit SIREN if input is valid
        - None otherwise
    """
    if not tax_id:
        return None

    s = str(tax_id).strip()

    if not s or s.lower() == "nan":
        return None

    # SIRET → SIREN
    if len(s) == 14 and s.isdigit():
        return s[:9]

    # Already SIREN
    if len(s) == 9 and s.isdigit():
        return s

    return None

def infer_legal_form(name):
    """
    Infer rough legal form token from name (safe for floats/NaN).
    """
    if name is None:
        return None
    s = str(name).strip()
    if not s or s.lower() == "nan":
        return None

    u = s.upper()
    tokens = ["S.L.U.", "S.L.", "S.A.", "S.COOP", "SCOOP", "S. COOP", "COOP"]
    for t in tokens:
        if t in u:
            return t.replace(" ", "")
    return None

import csv, io
import pandas as pd

def read_csv_flexible(content: bytes) -> pd.DataFrame:
    sample = content[:20000].decode("utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        sep = dialect.delimiter
    except Exception:
        sep = ","

    try:
        return pd.read_csv(io.BytesIO(content), sep=sep, encoding="utf-8-sig")
    except pd.errors.ParserError:
        return pd.read_csv(io.BytesIO(content), sep=sep, encoding="utf-8-sig", engine="python")


def base_family_from_legal_form(lf):
    if not lf:
        return None
    lf = str(lf).upper()
    if "COOP" in lf:
        return "COOPERATIVE"
    if "S.L" in lf or "S.A" in lf:
        return "COMPANY_LIMITED"
    return None

def _call_scraper(fn, url, ccaa, name, country):
    """
    Call scraper with country kwarg if it supports it; otherwise call without.
    Keeps backward compatibility with scrapers that don't accept country=...
    """
    try:
        return fn(url, ccaa, name, country=country)
    except TypeError:
        return fn(url, ccaa, name)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9,nl;q=0.8",
    "Connection": "keep-alive",
}

def resolve_local_file_path(path: str) -> str:
    """
    Resolve a local path robustly:
      - as provided
      - relative to the script directory
      - relative to /mnt/data (common in notebooks/sandboxes)
    """
    p = (path or "").strip()

    # Absolute path or already valid relative path
    if p and os.path.exists(p):
        return p

    # Relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(script_dir, p)
    if p and os.path.exists(cand):
        return cand

    # Common sandbox location
    cand = os.path.join("/mnt/data", p)
    if p and os.path.exists(cand):
        return cand

    # If nothing worked, return original (caller will raise a clear error)
    return p

# ---------------------------
# HTML extractor (optional; safe-fail)
# ---------------------------

def scrape_html_list(url, ccaa, register_name):
    r = requests.get(url, timeout=120, headers=DEFAULT_HEADERS)
    r.raise_for_status()

    # IMPORTANT: wrap in StringIO so pandas treats it as HTML content, not a filename
    try:
        tables = pd.read_html(io.StringIO(r.text))
    except Exception:
        tables = []

    # If no tables, return empty (no crash, no junk scraping)
    if not tables:
        return pd.DataFrame(columns=[
            "country","ccaa","ei_register_name","ei_registration_number","entity_name","tax_id",
            "legal_form_local","source_url","source_type","retrieved_at"
        ])

    # (keep your existing table-to-rows mapping below, or this simple default)
    df_all = pd.concat(tables, ignore_index=True)
    df_all.columns = [str(c).strip().lower() for c in df_all.columns]

    name_col = next((c for c in df_all.columns if c in [
        "denominación","denominacion","razón social","razon social","nombre","empresa","entidad"
    ]), df_all.columns[0])

    out = []
    for _, rec in df_all.iterrows():
        name = clean_text(rec.get(name_col))
        row_text = " ".join([str(v) for v in rec.to_dict().values()])
        tax = extract_tax_id(row_text) or extract_tax_id(name)

        out.append({
            "country": "ES",
            "ccaa": ccaa,
            "ei_register_name": register_name,
            "ei_registration_number": None,
            "entity_name": name,
            "tax_id": tax,
            "legal_form_local": infer_legal_form(name),
            "source_url": url,
            "source_type": "HTML",
            "retrieved_at": str(date.today())
        })

    return pd.DataFrame(out)

import re
import pandas as pd
import requests
from bs4 import BeautifulSoup


_DE_POSTCODE_CITY_LINE_RE = re.compile(
    r"(?P<prefix>.*?)(?P<postcode>\b\d{5}\b)\s+(?P<city>[A-Za-zÀ-ÿ\-\.\s/]+)",
    flags=re.UNICODE,
)


def scrape_bag_if_accessible_list(url, ccaa, register_name, country=None):
    """
    Germany (DE) — bag if accessible company list (postcode-split parser).
    Works even if the HTML is one huge text blob.
    """

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    content = soup.select_one(".entry-content") or soup.body
    if content is None:
        return pd.DataFrame()

    raw = content.get_text("\n", strip=False)
    raw = raw.replace("\r", "")
    # Collapse whitespace but keep newlines (we use them as soft separators)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{2,}", "\n", raw)

    # Split into chunks whenever a postcode appears (keep the postcode by using lookahead)
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]

    rows = []
    pending_name = None
    pending_street = None

    for ln in lines:
        t = ln.strip()

        # Skip headings / noise
        if len(t) < 3 or t.isupper():
            continue

        # Website
        if t.startswith("http") or "www." in t:
            if rows:
                rows[-1]["website"] = t
            continue

        # PLZ + city → finalize record (either "68199 Mannheim" or "... 68199 Mannheim")
        m = _DE_POSTCODE_CITY_LINE_RE.search(t)
        if m:
            postcode = m.group("postcode")
            city = (m.group("city") or "").strip()

            # If the line also contains a street/address prefix, keep it
            prefix = (m.group("prefix") or "").strip(" –-").strip()

            street = pending_street
            if prefix and any(ch.isdigit() for ch in prefix):
                street = prefix

            if pending_name and street:
                rows.append({
                    "entity_name": pending_name.strip(),
                    "address": street.strip(),
                    "postcode": postcode,
                    "city": city,
                })

            pending_name = None
            pending_street = None
            continue

        # Street line (has digits, no PLZ)
        if any(ch.isdigit() for ch in t):
            pending_street = t.rstrip("–- ").strip()
            continue

        # Otherwise: company name
        pending_name = t

    df = pd.DataFrame(rows)

    # Drop obvious duplicates (same name+postcode+city)
    if not df.empty and all(c in df.columns for c in ["entity_name", "postcode", "city"]):
        df = df.drop_duplicates(subset=["entity_name", "postcode", "city"], keep="first")

    if df.empty:
        return df

    df["country"] = country or "DE"
    df["ccaa"] = ccaa
    df["ei_register_name"] = register_name
    return df

import json
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

def scrape_kadence_query_loop(url, ccaa, register_name, country=None):
    """
    Scrape Kadence Query Loop results via WP REST endpoint exposed in page JS:
      kbp_query_loop_rest_endpoint = {"url": ".../wp-json/wp/v2/kadence_query/query", "nonce": "..."}
    Returns a DataFrame with entity_name + source_url (+ description if available).
    """

    # 1) Fetch the page to discover endpoint + nonce
    page_headers = dict(DEFAULT_HEADERS)
    page_headers["Referer"] = "https://codesocialeondernemingen.nl/"
    r = requests.get(url, timeout=60, headers=page_headers)
    r.raise_for_status()
    html = r.text

    m = re.search(r"kbp_query_loop_rest_endpoint\s*=\s*(\{.*?\});", html, flags=re.DOTALL)
    if not m:
        raise ValueError("Could not find kbp_query_loop_rest_endpoint in page HTML")

    # Parse the JS object safely (it is JSON-ish already)
    endpoint_obj = json.loads(m.group(1))
    endpoint_url = endpoint_obj["url"]
    nonce = endpoint_obj["nonce"]

    api_headers = {
        **DEFAULT_HEADERS,
        "Referer": "https://codesocialeondernemingen.nl/",
        "X-WP-Nonce": nonce,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # 2) Try a few likely Kadence payload shapes until one works
    def try_payload(payload):
        rr = requests.post(endpoint_url, headers=api_headers, json=payload, timeout=60)
        rr.raise_for_status()
        return rr

    working_mode = None
    page = 1
    first_response = None

    # These variants cover the common Kadence Blocks query endpoint implementations.
    candidates = [
        lambda p: {"page": p},
        lambda p: {"paged": p},
        lambda p: {"page": p, "per_page": 100},
        lambda p: {"paged": p, "per_page": 100},
        # Some versions wrap args:
        lambda p: {"args": {"paged": p}},
        lambda p: {"args": {"page": p}},
        lambda p: {"query": {"paged": p}},
        lambda p: {"query": {"page": p}},
    ]

    for make in candidates:
        try:
            resp = try_payload(make(1))
            # Must be JSON; and must contain something non-empty
            j = resp.json()
            if j:
                working_mode = make
                first_response = j
                break
        except Exception:
            continue

    if working_mode is None:
        raise ValueError("Kadence query endpoint reachable but no payload variant returned usable JSON")

    # 3) Extract items from response; supports either:
    # - list of posts objects
    # - dict with 'html' fragment
    # - dict with 'data'/'items' etc
    def extract_rows_from_json(j):
        rows = []

        # Case A: list of post-like objects
        if isinstance(j, list):
            for it in j:
                if not isinstance(it, dict):
                    continue
                title = None
                if isinstance(it.get("title"), dict):
                    title = it["title"].get("rendered")
                title = title or it.get("title") or it.get("name")
                link = it.get("link") or it.get("permalink") or it.get("url")
                if title:
                    rows.append({
                        "entity_name": BeautifulSoup(str(title), "html.parser").get_text(" ", strip=True),
                        "source_url": link or url,
                        "description": None,
                    })
            return rows

        # Case B: dict with HTML fragment
        if isinstance(j, dict):
            html_frag = j.get("html") or (j.get("data") or {}).get("html")
            if html_frag and isinstance(html_frag, str):
                soup = BeautifulSoup(html_frag, "html.parser")
                # Try common card patterns
                cards = soup.select(".kb-query-item, article, .hentry, .entry")
                for c in cards:
                    a = c.select_one("a[href]")
                    title = a.get_text(" ", strip=True) if a else None
                    href = a.get("href") if a else None
                    excerpt = None
                    p = c.select_one("p")
                    if p:
                        excerpt = p.get_text(" ", strip=True)
                    if title:
                        rows.append({"entity_name": title, "source_url": href or url, "description": excerpt})
                return rows

            # Case C: dict with items array
            for key in ("items", "posts", "results", "data"):
                v = j.get(key)
                if isinstance(v, list):
                    return extract_rows_from_json(v)

        return rows

    all_rows = []
    # consume first page
    all_rows.extend(extract_rows_from_json(first_response))

    # 4) Page until empty
    page = 2
    while True:
        resp = try_payload(working_mode(page))
        j = resp.json()
        chunk = extract_rows_from_json(j)
        if not chunk:
            break
        all_rows.extend(chunk)
        page += 1
        if page > 200:
            break

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    # Stamp metadata expected by your pipeline
    df["country"] = country or ""
    df["ccaa"] = ccaa
    df["ei_register_name"] = register_name
    return df

def scrape_dib_integrative_betriebe(url, ccaa, register_name, country=None):
    r = requests.get(url, timeout=60, headers=DEFAULT_HEADERS)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    content = soup.select_one("main") or soup.select_one("body")
    if content is None:
        return pd.DataFrame()

    # Line-oriented extraction works well on this page
    raw = content.get_text("\n", strip=True)
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]

    # AT postcodes are 4 digits
    plz_city_re = re.compile(r"^\d{4}\s+.+")
    phone_re = re.compile(r"^\+?\d[\d\s()/.-]{6,}$")

    legal_name_re = re.compile(
        r"\b(gGmbH|GmbH|Ges\.?\s*m\.?\s*b\.?\s*H|GmbH\s*&\s*Co\.?\s*KG|KG)\b",
        re.IGNORECASE,
    )

    dib_allow_re = re.compile(r"\b(wien\s*work|teamwork)\b", re.IGNORECASE)

    rows = []
    region = None
    pending_name = None
    pending_address = None

    AT_STATES = {
        "Wien",
        "Niederösterreich",
        "Oberösterreich",
        "Steiermark",
        "Salzburg",
        "Tirol",
        "Vorarlberg",
        "Kärnten",
        "Burgenland",
    }

    for ln in lines:
        t = ln.strip()
        low = t.lower()

        # Skip opening hours / generic page text
        if low.startswith(("mo-", "mo ", "di-", "mi-", "do-", "fr-", "sa-", "so-")) or "uhr" in low:
            continue

        # Skip question/CTA fragments
        if t.endswith("?"):
            continue

        # Skip obvious CTAs / headings / non-entities
        low = t.lower()
        if low in {"kontakt", "sie haben interesse"} or low.startswith(("sie haben", "interesse", "partner", "bundesland")):
            continue

        # Bundesland heading
        if t in AT_STATES:
            region = t
            continue

        # Likely entity name
        if pending_name is None and not plz_city_re.match(t) and "@" not in t and not t.startswith("http"):

            # Skip contact labels / UI text
            if low.startswith(("fax", "tel", "telefon", "phone", "schreiben", "kontakt")):
                continue
            if phone_re.match(t) or "uhr" in low:
                continue
            if t.endswith(":") or t.endswith(".:") or t.endswith("?"):
                continue

            # diB entries are legal entities (anchor on legal suffix)
            if not (legal_name_re.search(t) or dib_allow_re.search(t)):
                continue

            pending_name = t
            continue

        # Street address line
        if pending_name and pending_address is None and not phone_re.match(t) and any(ch.isdigit() for ch in t) and not plz_city_re.match(t):
            pending_address = t.rstrip("–- ").strip()
            continue

        # Postcode + city finalizes a record
        if pending_name and plz_city_re.match(t):
            postcode = t.split(" ", 1)[0]
            city = t.split(" ", 1)[1] if " " in t else None

            rows.append({
                "entity_name": pending_name,
                "address": pending_address,
                "postcode": postcode,
                "city": city,
                "region": region,
            })

            pending_name = None
            pending_address = None
            continue

        # Attach contact fields to last completed record
        if rows:
            if "@" in t and " " not in t:
                rows[-1]["email"] = t
            elif t.startswith("http") or "www." in t:
                rows[-1]["website"] = t
            elif phone_re.match(t):
                rows[-1]["phone"] = t

    return pd.DataFrame(rows)

def scrape_sena_verified_social_enterprises(url, ccaa, register_name, country=None):
    r = requests.get(url, timeout=60, headers=DEFAULT_HEADERS)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Grab links that look like VSE profile links (best-effort)
    rows = []
    seen = set()

    # Prefer main content area if present
    content = soup.select_one("main") or soup.select_one("body")
    if content is None:
        return pd.DataFrame()

    # 1) Use anchor text as entity name if it’s not boilerplate
    for a in content.select("a[href]"):
        name = a.get_text(" ", strip=True)
        href = (a.get("href") or "").strip()

        if not name or len(name) < 3:
            continue
        if name.lower().startswith(("mehr", "read", "learn", "kontakt", "impressum", "datenschutz")):
            continue
        if href.startswith("#") or href.lower().startswith("mailto:") or href.lower().startswith("tel:"):
            continue

        # Normalize absolute URL
        if href.startswith("/"):
            href = url.rstrip("/") + href

        # Drop photo credits / captions
        if "(c)" in name.lower() or "©" in name:
            continue

        key = (name, href)
        if key in seen:
            continue
        seen.add(key)

        rows.append({
            "entity_name": name,
            "website": href if href.startswith("http") or "www." in href else None,
        })

    # 2) If anchors are too noisy, fall back to headings in the page text
    if len(rows) < 5:
        text = content.get_text("\n", strip=True)
        for ln in [x.strip() for x in text.split("\n") if x.strip()]:

            # Strip photo credits / captions
            if "(c)" in ln.lower():
                ln = ln.split("(c)", 1)[0].strip()
            if "©" in ln:
                ln = ln.split("©", 1)[0].strip()
            if not ln:
                continue

            if 3 <= len(ln) <= 80 and ln[0].isalpha():
                if ln.lower().startswith(("erfahre", "verified", "social", "enterprise")):
                    continue
                if ln in seen:
                    continue

                # Keep a conservative set: Title-case or mixed-case tokens
                if any(ch.islower() for ch in ln) and any(ch.isupper() for ch in ln):
                    rows.append({"entity_name": ln})
                    seen.add(ln)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["se_recognition_type"] = "Label"
    df["se_recognition_name"] = "Verified Social Enterprise (SENA)"

    return df


# ---------------------------
# XLSX extractor
# ---------------------------

def scrape_xlsx(url, ccaa, register_name, country=None):

    # --------------------------------
    # Local file support (file:...)
    # --------------------------------

    if url.startswith("file:"):
        path = url.replace("file:", "", 1).strip()
        path = resolve_local_file_path(path)
        df = pd.read_excel(path)

    else:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content))

    # Normalize headers
    def _clean_header(h):
        h = str(h)
        h = " ".join(h.split())   # collapses whitespace incl. \r\n, tabs, double spaces
        h = h.strip().lower()
        return h

    df.columns = [_clean_header(c) for c in df.columns]

    if str(country or "").upper() == "IT":
        print("RUNTS XLSX columns:", df.columns.tolist()[:15])

    # --------------------------------
    # Load XLSX (supports file:... and http(s)://...)
    # --------------------------------
    if url.startswith("file:"):
        path = url.replace("file:", "", 1).strip()

        with open(path, "rb") as f:
            content = f.read()

    else:
        r = requests.get(url, timeout=60, headers=DEFAULT_HEADERS)
        r.raise_for_status()

        content = r.content

    df = pd.read_excel(io.BytesIO(content))

    # --------------------------------
    # Normalize headers
    # --------------------------------
    df.columns = [str(c).strip().lower() for c in df.columns]


    # --------------------------------
    # France — ESUS special handling
    # --------------------------------
    if country == "FR" and "siren" in df.columns:

        out = []

        for i, rec in df.iterrows():

            name = clean_text(rec.get("raison sociale"))
            siren = normalize_fr_siren(rec.get("siren"))

            if not name or not siren:
                continue

            out.append({
                "country": "FR",
                "ccaa": None,

                "ei_register_name": register_name,
                "ei_registration_number": None,

                "entity_name": name,
                "tax_id": siren,

                "legal_form_local": clean_text(
                    rec.get("statut juridique de l'entreprise")
                ),

                "address": None,
                "postcode": clean_text(rec.get("code postal")),
                "city": clean_text(rec.get("commune")),

                "source_url": url,
                "source_type": "XLSX",
                "retrieved_at": str(date.today()),
            })

        return pd.DataFrame(out)

    # Default: return raw dataframe so apply_mapping() can handle non-ES sources (e.g. RUNTS)
    return df


    # --------------------------------
    # Default XLSX (existing logic)
    # --------------------------------

    name_col = next(
        (
            c for c in df.columns
            if c in [
                "denominación",
                "denominacion",
                "nombre",
                "empresa",
                "razón social",
                "razon social",
            ]
        ),
        df.columns[0],
    )

    out = []

    for _, rec in df.iterrows():

        name = clean_text(rec.get(name_col))

        # Look for explicit CIF/NIF columns if present
        tax = None

        for c in df.columns:
            if "cif" in c or "nif" in c:
                tax = clean_text(rec.get(c))
                break

        if not tax:
            tax = extract_tax_id(name)

        out.append({
            "country": "ES",
            "ccaa": ccaa,

            "ei_register_name": register_name,
            "ei_registration_number": None,

            "entity_name": name,
            "tax_id": tax,

            "legal_form_local": infer_legal_form(name),

            "source_url": url,
            "source_type": "XLSX",
            "retrieved_at": str(date.today()),
        })

    return pd.DataFrame(out)

import csv
import io

def scrape_csv(url, ccaa, register_name):
    if url.startswith("file:"):
        path = url.replace("file:", "", 1).strip()

        with open(path, "rb") as f:
            content = f.read()

    else:
        r = requests.get(url, timeout=60)
        r.raise_for_status()

        content = r.content

    # --- delimiter sniffing (comma / semicolon / tab / pipe) ---
    sample = content[:20000].decode("utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        sep = dialect.delimiter
    except Exception:
        sep = ","  # safe fallback

    # --- robust CSV read ---
    try:
        df = pd.read_csv(
            io.BytesIO(content),
            sep=sep,
            encoding="utf-8-sig",
            dtype=str
        )
    except pd.errors.ParserError:
        # fallback for messy quoting
        df = pd.read_csv(
            io.BytesIO(content),
            sep=sep,
            encoding="utf-8-sig",
            dtype=str,
            engine="python"
        )

    return df

    # Guess separator
    sample = text[:2000]
    sep = ";" if sample.count(";") > sample.count(",") else ","
    df = read_csv_flexible(content)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # ... keep the rest of your existing scrape_csv logic below ...

    # Common name columns (Catalunya/JCyL often have something like this)
    name_col = next((c for c in df.columns if c in [
        "denominació", "denominacion", "denominación", "raó social", "razon social",
        "razón social", "nom", "nombre", "empresa"
    ]), df.columns[0])

    out = []
    for _, rec in df.iterrows():
        name = clean_text(rec.get(name_col))
        # try CIF/NIF columns, else extract from any text
        tax = None
        for c in df.columns:
            if "cif" in c or "nif" in c:
                tax = clean_text(rec.get(c))
                break
        if not tax:
            tax = extract_tax_id(name)

        out.append({
            "country": "ES",
            "ccaa": ccaa,
            "ei_register_name": register_name,
            "ei_registration_number": clean_text(rec.get("numero_registre") or rec.get("número_registro") or rec.get("registro") or rec.get("num_registro")),
            "entity_name": name,
            "tax_id": tax,
            "legal_form_local": infer_legal_form(name),
            "source_url": url,
            "source_type": "CSV",
            "retrieved_at": str(date.today()),
        })

    return pd.DataFrame(out)


# ---------------------------
# PDF extractor
# ---------------------------

def scrape_pdf(url, ccaa, register_name):
    """
    Table-first PDF extraction:
    1) Try extracting tables with pdfplumber.extract_tables()
    2) Normalize row cells and map likely columns
    3) Fallback to regex patterns (Andalucía + Lanbide styles) if tables not found
    """
    import pdfplumber

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    def norm_cell(x):
        x = clean_text(x)
        if x:
            # Remove repeated header artifacts and weird bullets
            x = x.replace("•", " ").strip()
            x = re.sub(r"\s+", " ", x).strip()
        return x

    def looks_like_header_row(row):
        joined = " ".join([c.lower() for c in row if c])
        header_terms = ["denomin", "razón", "razon", "cif", "nif", "registro", "ei", "domic", "provincia", "localidad"]
        return any(t in joined for t in header_terms)

    def is_mostly_empty(row):
        nonempty = [c for c in row if c]
        return len(nonempty) <= 1

    # -----------------------
    # 1) Table extraction
    # -----------------------
    extracted_rows = []
    with pdfplumber.open(io.BytesIO(r.content)) as pdf:
        for page in pdf.pages:
            # Try to extract multiple tables per page
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []

            for table in tables:
                for raw_row in table:
                    if not raw_row:
                        continue
                    row = [norm_cell(c) for c in raw_row]
                    if is_mostly_empty(row):
                        continue
                    extracted_rows.append(row)

    # Attempt to interpret extracted tables if we got any rows
    out = []
    if extracted_rows:
        # Find and skip header rows, then map likely columns
        for row in extracted_rows:
            if looks_like_header_row(row):
                continue

            # Heuristics: pick EI reg no, CIF/NIF, name from row cells
            #  - EI number: cell containing "EI/" or starting "EI-"
            #  - Tax id: matches CIF/NIF pattern
            #  - Name: longest text cell that isn't the tax id or EI code
            ei_no = None
            tax_id = None

            for c in row:
                if not c:
                    continue
                if (("EI/" in c) or c.startswith("EI-")) and ei_no is None:
                    ei_no = c
                if tax_id is None:
                    cand = extract_tax_id(c)
                    if cand:
                        tax_id = cand

            # Candidate name: choose best remaining cell
            candidates = []
            for c in row:
                if not c:
                    continue
                if ei_no and c == ei_no:
                    continue
                if tax_id and tax_id in c:
                    continue
                # Ignore very short tokens
                if len(c) < 3:
                    continue
                candidates.append(c)

            # Prefer the longest candidate as "name"
            name = max(candidates, key=len) if candidates else None
            if name:
                name = name.rstrip(",").strip()

            # If we still have nothing meaningful, skip
            if not any([ei_no, tax_id, name]):
                continue

            out.append({
                "country": "ES",
                "ccaa": ccaa,
                "ei_register_name": register_name,
                "ei_registration_number": clean_text(ei_no),
                "entity_name": clean_text(name),
                "tax_id": clean_text(tax_id),
                "legal_form_local": infer_legal_form(name),
                "source_url": url,
                "source_type": "PDF",
                "retrieved_at": str(date.today()),
            })

        df = pd.DataFrame(out)
        # If the table extraction worked, return it even if incomplete;
        # downstream dedupe/merge can improve it.
        if not df.empty:
            return df

    # -----------------------
    # 2) Fallback: regex extraction
    # -----------------------
    rows = []
    with pdfplumber.open(io.BytesIO(r.content)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text)

            # Pattern A (Andalucía-like): EI/005/2011 CIF NAME ...
            for m in re.finditer(
                r"\b(EI[/\-]\d+[/\-]\d{4})\b\s+([A-Z]\d{7}[A-Z0-9])\s+(.+?)\s",
                text
            ):
                ei_no, cif, name = m.groups()
                rows.append((ei_no, cif, name))

            # Pattern B (Lanbide-like): EI-2007/040/1 NAME CIF ...
            for m in re.finditer(
                r"\b(EI-\d{4}/\d{3}/\d)\b\s+(.+?)\s+([A-Z]\d{7}[A-Z0-9])\b",
                text
            ):
                ei_no, name, cif = m.groups()
                rows.append((ei_no, cif, name))

    out = []
    for ei_no, cif, name in rows:
        name = clean_text(name)
        if name:
            name = name.rstrip(",").strip()
        out.append({
            "country": "ES",
            "ccaa": ccaa,
            "ei_register_name": register_name,
            "ei_registration_number": clean_text(ei_no),
            "entity_name": name,
            "tax_id": clean_text(cif),
            "legal_form_local": infer_legal_form(name),
            "source_url": url,
            "source_type": "PDF",
            "retrieved_at": str(date.today()),
        })

    return pd.DataFrame(out)

# ---------------------------
# Pipeline helpers
# ---------------------------

def apply_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Apply a per-source column mapping from sources.yaml.

    mapping example:
      entity_name: ["denominazione", "ragione_sociale"]
      tax_id: ["piva", "partita_iva", "cf", "codice_fiscale"]
      ei_registration_number: ["numero_iscrizione"]
      legal_form_local: ["forma_giuridica"]
    """
    if df is None or df.empty or not mapping:
        return df

    def norm_key(s):
        s = str(s)
        s = " ".join(s.split())
        s = s.strip().lower()
        return s

    # Map normalized column names -> real column names
    norm_cols = {norm_key(c): c for c in df.columns}

    out = pd.DataFrame(index=df.index)

    for target_col, candidates in mapping.items():

        if candidates is None:
            continue

        if not isinstance(candidates, list):
            candidates = [candidates]

        selected = None

        for cand in candidates:
            key = norm_key(cand)

            if key in norm_cols:
                selected = norm_cols[key]
                break

        if selected is None:
            continue

        out[target_col] = df[selected]

    return out

import re
import unicodedata
from unidecode import unidecode

LEGAL_FORMS = [
    r"sociedad\s+cooperativa",
    r"cooperativa",
    r"s\.?\s*c\.?\s*a\.?",
    r"s\.?\s*l\.?",
    r"sll",
    r"s\.?\s*a\.?",
    r"sociedad\s+limitada",
    r"sociedad\s+anonima",
    r"sociedad\s+laboral",
    r"sca",
    r"sat"
]

STOPWORDS = [
    "grupo", "corporacion", "holding", "servicios",
    "integral", "general"
]

LEGAL_RE = re.compile(r"\b(" + "|".join(LEGAL_FORMS) + r")\b", re.I)
STOP_RE = re.compile(r"\b(" + "|".join(STOPWORDS) + r")\b", re.I)


def normalize_name(name: str) -> str:
    if not name:
        return ""

    # lowercase + remove accents
    s = unidecode(name.lower())

    # remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)

    # remove legal forms
    s = LEGAL_RE.sub(" ", s)

    # remove stopwords
    s = STOP_RE.sub(" ", s)

    # collapse spaces
    s = re.sub(r"\s+", " ", s)

    return s.strip()


def extract_tax_id_it(text):
    """
    Italy:
      - Partita IVA: 11 digits
      - Codice fiscale: 16 alphanumeric (often for persons; sometimes entities)
    """
    if text is None:
        return None

    s = str(text).strip().upper()
    if not s or s.lower() == "nan":
        return None

    m = re.search(r"\b\d{11}\b", s)
    if m:
        return m.group(0)

    m = re.search(r"\b[A-Z0-9]{16}\b", s)
    if m:
        return m.group(0)

    return None


# -------------------------
# Country → tax ID extractors
# -------------------------
TAX_ID_EXTRACTORS = {
    "IT": extract_tax_id_it,
    "ES": extract_tax_id_es,
}


# ---------------------------
# Main pipeline
# ---------------------------

def run_pipeline(
    config_path: str = "sources.yaml",
    out_path: str = "ei_registers_normalized.csv",
    report_path: str = "run_report.csv",
):
    run_date = date.today().isoformat()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    sources = cfg.get("sources", [])
    print(f"Loading sources from: {config_path}")
    print(f"Found {len(sources)} sources\n")

    frames = []
    report_rows = []

    for i, s in enumerate(sources, start=1):
        if not s.get("enabled", True):
            continue

        country = str(s.get("country", "")).upper()
        ccaa = s.get("ccaa")
        stype = s.get("type")
        name = (s.get("ei_register_name") or s.get("register_name") or s.get("name") or "").strip()
        url = s.get("url")
        mapping = s.get("mapping", {})

        print(f"[{i}] {ccaa or country} | {stype} | {name}")
        print(f"    {url}")

        status = "OK"
        err = ""
        rows = 0

        try:
            if stype == "HTML":
                parser_name = s.get("parser")
                if parser_name:
                    html_fn = globals().get(parser_name)
                    if html_fn is None:
                        raise ValueError(f"Unknown HTML parser: {parser_name}")
                else:
                    html_fn = scrape_html_list
                print(f"    HTML parser: {html_fn.__name__}")

                pages = expand_urls(url, s.get("pagination"))
                parts = []

                for u in pages:
                    part = _call_scraper(
                        html_fn,
                        u,
                        ccaa,
                        name,
                        country,
                    )
                    if part is not None and not part.empty:
                        parts.append(part)

                df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

                # OPTIONAL: follow linked PDFs and ingest them too
                if s.get("follow_pdf_links"):
                    r = requests.get(url, timeout=60, headers=DEFAULT_HEADERS)
                    r.raise_for_status()
                    pdf_urls = find_pdf_links(r.text, url)

                    pdf_parts = []
                    for pu in pdf_urls:
                        part = _call_scraper(scrape_pdf, pu, ccaa, name, country)
                        if part is not None and not part.empty:
                            pdf_parts.append(part)

                    if pdf_parts:
                        df = pd.concat([df] + pdf_parts, ignore_index=True)

            elif stype == "XLSX":
                df = _call_scraper(scrape_xlsx, url, ccaa, name, country)

            elif stype == "PDF":
                df = _call_scraper(scrape_pdf, url, ccaa, name, country)

            elif stype == "CSV":
                df = _call_scraper(scrape_csv, url, ccaa, name, country)

            else:
                raise ValueError(f"Unknown source type: {stype}")

            if df is None or df.empty:
                rows = 0
            else:
                # Apply mapping
                df = apply_mapping(df, mapping)

            # Derived identity field for no-tax-id matching
            if df is not None and not df.empty:
                if "entity_name" in df.columns:
                    df["entity_name_norm"] = df["entity_name"].apply(
                        lambda x: normalize_name(clean_text(x) or "")
                    )
                else:
                    df["entity_name_norm"] = None

            # Canonical metadata (fill only if missing or entirely empty)
            for meta_col, meta_val in {
                "country": country,
                "ccaa": ccaa,
                "ei_register_name": name,
                "source_url": url,
                "source_type": stype,
                "retrieved_at": run_date,
            }.items():
                if meta_col not in df.columns or df[meta_col].isna().all():
                    df[meta_col] = meta_val

            # Ensure optional fields exist
            for opt_col in [
                "ei_registration_number",
                "legal_form_local",
                "base_legal_form_code",
                "base_legal_form_family",
                "se_recognition_type",
                "se_recognition_name",
                "se_recognition_evidence",
            ]:
                if opt_col not in df.columns:
                    df[opt_col] = None

            # Ensure tax_id exists
            if "tax_id" not in df.columns:
                df["tax_id"] = None

            # Country-specific tax ID normalization
            fn = TAX_ID_EXTRACTORS.get(country)
            if fn is not None and df["tax_id"].notna().any():
                df["tax_id"] = df["tax_id"].apply(fn)

            rows = len(df)
            frames.append(df)


            print(f"    Extracted rows: {rows}")

        except Exception as e:
            status = "ERROR"
            err = f"{type(e).__name__}: {e}"
            print(f"    ERROR: {err}")

        report_rows.append({
            "run_date": run_date,
            "source_index": i,
            "country": country,
            "ccaa": ccaa,
            "source_type": stype,
            "ei_register_name": name,
            "source_url": url,
            "status": status,
            "rows_extracted": rows,
            "error": err,
        })

    # Write run report
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_path, index=False, encoding="utf-8")
    print(f"\nSaved: {report_path}")

    # Write normalized output
    if frames:
        final = pd.concat(frames, ignore_index=True)
    else:
        final = pd.DataFrame()

    base_cols = [
        "country",
        "ccaa",
        "address",
        "postcode",
        "city",
        "province",
        "ei_register_name",
        "ei_registration_number",
        "entity_name",
        "entity_name_norm",
        "tax_id",
        "legal_form_local",
        "base_legal_form_code",
        "base_legal_form_family",
        "se_recognition_type",
        "se_recognition_name",
        "se_recognition_evidence",
        "source_url",
        "source_type",
        "retrieved_at",
    ]

    for c in base_cols:
        if c not in final.columns:
            final[c] = None

    final = final[base_cols]

    # Add linkage-friendly root tax ID (especially useful for FR SIREN/SIRET)
    final["tax_id_root"] = final.apply(
        lambda r: derive_tax_id_root(r.get("country"), r.get("tax_id")),
        axis=1
    )

    # Ensure ID columns are written as strings (avoid float .0 on reload)
    for c in ["tax_id", "tax_id_root"]:
        if c in final.columns:
            final[c] = final[c].astype("string")

    # PRE-WRITE CHECK
    print(f"FINAL_ROWS_BEFORE_WRITE={len(final)}")

    final.to_csv(out_path, index=False, encoding="utf-8")

    # POST-WRITE CHECK (count lines on disk)
    with open(out_path, "rb") as f:
        line_count = sum(1 for _ in f)
    print(f"FINAL_FILE_LINES={line_count} (data_rows={max(0, line_count - 1)})")

    print(f"Saved: {out_path} ({len(final)} rows)")

if __name__ == "__main__":
    run_pipeline()
