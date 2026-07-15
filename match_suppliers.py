#!/usr/bin/env python3
# match_suppliers.py

import re
import pandas as pd
from unidecode import unidecode
from rapidfuzz import process, fuzz


# Expand this over time as you see patterns
LEGAL_SUFFIXES = [
    "sarl", "sas", "sa", "sasu", "scop", "scic", "eurl", "gmbh", "ug", "ag", "kg", "eg",
    "srl", "spa", "soc coop", "cooperativa", "sl", "s l", "sll", "cee",
    "asbl", "vzw", "aps", "onlus", "ets", "ev", "e v",
    "foundation", "association", "associazione"
]

LEGAL_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(map(re.escape, LEGAL_SUFFIXES)) + r")\b",
    re.I
)

# -----------------------------
# COUNTRY-AWARE NAME HEURISTICS
# -----------------------------

def _rx(patterns):
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]

NAME_HEURISTICS = {

    # -----------------
    # GERMANY
    # -----------------
    "DE": {
        "coop": _rx([
            r"\b(e\.?\s*g\.?)\b",
            r"\beingetragene\s+genossenschaft\b",
            r"\bgenossenschaft\b",
        ]),
        "ggmbh": _rx([
            r"\bggmbh\b",
            r"\bgemeinn[üu]tzige\s+gmbh\b",
            r"\bgemeinnuetzige\s+gmbh\b",
        ]),
    },

    # -----------------
    # NETHERLANDS
    # -----------------
    "NL": {
        "coop": _rx([
            r"\bco[öo]peratie\b",
            r"\bco[öo]peratief\b",
            r"\bco[öo]peratie\s*(?:u\.?\s*a\.?|b\.?\s*a\.?|w\.?\s*a\.?)\b",
            r"\b(?:u\.?\s*a\.?|b\.?\s*a\.?|w\.?\s*a\.?)\b(?=\s*$)",
        ]),
        "marker": _rx([
            r"\bANBI\b",
        ]),
        "foundation_assoc": _rx([
            r"\bstichting\b",
            r"\bvereniging\b",
        ]),
    },

    # -----------------
    # BELGIUM
    # -----------------
    "BE": {
        "coop": _rx([
            r"\bco[öo]perat(?:ieve|ief)\b",
            r"\b(?:soci[ée]t[ée]\s+coop[ée]rative|vennootschap\s+met\s+co[öo]peratief)\b",
            r"(?:^|[\s,;\(\[])\s*(?:CVBA|SCRL)\b(?=\s*$)",
            r"(?:^|[\s,;\(\[])\s*(?:CV|SC)\b(?=\s*$)",
        ]),
        "marker": _rx([
            r"\b(?:VZW|ASBL)\b",
            r"\b(?:IVZW|AISBL)\b",
            r"\bVoG\b",
        ]),
        "foundation_assoc": _rx([
            r"\b(?:stichting|fondation)\b",
            r"(?:^|[\s,;\(\[])\s*(?:FP|PS|FUP|SON)\b(?=\s*$)",
        ]),
    },

    # -----------------
    # FRANCE
    # -----------------
    "FR": {
        "coop": _rx([
            r"\bSCOP\b",
            r"\bSCIC\b",
            r"\bsoci[ée]t[ée]\s+coop[ée]rative\b",
        ]),
        "esus": _rx([
            r"\bESUS\b",
        ]),
        "marker": _rx([
            r"\bassociation\b",
            r"\bassociation\s+loi\s+1901\b",
            r"\bloi\s+1901\b",
            r"\bfondation\b",
            r"\bmutuelle\b",
        ]),
    },

    # -----------------
    # ITALY
    # -----------------
    "IT": {
        "coop": _rx([
            r"\bcooperativa\b",
            r"\bsociet[àa]\s+cooperativa\b",
            r"\bcooperativa\s+sociale\b",
            r"\b(?:soc\.?\s*)?coop\.?\b",
        ]),
        "impresa_sociale": _rx([
            r"\bimpresa\s+sociale\b",
        ]),
        "marker": _rx([
            r"\bONLUS\b",
            r"(?:^|[\s,;\(\[])\s*(?:ODV|APS)\b(?=\s*$)",
        ]),
        "foundation_assoc": _rx([
            r"\bfondazione\b",
            r"\bassociazione\b",
        ]),
    },

    # -----------------
    # SPAIN (NO sociedad laboral)
    # -----------------
    "ES": {
        "coop": _rx([
            r"\bsociedad\s+cooperativa\b",
            r"\bS\.\s*Coop\.?\b",
            r"\bS\.?\s*Coop\.\b",
            r"\bcooperativa\b",
        ]),
        "marker": _rx([
            r"\basociaci[óo]n\b",
            r"\bfundaci[óo]n\b",
            r"\bONGD?\b",
        ]),
    },

    # -----------------
    # PORTUGAL
    # -----------------
    "PT": {
        "coop": _rx([
            r"\bcooperativa\b",
            r"\bcooperativa\s+de\s+responsabilidade\s+limitada\b",
            r"(?:^|[\s,;\(\[])\s*C\.?\s*R\.?\s*L\.?\b(?=\s*$)",  # ", C.R.L." /(CRL) end-ish
            r"\bCRL\b(?=\s*$)",
        ]),
        "marker": _rx([
            r"\bIPSS\b",
            r"\bassocia[cç][ãa]o\b",
            r"\bfunda[cç][ãa]o\b",
            r"\bONGD?\b",
        ]),
    },

    # -----------------
    # AUSTRIA
    # -----------------
    "AT": {
        "coop": _rx([
            r"\begen\b(?=\s*$)",
            r"\bgenossenschaft\b",
        ]),
        "marker": _rx([
            r"\bverein\b",
            r"\bstiftung\b",
        ]),
    },

    # -----------------
    # DENMARK
    # -----------------
    "DK": {
        "coop": _rx([
            r"\bA\.?\s*M\.?\s*B\.?\s*A\.?\b(?=\s*$)",   # A.m.b.a.
        ]),
        "marker": _rx([
            r"\bF\.?\s*M\.?\s*B\.?\s*A\.?\b(?=\s*$)",   # F.m.b.a.
        ]),
    },

    # -----------------
    # FINLAND
    # -----------------
    "FI": {
        "coop": _rx([
            r"\bosuuskunta\b",
            r"\bandelslag\b",
            r"\b(?:osk|anl)\b(?=\s*$)",
        ]),
        "marker": _rx([
            r"\bs[äa][äa]ti[öo]\b",   # säätiö
            r"\bstiftelse\b",
            r"\bry\b(?=\s*$)",
        ]),
    },

    # -----------------
    # CZECHIA
    # -----------------
    "CZ": {
        "coop": _rx([
            r"\bdru[žz]stvo\b",
        ]),
        "marker": _rx([
            r"\bo\.?\s*p\.?\s*s\.?\b(?=\s*$)",   # o.p.s.
            r"\bz\.?\s*s\.?\b(?=\s*$)",          # z. s.
            r"\bnadace\b",
        ]),
    },

    # -----------------
    # IRELAND
    # -----------------
    "IE": {
        "coop": _rx([
            r"\bco-?operative\b",
        ]),
        "marker": _rx([
            r"\bCLG\b(?=\s*$)",
        ]),
    },

    # -----------------
    # POLAND
    # -----------------
    "PL": {
        "coop": _rx([
            r"\bsp[oó]łdzielnia\b",
        ]),
        "marker": _rx([
            r"\bfundacja\b",
            r"\bstowarzyszenie\b",
        ]),
    },

    # -----------------
    # SWEDEN
    # -----------------
    "SE": {
        "coop": _rx([
            r"\bekonomisk\s+f[öo]rening\b",
        ]),
        "marker": _rx([
            r"\bideell\s+f[öo]rening\b",
            r"\bstiftelse\b",
        ]),
    },

    # -----------------
    # GREECE
    # -----------------
    "GR": {
        "impresa_sociale": _rx([
            r"Κοιν\.?\s*Σ\.?\s*Επ\.?",
        ]),
    },

    # -----------------
    # SWITZERLAND
    # -----------------
    "CH": {
        "coop": _rx([
            r"\bgenossenschaft\b",
            r"\bsoci[ée]t[ée]\s+coop[ée]rative\b",
            r"\bsociet[àa]\s+cooperativa\b",
        ]),
        "foundation_assoc": _rx([
            # Associations
            r"\bverein\b",
            r"\bassociation\b",
            r"\bassociazione\b",

            # Foundations
            r"\bstiftung\b",
            r"\bfondation\b",
            r"\bfondazione\b",
        ]),
    },

}

GLOBAL_NAME_HEURISTICS = {
    "coop": _rx([
        r"\bcoop(?:erative|erativa)?\b",
        r"\bcooperativa\b",
        r"\bscop\b",
        r"\bscic\b",
        r"\bgenossenschaft\b",
        r"\bosuuskunta\b",
        r"\bsp[oó]łdzielnia\b",
        r"\bdru[žz]stvo\b",
    ]),
    "marker": _rx([
        r"\bvzw\b",
        r"\basbl\b",
        r"\baisbl\b",
        r"\bivzw\b",
        r"\bassociation\b",
        r"\bfoundation\b",
        r"\bfondation\b",
        r"\bfondazione\b",
        r"\bfundaci[óo]n\b",
        r"\bstichting\b",
        r"\bvereniging\b",
        r"\bonlus\b",
        r"(?:^|[\s,;\(\[])\s*(?:odv|aps)\b(?=\s*$)",
        r"\bmutuelle\b",
        r"\bongd?\b",
    ]),
    "foundation_assoc": _rx([
        r"\bassociation\b",
        r"\bassociazione\b",
        r"\basociaci[óo]n\b",
        r"\bassocia[cç][ãa]o\b",
        r"\bstichting\b",
        r"\bvereniging\b",
        r"\bverein\b",
        r"\bfoundation\b",
        r"\bfondation\b",
        r"\bfondazione\b",
        r"\bfundaci[óo]n\b",
        r"\bfunda[cç][ãa]o\b",
    ]),
}

HEURISTIC_COUNTRY_REQUIREMENTS = {
    "DE": {
        "coop": "DE",
        "ggmbh": "DE",
    },
    "NL": {
        "coop": "NL",
        "marker": "NL",
        "foundation_assoc": "NL",
    },
    "BE": {
        "coop": "BE",
        "marker": "BE",
        "foundation_assoc": "BE",
    },
    "FR": {
        "coop": "FR",
        "esus": "FR",
        "marker": "FR",
    },
    "IT": {
        "coop": "IT",
        "impresa_sociale": "IT",
        "marker": "IT",
        "foundation_assoc": "IT",
    },
    "ES": {
        "coop": "ES",
        "marker": "ES",
    },
    "PT": {
        "coop": "PT",
        "marker": "PT",
    },
    "AT": {
        "coop": "AT",
        "marker": "AT",
    },
    "DK": {
        "coop": "DK",
        "marker": "DK",
    },
    "FI": {
        "coop": "FI",
        "marker": "FI",
    },
    "CZ": {
        "coop": "CZ",
        "marker": "CZ",
    },
    "IE": {
        "coop": "IE",
        "marker": "IE",
    },
    "PL": {
        "coop": "PL",
        "marker": "PL",
    },
    "SE": {
        "coop": "SE",
        "marker": "SE",
    },
    "GR": {
        "impresa_sociale": "GR",
    },
    "CH": {
        "coop": "CH",
        "foundation_assoc": "CH",
    },
}

def classify_name_candidates(country: str, name: str) -> dict:
    c = "" if country is None else str(country)
    c = c.strip().upper()
    name = "" if name is None else str(name)

    country_rules = NAME_HEURISTICS.get(c, {})
    rules = {}

    for key in ("coop", "ggmbh", "esus", "impresa_sociale", "marker", "foundation_assoc"):
        merged = []
        merged.extend(GLOBAL_NAME_HEURISTICS.get(key, []))
        merged.extend(country_rules.get(key, []))
        rules[key] = merged

    out = {
        "name_coop_candidate": "NO",
        "name_marker_candidate": "NO",
        "name_candidate_reason": "",
        "name_candidate_trigger": "",
        "name_candidate_country_requirement": "",
    }

    triggered = []

    for rx in rules.get("coop", []):
        if rx.search(name):
            out["name_coop_candidate"] = "YES"
            triggered.append("coop")
            break

    for key in ("ggmbh", "esus", "impresa_sociale", "marker", "foundation_assoc"):
        for rx in rules.get(key, []):
            if rx.search(name):
                out["name_marker_candidate"] = "YES"
                triggered.append(key)
                break

    bucket_map = {
        "coop": "coop",
        "ggmbh": "social enterprise",
        "esus": "social enterprise",
        "impresa_sociale": "social enterprise",
        "marker": "not for profit",
        "foundation_assoc": "not for profit",
    }

    buckets = []
    for t in triggered:
        b = bucket_map.get(t)
        if b and b not in buckets:
            buckets.append(b)

    out["name_candidate_reason"] = ",".join(buckets)
    out["name_candidate_trigger"] = ",".join(triggered)

    # Infer country requirement from triggered country-specific rules
    required_countries = set()

    for cc, cc_map in HEURISTIC_COUNTRY_REQUIREMENTS.items():
        for trig in triggered:
            if cc_map.get(trig):
                if trig in NAME_HEURISTICS.get(cc, {}):
                    for rx in NAME_HEURISTICS[cc].get(trig, []):
                        if rx.search(name):
                            required_countries.add(cc)
                            break

    if len(required_countries) == 1:
        out["name_candidate_country_requirement"] = next(iter(required_countries))
    elif len(required_countries) > 1:
        out["name_candidate_country_requirement"] = "MULTI"
    else:
        out["name_candidate_country_requirement"] = ""

    return out

def normalize_name(s):
    if pd.isna(s) or not str(s).strip():
        return ""

    s = unidecode(str(s)).lower()
    s = re.sub(r"[&/]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = LEGAL_SUFFIX_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_tax_id(s):
    if pd.isna(s) or not str(s).strip():
        return ""
    s = str(s).upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def siren_of(tax_id):
    """Extract 9-digit SIREN from SIREN or SIRET. Returns None if invalid."""
    if pd.isna(tax_id):
        return None
    s = str(tax_id).strip()
    if not s or s.lower() == "nan":
        return None
    s = re.sub(r"\D", "", s)
    if len(s) == 14:
        return s[:9]
    if len(s) == 9:
        return s
    return None

def build_block_key(norm_name, n=4):
    return norm_name[:n] if norm_name else ""

# -----------------------------
# FUZZY MATCHING HARDENING
# -----------------------------

GENERIC_TOKENS = {
    # Org-form / generic heads
    "association", "associazione", "asociacion", "asociacao", "vereniging", "stichting",
    "foundation", "fondation", "fundacion", "fundacao",
    "federation", "federacion", "federacao",
    "institute", "institut", "instituto",
    # Health/lab generics that caused your false perfects
    "laboratoire", "laboratorio", "lab", "clinic", "hospital",
    # Business generics
    "company", "group", "holding", "service", "services",
    "international", "global", "europe", "eu",
}

def raw_contains_any(s: str, words):
    s = (s or "").lower()
    return any(w in s for w in words)

COOP_WORDS = [
    "cooperativa", "cooperative", "coop", "coöperatie",
    "genossenschaft", "osuuskunta", "spoldzielnia", "druzstvo",
]

ASSOC_WORDS = [
    "association", "associazione", "asociacion", "associacao",
    "verein", "stichting", "vereniging",
    "fondation", "foundation", "fundacion", "fundacao",
]


def _tokens(s: str):
    return [t for t in (s or "").split() if t]

def meaningful_tokens(norm_name: str):
    return {t for t in _tokens(norm_name) if t not in GENERIC_TOKENS and len(t) >= 4}

def is_too_generic(norm_name: str) -> bool:
    mt = meaningful_tokens(norm_name)
    if not mt:
        return True
    # Reject single-token names unless token is long enough to be distinctive
    if len(mt) == 1:
        t = next(iter(mt))
        if len(t) < 8:
            return True
    return False

def has_min_shared_tokens(a: str, b: str, k: int = 2) -> bool:
    return len(meaningful_tokens(a) & meaningful_tokens(b)) >= k

# -----------------------------
# Commercial supplier detector
# -----------------------------

COMMERCIAL_FORMS_RE = re.compile(
    r"\b("
    r"s\.?\s*a\.?\s*r\.?\s*l\.?|"
    r"s\.?\s*a\.?\s*s\.?|"
    r"s\.?\s*a\.?|"
    r"s\.?\s*a\.?\s*s\.?\s*u\.?|"
    r"gmbh|ug|ag|kg|"
    r"s\.?\s*r\.?\s*l\.?|"
    r"s\.?\s*p\.?\s*a\.?|"
    r"s\.?\s*l\.?|"
    r"ltd|inc"
    r")\b",
    re.IGNORECASE,
)

def supplier_is_commercial(raw_name: str) -> bool:
    return bool(COMMERCIAL_FORMS_RE.search(raw_name or ""))

SOCIAL_BRANDS = {
    # Spain / ONCE group
    "ilunion",

    # Spain / work-integration groups
    "gureak",
    "amiab",

    # Germany / social enterprises
    "auticon",
    "afb",          # AfB gemeinnützige GmbH (short brand form)

    # France / insertion groups
    "groupe sos",

    # Nordics
    "samhall",
}

def is_whitelisted_social_brand(raw: str) -> bool:
    s = (raw or "").lower()
    return any(re.search(rf"\b{re.escape(b)}\b", s) for b in SOCIAL_BRANDS)

def match_suppliers(
    suppliers_path,
    master_path,
    suppliers_name_col="entity_name",
    suppliers_match_name_col=None,
    suppliers_tax_col=None,
    suppliers_country_col=None,
    master_name_col="entity_name",
    master_tax_col="tax_id",
    master_register_col="source_register",
    master_country_col="country",
    master_region_col="region",
    score_cutoff=92,
    trusted_entities_path=None,
    allow_commercial_fuzzy=False,
    out_path="suppliers_matched.csv",
):

    print("Loading files...")

    if suppliers_path.lower().endswith((".xlsx", ".xls")):
        sup = pd.read_excel(suppliers_path)
    else:
        sup = pd.read_csv(suppliers_path)

    mst = pd.read_csv(master_path, dtype=str, keep_default_na=False, low_memory=False)

    print("Normalizing names...")

    # Supplier legal/raw name column (expects headered file)
    if suppliers_name_col not in sup.columns:
        raise ValueError(f"Supplier legal name column not found: {suppliers_name_col}")

    sup["_supplier_raw_name"] = sup[suppliers_name_col].fillna("").astype(str)

    # Optional separate supplier match-name column
    if suppliers_match_name_col:
        if suppliers_match_name_col not in sup.columns:
            raise ValueError(f"Supplier match name column not found: {suppliers_match_name_col}")
        sup["_supplier_match_name"] = sup[suppliers_match_name_col].fillna("").astype(str)
    else:
        sup["_supplier_match_name"] = sup["_supplier_raw_name"]

    # Supplier country (optional)
    if suppliers_country_col is not None:
        if isinstance(suppliers_country_col, str) and suppliers_country_col.isdigit():
            c_idx = int(suppliers_country_col)
            sup["supplier_country"] = sup.iloc[:, c_idx].fillna("").astype(str).str.strip().str.upper()
        elif isinstance(suppliers_country_col, int):
            sup["supplier_country"] = sup.iloc[:, suppliers_country_col].fillna("").astype(str).str.strip().str.upper()
        else:
            if suppliers_country_col not in sup.columns:
                raise ValueError(f"Supplier country column not found: {suppliers_country_col}")
            sup["supplier_country"] = sup[suppliers_country_col].fillna("").astype(str).str.strip().str.upper()
    else:
        sup["supplier_country"] = ""

    # Clean: drop blanks and drop rows that are just the literal header label appearing in the data
    sup = sup[sup["_supplier_raw_name"].notna()]
    sup = sup[
        sup["_supplier_raw_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        .ne("column2")
    ]

    sup["_norm_name"] = sup["_supplier_match_name"].map(normalize_name)
    mst["_norm_name"] = mst[master_name_col].map(normalize_name)

    # Country-aware name heuristics
    flags = sup.apply(
        lambda r: classify_name_candidates(
            r.get("supplier_country", ""),
            r["_supplier_raw_name"],
        ),
        axis=1,
    )
    flags_df = flags.apply(pd.Series)
    sup = pd.concat([sup, flags_df], axis=1)

    # Combined social-economy name-based flag
    sup["social_economy_name_candidate"] = (
        (sup.get("name_coop_candidate", "") == "YES")
        | (sup.get("name_marker_candidate", "") == "YES")
    ).map(lambda x: "YES" if x else "NO")

    # Tax columns
    if suppliers_tax_col:
        sup["_norm_tax"] = sup[suppliers_tax_col].map(normalize_tax_id)
    else:
        sup["_norm_tax"] = ""

    mst["_norm_tax"] = (
        mst[master_tax_col].map(normalize_tax_id)
        if master_tax_col in mst.columns
        else ""
    )

    # Build SIREN bridge keys (FR: SIREN / SIRET linking)
    sup["_siren"] = sup["_norm_tax"].map(siren_of)
    mst["_siren"] = mst["_norm_tax"].map(siren_of)


    # Output columns
    sup["social_enterprise_supplier"] = "NO"
    sup["matched_register"] = ""
    sup["matched_entity_name"] = ""
    sup["match_type"] = ""
    sup["match_score"] = ""
    sup["match_country"] = ""
    sup["match_region"] = ""

    # Force text-safe dtypes for output columns
    for col in [
        "social_enterprise_supplier",
        "matched_register",
        "matched_entity_name",
        "match_type",
        "match_score",
        "match_country",
        "match_region",
    ]:
        sup[col] = sup[col].fillna("").astype(object)

    # -----------------------------
    # 1) TAX ID MATCHING
    # -----------------------------
    if suppliers_tax_col and mst["_norm_tax"].astype(bool).any():
        print("Running tax ID matching...")

        mst_tax_index = mst[mst["_norm_tax"].astype(bool)].set_index("_norm_tax", drop=False)

        hits = sup["_norm_tax"].astype(bool) & sup["_norm_tax"].isin(mst_tax_index.index)

        if hits.any():
            matched = mst_tax_index.loc[sup.loc[hits, "_norm_tax"]].reset_index(drop=True)

            sup.loc[hits, "social_enterprise_supplier"] = "YES"
            sup.loc[hits, "matched_register"] = matched[master_register_col].values
            sup.loc[hits, "matched_entity_name"] = matched[master_name_col].values
            sup.loc[hits, "match_type"] = "tax_id_exact"
            sup.loc[hits, "match_score"] = 100

            if master_country_col in matched.columns:
                sup.loc[hits, "match_country"] = matched[master_country_col].astype(str).str.upper().values
            if master_region_col in matched.columns:
                sup.loc[hits, "match_region"] = matched[master_region_col].values

    # -----------------------------
    # 1b) SIREN / SIRET BRIDGE (FR)
    # -----------------------------
    if mst["_siren"].astype(bool).any() and sup["_siren"].astype(bool).any():
        print("Running SIREN/SIRET bridge matching...")

        mst_siren_index = mst[mst["_siren"].astype(bool)].set_index("_siren", drop=False)

        hits = (
            sup["social_enterprise_supplier"].ne("YES")
            & sup["_siren"].astype(bool)
            & sup["_siren"].isin(mst_siren_index.index)
        )

        if hits.any():
            matched = mst_siren_index.loc[sup.loc[hits, "_siren"]].reset_index(drop=True)

            sup.loc[hits, "social_enterprise_supplier"] = "YES"
            sup.loc[hits, "matched_register"] = matched[master_register_col].values
            sup.loc[hits, "matched_entity_name"] = matched[master_name_col].values
            sup.loc[hits, "match_type"] = "tax_id_siren_bridge"
            sup.loc[hits, "match_score"] = 98

            if master_country_col in matched.columns:
                sup.loc[hits, "match_country"] = matched[master_country_col].astype(str).str.upper().values
            if master_region_col in matched.columns:
                sup.loc[hits, "match_region"] = matched[master_region_col].values

    # -----------------------------
    # Exact normalized name matching
    # -----------------------------
    print("Running exact normalized name matching...")

    # Build lookup from master
    mst_lookup = (
        mst
        .dropna(subset=["_norm_name"])
        .groupby("_norm_name", as_index=False)
        .first()
    )

    # Merge on normalized name
    merged = sup.merge(
        mst_lookup,
        how="left",
        left_on="_norm_name",
        right_on="_norm_name",
        suffixes=("", "_mst")
    )

    hits = merged[master_name_col].notna()

    if hits.any():
        sup.loc[hits, "social_enterprise_supplier"] = "YES"
        sup.loc[hits, "matched_entity_name"] = merged.loc[hits, master_name_col].fillna("").astype(str).values
        sup.loc[hits, "matched_register"] = merged.loc[hits, master_register_col].fillna("").astype(str).values
        sup.loc[hits, "match_type"] = "name_exact_norm"
        sup.loc[hits, "match_score"] = "100"

        if master_country_col in merged.columns:
            sup.loc[hits, "match_country"] = merged.loc[hits, master_country_col].fillna("").astype(str).values

        if master_region_col in merged.columns:
            sup.loc[hits, "match_region"] = merged.loc[hits, master_region_col].fillna("").astype(str).values

    print(f"Exact normalized matches: {hits.sum()}")

    # -----------------------------
    # 2) FUZZY NAME MATCHING (HARDENED)
    # -----------------------------
    print("Preparing fuzzy blocks...")

    remaining = (
        sup["social_enterprise_supplier"].ne("YES")
        & sup["_norm_name"].astype(bool)
    )

    # Prepare master blocks by (country, block_prefix)
    mst["_block"] = mst["_norm_name"].map(build_block_key)

    if master_country_col in mst.columns:
        mst["_m_country"] = mst[master_country_col].fillna("").astype(str).str.strip().str.upper()
    else:
        mst["_m_country"] = ""

    blocks_by_country = {}
    for (cty, b), grp in mst.groupby(["_m_country", "_block"], sort=False):
        blocks_by_country[(cty, b)] = list(zip(grp["_norm_name"].tolist(), grp.index.tolist()))

    # Fallback global blocks for suppliers with unknown country
    blocks_global = {}
    for b, grp in mst.groupby("_block", sort=False):
        blocks_global[b] = list(zip(grp["_norm_name"].tolist(), grp.index.tolist()))

    def fuzzy_best(norm_name: str, supplier_country: str):
        b = build_block_key(norm_name)
        c = (supplier_country or "").strip().upper()

        # If supplier country known, only match within that country
        if c:
            candidates = blocks_by_country.get((c, b), [])
        else:
            candidates = blocks_global.get(b, [])

        if not candidates:
            return None

        names = [c0 for (c0, _) in candidates]

        res = process.extractOne(
            norm_name,
            names,
            scorer=fuzz.WRatio,      # less permissive than token_set_ratio
            score_cutoff=score_cutoff,
        )

        if not res:
            return None

        _, score, j = res
        matched_norm = names[j]
        return candidates[j][1], score, matched_norm

    print("Running fuzzy matching...")

    # Pre-fetch supplier country series for speed
    sup_country_series = sup["supplier_country"] if "supplier_country" in sup.columns else pd.Series([""] * len(sup))

    for i, nm in sup.loc[remaining, "_norm_name"].items():

        # Skip generic / low-information names
        if is_too_generic(nm):
            continue

        # -----------------------------
        # Skip commercial suppliers unless strong name heuristic
        # -----------------------------
        raw = sup.at[i, "_supplier_raw_name"]
        is_name_candidate = (sup.at[i, "social_economy_name_candidate"] == "YES")

        if (
            not allow_commercial_fuzzy
            and supplier_is_commercial(raw)
            and not is_name_candidate
            and not is_whitelisted_social_brand(raw)
        ):
            continue

        # -----------------------------
        # Continue with fuzzy matching
        # -----------------------------
        s_country = sup_country_series.at[i] if i in sup_country_series.index else ""
        out = fuzzy_best(nm, s_country)

        if out is None:
            continue

        m_idx, score, matched_norm = out

        # -----------------------------
        # NEW: veto coop ↔ association cross-matches
        # -----------------------------
        raw = sup.at[i, "_supplier_raw_name"]
        matched_raw = mst.at[m_idx, master_name_col]

        supplier_is_coop = raw_contains_any(raw, COOP_WORDS)
        supplier_is_assoc = raw_contains_any(raw, ASSOC_WORDS)

        matched_is_coop = raw_contains_any(matched_raw, COOP_WORDS)
        matched_is_assoc = raw_contains_any(matched_raw, ASSOC_WORDS)

        if supplier_is_coop and matched_is_assoc:
            continue

        if supplier_is_assoc and matched_is_coop:
            continue

        # Must share >=2 meaningful tokens (prevents LABORATOIRE ↔ LABORATOIRE, ASSOCIATION ↔ ASSOCIATION, names ↔ first names)
        if not has_min_shared_tokens(nm, matched_norm, k=2):
            continue

        # Additional safety: never let fuzzy be "100"
        score = min(99, int(score))

        sup.at[i, "social_enterprise_supplier"] = "YES"
        sup.at[i, "matched_entity_name"] = str(mst.at[m_idx, master_name_col] or "")
        sup.at[i, "matched_register"] = str(mst.at[m_idx, master_register_col] or "")
        sup.at[i, "match_type"] = "name_fuzzy"
        sup.at[i, "match_score"] = str(score)

        if master_country_col in mst.columns:
            sup.at[i, "match_country"] = str(mst.at[m_idx, master_country_col]).strip().upper()
        if master_region_col in mst.columns:
            sup.at[i, "match_region"] = mst.at[m_idx, master_region_col]

    # -----------------------------
    # SAVE
    # -----------------------------
    print("Saving output:", out_path)

    # keep name-candidate columns; drop internal norm fields
    sup = sup.drop(columns=[c for c in ["_norm_name", "_norm_tax", "_siren"] if c in sup.columns])

    # De-duplicate columns (fixes repeated name_* headers if any upstream concat repeats)
    sup = sup.loc[:, ~sup.columns.duplicated()]

    sup.to_csv(out_path, index=False)
    print("Done.")


# -----------------------------
# COMMAND LINE ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--suppliers", required=True)
    parser.add_argument("--master", required=True)

    # Suppliers file is often headerless; pass a column index (e.g. 1 for name, 2 for country)
    parser.add_argument("--sup-name-col", default="entity_name")
    parser.add_argument("--sup-match-name-col", default=None)
    parser.add_argument("--sup-tax-col", default=None)
    parser.add_argument("--sup-country-col", default=None)

    # Master (normalized DB) columns
    parser.add_argument("--master-name-col", default="entity_name")
    parser.add_argument("--master-tax-col", default="tax_id")
    parser.add_argument("--master-register-col", default="ei_register_name")
    parser.add_argument("--master-country-col", default="country")
    parser.add_argument("--master-region-col", default="ccaa")

    # Matching controls
    parser.add_argument("--threshold", type=int, default=92)

    parser.add_argument(
        "--allow-commercial-fuzzy",
        action="store_true",
        help="Allow fuzzy matching for commercial suppliers"
    )

    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    match_suppliers(
        suppliers_path=args.suppliers,
        master_path=args.master,

        suppliers_name_col=args.sup_name_col,
        suppliers_match_name_col=args.sup_match_name_col,
        suppliers_tax_col=args.sup_tax_col,
        suppliers_country_col=args.sup_country_col,

        master_name_col=args.master_name_col,
        master_tax_col=args.master_tax_col,
        master_register_col=args.master_register_col,
        master_country_col=args.master_country_col,
        master_region_col=args.master_region_col,

        score_cutoff=args.threshold,
        allow_commercial_fuzzy=args.allow_commercial_fuzzy,
        out_path=args.out,
    )
