#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Config
# -----------------------------

REQUIRED_QUEUE_COLUMNS = [
    "country",
    "raw_name",
    "city",
    "postcode",
    "reason",
    "priority",
    "suggested_source",
    "client_status",
    "verified",
    "verified_name",
    "verified_id_type",
    "verified_id",
    "verified_source",
    "verified_url",
    "notes",
]

OVERLAY_COLUMNS = [
    "country",
    "raw_name",
    "verified_name",
    "verified_id_type",
    "verified_id",
    "verified_id_normalized",
    "verified_source",
    "verified_url",
    "city",
    "postcode",
    "reason",
    "priority",
    "suggested_source",
    "client_status",
    "notes",
    "overlay_key",
    "ingested_at_utc",
    "ingest_source_file",
    "ingest_batch_id",
]

# Heuristic aliases for mapping overlay -> core master schema
MASTER_COLUMN_ALIASES = {
    "country": ["country", "country_code", "register_country"],
    "name": ["name", "entity_name", "legal_name", "supplier_name", "normalized_name", "register_name"],
    "id_type": ["id_type", "identifier_type", "registry_id_type", "company_id_type", "entity_id_type"],
    "id": ["id", "identifier", "registry_id", "company_id", "entity_id", "register_id"],
    "source": ["source", "registry_source", "register_source", "matched_source"],
    "url": ["url", "source_url", "registry_url", "record_url"],
    "city": ["city", "town"],
    "postcode": ["postcode", "postal_code", "zip", "zip_code"],
    "notes": ["notes", "comment", "comments"],
}


# -----------------------------
# Helpers
# -----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def backup_file(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak.{ts}")
    shutil.copy2(path, backup)
    return backup


def clean_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_verified_flag(x) -> str:
    v = clean_str(x).upper()
    return "Y" if v == "Y" else "N"


def read_queue_file(path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Queue file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if suffix in [".xlsx", ".xlsm", ".xls"]:
        chosen_sheet = sheet_name or "verify_queue"
        xl = pd.ExcelFile(path)
        if chosen_sheet not in xl.sheet_names:
            # fall back to first sheet
            chosen_sheet = xl.sheet_names[0]
        return pd.read_excel(path, sheet_name=chosen_sheet, dtype=str).fillna("")
    raise ValueError(f"Unsupported input format: {path.suffix}")


def ensure_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Queue file missing required columns: {missing}")


def canonicalize_country(country: str) -> str:
    return clean_str(country).upper()


def canonicalize_id_type(country: str, id_type: str) -> str:
    c = canonicalize_country(country)
    t = clean_str(id_type).strip().lower()

    # Belgium
    if c == "BE":
        if any(k in t for k in ["enterprise", "ondernemingsnummer", "numéro d'entreprise", "kbo", "cbe", "bce"]):
            return "BE_ENTERPRISE_NUMBER"

    # France
    if c == "FR":
        if "siret" in t:
            return "FR_SIRET"
        if "siren" in t:
            return "FR_SIREN"
        if "rna" in t:
            return "FR_RNA"

    # Germany
    if c == "DE":
        if any(k in t for k in ["handelsregister", "hrb", "hra"]):
            return "DE_HANDELSREGISTER"
        if any(k in t for k in ["vat", "ust", "ust-id", "ustid"]):
            return "DE_VAT"

    # Spain
    if c == "ES":
        if "cif" in t:
            return "ES_CIF"
        if "nif" in t:
            return "ES_NIF"

    # Italy
    if c == "IT":
        if any(k in t for k in ["partita iva", "p.iva", "vat"]):
            return "IT_PARTITA_IVA"
        if "codice fiscale" in t:
            return "IT_CODICE_FISCALE"
        if "rea" in t:
            return "IT_REA"

    # Switzerland
    if c == "CH":
        if any(k in t for k in ["uid", "che"]):
            return "CH_UID"
        if "ide" in t:
            return "CH_UID"

    # Netherlands
    if c == "NL":
        if "kvk" in t:
            return "NL_KVK"
        if "btw" in t or "vat" in t:
            return "NL_VAT"

    # EU / generic VAT / fallback
    if "vat" in t:
        return f"{c}_VAT" if c else "VAT"
    if "company" in t or "registry" in t or "register" in t:
        return f"{c}_REGISTRY_ID" if c else "REGISTRY_ID"

    # Preserve something deterministic
    normalized = re.sub(r"[^A-Z0-9]+", "_", t.upper()).strip("_")
    return normalized or "UNKNOWN"


def normalize_id(country: str, id_type: str, raw_id: str) -> str:
    c = canonicalize_country(country)
    t = canonicalize_id_type(country, id_type)
    raw = clean_str(raw_id).upper()

    # Common cleanup
    raw = raw.replace("\u00A0", " ")
    raw = re.sub(r"\s+", " ", raw).strip()

    # Country-specific normalization
    if t == "BE_ENTERPRISE_NUMBER":
        digits = re.sub(r"\D", "", raw)
        if len(digits) == 9:
            digits = "0" + digits
        return digits

    if t == "FR_SIREN":
        return re.sub(r"\D", "", raw)

    if t == "FR_SIRET":
        return re.sub(r"\D", "", raw)

    if t == "DE_VAT":
        alnum = re.sub(r"[^A-Z0-9]", "", raw)
        if not alnum.startswith("DE") and re.fullmatch(r"\d{9}", alnum):
            return "DE" + alnum
        return alnum

    if t == "DE_HANDELSREGISTER":
        return re.sub(r"[^A-Z0-9]", "", raw)

    if t in {"ES_CIF", "ES_NIF"}:
        return re.sub(r"[^A-Z0-9]", "", raw)

    if t == "IT_PARTITA_IVA":
        return re.sub(r"\D", "", raw)

    if t == "IT_CODICE_FISCALE":
        return re.sub(r"[^A-Z0-9]", "", raw)

    if t == "CH_UID":
        alnum = re.sub(r"[^A-Z0-9]", "", raw)
        if alnum.startswith("CHE"):
            return alnum
        if re.fullmatch(r"\d{9}", alnum):
            return "CHE" + alnum
        return alnum

    if t == "NL_KVK":
        return re.sub(r"\D", "", raw)

    if t.endswith("_VAT") or "VAT" in t:
        return re.sub(r"[^A-Z0-9]", "", raw)

    # Generic fallback: uppercase alnum only
    return re.sub(r"[^A-Z0-9]", "", raw)


def make_overlay_key(country: str, id_type: str, normalized_id: str) -> str:
    return f"{canonicalize_country(country)}|{canonicalize_id_type(country, id_type)}|{normalized_id}"


def find_master_column(master_cols: List[str], canonical_key: str) -> Optional[str]:
    aliases = MASTER_COLUMN_ALIASES.get(canonical_key, [])
    lower_map = {c.lower(): c for c in master_cols}
    for a in aliases:
        if a.lower() in lower_map:
            return lower_map[a.lower()]
    return None


@dataclass
class ValidationResult:
    accepted: pd.DataFrame
    rejected: pd.DataFrame
    warnings: List[str]
    stats: Dict[str, int]


# -----------------------------
# Validation + transformation
# -----------------------------

def validate_and_transform_queue(df: pd.DataFrame, source_file: str, batch_id: str) -> ValidationResult:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    ensure_required_columns(df, REQUIRED_QUEUE_COLUMNS)

    # Normalize basics
    for col in REQUIRED_QUEUE_COLUMNS:
        df[col] = df[col].map(clean_str)

    df["verified"] = df["verified"].map(normalize_verified_flag)
    df["country"] = df["country"].map(canonicalize_country)

    reviewed = df[df["verified"] == "Y"].copy()

    if reviewed.empty:
        return ValidationResult(
            accepted=pd.DataFrame(columns=OVERLAY_COLUMNS),
            rejected=pd.DataFrame(columns=list(df.columns) + ["reject_reason"]),
            warnings=["No rows with verified == Y"],
            stats={
                "input_rows": len(df),
                "verified_y_rows": 0,
                "accepted_rows": 0,
                "rejected_rows": 0,
            },
        )

    # Required validation for verified rows
    reviewed["reject_reason"] = ""

    missing_type_mask = reviewed["verified_id_type"].eq("")
    reviewed.loc[missing_type_mask, "reject_reason"] = "missing_verified_id_type"

    missing_id_mask = reviewed["verified_id"].eq("")
    reviewed.loc[missing_id_mask, "reject_reason"] = reviewed.loc[missing_id_mask, "reject_reason"].mask(
        reviewed.loc[missing_id_mask, "reject_reason"].eq(""),
        "missing_verified_id",
    ).fillna(reviewed.loc[missing_id_mask, "reject_reason"])

    reviewed["verified_id_type"] = reviewed.apply(
        lambda r: canonicalize_id_type(r["country"], r["verified_id_type"]),
        axis=1,
    )
    reviewed["verified_id_normalized"] = reviewed.apply(
        lambda r: normalize_id(r["country"], r["verified_id_type"], r["verified_id"]),
        axis=1,
    )

    # Empty after normalization is also reject
    empty_norm_mask = reviewed["verified_id_normalized"].eq("")
    reviewed.loc[empty_norm_mask & reviewed["reject_reason"].eq(""), "reject_reason"] = "verified_id_normalized_empty"

    reviewed["overlay_key"] = reviewed.apply(
        lambda r: make_overlay_key(r["country"], r["verified_id_type"], r["verified_id_normalized"]),
        axis=1,
    )

    # Duplicate verified IDs inside the incoming batch
    dup_mask = reviewed["overlay_key"].duplicated(keep=False)
    dupes = reviewed[dup_mask].copy()

    warnings: List[str] = []

    if not dupes.empty:
        # allow exact duplicate rows collapsing later, but reject conflicting duplicates
        for key, g in dupes.groupby("overlay_key", dropna=False):
            distinct_names = set(x for x in g["verified_name"].astype(str).str.strip() if x)
            distinct_urls = set(x for x in g["verified_url"].astype(str).str.strip() if x)
            if len(distinct_names) > 1 or len(distinct_urls) > 1:
                reviewed.loc[reviewed["overlay_key"] == key, "reject_reason"] = "conflicting_duplicate_verified_id"
            else:
                warnings.append(f"Collapsed identical duplicate rows for overlay_key={key}")

    rejected = reviewed[reviewed["reject_reason"] != ""].copy()
    accepted = reviewed[reviewed["reject_reason"] == ""].copy()

    # Collapse exact duplicates in accepted
    accepted = accepted.drop_duplicates(subset=["overlay_key"], keep="last").copy()

    accepted["ingested_at_utc"] = utc_now_iso()
    accepted["ingest_source_file"] = source_file
    accepted["ingest_batch_id"] = batch_id

    accepted = accepted[OVERLAY_COLUMNS].copy()

    stats = {
        "input_rows": len(df),
        "verified_y_rows": len(reviewed),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
    }

    return ValidationResult(
        accepted=accepted,
        rejected=rejected,
        warnings=warnings,
        stats=stats,
    )


# -----------------------------
# Overlay upsert
# -----------------------------

def load_existing_overlay(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=OVERLAY_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in OVERLAY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[OVERLAY_COLUMNS].copy()


def upsert_overlay(existing: pd.DataFrame, incoming: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []

    if existing.empty:
        return incoming.copy(), warnings
    if incoming.empty:
        return existing.copy(), warnings

    existing = existing.copy()
    incoming = incoming.copy()

    # Normalize essential columns for existing overlay too
    for col in OVERLAY_COLUMNS:
        if col not in existing.columns:
            existing[col] = ""
        existing[col] = existing[col].map(clean_str)

    # Check conflicts between existing and incoming
    if "overlay_key" in existing.columns and "overlay_key" in incoming.columns:
        overlapping = set(existing["overlay_key"]).intersection(set(incoming["overlay_key"]))
        for key in sorted(overlapping):
            e = existing[existing["overlay_key"] == key].tail(1)
            i = incoming[incoming["overlay_key"] == key].tail(1)

            e_name = clean_str(e.iloc[0]["verified_name"])
            i_name = clean_str(i.iloc[0]["verified_name"])
            e_url = clean_str(e.iloc[0]["verified_url"])
            i_url = clean_str(i.iloc[0]["verified_url"])

            if (e_name and i_name and e_name != i_name) or (e_url and i_url and e_url != i_url):
                warnings.append(
                    f"overlay_key={key} already existed and was replaced by incoming row "
                    f"(name/url changed; inspect backup if needed)"
                )

    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.drop_duplicates(subset=["overlay_key"], keep="last").copy()
    combined = combined[OVERLAY_COLUMNS].copy()
    return combined, warnings


# -----------------------------
# Build combined master
# -----------------------------

def overlay_to_master_schema(overlay_df: pd.DataFrame, master_columns: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(columns=master_columns)
    for c in master_columns:
        out[c] = ""

    col_country = find_master_column(master_columns, "country")
    col_name = find_master_column(master_columns, "name")
    col_id_type = find_master_column(master_columns, "id_type")
    col_id = find_master_column(master_columns, "id")
    col_source = find_master_column(master_columns, "source")
    col_url = find_master_column(master_columns, "url")
    col_city = find_master_column(master_columns, "city")
    col_postcode = find_master_column(master_columns, "postcode")
    col_notes = find_master_column(master_columns, "notes")

    if col_country:
        out[col_country] = overlay_df["country"]
    if col_name:
        out[col_name] = overlay_df["verified_name"]
    if col_id_type:
        out[col_id_type] = overlay_df["verified_id_type"]
    if col_id:
        out[col_id] = overlay_df["verified_id_normalized"]
    if col_source:
        out[col_source] = overlay_df["verified_source"]
    if col_url:
        out[col_url] = overlay_df["verified_url"]
    if col_city:
        out[col_city] = overlay_df["city"]
    if col_postcode:
        out[col_postcode] = overlay_df["postcode"]
    if col_notes:
        out[col_notes] = overlay_df["notes"]

    # Opportunistic fill if audit-style columns exist in master
    lower_map = {c.lower(): c for c in master_columns}
    for src, candidates in [
        ("overlay_key", ["overlay_key"]),
        ("ingested_at_utc", ["ingested_at_utc", "overlay_ingested_at_utc"]),
        ("ingest_source_file", ["ingest_source_file", "overlay_source_file"]),
        ("ingest_batch_id", ["ingest_batch_id", "overlay_batch_id"]),
        ("raw_name", ["raw_name"]),
    ]:
        for cand in candidates:
            if cand.lower() in lower_map:
                out[lower_map[cand.lower()]] = overlay_df[src]
                break

    return out


def build_master_plus_overlay(core_master_path: Path, overlay_path: Path, combined_path: Path) -> Dict[str, int]:
    core = pd.read_csv(core_master_path, dtype=str, keep_default_na=False)
    overlay = pd.read_csv(overlay_path, dtype=str, keep_default_na=False) if overlay_path.exists() else pd.DataFrame(columns=OVERLAY_COLUMNS)

    master_cols = list(core.columns)
    overlay_as_master = overlay_to_master_schema(overlay, master_cols)

    # Deduplicate by best available key
    col_country = find_master_column(master_cols, "country")
    col_id_type = find_master_column(master_cols, "id_type")
    col_id = find_master_column(master_cols, "id")

    core = core.copy()
    overlay_as_master = overlay_as_master.copy()

    core["__origin"] = "core"
    overlay_as_master["__origin"] = "overlay"

    combined = pd.concat([core, overlay_as_master], ignore_index=True)

    if col_country and col_id_type and col_id:
        combined["__dedupe_key"] = (
            combined[col_country].map(clean_str).str.upper() + "|" +
            combined[col_id_type].map(clean_str).str.upper() + "|" +
            combined[col_id].map(clean_str).str.upper()
        )
        combined = combined.drop_duplicates(subset=["__dedupe_key"], keep="last").copy()
        combined = combined.drop(columns=["__dedupe_key"])

    combined = combined.drop(columns=["__origin"])
    combined.to_csv(combined_path, index=False, quoting=csv.QUOTE_MINIMAL)

    return {
        "core_rows": len(core),
        "overlay_rows": len(overlay),
        "combined_rows": len(combined),
    }


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest reviewed verify_queue into verified overlay and build combined master.")
    parser.add_argument("--queue", required=True, help="Path to reviewed verify_queue.csv or .xlsx")
    parser.add_argument("--overlay", default="ei_registers_verified_overlay.csv", help="Overlay CSV path")
    parser.add_argument("--core-master", default="ei_registers_normalized_headered.csv", help="Core master CSV path")
    parser.add_argument("--combined-master", default="ei_registers_master_plus_overlay.csv", help="Combined master CSV path")
    parser.add_argument("--sheet-name", default="verify_queue", help="Excel sheet name if queue is xlsx")
    parser.add_argument("--audit-dir", default="audit", help="Directory for validation reports")
    args = parser.parse_args()

    queue_path = Path(args.queue).expanduser().resolve()
    overlay_path = Path(args.overlay).expanduser().resolve()
    core_master_path = Path(args.core_master).expanduser().resolve()
    combined_master_path = Path(args.combined_master).expanduser().resolve()
    audit_dir = Path(args.audit_dir).expanduser().resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    queue_df = read_queue_file(queue_path, sheet_name=args.sheet_name)

    result = validate_and_transform_queue(
        queue_df,
        source_file=queue_path.name,
        batch_id=batch_id,
    )

    # Save rejected rows for audit
    rejected_path = audit_dir / f"verify_queue_rejected_{batch_id}.csv"
    result.rejected.to_csv(rejected_path, index=False)

    # Backup and upsert overlay
    existing_overlay = load_existing_overlay(overlay_path)
    overlay_backup = backup_file(overlay_path)
    combined_overlay, upsert_warnings = upsert_overlay(existing_overlay, result.accepted)
    combined_overlay.to_csv(overlay_path, index=False, quoting=csv.QUOTE_MINIMAL)

    # Build combined master
    core_backup = backup_file(combined_master_path) if combined_master_path.exists() else None
    master_stats = build_master_plus_overlay(core_master_path, overlay_path, combined_master_path)

    # Summary report
    summary = {
        "batch_id": batch_id,
        "queue_file": str(queue_path),
        "overlay_file": str(overlay_path),
        "core_master_file": str(core_master_path),
        "combined_master_file": str(combined_master_path),
        "overlay_backup": str(overlay_backup) if overlay_backup else None,
        "combined_master_backup": str(core_backup) if core_backup else None,
        "validation_stats": result.stats,
        "build_stats": master_stats,
        "warnings": result.warnings + upsert_warnings,
        "rejected_rows_file": str(rejected_path),
        "generated_at_utc": utc_now_iso(),
    }

    summary_path = audit_dir / f"ingest_summary_{batch_id}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
