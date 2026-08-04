#!/usr/bin/env python3
"""Ingest manually approved trusted terms into canonical tables.

Manual additions remain easy to maintain in CSV, but every accepted term must:

- link to an existing persistent entity_id;
- use an approved term type;
- include a country;
- be explicitly approved for matching;
- avoid conflicted identifiers;
- retain reviewer, reason and evidence provenance.

Valid records are materialised into entity_aliases.parquet or
entity_identifiers.parquet. Invalid records are written to an explicit
review CSV and Parquet file.
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from canonical_entity_linkage import (
    clean_text,
    normalize_identifier,
    normalize_name,
)
from materialise_canonical_links import (
    ALIAS_COLUMNS,
    IDENTIFIER_ID_PREFIX,
)


SCHEMA_VERSION = "1.0.0"
ALIAS_ID_PREFIX = "sko_alias_"

ALLOWED_TERM_TYPES = {
    "reviewed_alias",
    "client_variant",
    "brand",
    "identifier",
}

TRUE_VALUES = {"true", "yes", "y", "1"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def allocate_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4()}"


def read_optional_parquet(
    path: Path,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=columns or [])


def ensure_columns(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = None
    return result


def ingest_manual_terms(
    manual_csv: Path,
    canonical_dir: Path,
) -> dict[str, Any]:
    entities_path = canonical_dir / "canonical_entities.parquet"
    aliases_path = canonical_dir / "entity_aliases.parquet"
    identifiers_path = canonical_dir / "entity_identifiers.parquet"

    if not entities_path.exists():
        raise FileNotFoundError(
            f"Required canonical table is missing: {entities_path}"
        )
    if not identifiers_path.exists():
        raise FileNotFoundError(
            f"Required canonical table is missing: {identifiers_path}"
        )

    manual = pd.read_csv(
        manual_csv,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "entity_id",
        "term_raw",
        "term_type",
        "country",
        "approved_for_matching",
        "approved_by",
        "approved_at",
        "reason",
        "evidence_reference",
    }
    missing = required_columns - set(manual.columns)
    if missing:
        raise ValueError(
            "Manual trusted-term file is missing columns: "
            + ", ".join(sorted(missing))
        )

    for column in [
        "identifier_type",
        "identifier_value",
    ]:
        if column not in manual.columns:
            manual[column] = ""

    entities = pd.read_parquet(entities_path)
    identifiers = pd.read_parquet(identifiers_path)
    aliases = read_optional_parquet(
        aliases_path,
        ALIAS_COLUMNS,
    )

    active_entity_ids = set(
        entities.loc[
            entities["entity_id"].notna()
            & entities["entity_id"].astype(str).str.strip().ne("")
            & ~entities["record_status"]
            .astype(str)
            .str.lower()
            .isin({"merged", "deprecated", "quarantined"}),
            "entity_id",
        ].astype(str)
    )

    existing_alias_keys = {
        "|".join(
            [
                clean_text(row.get("entity_id")),
                clean_text(row.get("country")).upper(),
                normalize_name(row.get("alias_name_norm")),
                clean_text(row.get("alias_type")).lower(),
            ]
        )
        for row in aliases.to_dict("records")
    }

    existing_identifier_map: dict[str, set[str]] = {}
    for row in identifiers.to_dict("records"):
        status = clean_text(
            row.get("verification_status")
        ).lower()
        if status in {"rejected", "conflicted", "superseded"}:
            continue

        key = "|".join(
            [
                clean_text(row.get("country")).upper(),
                clean_text(row.get("identifier_type")).upper(),
                normalize_identifier(
                    row.get("identifier_value_normalized")
                ),
            ]
        )
        entity_id = clean_text(row.get("entity_id"))

        if key and entity_id:
            existing_identifier_map.setdefault(
                key,
                set(),
            ).add(entity_id)

    alias_rows: list[dict[str, Any]] = []
    identifier_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []

    for row_number, row in enumerate(
        manual.to_dict("records"),
        start=2,
    ):
        entity_id = clean_text(row.get("entity_id"))
        term_raw = clean_text(row.get("term_raw"))
        term_type = clean_text(
            row.get("term_type")
        ).lower()
        country = clean_text(row.get("country")).upper()
        approved = clean_text(
            row.get("approved_for_matching")
        ).lower() in TRUE_VALUES
        approved_by = clean_text(row.get("approved_by"))
        approved_at = clean_text(row.get("approved_at"))
        reason = clean_text(row.get("reason"))
        evidence_reference = clean_text(
            row.get("evidence_reference")
        )

        errors = []

        if entity_id not in active_entity_ids:
            errors.append(
                "entity_id does not identify an active canonical entity"
            )
        if not term_raw:
            errors.append("term_raw is required")
        if term_type not in ALLOWED_TERM_TYPES:
            errors.append("term_type is not allowed")
        if not country:
            errors.append("country is required")
        if not approved:
            errors.append(
                "approved_for_matching must be explicitly true"
            )
        if not approved_by:
            errors.append("approved_by is required")
        if not approved_at:
            errors.append("approved_at is required")
        if not reason:
            errors.append("reason is required")

        if term_type == "identifier":
            identifier_type = clean_text(
                row.get("identifier_type")
            ).upper()
            identifier_value_raw = clean_text(
                row.get("identifier_value")
            )
            identifier_value_normalized = normalize_identifier(
                identifier_value_raw
            )

            if not identifier_type:
                errors.append(
                    "identifier_type is required for identifier terms"
                )
            if not identifier_value_normalized:
                errors.append(
                    "identifier_value is required for identifier terms"
                )

            identifier_key = "|".join(
                [
                    country,
                    identifier_type,
                    identifier_value_normalized,
                ]
            )

            existing_entities = existing_identifier_map.get(
                identifier_key,
                set(),
            )
            if (
                existing_entities
                and entity_id not in existing_entities
            ):
                errors.append(
                    "identifier is already linked to another entity"
                )
        else:
            identifier_type = ""
            identifier_value_raw = ""
            identifier_value_normalized = ""

        if errors:
            review_rows.append(
                {
                    "source_row_number": row_number,
                    **row,
                    "review_reason": "; ".join(errors),
                }
            )
            continue

        if term_type == "identifier":
            if not existing_identifier_map.get(identifier_key):
                identifier_rows.append(
                    {
                        "identifier_id": allocate_id(
                            IDENTIFIER_ID_PREFIX
                        ),
                        "entity_id": entity_id,
                        "identifier_type": identifier_type,
                        "identifier_value_raw": (
                            identifier_value_raw
                        ),
                        "identifier_value_normalized": (
                            identifier_value_normalized
                        ),
                        "country": country,
                        "identifier_scope": "legal_entity",
                        "is_primary": False,
                        "verification_status": "human_verified",
                        "source_record_id": None,
                        "created_at": approved_at or utc_now(),
                        "schema_version": SCHEMA_VERSION,
                    }
                )
                existing_identifier_map.setdefault(
                    identifier_key,
                    set(),
                ).add(entity_id)
            continue

        alias_type = (
            "brand"
            if term_type == "brand"
            else term_type
        )
        alias_name_norm = normalize_name(term_raw)
        alias_key = "|".join(
            [
                entity_id,
                country,
                alias_name_norm,
                alias_type,
            ]
        )

        if alias_key in existing_alias_keys:
            continue

        alias_rows.append(
            {
                "alias_id": allocate_id(ALIAS_ID_PREFIX),
                "entity_id": entity_id,
                "alias_name": term_raw,
                "alias_name_norm": alias_name_norm,
                "alias_type": alias_type,
                "language": "",
                "country": country,
                "is_preferred": False,
                "valid_from": None,
                "valid_to": None,
                "source_record_id": None,
                "evidence_id": evidence_reference or None,
                "verification_status": "human_verified",
                "review_status": "accepted",
                "source_client": "",
                "supplier_record_key": "",
                "created_at": approved_at or utc_now(),
                "schema_version": SCHEMA_VERSION,
            }
        )
        existing_alias_keys.add(alias_key)

    if alias_rows:
        aliases = pd.concat(
            [
                aliases,
                pd.DataFrame(
                    alias_rows,
                    columns=ALIAS_COLUMNS,
                ),
            ],
            ignore_index=True,
        ).drop_duplicates(
            subset=[
                "entity_id",
                "country",
                "alias_name_norm",
                "alias_type",
            ],
            keep="first",
        )

    if identifier_rows:
        new_identifiers = ensure_columns(
            pd.DataFrame(identifier_rows),
            list(identifiers.columns),
        )[identifiers.columns]

        identifiers = pd.concat(
            [identifiers, new_identifiers],
            ignore_index=True,
        ).drop_duplicates(
            subset=[
                "entity_id",
                "country",
                "identifier_type",
                "identifier_value_normalized",
            ],
            keep="first",
        )

    aliases.to_parquet(aliases_path, index=False)
    identifiers.to_parquet(identifiers_path, index=False)

    review = pd.DataFrame(review_rows)
    review_parquet = (
        canonical_dir / "manual_trusted_terms_review.parquet"
    )
    review_csv = (
        canonical_dir / "manual_trusted_terms_review.csv"
    )

    if review.empty:
        review = pd.DataFrame(
            columns=[
                "source_row_number",
                *manual.columns,
                "review_reason",
            ]
        )

    review.to_parquet(review_parquet, index=False)
    review.to_csv(review_csv, index=False)

    summary = {
        "input_rows": int(len(manual)),
        "aliases_added": int(len(alias_rows)),
        "identifiers_added": int(len(identifier_rows)),
        "review_rows": int(len(review)),
    }

    manifest_path = (
        canonical_dir / "manual_trusted_terms_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest manually approved canonical trusted terms."
    )
    parser.add_argument(
        "--manual-terms",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--canonical-dir",
        required=True,
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = ingest_manual_terms(
        manual_csv=args.manual_terms,
        canonical_dir=args.canonical_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
