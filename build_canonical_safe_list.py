#!/usr/bin/env python3
"""Build the governed canonical trusted/safe-list export.

E1.4 increment 4:

- export only accepted entity-linked names, aliases, brands and identifiers;
- retain canonical entity and source-record linkage;
- preserve governance and provenance fields;
- write Parquet as the authoritative output;
- write CSV only as a derived matcher/review compatibility export;
- produce DuckDB QA views.

This module does not alter matcher thresholds, heuristics, classification or
publication behaviour.
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from canonical_entity_linkage import clean_text, normalize_identifier, normalize_name


SCHEMA_VERSION = "1.0.0"
TRUSTED_TERM_ID_PREFIX = "sko_term_"

TRUSTED_COLUMNS = [
    "trusted_term_id",
    "entity_id",
    "source_record_id",
    "term_type",
    "term_raw",
    "term_normalized",
    "country",
    "identifier_type",
    "identifier_value_normalized",
    "identifier_scope",
    "relationship_type",
    "verification_status",
    "review_status",
    "approved_for_matching",
    "approved_by",
    "approved_at",
    "evidence_reference",
    "source_client",
    "supplier_record_key",
    "created_at",
    "schema_version",
]

ACCEPTED_ALIAS_STATUSES = {
    "accepted",
    "reviewed",
    "human_verified",
    "source_verified",
}

DISALLOWED_STATUSES = {
    "rejected",
    "conflicted",
    "superseded",
}


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def allocate_id(prefix: str) -> str:
    """Allocate an opaque row identifier."""
    return f"{prefix}{uuid.uuid4()}"


def read_required_parquet(path: Path) -> pd.DataFrame:
    """Read a required Parquet table."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required canonical table is missing: {path}"
        )
    return pd.read_parquet(path)


def read_optional_parquet(
    path: Path,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read an optional Parquet table or return an empty frame."""
    if path.exists():
        return pd.read_parquet(path)

    return pd.DataFrame(columns=columns or [])


def trusted_term_key(
    entity_id: Any,
    country: Any,
    term_type: Any,
    term_normalized: Any,
    identifier_type: Any = "",
) -> str:
    """Return a stable deduplication key for one trusted term."""
    return "|".join(
        [
            clean_text(entity_id),
            clean_text(country).upper(),
            clean_text(term_type).lower(),
            normalize_name(term_normalized),
            clean_text(identifier_type).upper(),
        ]
    )


def build_safe_list(
    canonical_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Build authoritative and derived safe-list outputs."""
    if output_dir is None:
        output_dir = canonical_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    entities_path = canonical_dir / "canonical_entities.parquet"
    source_records_path = canonical_dir / "source_records.parquet"
    identifiers_path = canonical_dir / "entity_identifiers.parquet"
    aliases_path = canonical_dir / "entity_aliases.parquet"

    entities = read_required_parquet(entities_path)
    source_records = read_required_parquet(source_records_path)
    identifiers = read_required_parquet(identifiers_path)
    aliases = read_optional_parquet(aliases_path)

    active_entities = entities.loc[
        entities["entity_id"].notna()
        & entities["entity_id"].astype(str).str.strip().ne("")
        & ~entities["record_status"]
        .astype(str)
        .str.lower()
        .isin({"merged", "deprecated", "quarantined"})
    ].copy()

    active_entity_ids = set(
        active_entities["entity_id"].astype(str)
    )

    source_by_id: dict[str, dict[str, Any]] = {}
    if not source_records.empty:
        for row in source_records.to_dict("records"):
            source_record_id = clean_text(
                row.get("source_record_id")
            )
            if source_record_id:
                source_by_id[source_record_id] = row

    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for row in active_entities.to_dict("records"):
        entity_id = clean_text(row.get("entity_id"))
        term_raw = clean_text(row.get("canonical_name"))
        term_normalized = normalize_name(
            row.get("canonical_name_norm") or term_raw
        )
        country = clean_text(row.get("country")).upper()
        source_record_id = clean_text(
            row.get("primary_source_record_id")
        )
        source_record = source_by_id.get(
            source_record_id,
            {},
        )

        if not entity_id or not term_raw or not country:
            continue

        key = trusted_term_key(
            entity_id,
            country,
            "legal_name",
            term_normalized,
        )

        if key in seen_keys:
            continue

        rows.append(
            {
                "trusted_term_id": allocate_id(
                    TRUSTED_TERM_ID_PREFIX
                ),
                "entity_id": entity_id,
                "source_record_id": (
                    source_record_id or None
                ),
                "term_type": "legal_name",
                "term_raw": term_raw,
                "term_normalized": term_normalized,
                "country": country,
                "identifier_type": "",
                "identifier_value_normalized": "",
                "identifier_scope": "",
                "relationship_type": "legal_entity",
                "verification_status": (
                    "human_verified"
                    if clean_text(
                        row.get("identity_review_status")
                    ).lower()
                    == "reviewed"
                    else "source_verified"
                ),
                "review_status": clean_text(
                    row.get("identity_review_status")
                ),
                "approved_for_matching": True,
                "approved_by": clean_text(
                    row.get("created_by")
                ),
                "approved_at": clean_text(
                    row.get("updated_at")
                ),
                "evidence_reference": clean_text(
                    source_record.get("source_url")
                ),
                "source_client": "",
                "supplier_record_key": "",
                "created_at": utc_now(),
                "schema_version": SCHEMA_VERSION,
            }
        )
        seen_keys.add(key)

    if not aliases.empty:
        for row in aliases.to_dict("records"):
            entity_id = clean_text(row.get("entity_id"))
            if entity_id not in active_entity_ids:
                continue

            verification_status = clean_text(
                row.get("verification_status")
            ).lower()
            review_status = clean_text(
                row.get("review_status")
            ).lower()

            accepted = (
                verification_status in ACCEPTED_ALIAS_STATUSES
                or review_status in ACCEPTED_ALIAS_STATUSES
            )

            if (
                not accepted
                or verification_status in DISALLOWED_STATUSES
                or review_status in DISALLOWED_STATUSES
            ):
                continue

            term_raw = clean_text(row.get("alias_name"))
            term_normalized = normalize_name(
                row.get("alias_name_norm") or term_raw
            )
            country = clean_text(row.get("country")).upper()
            alias_type = clean_text(
                row.get("alias_type")
            ).lower()

            if alias_type == "brand":
                term_type = "brand"
            elif alias_type == "client_variant":
                term_type = "client_variant"
            else:
                term_type = "reviewed_alias"

            if not term_raw or not country:
                continue

            key = trusted_term_key(
                entity_id,
                country,
                term_type,
                term_normalized,
            )

            if key in seen_keys:
                continue

            rows.append(
                {
                    "trusted_term_id": allocate_id(
                        TRUSTED_TERM_ID_PREFIX
                    ),
                    "entity_id": entity_id,
                    "source_record_id": (
                        clean_text(
                            row.get("source_record_id")
                        )
                        or None
                    ),
                    "term_type": term_type,
                    "term_raw": term_raw,
                    "term_normalized": term_normalized,
                    "country": country,
                    "identifier_type": "",
                    "identifier_value_normalized": "",
                    "identifier_scope": "",
                    "relationship_type": "legal_entity",
                    "verification_status": verification_status,
                    "review_status": review_status,
                    "approved_for_matching": True,
                    "approved_by": "",
                    "approved_at": clean_text(
                        row.get("created_at")
                    ),
                    "evidence_reference": clean_text(
                        row.get("evidence_id")
                    ),
                    "source_client": clean_text(
                        row.get("source_client")
                    ),
                    "supplier_record_key": clean_text(
                        row.get("supplier_record_key")
                    ),
                    "created_at": utc_now(),
                    "schema_version": SCHEMA_VERSION,
                }
            )
            seen_keys.add(key)

    if not identifiers.empty:
        for row in identifiers.to_dict("records"):
            entity_id = clean_text(row.get("entity_id"))
            if entity_id not in active_entity_ids:
                continue

            verification_status = clean_text(
                row.get("verification_status")
            ).lower()

            if (
                not verification_status
                or verification_status in DISALLOWED_STATUSES
            ):
                continue

            country = clean_text(row.get("country")).upper()
            identifier_type = clean_text(
                row.get("identifier_type")
            ).upper()
            identifier_value = normalize_identifier(
                row.get("identifier_value_normalized")
            )
            identifier_scope = clean_text(
                row.get("identifier_scope")
            ).lower()

            if (
                not country
                or not identifier_type
                or not identifier_value
            ):
                continue

            key = trusted_term_key(
                entity_id,
                country,
                "identifier",
                identifier_value,
                identifier_type,
            )

            if key in seen_keys:
                continue

            rows.append(
                {
                    "trusted_term_id": allocate_id(
                        TRUSTED_TERM_ID_PREFIX
                    ),
                    "entity_id": entity_id,
                    "source_record_id": (
                        clean_text(
                            row.get("source_record_id")
                        )
                        or None
                    ),
                    "term_type": "identifier",
                    "term_raw": clean_text(
                        row.get("identifier_value_raw")
                    ),
                    "term_normalized": identifier_value,
                    "country": country,
                    "identifier_type": identifier_type,
                    "identifier_value_normalized": identifier_value,
                    "identifier_scope": identifier_scope,
                    "relationship_type": "legal_entity",
                    "verification_status": verification_status,
                    "review_status": "accepted",
                    "approved_for_matching": True,
                    "approved_by": "",
                    "approved_at": clean_text(
                        row.get("created_at")
                    ),
                    "evidence_reference": "",
                    "source_client": "",
                    "supplier_record_key": "",
                    "created_at": utc_now(),
                    "schema_version": SCHEMA_VERSION,
                }
            )
            seen_keys.add(key)

    trusted_terms = pd.DataFrame(
        rows,
        columns=TRUSTED_COLUMNS,
    )

    trusted_terms = trusted_terms.loc[
        trusted_terms["entity_id"].notna()
        & trusted_terms["entity_id"].astype(str).str.strip().ne("")
        & trusted_terms["approved_for_matching"].astype(bool)
    ].copy()

    trusted_terms = trusted_terms.drop_duplicates(
        subset=[
            "entity_id",
            "country",
            "term_type",
            "term_normalized",
            "identifier_type",
        ],
        keep="first",
    ).reset_index(drop=True)

    parquet_path = output_dir / "trusted_match_terms.parquet"
    csv_path = output_dir / "trusted_match_terms.csv"

    trusted_terms.to_parquet(parquet_path, index=False)
    trusted_terms.to_csv(csv_path, index=False)

    qa_path = output_dir / "trusted_match_terms_qa.duckdb"
    if qa_path.exists():
        qa_path.unlink()

    connection = duckdb.connect(str(qa_path))
    try:
        connection.execute(
            f"""
            CREATE VIEW trusted_match_terms AS
            SELECT *
            FROM read_parquet('{parquet_path.as_posix()}')
            """
        )
        connection.execute(
            """
            CREATE VIEW qa_terms_without_entity_id AS
            SELECT *
            FROM trusted_match_terms
            WHERE entity_id IS NULL OR entity_id = ''
            """
        )
        connection.execute(
            """
            CREATE VIEW qa_unapproved_terms AS
            SELECT *
            FROM trusted_match_terms
            WHERE approved_for_matching IS NOT TRUE
            """
        )
        connection.execute(
            """
            CREATE VIEW qa_duplicate_trusted_terms AS
            SELECT
                entity_id,
                country,
                term_type,
                term_normalized,
                identifier_type,
                COUNT(*) AS row_count
            FROM trusted_match_terms
            GROUP BY
                entity_id,
                country,
                term_type,
                term_normalized,
                identifier_type
            HAVING COUNT(*) > 1
            """
        )
        connection.execute(
            """
            CREATE VIEW qa_disallowed_status_terms AS
            SELECT *
            FROM trusted_match_terms
            WHERE lower(verification_status) IN (
                'rejected',
                'conflicted',
                'superseded'
            )
               OR lower(review_status) IN (
                'rejected',
                'conflicted',
                'superseded'
            )
            """
        )
        connection.execute(
            """
            CREATE VIEW qa_term_type_counts AS
            SELECT term_type, COUNT(*) AS row_count
            FROM trusted_match_terms
            GROUP BY term_type
            ORDER BY term_type
            """
        )
    finally:
        connection.close()

    summary = {
        "trusted_terms": int(len(trusted_terms)),
        "legal_names": int(
            trusted_terms["term_type"].eq("legal_name").sum()
        ),
        "reviewed_aliases": int(
            trusted_terms["term_type"]
            .eq("reviewed_alias")
            .sum()
        ),
        "client_variants": int(
            trusted_terms["term_type"]
            .eq("client_variant")
            .sum()
        ),
        "brands": int(
            trusted_terms["term_type"].eq("brand").sum()
        ),
        "identifiers": int(
            trusted_terms["term_type"].eq("identifier").sum()
        ),
    }

    manifest_path = output_dir / "trusted_match_terms_manifest.json"
    manifest_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical trusted/safe-list export."
        )
    )
    parser.add_argument(
        "--canonical-dir",
        required=True,
        type=Path,
        help="Directory containing canonical Parquet tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to the canonical directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Build the safe-list output."""
    args = parse_args()
    summary = build_safe_list(
        canonical_dir=args.canonical_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
