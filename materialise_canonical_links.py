#!/usr/bin/env python3
"""Materialise accepted supplier links into canonical entity tables.

E1.4 increment 3:

- add genuinely new accepted entities to canonical_entities;
- add accepted identifiers to entity_identifiers;
- add reviewed supplier names to entity_aliases;
- retain supplier and client provenance;
- store organisational relationships separately from legal identity;
- preserve opaque IDs and repeated-run stability;
- produce Parquet authoritative outputs and DuckDB QA.

This process does not alter matching, classification or publication rules.
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

from canonical_entity_linkage import (
    ACCEPTED_STATUSES,
    clean_text,
    normalize_identifier,
    normalize_name,
)


SCHEMA_VERSION = "1.0.0"
ALIAS_ID_PREFIX = "sko_alias_"
IDENTIFIER_ID_PREFIX = "sko_id_"
EVENT_ID_PREFIX = "sko_evt_"
RELATIONSHIP_ID_PREFIX = "sko_rel_"

ALIAS_COLUMNS = [
    "alias_id",
    "entity_id",
    "alias_name",
    "alias_name_norm",
    "alias_type",
    "language",
    "country",
    "is_preferred",
    "valid_from",
    "valid_to",
    "source_record_id",
    "evidence_id",
    "verification_status",
    "review_status",
    "source_client",
    "supplier_record_key",
    "created_at",
    "schema_version",
]

RELATIONSHIP_COLUMNS = [
    "relationship_id",
    "subject_entity_id",
    "object_entity_id",
    "relationship_type",
    "relationship_status",
    "evidence_reference",
    "source_client",
    "supplier_record_key",
    "reviewed_by",
    "reviewed_at",
    "created_at",
    "schema_version",
]


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def allocate_id(prefix: str) -> str:
    """Allocate an opaque persisted row identifier."""
    return f"{prefix}{uuid.uuid4()}"


def read_parquet_or_empty(
    path: Path,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read a Parquet table or return an empty schema."""
    if path.exists():
        return pd.read_parquet(path)

    if columns is None:
        return pd.DataFrame()

    return pd.DataFrame(columns=columns)


def ensure_columns(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Add missing columns without removing existing fields."""
    result = frame.copy()

    for column in columns:
        if column not in result.columns:
            result[column] = None

    return result


def identifier_key(
    country: Any,
    identifier_type: Any,
    identifier_value: Any,
) -> str:
    """Return a country/type/value uniqueness key."""
    country_norm = clean_text(country).upper()
    type_norm = clean_text(identifier_type).upper()
    value_norm = normalize_identifier(identifier_value)

    if not country_norm or not type_norm or not value_norm:
        return ""

    return "|".join(
        [
            country_norm,
            type_norm,
            value_norm,
        ]
    )


def alias_key(
    entity_id: Any,
    country: Any,
    alias_name: Any,
    alias_type: Any,
) -> str:
    """Return a stable alias deduplication key."""
    return "|".join(
        [
            clean_text(entity_id),
            clean_text(country).upper(),
            normalize_name(alias_name),
            clean_text(alias_type).lower(),
        ]
    )


def relationship_key(
    subject_entity_id: Any,
    object_entity_id: Any,
    relationship_type: Any,
) -> str:
    """Return a stable relationship deduplication key."""
    return "|".join(
        [
            clean_text(subject_entity_id),
            clean_text(object_entity_id),
            clean_text(relationship_type).lower(),
        ]
    )


def materialise_links(
    canonical_dir: Path,
) -> dict[str, Any]:
    """Apply accepted supplier linkage decisions to canonical tables."""
    links_path = canonical_dir / "supplier_entity_links.parquet"
    entities_path = canonical_dir / "canonical_entities.parquet"
    identifiers_path = canonical_dir / "entity_identifiers.parquet"
    source_records_path = canonical_dir / "source_records.parquet"
    events_path = canonical_dir / "entity_events.parquet"
    aliases_path = canonical_dir / "entity_aliases.parquet"
    relationships_path = (
        canonical_dir / "entity_relationships.parquet"
    )

    for path in [
        links_path,
        entities_path,
        identifiers_path,
        source_records_path,
        events_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required canonical table is missing: {path}"
            )

    links = pd.read_parquet(links_path)
    entities = pd.read_parquet(entities_path)
    identifiers = pd.read_parquet(identifiers_path)
    source_records = pd.read_parquet(source_records_path)
    events = pd.read_parquet(events_path)

    aliases = read_parquet_or_empty(
        aliases_path,
        ALIAS_COLUMNS,
    )
    relationships = read_parquet_or_empty(
        relationships_path,
        RELATIONSHIP_COLUMNS,
    )

    accepted_links = links.loc[
        links["acceptance_status"]
        .astype(str)
        .str.lower()
        .isin(ACCEPTED_STATUSES)
        & links["resolution_status"].isin(["reused", "new"])
        & links["matched_entity_id"].notna()
        & links["matched_entity_id"].astype(str).str.strip().ne("")
        & ~links["review_required"].astype(bool)
    ].copy()

    existing_entity_ids = set(
        entities["entity_id"].astype(str)
    )

    existing_identifier_keys: dict[str, set[str]] = {}
    for row in identifiers.to_dict("records"):
        if clean_text(row.get("verification_status")).lower() in {
            "rejected",
            "conflicted",
            "superseded",
        }:
            continue

        key = identifier_key(
            row.get("country"),
            row.get("identifier_type"),
            row.get("identifier_value_normalized"),
        )
        entity_id = clean_text(row.get("entity_id"))

        if key and entity_id:
            existing_identifier_keys.setdefault(
                key,
                set(),
            ).add(entity_id)

    alias_keys = {
        alias_key(
            row.get("entity_id"),
            row.get("country"),
            row.get("alias_name"),
            row.get("alias_type"),
        )
        for row in aliases.to_dict("records")
    }

    relationship_keys = {
        relationship_key(
            row.get("subject_entity_id"),
            row.get("object_entity_id"),
            row.get("relationship_type"),
        )
        for row in relationships.to_dict("records")
    }

    new_entity_rows: list[dict[str, Any]] = []
    new_identifier_rows: list[dict[str, Any]] = []
    new_alias_rows: list[dict[str, Any]] = []
    new_relationship_rows: list[dict[str, Any]] = []
    new_event_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []

    for row in accepted_links.to_dict("records"):
        entity_id = clean_text(row.get("matched_entity_id"))
        supplier_name = clean_text(
            row.get("supplier_name_original")
        )
        supplier_name_norm = (
            clean_text(row.get("supplier_name_norm"))
            or normalize_name(supplier_name)
        )
        country = clean_text(
            row.get("supplier_country")
        ).upper()
        supplier_record_key = clean_text(
            row.get("supplier_record_key")
        )
        source_client = clean_text(row.get("source_client"))
        reviewed_by = clean_text(row.get("reviewed_by"))
        reviewed_at = clean_text(row.get("reviewed_at"))
        matched_source_record_id = clean_text(
            row.get("matched_source_record_id")
        )

        if entity_id not in existing_entity_ids:
            new_entity_rows.append(
                {
                    "entity_id": entity_id,
                    "canonical_name": supplier_name,
                    "canonical_name_norm": normalize_name(
                        supplier_name_norm
                    ),
                    "country": country,
                    "entity_status": "unknown",
                    "record_status": "active",
                    "legal_form_local": "",
                    "base_legal_form_family": "",
                    "primary_identifier_id": None,
                    "primary_source_record_id": (
                        matched_source_record_id or None
                    ),
                    "identity_confidence": (
                        "high"
                        if clean_text(
                            row.get(
                                "supplier_identifier_value"
                            )
                        )
                        else "reviewed"
                    ),
                    "identity_review_status": "reviewed",
                    "created_at": utc_now(),
                    "created_by": (
                        reviewed_by
                        or "materialise_canonical_links.py"
                    ),
                    "updated_at": utc_now(),
                    "schema_version": SCHEMA_VERSION,
                }
            )
            existing_entity_ids.add(entity_id)

            new_event_rows.append(
                {
                    "event_id": allocate_id(EVENT_ID_PREFIX),
                    "event_type": "create",
                    "subject_entity_id": entity_id,
                    "target_entity_id": None,
                    "reason": (
                        "Created from an accepted supplier linkage."
                    ),
                    "decision_status": "approved",
                    "decided_by": (
                        reviewed_by
                        or "materialise_canonical_links.py"
                    ),
                    "decided_at": reviewed_at or utc_now(),
                    "effective_at": reviewed_at or utc_now(),
                    "migration_batch_id": (
                        f"accepted_supplier|{supplier_record_key}"
                    ),
                    "schema_version": SCHEMA_VERSION,
                }
            )

        identifier_type = clean_text(
            row.get("supplier_identifier_type")
        )
        identifier_value_raw = clean_text(
            row.get("supplier_identifier_value")
        )
        identifier_value_normalized = normalize_identifier(
            identifier_value_raw
        )

        key = identifier_key(
            country,
            identifier_type,
            identifier_value_normalized,
        )

        if key:
            linked_entities = existing_identifier_keys.get(
                key,
                set(),
            )

            if linked_entities and entity_id not in linked_entities:
                conflict_rows.append(
                    {
                        "supplier_record_key": supplier_record_key,
                        "entity_id": entity_id,
                        "identifier_key": key,
                        "conflicting_entity_ids": "|".join(
                            sorted(linked_entities)
                        ),
                        "reason": (
                            "Accepted linkage identifier is already "
                            "attached to another canonical entity."
                        ),
                    }
                )
            elif not linked_entities:
                identifier_id = allocate_id(
                    IDENTIFIER_ID_PREFIX
                )

                new_identifier_rows.append(
                    {
                        "identifier_id": identifier_id,
                        "entity_id": entity_id,
                        "identifier_type": identifier_type.upper(),
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
                        "source_record_id": (
                            matched_source_record_id or None
                        ),
                        "created_at": utc_now(),
                        "schema_version": SCHEMA_VERSION,
                    }
                )

                existing_identifier_keys.setdefault(
                    key,
                    set(),
                ).add(entity_id)

        current_alias_key = alias_key(
            entity_id,
            country,
            supplier_name,
            "client_variant",
        )

        if supplier_name and current_alias_key not in alias_keys:
            new_alias_rows.append(
                {
                    "alias_id": allocate_id(ALIAS_ID_PREFIX),
                    "entity_id": entity_id,
                    "alias_name": supplier_name,
                    "alias_name_norm": normalize_name(
                        supplier_name_norm
                    ),
                    "alias_type": "client_variant",
                    "language": "",
                    "country": country,
                    "is_preferred": False,
                    "valid_from": None,
                    "valid_to": None,
                    "source_record_id": (
                        matched_source_record_id or None
                    ),
                    "evidence_id": None,
                    "verification_status": "human_verified",
                    "review_status": "accepted",
                    "source_client": source_client,
                    "supplier_record_key": supplier_record_key,
                    "created_at": utc_now(),
                    "schema_version": SCHEMA_VERSION,
                }
            )
            alias_keys.add(current_alias_key)

        relationship_type = clean_text(
            row.get("relationship_type")
        ).lower()
        related_entity_id = clean_text(
            row.get("related_entity_id")
        )

        if (
            relationship_type
            and relationship_type != "legal_entity"
            and related_entity_id
        ):
            current_relationship_key = relationship_key(
                entity_id,
                related_entity_id,
                relationship_type,
            )

            if (
                entity_id == related_entity_id
                or related_entity_id not in existing_entity_ids
            ):
                conflict_rows.append(
                    {
                        "supplier_record_key": supplier_record_key,
                        "entity_id": entity_id,
                        "identifier_key": "",
                        "conflicting_entity_ids": (
                            related_entity_id
                        ),
                        "reason": (
                            "Relationship target is invalid, missing "
                            "or identical to the subject entity."
                        ),
                    }
                )
            elif (
                current_relationship_key
                not in relationship_keys
            ):
                new_relationship_rows.append(
                    {
                        "relationship_id": allocate_id(
                            RELATIONSHIP_ID_PREFIX
                        ),
                        "subject_entity_id": entity_id,
                        "object_entity_id": related_entity_id,
                        "relationship_type": relationship_type,
                        "relationship_status": "accepted",
                        "evidence_reference": "",
                        "source_client": source_client,
                        "supplier_record_key": (
                            supplier_record_key
                        ),
                        "reviewed_by": reviewed_by,
                        "reviewed_at": reviewed_at,
                        "created_at": utc_now(),
                        "schema_version": SCHEMA_VERSION,
                    }
                )
                relationship_keys.add(
                    current_relationship_key
                )

    if new_entity_rows:
        entities = pd.concat(
            [
                entities,
                ensure_columns(
                    pd.DataFrame(new_entity_rows),
                    list(entities.columns),
                )[entities.columns],
            ],
            ignore_index=True,
        )

    if new_identifier_rows:
        identifiers = pd.concat(
            [
                identifiers,
                ensure_columns(
                    pd.DataFrame(new_identifier_rows),
                    list(identifiers.columns),
                )[identifiers.columns],
            ],
            ignore_index=True,
        )

    if new_alias_rows:
        aliases = pd.concat(
            [
                aliases,
                pd.DataFrame(
                    new_alias_rows,
                    columns=ALIAS_COLUMNS,
                ),
            ],
            ignore_index=True,
        )

    if new_relationship_rows:
        relationships = pd.concat(
            [
                relationships,
                pd.DataFrame(
                    new_relationship_rows,
                    columns=RELATIONSHIP_COLUMNS,
                ),
            ],
            ignore_index=True,
        )

    if new_event_rows:
        events = pd.concat(
            [
                events,
                ensure_columns(
                    pd.DataFrame(new_event_rows),
                    list(events.columns),
                )[events.columns],
            ],
            ignore_index=True,
        )

    entities = entities.drop_duplicates(
        subset=["entity_id"],
        keep="first",
    )
    identifiers = identifiers.drop_duplicates(
        subset=[
            "entity_id",
            "country",
            "identifier_type",
            "identifier_value_normalized",
        ],
        keep="first",
    )
    aliases = aliases.drop_duplicates(
        subset=[
            "entity_id",
            "country",
            "alias_name_norm",
            "alias_type",
        ],
        keep="first",
    )
    relationships = relationships.drop_duplicates(
        subset=[
            "subject_entity_id",
            "object_entity_id",
            "relationship_type",
        ],
        keep="first",
    )
    events = events.drop_duplicates(
        subset=["event_id"],
        keep="first",
    )

    entities.to_parquet(entities_path, index=False)
    identifiers.to_parquet(identifiers_path, index=False)
    aliases.to_parquet(aliases_path, index=False)
    relationships.to_parquet(
        relationships_path,
        index=False,
    )
    events.to_parquet(events_path, index=False)

    conflicts = pd.DataFrame(conflict_rows)
    conflicts_path = (
        canonical_dir / "canonical_materialisation_review.parquet"
    )
    conflicts_csv_path = (
        canonical_dir / "canonical_materialisation_review.csv"
    )

    if conflicts.empty:
        conflicts = pd.DataFrame(
            columns=[
                "supplier_record_key",
                "entity_id",
                "identifier_key",
                "conflicting_entity_ids",
                "reason",
            ]
        )

    conflicts.to_parquet(conflicts_path, index=False)
    conflicts.to_csv(conflicts_csv_path, index=False)

    qa_path = canonical_dir / "canonical_materialisation_qa.duckdb"
    if qa_path.exists():
        qa_path.unlink()

    connection = duckdb.connect(str(qa_path))
    try:
        connection.execute(
            f"""
            CREATE VIEW canonical_entities AS
            SELECT *
            FROM read_parquet('{entities_path.as_posix()}')
            """
        )
        connection.execute(
            f"""
            CREATE VIEW entity_identifiers AS
            SELECT *
            FROM read_parquet('{identifiers_path.as_posix()}')
            """
        )
        connection.execute(
            f"""
            CREATE VIEW entity_aliases AS
            SELECT *
            FROM read_parquet('{aliases_path.as_posix()}')
            """
        )
        connection.execute(
            f"""
            CREATE VIEW entity_relationships AS
            SELECT *
            FROM read_parquet(
                '{relationships_path.as_posix()}'
            )
            """
        )
        connection.execute(
            f"""
            CREATE VIEW materialisation_review AS
            SELECT *
            FROM read_parquet('{conflicts_path.as_posix()}')
            """
        )
        connection.execute(
            """
            CREATE VIEW qa_duplicate_entity_ids AS
            SELECT entity_id, COUNT(*) AS row_count
            FROM canonical_entities
            GROUP BY entity_id
            HAVING COUNT(*) > 1
            """
        )
        connection.execute(
            """
            CREATE VIEW qa_identifier_conflicts AS
            SELECT
                country,
                identifier_type,
                identifier_value_normalized,
                COUNT(DISTINCT entity_id) AS entity_count
            FROM entity_identifiers
            WHERE verification_status NOT IN (
                'rejected',
                'conflicted',
                'superseded'
            )
            GROUP BY
                country,
                identifier_type,
                identifier_value_normalized
            HAVING COUNT(DISTINCT entity_id) > 1
            """
        )
        connection.execute(
            """
            CREATE VIEW qa_aliases_without_entities AS
            SELECT a.*
            FROM entity_aliases a
            LEFT JOIN canonical_entities e
              ON a.entity_id = e.entity_id
            WHERE e.entity_id IS NULL
            """
        )
        connection.execute(
            """
            CREATE VIEW qa_relationships_without_entities AS
            SELECT r.*
            FROM entity_relationships r
            LEFT JOIN canonical_entities subject
              ON r.subject_entity_id = subject.entity_id
            LEFT JOIN canonical_entities object
              ON r.object_entity_id = object.entity_id
            WHERE subject.entity_id IS NULL
               OR object.entity_id IS NULL
            """
        )
    finally:
        connection.close()

    summary = {
        "eligible_link_rows": int(len(accepted_links)),
        "new_entities_materialised": int(len(new_entity_rows)),
        "new_identifiers_materialised": int(
            len(new_identifier_rows)
        ),
        "new_aliases_materialised": int(len(new_alias_rows)),
        "new_relationships_materialised": int(
            len(new_relationship_rows)
        ),
        "materialisation_review_rows": int(len(conflicts)),
        "canonical_entities": int(len(entities)),
        "entity_identifiers": int(len(identifiers)),
        "entity_aliases": int(len(aliases)),
        "entity_relationships": int(len(relationships)),
    }

    manifest_path = (
        canonical_dir / "materialisation_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Materialise accepted supplier links into canonical tables."
        )
    )
    parser.add_argument(
        "--canonical-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing canonical tables and "
            "supplier_entity_links.parquet."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run canonical materialisation."""
    args = parse_args()
    summary = materialise_links(args.canonical_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
