#!/usr/bin/env python3
"""Build the first persisted skopia canonical entity layer.

E1.3 scope:
- persistent opaque canonical entity IDs;
- persistent source-record IDs;
- identifier-led source-record resolution;
- Parquet authoritative outputs;
- DuckDB QA views;
- no matching or classification changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


SCHEMA_VERSION = "1.0.0"
ENTITY_ID_PREFIX = "sko_ent_"
SOURCE_RECORD_ID_PREFIX = "sko_src_"
IDENTIFIER_ID_PREFIX = "sko_id_"
EVENT_ID_PREFIX = "sko_evt_"

SOURCE_COLUMNS = [
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
    "tax_id_root",
]


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def allocate_id(prefix: str) -> str:
    """Allocate an opaque persisted ID."""
    return f"{prefix}{uuid.uuid4()}"


def clean_text(value: Any) -> str:
    """Return a trimmed string representation."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_token(value: Any) -> str:
    """Normalize a value for internal continuity keys."""
    text = clean_text(value).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def stable_hash(value: str, length: int = 24) -> str:
    """Create a deterministic non-canonical helper hash."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def derive_source_id(row: pd.Series) -> str:
    """Derive a stable source-definition ID from source metadata."""
    source_basis = "|".join(
        [
            clean_text(row.get("country")).upper(),
            clean_text(row.get("ei_register_name")),
            clean_text(row.get("source_url")),
            clean_text(row.get("source_type")),
        ]
    )
    return f"sko_source_{stable_hash(source_basis)}"


def derive_record_fingerprint(row: pd.Series) -> str:
    """Fingerprint a complete source observation for change detection."""
    payload = {
        column: clean_text(row.get(column))
        for column in SOURCE_COLUMNS
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def derive_continuity_key(row: pd.Series) -> tuple[str, str]:
    """Return a conservative source-record continuity key and method."""
    source_id = clean_text(row["source_id"])
    country = clean_text(row.get("country")).upper()

    registration_number = normalize_token(
        row.get("ei_registration_number")
    )
    if registration_number:
        return (
            f"{source_id}|registration|{country}|{registration_number}",
            "source_registration_number",
        )

    tax_id = normalize_token(row.get("tax_id"))
    if tax_id:
        return (
            f"{source_id}|tax|{country}|{tax_id}",
            "source_tax_id",
        )

    fingerprint = clean_text(row["record_fingerprint"])
    return (
        f"{source_id}|fingerprint|{fingerprint}",
        "record_fingerprint",
    )


def derive_entity_identifier(
    row: pd.Series,
) -> dict[str, str] | None:
    """Return the strongest accepted country-scoped entity identifier.

    French SIRETs identify establishments. Their nine-digit SIREN root is
    therefore used as the legal-entity identifier where available.

    Other countries initially use the normalized tax ID directly. This is a
    deliberately narrow E1.3 rule and does not classify the organisation.
    """
    country = clean_text(row.get("country")).upper()
    tax_id = normalize_token(row.get("tax_id"))
    tax_id_root = normalize_token(row.get("tax_id_root"))

    if not country:
        return None

    if country == "FR":
        if len(tax_id_root) == 9 and tax_id_root.isdigit():
            return {
                "identifier_type": "FR_SIREN",
                "identifier_value_raw": clean_text(
                    row.get("tax_id_root")
                ),
                "identifier_value_normalized": tax_id_root,
                "country": country,
                "identifier_scope": "legal_entity",
            }
        return None

    if not tax_id:
        return None

    return {
        "identifier_type": f"{country}_TAX_ID",
        "identifier_value_raw": clean_text(row.get("tax_id")),
        "identifier_value_normalized": tax_id,
        "country": country,
        "identifier_scope": "legal_entity",
    }


def identifier_key(identifier: dict[str, str]) -> str:
    """Create a country/type/value identifier lookup key."""
    return "|".join(
        [
            identifier["country"],
            identifier["identifier_type"],
            identifier["identifier_value_normalized"],
        ]
    )


def read_input_csv(path: Path) -> pd.DataFrame:
    """Read source records while preserving all values as strings."""
    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    for column in SOURCE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    return frame


def read_prior_table(path: Path) -> pd.DataFrame:
    """Read an existing Parquet table or return an empty frame."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def build_prior_maps(
    prior_source_records: pd.DataFrame,
    prior_identifiers: pd.DataFrame,
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, set[str]],
]:
    """Build persisted observation and identifier lookup maps."""
    source_map: dict[str, dict[str, str]] = {}
    identifier_map: dict[str, set[str]] = {}

    if not prior_source_records.empty:
        for row in prior_source_records.to_dict("records"):
            observation_key = clean_text(
                row.get("observation_key")
            )
            if not observation_key:
                continue

            source_map[observation_key] = {
                "source_record_id": clean_text(
                    row.get("source_record_id")
                ),
                "entity_id": clean_text(row.get("entity_id")),
            }

    if not prior_identifiers.empty:
        for row in prior_identifiers.to_dict("records"):
            if clean_text(row.get("verification_status")) == "conflicted":
                continue

            key = "|".join(
                [
                    clean_text(row.get("country")).upper(),
                    clean_text(row.get("identifier_type")),
                    clean_text(
                        row.get("identifier_value_normalized")
                    ),
                ]
            )
            entity_id = clean_text(row.get("entity_id"))

            if key and entity_id:
                identifier_map.setdefault(key, set()).add(entity_id)

    return source_map, identifier_map


def consume_prior_source_mapping(
    prior_source_map: dict[str, dict[str, str]],
    observation_key: str,
) -> dict[str, str] | None:
    """Return the prior mapping for one persisted source observation."""
    return prior_source_map.get(observation_key)


def select_canonical_name(records: pd.DataFrame) -> str:
    """Select a simple current canonical name without altering identity."""
    name_column = (
        "entity_name_raw"
        if "entity_name_raw" in records.columns
        else "entity_name"
    )

    names = [
        clean_text(value)
        for value in records[name_column].tolist()
        if clean_text(value)
    ]

    if not names:
        return "Unnamed entity"

    return sorted(
        names,
        key=lambda value: (-len(value), value),
    )[0]


def build_canonical_layer(
    input_csv: Path,
    output_dir: Path,
    selection_csv: Path | None = None,
) -> dict[str, Any]:
    """Build or update the persisted canonical entity layer."""
    output_dir.mkdir(parents=True, exist_ok=True)

    source_records_path = output_dir / "source_records.parquet"
    canonical_entities_path = output_dir / "canonical_entities.parquet"
    identifiers_path = output_dir / "entity_identifiers.parquet"
    events_path = output_dir / "entity_events.parquet"

    prior_source_records = read_prior_table(source_records_path)
    prior_entities = read_prior_table(canonical_entities_path)
    prior_identifiers = read_prior_table(identifiers_path)
    prior_events = read_prior_table(events_path)

    prior_source_map, prior_identifier_map = build_prior_maps(
        prior_source_records,
        prior_identifiers,
    )

    source = read_input_csv(input_csv).copy()
    input_rows_before_selection = len(source)

    if selection_csv is not None:
        selection = pd.read_csv(
            selection_csv,
            dtype=str,
            keep_default_na=False,
        )

        required_selection_columns = {
            "country",
            "ei_register_name",
        }
        missing_columns = (
            required_selection_columns - set(selection.columns)
        )
        if missing_columns:
            raise ValueError(
                "Selection file is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        selection["country"] = (
            selection["country"]
            .astype(str)
            .str.strip()
            .str.upper()
        )
        selection["ei_register_name"] = (
            selection["ei_register_name"]
            .astype(str)
            .str.strip()
        )

        source["country"] = (
            source["country"]
            .astype(str)
            .str.strip()
            .str.upper()
        )
        source["ei_register_name"] = (
            source["ei_register_name"]
            .astype(str)
            .str.strip()
        )

        selected_parts: list[pd.DataFrame] = []

        if "tax_id" in selection.columns:
            tax_selection = selection.loc[
                selection["tax_id"].astype(str).str.strip().ne("")
            ].copy()

            if not tax_selection.empty:
                tax_selection["_selection_tax_id"] = (
                    tax_selection["tax_id"].map(normalize_token)
                )
                source["_selection_tax_id"] = (
                    source["tax_id"].map(normalize_token)
                )

                selected_parts.append(
                    source.merge(
                        tax_selection[
                            [
                                "country",
                                "ei_register_name",
                                "_selection_tax_id",
                            ]
                        ].drop_duplicates(),
                        on=[
                            "country",
                            "ei_register_name",
                            "_selection_tax_id",
                        ],
                        how="inner",
                    )
                )

        if "ei_registration_number" in selection.columns:
            registration_selection = selection.loc[
                selection["ei_registration_number"]
                .astype(str)
                .str.strip()
                .ne("")
            ].copy()

            if not registration_selection.empty:
                registration_selection[
                    "_selection_registration_number"
                ] = registration_selection[
                    "ei_registration_number"
                ].map(normalize_token)

                source[
                    "_selection_registration_number"
                ] = source[
                    "ei_registration_number"
                ].map(normalize_token)

                selected_parts.append(
                    source.merge(
                        registration_selection[
                            [
                                "country",
                                "ei_register_name",
                                "_selection_registration_number",
                            ]
                        ].drop_duplicates(),
                        on=[
                            "country",
                            "ei_register_name",
                            "_selection_registration_number",
                        ],
                        how="inner",
                    )
                )

        if not selected_parts:
            raise ValueError(
                "Selection file must contain at least one populated "
                "tax_id or ei_registration_number column."
            )

        source = pd.concat(
            selected_parts,
            ignore_index=True,
        ).drop_duplicates(
            subset=[
                column
                for column in SOURCE_COLUMNS
                if column in source.columns
            ],
            keep="first",
        )

        source = source.drop(
            columns=[
                "_selection_tax_id",
                "_selection_registration_number",
            ],
            errors="ignore",
        )

        if source.empty:
            raise ValueError(
                "The selection file did not resolve to any source records."
            )

    source["country"] = source["country"].map(
        lambda value: clean_text(value).upper()
    )
    source["source_id"] = source.apply(derive_source_id, axis=1)
    source["record_fingerprint"] = source.apply(
        derive_record_fingerprint,
        axis=1,
    )

    continuity = source.apply(
        derive_continuity_key,
        axis=1,
        result_type="expand",
    )
    continuity.columns = ["continuity_key", "continuity_method"]
    source = pd.concat([source, continuity], axis=1)

    source["duplicate_ordinal"] = (
        source.groupby(
            ["continuity_key", "record_fingerprint"],
            sort=False,
            dropna=False,
        )
        .cumcount()
        .astype(int)
    )

    source["observation_key"] = (
        source["continuity_key"]
        + "|fingerprint|"
        + source["record_fingerprint"]
        + "|occurrence|"
        + source["duplicate_ordinal"].astype(str)
    )

    source_rows: list[dict[str, Any]] = []
    identifier_rows: list[dict[str, Any]] = []
    newly_created_entities: set[str] = set()
    current_identifier_entities: dict[str, str] = {}
    conflict_keys: set[str] = set()

    existing_entity_ids = set()
    if not prior_entities.empty:
        existing_entity_ids = set(
            prior_entities["entity_id"].astype(str)
        )

    for row_number, (_, row) in enumerate(source.iterrows(), start=1):
        continuity_key = clean_text(row["continuity_key"])
        observation_key = clean_text(row["observation_key"])
        prior_mapping = consume_prior_source_mapping(
            prior_source_map,
            observation_key,
        )

        source_record_id = (
            prior_mapping["source_record_id"]
            if prior_mapping
            else allocate_id(SOURCE_RECORD_ID_PREFIX)
        )

        country = clean_text(row.get("country")).upper()
        entity_name = clean_text(row.get("entity_name"))

        entity_id = ""
        resolution_status = "linked"
        resolution_method = ""
        resolution_confidence = "low"
        resolution_reason = ""

        if not country or not entity_name:
            resolution_status = "quarantined"
            resolution_method = "required_field_validation"
            resolution_confidence = "none"
            resolution_reason = (
                "Missing required country or entity name."
            )
        elif prior_mapping and prior_mapping["entity_id"]:
            entity_id = prior_mapping["entity_id"]
            resolution_method = "persisted_source_mapping"
            resolution_confidence = "high"
            resolution_reason = (
                "Reused persisted source-record-to-entity mapping."
            )
        else:
            identifier = derive_entity_identifier(row)

            if identifier is not None:
                key = identifier_key(identifier)
                prior_entities_for_identifier = prior_identifier_map.get(
                    key,
                    set(),
                )

                if len(prior_entities_for_identifier) > 1:
                    conflict_keys.add(key)
                    resolution_status = "quarantined"
                    resolution_method = "identifier_conflict"
                    resolution_confidence = "none"
                    resolution_reason = (
                        "Identifier is already linked to multiple "
                        "canonical entities."
                    )
                elif len(prior_entities_for_identifier) == 1:
                    entity_id = next(iter(prior_entities_for_identifier))
                    resolution_method = "authoritative_identifier"
                    resolution_confidence = "high"
                    resolution_reason = (
                        "Linked using a persisted country-scoped "
                        "legal-entity identifier."
                    )
                elif key in current_identifier_entities:
                    entity_id = current_identifier_entities[key]
                    resolution_method = "authoritative_identifier"
                    resolution_confidence = "high"
                    resolution_reason = (
                        "Linked using a country-scoped legal-entity "
                        "identifier from the current build."
                    )
                else:
                    entity_id = allocate_id(ENTITY_ID_PREFIX)
                    current_identifier_entities[key] = entity_id
                    newly_created_entities.add(entity_id)
                    resolution_method = (
                        "new_authoritative_identifier_cluster"
                    )
                    resolution_confidence = "high"
                    resolution_reason = (
                        "Created a new entity from a non-conflicted "
                        "country-scoped legal-entity identifier."
                    )

                if entity_id:
                    identifier_rows.append(
                        {
                            "identifier_id": allocate_id(
                                IDENTIFIER_ID_PREFIX
                            ),
                            "entity_id": entity_id,
                            **identifier,
                            "is_primary": True,
                            "verification_status": "source_verified",
                            "source_record_id": source_record_id,
                            "created_at": utc_now(),
                            "schema_version": SCHEMA_VERSION,
                        }
                    )
            else:
                entity_id = allocate_id(ENTITY_ID_PREFIX)
                newly_created_entities.add(entity_id)
                resolution_method = "new_singleton_entity"
                resolution_confidence = "low"
                resolution_reason = (
                    "No accepted legal-entity identifier was available; "
                    "created a separate singleton entity."
                )

        source_payload = {
            column: clean_text(row.get(column))
            for column in source.columns
            if column not in {
                "source_id",
                "record_fingerprint",
                "continuity_key",
                "continuity_method",
            }
        }

        source_rows.append(
            {
                "source_record_id": source_record_id,
                "entity_id": entity_id or None,
                "source_id": clean_text(row["source_id"]),
                "source_record_key": continuity_key,
                "continuity_key": continuity_key,
                "continuity_method": clean_text(
                    row["continuity_method"]
                ),
                "record_occurrence": int(
                    row["duplicate_ordinal"]
                ),
                "observation_key": observation_key,
                "ingest_batch_id": stable_hash(
                    f"{input_csv.resolve()}|{clean_text(row.get('retrieved_at'))}"
                ),
                "source_row_number": row_number,
                "entity_name_raw": entity_name,
                "entity_name_norm": clean_text(
                    row.get("entity_name_norm")
                ),
                "country": country,
                "address_raw": clean_text(row.get("address")),
                "city": clean_text(row.get("city")),
                "postcode": clean_text(row.get("postcode")),
                "legal_form_local": clean_text(
                    row.get("legal_form_local")
                ),
                "source_status": "",
                "retrieved_at": clean_text(row.get("retrieved_at")),
                "source_url": clean_text(row.get("source_url")),
                "source_page": "",
                "record_fingerprint": clean_text(
                    row["record_fingerprint"]
                ),
                "resolution_status": resolution_status,
                "resolution_method": resolution_method,
                "resolution_confidence": resolution_confidence,
                "resolution_reason": resolution_reason,
                "raw_payload_json": json.dumps(
                    source_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "schema_version": SCHEMA_VERSION,
            }
        )

    current_source_records = pd.DataFrame(source_rows)

    if prior_source_records.empty:
        persisted_source_records = current_source_records
    else:
        current_observation_keys = set(
            current_source_records["observation_key"].astype(str)
        )

        retained_prior_source_records = prior_source_records.loc[
            ~prior_source_records["observation_key"]
            .astype(str)
            .isin(current_observation_keys)
        ]

        persisted_source_records = pd.concat(
            [
                retained_prior_source_records,
                current_source_records,
            ],
            ignore_index=True,
        )

    persisted_source_records = (
        persisted_source_records
        .drop_duplicates(
            subset=["source_record_id"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    prior_entity_records: dict[str, dict[str, Any]] = {}
    if not prior_entities.empty:
        prior_entity_records = {
            clean_text(row["entity_id"]): row
            for row in prior_entities.to_dict("records")
        }

    entity_rows: list[dict[str, Any]] = []
    linked = persisted_source_records.loc[
        persisted_source_records["entity_id"].notna()
        & persisted_source_records["entity_id"].ne("")
    ]

    for entity_id, group in linked.groupby("entity_id", sort=False):
        prior = prior_entity_records.get(entity_id, {})
        canonical_name = (
            clean_text(prior.get("canonical_name"))
            or select_canonical_name(group)
        )
        country = (
            clean_text(prior.get("country"))
            or clean_text(group.iloc[0]["country"])
        )

        entity_rows.append(
            {
                "entity_id": entity_id,
                "canonical_name": canonical_name,
                "canonical_name_norm": normalize_token(
                    canonical_name
                ),
                "country": country,
                "entity_status": clean_text(
                    prior.get("entity_status")
                )
                or "unknown",
                "record_status": clean_text(
                    prior.get("record_status")
                )
                or "active",
                "legal_form_local": clean_text(
                    prior.get("legal_form_local")
                ),
                "base_legal_form_family": clean_text(
                    prior.get("base_legal_form_family")
                ),
                "primary_identifier_id": clean_text(
                    prior.get("primary_identifier_id")
                )
                or None,
                "primary_source_record_id": clean_text(
                    group.iloc[0]["source_record_id"]
                ),
                "identity_confidence": (
                    "high"
                    if (
                        group["resolution_confidence"] == "high"
                    ).any()
                    else "low"
                ),
                "identity_review_status": (
                    "machine_resolved"
                    if (
                        group["resolution_confidence"] == "high"
                    ).any()
                    else "unreviewed"
                ),
                "created_at": clean_text(prior.get("created_at"))
                or utc_now(),
                "created_by": clean_text(prior.get("created_by"))
                or "canonical_entity_layer.py",
                "updated_at": utc_now(),
                "schema_version": SCHEMA_VERSION,
            }
        )

    canonical_entities = pd.DataFrame(entity_rows)

    new_identifiers = pd.DataFrame(identifier_rows)

    identifier_columns = [
        "identifier_id",
        "entity_id",
        "identifier_type",
        "identifier_value_raw",
        "identifier_value_normalized",
        "country",
        "identifier_scope",
        "is_primary",
        "verification_status",
        "source_record_id",
        "created_at",
        "schema_version",
    ]

    if new_identifiers.empty:
        new_identifiers = pd.DataFrame(
            columns=identifier_columns
        )

    if prior_identifiers.empty:
        current_identifiers = new_identifiers
    elif new_identifiers.empty:
        current_identifiers = prior_identifiers.copy()
    else:
        current_identifiers = pd.concat(
            [
                prior_identifiers,
                new_identifiers,
            ],
            ignore_index=True,
        )

    current_identifiers = (
        current_identifiers
        .drop_duplicates(
            subset=[
                "entity_id",
                "country",
                "identifier_type",
                "identifier_value_normalized",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )


    existing_event_entities = set()
    if not prior_events.empty:
        existing_event_entities = set(
            prior_events.loc[
                prior_events["event_type"].eq("create"),
                "subject_entity_id",
            ].astype(str)
        )

    event_rows = []
    for entity_id in canonical_entities["entity_id"].tolist():
        if entity_id in existing_event_entities:
            continue
        event_rows.append(
            {
                "event_id": allocate_id(EVENT_ID_PREFIX),
                "event_type": "create",
                "subject_entity_id": entity_id,
                "target_entity_id": None,
                "reason": "Initial canonical entity creation.",
                "decision_status": "approved",
                "decided_by": "canonical_entity_layer.py",
                "decided_at": utc_now(),
                "effective_at": utc_now(),
                "migration_batch_id": stable_hash(
                    str(input_csv.resolve())
                ),
                "schema_version": SCHEMA_VERSION,
            }
        )

    current_events = pd.DataFrame(event_rows)
    if prior_events.empty:
        entity_events = current_events
    elif current_events.empty:
        entity_events = prior_events
    else:
        entity_events = pd.concat(
            [prior_events, current_events],
            ignore_index=True,
        ).drop_duplicates("event_id")

    canonical_entities.to_parquet(
        canonical_entities_path,
        index=False,
    )
    persisted_source_records.to_parquet(
        source_records_path,
        index=False,
    )
    current_identifiers.to_parquet(
        identifiers_path,
        index=False,
    )
    entity_events.to_parquet(
        events_path,
        index=False,
    )

    qa_database_path = output_dir / "canonical_qa.duckdb"
    if qa_database_path.exists():
        qa_database_path.unlink()

    connection = duckdb.connect(str(qa_database_path))
    try:
        connection.execute(
            f"""
            CREATE VIEW canonical_entities AS
            SELECT * FROM read_parquet(
                '{canonical_entities_path.as_posix()}'
            )
            """
        )
        connection.execute(
            f"""
            CREATE VIEW source_records AS
            SELECT * FROM read_parquet(
                '{source_records_path.as_posix()}'
            )
            """
        )
        connection.execute(
            f"""
            CREATE VIEW entity_identifiers AS
            SELECT * FROM read_parquet(
                '{identifiers_path.as_posix()}'
            )
            """
        )
        connection.execute(
            """
            CREATE VIEW qa_duplicate_source_record_ids AS
            SELECT source_record_id, COUNT(*) AS row_count
            FROM source_records
            GROUP BY source_record_id
            HAVING COUNT(*) > 1
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
            GROUP BY
                country,
                identifier_type,
                identifier_value_normalized
            HAVING COUNT(DISTINCT entity_id) > 1
            """
        )
        connection.execute(
            """
            CREATE VIEW qa_source_records_by_resolution_status AS
            SELECT resolution_status, COUNT(*) AS row_count
            FROM source_records
            GROUP BY resolution_status
            ORDER BY resolution_status
            """
        )
        connection.execute(
            """
            CREATE VIEW qa_entity_cluster_sizes AS
            SELECT entity_id, COUNT(*) AS source_record_count
            FROM source_records
            WHERE entity_id IS NOT NULL
            GROUP BY entity_id
            """
        )
        connection.execute(
            """
            CREATE VIEW qa_entities_without_source_records AS
            SELECT entity_id
            FROM canonical_entities
            WHERE entity_id NOT IN (
                SELECT DISTINCT entity_id
                FROM source_records
                WHERE entity_id IS NOT NULL
            )
            """
        )
    finally:
        connection.close()

    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_rows_before_selection": int(
            input_rows_before_selection
        ),
        "selected_input_rows": int(len(source)),
        "selection_applied": selection_csv is not None,
        "selection_file": (
            str(selection_csv)
            if selection_csv is not None
            else None
        ),
        "selected_source_records": int(
            len(current_source_records)
        ),
        "source_records": int(
            len(persisted_source_records)
        ),
        "canonical_entities": int(len(canonical_entities)),
        "entity_identifiers": int(len(current_identifiers)),
        "linked_records": int(
            persisted_source_records["resolution_status"]
            .eq("linked")
            .sum()
        ),
        "quarantined_records": int(
            persisted_source_records["resolution_status"]
            .eq("quarantined")
            .sum()
        ),
        "new_entities": int(len(newly_created_entities)),
        "identifier_conflict_keys": int(len(conflict_keys)),
        "generated_at": utc_now(),
    }

    manifest_path = output_dir / "schema_manifest.json"
    manifest_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build persisted skopia canonical entity IDs."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Normalized source-record CSV.",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=None,
        help=(
            "Optional CSV identifying source records to materialise. "
            "Must contain country, ei_register_name and at least one "
            "populated tax_id or ei_registration_number."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Canonical v1 output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_canonical_layer(
        input_csv=args.input,
        output_dir=args.output_dir,
        selection_csv=args.selection,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
