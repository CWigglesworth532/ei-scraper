"""Apply the owner-approved ILUNION consolidation to an isolated store.

This script must never modify the operational canonical store. It applies
DEC-007 to a separate working copy, preserves the superseded entity row, and
writes deterministic audit evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from canonical_entity_layer import normalize_token


SURVIVOR_ENTITY_ID = (
    "sko_ent_184dcde9-369f-4ac4-a1f6-27469964bb28"
)
DUPLICATE_ENTITY_ID = (
    "sko_ent_652899e9-5728-4c88-bfcc-f8b2a587e48d"
)
DECISION_ID = "DEC-007"

ILUNION_SUPPLIERS = {
    "SB00173": "ILUNION",
    "SB00174": "ILUNION LAVANDERIAS Y SERVICIOS",
    "SB00175": "ILUNION LIMPIEZA Y MEDIOAMBIENTE",
    "SB00176": "ILUNION SEGURIDAD",
}

REQUIRED_TABLES = (
    "canonical_entities.parquet",
    "supplier_entity_links.parquet",
    "entity_aliases.parquet",
    "entity_identifiers.parquet",
    "source_records.parquet",
    "trusted_match_terms.parquet",
    "entity_events.parquet",
    "entity_relationships.parquet",
)

ID_NAMESPACE = uuid.UUID(
    "b74cab6c-bf7a-4f62-bb70-83dbdc4ac455"
)


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{uuid.uuid5(ID_NAMESPACE, value)}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_directory(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise FileNotFoundError(path)

    checksums = {
        file.relative_to(path).as_posix(): sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }

    if not checksums:
        raise ValueError(f"Store is empty: {path}")

    return checksums


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}"
        )


def load_tables(store: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    for filename in REQUIRED_TABLES:
        path = store / filename
        if not path.exists():
            raise FileNotFoundError(path)

        tables[filename] = pd.read_parquet(path)

    return tables


def assert_isolated_store(
    working_store: Path,
    operational_store: Path,
) -> None:
    if working_store.resolve() == operational_store.resolve():
        raise ValueError(
            "Working and operational stores must be different."
        )

    try:
        working_store.resolve().relative_to(
            operational_store.resolve()
        )
    except ValueError:
        pass
    else:
        raise ValueError(
            "Working store must not be inside the operational store."
        )


def validate_owner_review(
    owner_review: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "supplier_baseline_id",
        "supplier_name",
        "country_code",
        "owner_decision",
        "owner_approved_entity_id",
        "owner_review_notes",
    }
    require_columns(owner_review, required, "Owner review")

    selected = owner_review.loc[
        owner_review["supplier_baseline_id"].isin(
            ILUNION_SUPPLIERS
        )
    ].copy()

    if len(selected) != 4:
        raise ValueError(
            f"Expected four ILUNION decisions, found {len(selected)}."
        )

    if set(selected["supplier_baseline_id"]) != set(
        ILUNION_SUPPLIERS
    ):
        raise ValueError("Owner review does not contain the expected IDs.")

    wrong_decision = ~selected["owner_decision"].map(
        clean_text
    ).eq("reuse_existing_entity")

    wrong_entity = ~selected[
        "owner_approved_entity_id"
    ].map(clean_text).eq(SURVIVOR_ENTITY_ID)

    wrong_country = ~selected["country_code"].map(
        clean_text
    ).str.upper().eq("ES")

    if wrong_decision.any() or wrong_entity.any() or wrong_country.any():
        raise ValueError(
            "ILUNION owner decisions are incomplete or inconsistent."
        )

    return selected.sort_values(
        "supplier_baseline_id"
    ).reset_index(drop=True)


def update_entities(
    entities: pd.DataFrame,
    decision_at: str,
    decided_by: str,
) -> pd.DataFrame:
    required = {
        "entity_id",
        "canonical_name",
        "canonical_name_norm",
        "country",
        "entity_status",
        "record_status",
        "identity_confidence",
        "identity_review_status",
        "updated_at",
    }
    require_columns(entities, required, "canonical_entities")

    output = entities.copy()

    survivor_mask = output["entity_id"].map(
        clean_text
    ).eq(SURVIVOR_ENTITY_ID)
    duplicate_mask = output["entity_id"].map(
        clean_text
    ).eq(DUPLICATE_ENTITY_ID)

    if survivor_mask.sum() != 1:
        raise ValueError("Surviving ILUNION entity is not unique.")
    if duplicate_mask.sum() != 1:
        raise ValueError("Duplicate ILUNION entity is not unique.")

    output.loc[survivor_mask, "canonical_name"] = "ILUNION"
    output.loc[
        survivor_mask,
        "canonical_name_norm",
    ] = normalize_token("ILUNION")
    output.loc[survivor_mask, "country"] = "ES"
    output.loc[survivor_mask, "entity_status"] = "active"
    output.loc[survivor_mask, "record_status"] = "active"
    output.loc[
        survivor_mask,
        "identity_confidence",
    ] = "high"
    output.loc[
        survivor_mask,
        "identity_review_status",
    ] = "reviewed"
    output.loc[survivor_mask, "updated_at"] = decision_at

    output.loc[
        duplicate_mask,
        "entity_status",
    ] = "superseded"
    output.loc[
        duplicate_mask,
        "record_status",
    ] = "merged"
    output.loc[
        duplicate_mask,
        "identity_review_status",
    ] = "reviewed"
    output.loc[duplicate_mask, "updated_at"] = decision_at

    if "created_by" in output.columns:
        output.loc[
            survivor_mask,
            "created_by",
        ] = output.loc[
            survivor_mask,
            "created_by",
        ].fillna(decided_by)

    return output


def update_links(
    links: pd.DataFrame,
    owner_rows: pd.DataFrame,
    decision_at: str,
    decided_by: str,
    source_file: str,
) -> pd.DataFrame:
    required = {
        "supplier_record_key",
        "supplier_name_original",
        "supplier_name_norm",
        "supplier_country",
        "supplier_identifier_type",
        "supplier_identifier_value",
        "acceptance_status",
        "source_client",
        "source_client_file",
        "reviewed_by",
        "reviewed_at",
        "matched_entity_id",
        "matched_source_record_id",
        "resolution_status",
        "resolution_method",
        "allocate_new_entity",
        "review_required",
        "candidate_entity_ids",
        "resolution_reason",
        "relationship_type",
        "related_entity_id",
    }
    require_columns(links, required, "supplier_entity_links")

    output = links.copy()

    duplicate_mask = output["matched_entity_id"].map(
        clean_text
    ).eq(DUPLICATE_ENTITY_ID)

    output.loc[
        duplicate_mask,
        "matched_entity_id",
    ] = SURVIVOR_ENTITY_ID
    output.loc[
        duplicate_mask,
        "candidate_entity_ids",
    ] = SURVIVOR_ENTITY_ID
    output.loc[
        duplicate_mask,
        "resolution_status",
    ] = "reused"
    output.loc[
        duplicate_mask,
        "resolution_method",
    ] = "owner_approved_canonical_merge"
    output.loc[
        duplicate_mask,
        "allocate_new_entity",
    ] = False
    output.loc[
        duplicate_mask,
        "review_required",
    ] = False
    output.loc[
        duplicate_mask,
        "resolution_reason",
    ] = (
        "DEC-007 consolidated the Bayer-created ILUNION "
        "entity into the governed ILUNION directory entry."
    )

    new_rows: list[dict[str, Any]] = []

    for row in owner_rows.to_dict("records"):
        supplier_id = clean_text(
            row["supplier_baseline_id"]
        )
        supplier_name = clean_text(row["supplier_name"])
        supplier_record_key = (
            f"CBRE_AGGREGATE|{supplier_id}"
        )

        new_rows.append(
            {
                "supplier_record_key": supplier_record_key,
                "supplier_name_original": supplier_name,
                "supplier_name_norm": normalize_token(
                    supplier_name
                ),
                "supplier_country": "ES",
                "supplier_identifier_type": "",
                "supplier_identifier_value": "",
                "acceptance_status": "reviewed_confirmed",
                "source_client": "CBRE",
                "source_client_file": source_file,
                "reviewed_by": decided_by,
                "reviewed_at": decision_at,
                "matched_entity_id": SURVIVOR_ENTITY_ID,
                "matched_source_record_id": "",
                "resolution_status": "reused",
                "resolution_method": (
                    "owner_approved_group_canonical_reuse"
                ),
                "allocate_new_entity": False,
                "review_required": False,
                "candidate_entity_ids": SURVIVOR_ENTITY_ID,
                "resolution_reason": (
                    "DEC-007 owner-approved one governed ILUNION "
                    "canonical/directory entry."
                ),
                "relationship_type": "",
                "related_entity_id": "",
            }
        )

    additions = pd.DataFrame(new_rows, columns=output.columns)

    existing_keys = set(
        output["supplier_record_key"].map(clean_text)
    )
    additions = additions.loc[
        ~additions["supplier_record_key"].isin(existing_keys)
    ]

    output = pd.concat(
        [output, additions],
        ignore_index=True,
    )

    return output.sort_values(
        "supplier_record_key"
    ).reset_index(drop=True)


def update_aliases(
    aliases: pd.DataFrame,
    owner_rows: pd.DataFrame,
    decision_at: str,
) -> pd.DataFrame:
    required = {
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
    }
    require_columns(aliases, required, "entity_aliases")

    output = aliases.copy()

    output.loc[
        output["entity_id"].map(clean_text).eq(
            DUPLICATE_ENTITY_ID
        ),
        "entity_id",
    ] = SURVIVOR_ENTITY_ID

    alias_rows: list[dict[str, Any]] = []

    for row in owner_rows.to_dict("records"):
        supplier_id = clean_text(
            row["supplier_baseline_id"]
        )
        supplier_name = clean_text(row["supplier_name"])
        supplier_record_key = (
            f"CBRE_AGGREGATE|{supplier_id}"
        )

        alias_rows.append(
            {
                "alias_id": stable_id(
                    "sko_alias",
                    f"{DECISION_ID}|{supplier_record_key}",
                ),
                "entity_id": SURVIVOR_ENTITY_ID,
                "alias_name": supplier_name,
                "alias_name_norm": normalize_token(
                    supplier_name
                ),
                "alias_type": "client_variant",
                "language": "",
                "country": "ES",
                "is_preferred": (
                    supplier_name == "ILUNION"
                ),
                "valid_from": None,
                "valid_to": None,
                "source_record_id": "",
                "evidence_id": DECISION_ID,
                "verification_status": "human_verified",
                "review_status": "accepted",
                "source_client": "CBRE",
                "supplier_record_key": supplier_record_key,
                "created_at": decision_at,
                "schema_version": "1.0.0",
            }
        )

    additions = pd.DataFrame(
        alias_rows,
        columns=output.columns,
    )

    existing_ids = set(output["alias_id"].map(clean_text))
    additions = additions.loc[
        ~additions["alias_id"].isin(existing_ids)
    ]

    output = pd.concat(
        [output, additions],
        ignore_index=True,
    )

    return output.sort_values("alias_id").reset_index(drop=True)


def update_trusted_terms(
    trusted: pd.DataFrame,
    owner_rows: pd.DataFrame,
    decision_at: str,
    decided_by: str,
) -> pd.DataFrame:
    required = {
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
    }
    require_columns(
        trusted,
        required,
        "trusted_match_terms",
    )

    output = trusted.copy()

    output.loc[
        output["entity_id"].map(clean_text).eq(
            DUPLICATE_ENTITY_ID
        ),
        "entity_id",
    ] = SURVIVOR_ENTITY_ID

    rows: list[dict[str, Any]] = []

    for row in owner_rows.to_dict("records"):
        supplier_id = clean_text(
            row["supplier_baseline_id"]
        )
        supplier_name = clean_text(row["supplier_name"])
        supplier_record_key = (
            f"CBRE_AGGREGATE|{supplier_id}"
        )

        rows.append(
            {
                "trusted_term_id": stable_id(
                    "sko_term",
                    f"{DECISION_ID}|{supplier_record_key}",
                ),
                "entity_id": SURVIVOR_ENTITY_ID,
                "source_record_id": "",
                "term_type": "client_variant",
                "term_raw": supplier_name,
                "term_normalized": normalize_token(
                    supplier_name
                ),
                "country": "ES",
                "identifier_type": "",
                "identifier_value_normalized": "",
                "identifier_scope": "",
                "relationship_type": "group_directory_entry",
                "verification_status": "human_verified",
                "review_status": "accepted",
                "approved_for_matching": True,
                "approved_by": decided_by,
                "approved_at": decision_at,
                "evidence_reference": DECISION_ID,
                "source_client": "CBRE",
                "supplier_record_key": supplier_record_key,
                "created_at": decision_at,
                "schema_version": "1.0.0",
            }
        )

    additions = pd.DataFrame(rows, columns=output.columns)

    existing_ids = set(
        output["trusted_term_id"].map(clean_text)
    )
    additions = additions.loc[
        ~additions["trusted_term_id"].isin(existing_ids)
    ]

    output = pd.concat(
        [output, additions],
        ignore_index=True,
    )

    return output.sort_values(
        "trusted_term_id"
    ).reset_index(drop=True)


def update_relationships(
    relationships: pd.DataFrame,
    decision_at: str,
    decided_by: str,
) -> pd.DataFrame:
    required = {
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
    }
    require_columns(
        relationships,
        required,
        "entity_relationships",
    )

    relationship_id = stable_id(
        "sko_rel",
        f"{DECISION_ID}|{DUPLICATE_ENTITY_ID}|"
        f"{SURVIVOR_ENTITY_ID}|merged_into",
    )

    if relationship_id in set(
        relationships["relationship_id"].map(clean_text)
    ):
        return relationships.copy()

    addition = pd.DataFrame(
        [
            {
                "relationship_id": relationship_id,
                "subject_entity_id": DUPLICATE_ENTITY_ID,
                "object_entity_id": SURVIVOR_ENTITY_ID,
                "relationship_type": "merged_into",
                "relationship_status": "accepted",
                "evidence_reference": DECISION_ID,
                "source_client": "CBRE",
                "supplier_record_key": "",
                "reviewed_by": decided_by,
                "reviewed_at": decision_at,
                "created_at": decision_at,
                "schema_version": "1.0.0",
            }
        ],
        columns=relationships.columns,
    )

    return pd.concat(
        [relationships, addition],
        ignore_index=True,
    ).sort_values(
        "relationship_id"
    ).reset_index(drop=True)


def update_events(
    events: pd.DataFrame,
    decision_at: str,
    decided_by: str,
) -> pd.DataFrame:
    required = {
        "event_id",
        "event_type",
        "subject_entity_id",
        "target_entity_id",
        "reason",
        "decision_status",
        "decided_by",
        "decided_at",
        "effective_at",
        "migration_batch_id",
        "schema_version",
    }
    require_columns(events, required, "entity_events")

    event_id = stable_id(
        "sko_evt",
        f"{DECISION_ID}|{DUPLICATE_ENTITY_ID}|"
        f"{SURVIVOR_ENTITY_ID}|merge",
    )

    if event_id in set(events["event_id"].map(clean_text)):
        return events.copy()

    addition = pd.DataFrame(
        [
            {
                "event_id": event_id,
                "event_type": "merge",
                "subject_entity_id": DUPLICATE_ENTITY_ID,
                "target_entity_id": SURVIVOR_ENTITY_ID,
                "reason": (
                    "DEC-007 owner-approved one governed ILUNION "
                    "canonical and directory entry."
                ),
                "decision_status": "approved",
                "decided_by": decided_by,
                "decided_at": decision_at,
                "effective_at": decision_at,
                "migration_batch_id": (
                    "cbre_ilunion_merge_dec_007"
                ),
                "schema_version": "1.0.0",
            }
        ],
        columns=events.columns,
    )

    return pd.concat(
        [events, addition],
        ignore_index=True,
    ).sort_values(
        "event_id"
    ).reset_index(drop=True)


def validate_result(
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    entities = tables["canonical_entities.parquet"]
    links = tables["supplier_entity_links.parquet"]
    aliases = tables["entity_aliases.parquet"]
    identifiers = tables["entity_identifiers.parquet"]
    source_records = tables["source_records.parquet"]
    trusted = tables["trusted_match_terms.parquet"]
    events = tables["entity_events.parquet"]
    relationships = tables[
        "entity_relationships.parquet"
    ]

    survivor = entities.loc[
        entities["entity_id"].map(clean_text).eq(
            SURVIVOR_ENTITY_ID
        )
    ]
    duplicate = entities.loc[
        entities["entity_id"].map(clean_text).eq(
            DUPLICATE_ENTITY_ID
        )
    ]

    if len(survivor) != 1 or len(duplicate) != 1:
        raise ValueError("ILUNION entity rows are not unique.")

    if clean_text(
        survivor.iloc[0]["canonical_name"]
    ) != "ILUNION":
        raise ValueError("Surviving canonical name is not ILUNION.")

    if clean_text(
        duplicate.iloc[0]["record_status"]
    ) != "merged":
        raise ValueError("Duplicate entity is not marked merged.")

    duplicate_live_links = links[
        "matched_entity_id"
    ].map(clean_text).eq(DUPLICATE_ENTITY_ID).sum()

    duplicate_aliases = aliases["entity_id"].map(
        clean_text
    ).eq(DUPLICATE_ENTITY_ID).sum()

    duplicate_identifiers = identifiers[
        "entity_id"
    ].map(clean_text).eq(DUPLICATE_ENTITY_ID).sum()

    duplicate_sources = source_records["entity_id"].map(
        clean_text
    ).eq(DUPLICATE_ENTITY_ID).sum()

    duplicate_trusted = trusted["entity_id"].map(
        clean_text
    ).eq(DUPLICATE_ENTITY_ID).sum()

    if any(
        [
            duplicate_live_links,
            duplicate_aliases,
            duplicate_identifiers,
            duplicate_sources,
            duplicate_trusted,
        ]
    ):
        raise ValueError(
            "Live operational references remain on the duplicate "
            "ILUNION entity."
        )

    expected_cbre_keys = {
        f"CBRE_AGGREGATE|{supplier_id}"
        for supplier_id in ILUNION_SUPPLIERS
    }

    cbre_links = links.loc[
        links["supplier_record_key"].isin(
            expected_cbre_keys
        )
    ]

    if len(cbre_links) != 4:
        raise ValueError(
            f"Expected four CBRE links, found {len(cbre_links)}."
        )

    if not cbre_links["matched_entity_id"].map(
        clean_text
    ).eq(SURVIVOR_ENTITY_ID).all():
        raise ValueError("CBRE links do not all use the survivor.")

    merge_relationships = relationships.loc[
        relationships["subject_entity_id"]
        .map(clean_text)
        .eq(DUPLICATE_ENTITY_ID)
        & relationships["object_entity_id"]
        .map(clean_text)
        .eq(SURVIVOR_ENTITY_ID)
        & relationships["relationship_type"]
        .map(clean_text)
        .eq("merged_into")
    ]

    merge_events = events.loc[
        events["subject_entity_id"]
        .map(clean_text)
        .eq(DUPLICATE_ENTITY_ID)
        & events["target_entity_id"]
        .map(clean_text)
        .eq(SURVIVOR_ENTITY_ID)
        & events["event_type"]
        .map(clean_text)
        .eq("merge")
    ]

    if len(merge_relationships) != 1:
        raise ValueError(
            "Expected exactly one merged_into relationship."
        )
    if len(merge_events) != 1:
        raise ValueError("Expected exactly one merge event.")

    redirected_bayer_links = links.loc[
        links["source_client"]
        .map(clean_text)
        .str.casefold()
        .eq("bayer")
        & links["matched_entity_id"]
        .map(clean_text)
        .eq(SURVIVOR_ENTITY_ID)
        & links["resolution_method"]
        .map(clean_text)
        .eq("owner_approved_canonical_merge")
    ].copy()

    if len(redirected_bayer_links) != 1:
        raise ValueError(
            "Expected exactly one Bayer ILUNION supplier link "
            "redirected to the surviving entity; found "
            f"{len(redirected_bayer_links)}."
        )

    redirected_bayer_key = clean_text(
        redirected_bayer_links.iloc[0][
            "supplier_record_key"
        ]
    )

    if not redirected_bayer_key:
        raise ValueError(
            "Redirected Bayer ILUNION link has no supplier record key."
        )

    return {
        "canonical_entities": int(len(entities)),
        "active_surviving_ilunion_entities": int(
            entities["entity_id"]
            .map(clean_text)
            .eq(SURVIVOR_ENTITY_ID)
            .sum()
        ),
        "merged_duplicate_entities": int(
            entities["entity_id"]
            .map(clean_text)
            .eq(DUPLICATE_ENTITY_ID)
            .sum()
        ),
        "cbre_supplier_links_added": 4,
        "bayer_supplier_links_redirected": 1,
        "duplicate_live_links_remaining": int(
            duplicate_live_links
        ),
        "duplicate_aliases_remaining": int(
            duplicate_aliases
        ),
        "duplicate_identifiers_remaining": int(
            duplicate_identifiers
        ),
        "duplicate_source_records_remaining": int(
            duplicate_sources
        ),
        "duplicate_trusted_terms_remaining": int(
            duplicate_trusted
        ),
        "merge_relationships": int(
            len(merge_relationships)
        ),
        "merge_events": int(len(merge_events)),
    }


def write_tables(
    store: Path,
    tables: dict[str, pd.DataFrame],
) -> list[str]:
    """Write only tables whose logical contents changed.

    Avoiding unnecessary Parquet rewrites preserves byte-for-byte
    idempotency on repeated runs.
    """
    changed_files: list[str] = []

    for filename, frame in tables.items():
        path = store / filename
        proposed = frame.reset_index(drop=True)

        if path.exists():
            existing = pd.read_parquet(path).reset_index(
                drop=True
            )

            if existing.equals(proposed):
                continue

        proposed.to_parquet(
            path,
            index=False,
        )
        changed_files.append(filename)

    return changed_files


def apply_merge(
    *,
    working_store: Path,
    operational_store: Path,
    owner_review_path: Path,
    evidence_dir: Path,
    decision_at: str,
    decided_by: str,
    source_file: str,
) -> dict[str, Any]:
    assert_isolated_store(
        working_store,
        operational_store,
    )

    operational_before = checksum_directory(
        operational_store
    )
    working_before = checksum_directory(working_store)

    owner_review = pd.read_csv(
        owner_review_path,
        low_memory=False,
    )
    owner_rows = validate_owner_review(owner_review)

    tables = load_tables(working_store)

    tables["canonical_entities.parquet"] = (
        update_entities(
            tables["canonical_entities.parquet"],
            decision_at,
            decided_by,
        )
    )
    tables["supplier_entity_links.parquet"] = (
        update_links(
            tables["supplier_entity_links.parquet"],
            owner_rows,
            decision_at,
            decided_by,
            source_file,
        )
    )
    tables["entity_aliases.parquet"] = update_aliases(
        tables["entity_aliases.parquet"],
        owner_rows,
        decision_at,
    )
    tables["trusted_match_terms.parquet"] = (
        update_trusted_terms(
            tables["trusted_match_terms.parquet"],
            owner_rows,
            decision_at,
            decided_by,
        )
    )
    tables["entity_relationships.parquet"] = (
        update_relationships(
            tables["entity_relationships.parquet"],
            decision_at,
            decided_by,
        )
    )
    tables["entity_events.parquet"] = update_events(
        tables["entity_events.parquet"],
        decision_at,
        decided_by,
    )

    qa = validate_result(tables)
    changed_files = write_tables(
        working_store,
        tables,
    )

    reloaded = load_tables(working_store)
    qa_after_write = validate_result(reloaded)

    if qa != qa_after_write:
        raise RuntimeError(
            "QA changed after writing and reloading the store."
        )

    operational_after = checksum_directory(
        operational_store
    )

    if operational_before != operational_after:
        raise RuntimeError(
            "Operational store changed during isolated merge."
        )

    working_after = checksum_directory(working_store)

    evidence_dir.mkdir(parents=True, exist_ok=True)

    (
        evidence_dir / "operational_store_before.json"
    ).write_text(
        json.dumps(
            operational_before,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (
        evidence_dir / "operational_store_after.json"
    ).write_text(
        json.dumps(
            operational_after,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (
        evidence_dir / "working_store_before.json"
    ).write_text(
        json.dumps(
            working_before,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (
        evidence_dir / "working_store_after.json"
    ).write_text(
        json.dumps(
            working_after,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "status": "isolated_merge_implemented",
        "decision_id": DECISION_ID,
        "surviving_entity_id": SURVIVOR_ENTITY_ID,
        "duplicate_entity_id": DUPLICATE_ENTITY_ID,
        "decision_at": decision_at,
        "decided_by": decided_by,
        "working_store": str(working_store),
        "operational_store": str(operational_store),
        "operational_store_unchanged": True,
        "working_store_changed": (
            working_before != working_after
        ),
        "working_store_files_written": sorted(
            changed_files
        ),
        "working_store_file_write_count": len(
            changed_files
        ),
        "qa": qa_after_write,
        "canonical_merge_implemented": True,
        "canonical_merge_tested": False,
        "canonical_merge_accepted": False,
    }

    (
        evidence_dir / "ilunion_merge_manifest.json"
    ).write_text(
        json.dumps(summary, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply DEC-007 ILUNION consolidation to an "
            "isolated canonical working copy."
        )
    )
    parser.add_argument(
        "--working-store",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--operational-store",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--owner-review",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--decision-at",
        required=True,
    )
    parser.add_argument(
        "--decided-by",
        required=True,
    )
    parser.add_argument(
        "--source-file",
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = apply_merge(
        working_store=args.working_store,
        operational_store=args.operational_store,
        owner_review_path=args.owner_review,
        evidence_dir=args.evidence_dir,
        decision_at=args.decision_at,
        decided_by=args.decided_by,
        source_file=args.source_file,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
