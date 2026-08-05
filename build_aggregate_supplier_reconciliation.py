"""Read-only aggregate-supplier reconciliation against the canonical store.

This module prepares governed identity evidence for owner review. It must not
allocate canonical entity IDs, persist supplier links, or modify the operational
canonical store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ALLOWED_CATEGORIES = {
    "reuse_existing_identifier_or_source_supported",
    "reuse_existing_reviewed_evidence",
    "create_new_entity_candidate",
    "ambiguous_identity_owner_review",
    "unsupported_aggregate_only",
    "blocked",
}

REQUIRED_AGGREGATE_COLUMNS = {
    "supplier_baseline_id",
    "baseline_id",
    "supplier_name",
    "supplier_name_norm",
    "country_code",
    "country_name",
    "spend_total_eur",
    "source_record_ids_merged",
    "supplier_name_variants",
}

REQUIRED_ROW_LEVEL_COLUMNS = {
    "record_id",
    "source_input",
    "source_row_number",
    "supplier_name",
    "supplier_name_norm",
    "original_supplier_name",
    "erp_supplier_name",
    "erp_supplier_number",
    "country_code",
    "spend_amount_eur",
}

CANONICAL_TABLES = {
    "canonical_entities": "canonical_entities.parquet",
    "supplier_entity_links": "supplier_entity_links.parquet",
    "entity_aliases": "entity_aliases.parquet",
    "entity_identifiers": "entity_identifiers.parquet",
    "source_records": "source_records.parquet",
}

OUTPUT_COLUMNS = [
    "supplier_baseline_id",
    "baseline_id",
    "supplier_name",
    "supplier_name_norm",
    "country_code",
    "country_name",
    "spend_total_eur",
    "source_record_ids_merged",
    "supplier_name_variants",
    "source_workbook",
    "aggregate_sheet",
    "row_level_sheet",
    "expanded_source_row_count",
    "expanded_source_record_ids",
    "historical_matched_row_count",
    "historical_positive_row_count",
    "historical_safe_row_count",
    "historical_match_methods",
    "historical_matched_registers",
    "historical_matched_entity_names",
    "historical_identifiers",
    "canonical_supplier_link_candidate_ids",
    "canonical_identifier_candidate_ids",
    "canonical_source_record_candidate_ids",
    "canonical_reviewed_alias_candidate_ids",
    "canonical_name_country_candidate_ids",
    "canonical_candidate_entity_ids",
    "canonical_candidate_count",
    "candidate_entity_id",
    "candidate_canonical_name",
    "candidate_canonical_country",
    "evidence_basis",
    "identity_conflict",
    "reconciliation_category",
    "materiality_rank",
    "materiality_at_least_100k",
    "owner_decision",
    "owner_approved_entity_id",
    "owner_review_notes",
]


def clean_text(value: Any) -> str:
    """Return a stripped string, treating pandas nulls as blank."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_country(value: Any) -> str:
    """Normalize a country value to an uppercase token."""
    return clean_text(value).upper()


def normalize_identifier(value: Any) -> str:
    """Normalize an identifier for exact comparison."""
    return re.sub(r"[^A-Z0-9]", "", clean_text(value).upper())


def normalize_name(value: Any) -> str:
    """Normalize a name conservatively for candidate generation only."""
    text = clean_text(value).casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def split_semicolon_values(value: Any) -> list[str]:
    """Split a semicolon-delimited aggregate field deterministically."""
    text = clean_text(value)
    if not text:
        return []

    return [
        token.strip()
        for token in re.split(r"\s*;\s*", text)
        if token.strip()
    ]


def join_sorted(values: Iterable[Any]) -> str:
    """Return unique nonblank values in stable lexical order."""
    cleaned = {
        clean_text(value)
        for value in values
        if clean_text(value)
    }
    return " | ".join(sorted(cleaned))


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    """Fail clearly when an input schema is incomplete."""
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}"
        )


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 digest for one file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def checksum_directory(path: Path) -> dict[str, str]:
    """Calculate deterministic checksums for all files in a directory."""
    if not path.is_dir():
        raise FileNotFoundError(
            f"Canonical store directory not found: {path}"
        )

    checksums: dict[str, str] = {}

    for file_path in sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
    ):
        relative = file_path.relative_to(path).as_posix()
        checksums[relative] = sha256_file(file_path)

    if not checksums:
        raise ValueError(f"Canonical store is empty: {path}")

    return checksums


def checksum_manifest_digest(
    checksums: dict[str, str],
) -> str:
    """Return one stable digest representing a checksum manifest."""
    payload = json.dumps(
        checksums,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_baseline_rows(
    aggregate: pd.DataFrame,
    baseline_id: str,
) -> pd.DataFrame:
    """Select one authoritative aggregate client population."""
    require_columns(
        aggregate,
        REQUIRED_AGGREGATE_COLUMNS,
        "Aggregate sheet",
    )

    target = clean_text(baseline_id).casefold()
    selected = aggregate.loc[
        aggregate["baseline_id"]
        .map(clean_text)
        .str.casefold()
        .eq(target)
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No aggregate rows found for baseline_id={baseline_id!r}"
        )

    selected["supplier_baseline_id"] = selected[
        "supplier_baseline_id"
    ].map(clean_text)

    blank_keys = selected["supplier_baseline_id"].eq("")
    if blank_keys.any():
        raise ValueError(
            "Selected aggregate population contains blank "
            "supplier_baseline_id values."
        )

    duplicated_keys = selected[
        "supplier_baseline_id"
    ].duplicated(keep=False)

    if duplicated_keys.any():
        values = sorted(
            selected.loc[
                duplicated_keys,
                "supplier_baseline_id",
            ].unique()
        )
        raise ValueError(
            "Selected aggregate population contains duplicate "
            f"supplier_baseline_id values: {values}"
        )

    selected["spend_total_eur"] = pd.to_numeric(
        selected["spend_total_eur"],
        errors="raise",
    )

    return selected.reset_index(drop=True)


def expand_source_rows(
    aggregate_rows: pd.DataFrame,
    row_level: pd.DataFrame,
) -> pd.DataFrame:
    """Expand aggregate source IDs and resolve each exactly once.

    A source record ID may not occur under more than one aggregate supplier.
    Every expanded source ID must resolve exactly once in row_level.record_id.
    """
    require_columns(
        aggregate_rows,
        REQUIRED_AGGREGATE_COLUMNS,
        "Selected aggregate population",
    )
    require_columns(
        row_level,
        REQUIRED_ROW_LEVEL_COLUMNS,
        "Row-level sheet",
    )

    expanded_rows: list[dict[str, Any]] = []

    for row in aggregate_rows.to_dict("records"):
        source_ids = split_semicolon_values(
            row.get("source_record_ids_merged")
        )

        if not source_ids:
            raise ValueError(
                "Aggregate supplier has no source_record_ids_merged: "
                f"{row.get('supplier_baseline_id')}"
            )

        if len(source_ids) != len(set(source_ids)):
            raise ValueError(
                "Aggregate supplier contains duplicate source record IDs: "
                f"{row.get('supplier_baseline_id')}"
            )

        for source_record_id in source_ids:
            expanded_rows.append(
                {
                    "supplier_baseline_id": clean_text(
                        row.get("supplier_baseline_id")
                    ),
                    "aggregate_supplier_name": clean_text(
                        row.get("supplier_name")
                    ),
                    "aggregate_country": normalize_country(
                        row.get("country_code")
                    ),
                    "aggregate_spend_eur": float(
                        row.get("spend_total_eur")
                    ),
                    "source_record_id": source_record_id,
                }
            )

    expanded = pd.DataFrame(expanded_rows)

    duplicated_across_suppliers = expanded[
        "source_record_id"
    ].duplicated(keep=False)

    if duplicated_across_suppliers.any():
        conflicts = (
            expanded.loc[
                duplicated_across_suppliers,
                [
                    "source_record_id",
                    "supplier_baseline_id",
                ],
            ]
            .sort_values(
                ["source_record_id", "supplier_baseline_id"]
            )
            .to_dict("records")
        )
        raise ValueError(
            "Source record IDs occur under multiple aggregate "
            f"suppliers: {conflicts}"
        )

    row_level = row_level.copy()
    row_level["record_id"] = row_level["record_id"].map(clean_text)

    row_counts = row_level["record_id"].value_counts()
    expanded["_row_level_match_count"] = (
        expanded["source_record_id"]
        .map(row_counts)
        .fillna(0)
        .astype(int)
    )

    unresolved = expanded.loc[
        expanded["_row_level_match_count"].eq(0),
        "source_record_id",
    ].tolist()

    if unresolved:
        raise ValueError(
            "Source record IDs do not resolve in the row-level sheet: "
            f"{sorted(unresolved)}"
        )

    duplicated_row_level = expanded.loc[
        expanded["_row_level_match_count"].gt(1),
        "source_record_id",
    ].tolist()

    if duplicated_row_level:
        raise ValueError(
            "Source record IDs resolve multiple times in the row-level "
            f"sheet: {sorted(duplicated_row_level)}"
        )

    resolved = expanded.merge(
        row_level,
        how="left",
        left_on="source_record_id",
        right_on="record_id",
        validate="one_to_one",
        suffixes=("_aggregate", "_source"),
    )

    source_mismatch = (
        resolved["source_input"]
        .map(clean_text)
        .str.casefold()
        .ne(
            aggregate_rows["baseline_id"]
            .iloc[0]
            .strip()
            .casefold()
        )
    )

    if source_mismatch.any():
        mismatches = resolved.loc[
            source_mismatch,
            [
                "supplier_baseline_id",
                "source_record_id",
                "source_input",
            ],
        ].to_dict("records")
        raise ValueError(
            "Resolved source rows belong to a different baseline: "
            f"{mismatches}"
        )

    return resolved.drop(
        columns=["_row_level_match_count"]
    ).sort_values(
        ["supplier_baseline_id", "source_record_id"]
    ).reset_index(drop=True)


def load_canonical_tables(
    canonical_store: Path,
) -> dict[str, pd.DataFrame]:
    """Load required canonical tables without modifying the store."""
    tables: dict[str, pd.DataFrame] = {}

    for label, filename in CANONICAL_TABLES.items():
        path = canonical_store / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Required canonical table missing: {path}"
            )
        tables[label] = pd.read_parquet(path)

    return tables


def accepted_link_rows(
    links: pd.DataFrame,
) -> pd.DataFrame:
    """Return persisted supplier links eligible as accepted evidence."""
    if links.empty:
        return links.copy()

    required = {
        "supplier_record_key",
        "matched_entity_id",
        "acceptance_status",
        "review_required",
        "resolution_status",
    }
    require_columns(links, required, "supplier_entity_links")

    accepted_statuses = {
        "accepted",
        "reviewed_confirmed",
        "confirmed",
    }

    return links.loc[
        links["matched_entity_id"].map(clean_text).ne("")
        & links["acceptance_status"]
        .map(clean_text)
        .str.casefold()
        .isin(accepted_statuses)
        & ~links["review_required"].fillna(False).astype(bool)
        & ~links["resolution_status"]
        .map(clean_text)
        .str.casefold()
        .isin({"review", "blocked", "rejected"})
    ].copy()


def accepted_identifier_rows(
    identifiers: pd.DataFrame,
) -> pd.DataFrame:
    """Return identifiers eligible as exact accepted evidence."""
    if identifiers.empty:
        return identifiers.copy()

    required = {
        "entity_id",
        "identifier_value_normalized",
        "verification_status",
    }
    require_columns(identifiers, required, "entity_identifiers")

    rejected = {
        "rejected",
        "conflicted",
        "invalid",
        "unverified",
    }

    return identifiers.loc[
        identifiers["entity_id"].map(clean_text).ne("")
        & identifiers[
            "identifier_value_normalized"
        ].map(normalize_identifier).ne("")
        & ~identifiers["verification_status"]
        .map(clean_text)
        .str.casefold()
        .isin(rejected)
    ].copy()


def reviewed_alias_rows(
    aliases: pd.DataFrame,
) -> pd.DataFrame:
    """Return aliases explicitly eligible as reviewed identity evidence."""
    if aliases.empty:
        return aliases.copy()

    required = {
        "entity_id",
        "alias_name",
        "country",
        "verification_status",
        "review_status",
    }
    require_columns(aliases, required, "entity_aliases")

    accepted_review = {
        "reviewed",
        "approved",
        "accepted",
        "reviewed_confirmed",
    }
    rejected_verification = {
        "rejected",
        "conflicted",
        "invalid",
    }

    return aliases.loc[
        aliases["entity_id"].map(clean_text).ne("")
        & aliases["alias_name"].map(clean_text).ne("")
        & aliases["review_status"]
        .map(clean_text)
        .str.casefold()
        .isin(accepted_review)
        & ~aliases["verification_status"]
        .map(clean_text)
        .str.casefold()
        .isin(rejected_verification)
    ].copy()


def build_entity_metadata(
    entities: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    """Create a minimal entity metadata lookup."""
    require_columns(
        entities,
        {"entity_id", "canonical_name", "country"},
        "canonical_entities",
    )

    metadata: dict[str, dict[str, str]] = {}

    for row in entities.to_dict("records"):
        entity_id = clean_text(row.get("entity_id"))
        if not entity_id:
            continue

        metadata[entity_id] = {
            "canonical_name": clean_text(
                row.get("canonical_name")
            ),
            "country": normalize_country(row.get("country")),
        }

    return metadata


def build_name_country_lookup(
    frame: pd.DataFrame,
    entity_column: str,
    name_column: str,
    country_column: str,
) -> dict[str, set[str]]:
    """Build a country-aware candidate lookup."""
    if frame.empty:
        return {}

    require_columns(
        frame,
        {entity_column, name_column, country_column},
        "Name-country candidate frame",
    )

    lookup: dict[str, set[str]] = defaultdict(set)

    for row in frame.to_dict("records"):
        entity_id = clean_text(row.get(entity_column))
        name = normalize_name(row.get(name_column))
        country = normalize_country(row.get(country_column))

        if entity_id and name and country:
            lookup[f"{country}|{name}"].add(entity_id)

    return dict(lookup)


def build_identifier_lookup(
    identifiers: pd.DataFrame,
) -> dict[str, set[str]]:
    """Build an exact normalized-identifier entity lookup."""
    lookup: dict[str, set[str]] = defaultdict(set)

    for row in accepted_identifier_rows(
        identifiers
    ).to_dict("records"):
        identifier = normalize_identifier(
            row.get("identifier_value_normalized")
        )
        entity_id = clean_text(row.get("entity_id"))

        if identifier and entity_id:
            lookup[identifier].add(entity_id)

    return dict(lookup)


def decide_reconciliation_category(
    *,
    identifier_or_source_candidates: set[str],
    reviewed_evidence_candidates: set[str],
    name_only_candidates: set[str],
    has_historical_positive_evidence: bool,
    is_blocked: bool = False,
) -> tuple[str, set[str], str]:
    """Select a conservative reconciliation category.

    Returns category, all candidate IDs considered, and an evidence basis.
    """
    if is_blocked:
        return "blocked", set(), "blocked_input_or_schema_condition"

    strong = set(identifier_or_source_candidates)
    reviewed = set(reviewed_evidence_candidates)
    names = set(name_only_candidates)

    all_candidates = strong | reviewed | names

    if len(strong) > 1:
        return (
            "ambiguous_identity_owner_review",
            all_candidates,
            "conflicting_identifier_or_source_candidates",
        )

    if len(strong) == 1:
        strong_id = next(iter(strong))
        conflicting = (
            reviewed | names
        ) - {strong_id}

        if conflicting:
            return (
                "ambiguous_identity_owner_review",
                all_candidates,
                "strong_candidate_conflicts_with_other_evidence",
            )

        return (
            "reuse_existing_identifier_or_source_supported",
            all_candidates,
            "accepted_identifier_or_persisted_source_evidence",
        )

    if len(reviewed) > 1:
        return (
            "ambiguous_identity_owner_review",
            all_candidates,
            "conflicting_reviewed_evidence_candidates",
        )

    if len(reviewed) == 1:
        reviewed_id = next(iter(reviewed))
        conflicting = names - {reviewed_id}

        if conflicting:
            return (
                "ambiguous_identity_owner_review",
                all_candidates,
                "reviewed_candidate_conflicts_with_name_evidence",
            )

        return (
            "reuse_existing_reviewed_evidence",
            all_candidates,
            "accepted_reviewed_alias_or_cross_client_evidence",
        )

    if names:
        return (
            "ambiguous_identity_owner_review",
            all_candidates,
            "name_country_candidate_only",
        )

    if has_historical_positive_evidence:
        return (
            "create_new_entity_candidate",
            set(),
            "historical_positive_evidence_without_existing_entity",
        )

    return (
        "unsupported_aggregate_only",
        set(),
        "no_accepted_identity_or_historical_positive_evidence",
    )


def validate_reconciliation(
    reconciliation: pd.DataFrame,
    *,
    expected_rows: int,
    expected_spend_eur: float,
) -> None:
    """Validate complete row, key, category and spend accounting."""
    require_columns(
        reconciliation,
        {
            "supplier_baseline_id",
            "spend_total_eur",
            "reconciliation_category",
        },
        "Reconciliation output",
    )

    if len(reconciliation) != expected_rows:
        raise ValueError(
            "Reconciliation row count mismatch: "
            f"expected {expected_rows}, got {len(reconciliation)}"
        )

    if reconciliation[
        "supplier_baseline_id"
    ].duplicated().any():
        raise ValueError(
            "Reconciliation contains duplicate supplier_baseline_id "
            "values."
        )

    categories = set(
        reconciliation["reconciliation_category"].map(clean_text)
    )
    invalid_categories = sorted(
        categories - ALLOWED_CATEGORIES
    )
    if invalid_categories:
        raise ValueError(
            "Reconciliation contains invalid categories: "
            f"{invalid_categories}"
        )

    spend = pd.to_numeric(
        reconciliation["spend_total_eur"],
        errors="raise",
    ).sum()

    if abs(float(spend) - float(expected_spend_eur)) > 0.01:
        raise ValueError(
            "Reconciliation spend mismatch: "
            f"expected {expected_spend_eur:.2f}, got {spend:.2f}"
        )


def build_manifest_summary(
    reconciliation: pd.DataFrame,
    *,
    expanded_source_rows: int,
    unique_source_record_ids: int,
    expected_rows: int,
    expected_spend_eur: float,
    canonical_checksums_before: dict[str, str],
    canonical_checksums_after: dict[str, str],
) -> dict[str, Any]:
    """Build a deterministic reconciliation manifest summary."""
    validate_reconciliation(
        reconciliation,
        expected_rows=expected_rows,
        expected_spend_eur=expected_spend_eur,
    )

    if canonical_checksums_before != canonical_checksums_after:
        raise RuntimeError(
            "Canonical-store checksums changed during read-only "
            "reconciliation."
        )

    category_counts = (
        reconciliation["reconciliation_category"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )

    category_spend = (
        reconciliation.groupby(
            "reconciliation_category",
            dropna=False,
        )["spend_total_eur"]
        .sum()
        .sort_index()
        .round(2)
        .to_dict()
    )

    evidence_counts = (
        reconciliation["evidence_basis"]
        .map(clean_text)
        .replace("", "<BLANK>")
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
        if "evidence_basis" in reconciliation.columns
        else {}
    )

    return {
        "aggregate_rows": int(len(reconciliation)),
        "aggregate_spend_eur": round(
            float(reconciliation["spend_total_eur"].sum()),
            2,
        ),
        "expanded_source_rows": int(expanded_source_rows),
        "unique_source_record_ids": int(
            unique_source_record_ids
        ),
        "category_counts": category_counts,
        "category_spend_eur": category_spend,
        "evidence_basis_counts": evidence_counts,
        "canonical_store_file_count": len(
            canonical_checksums_before
        ),
        "canonical_store_checksum_digest": (
            checksum_manifest_digest(
                canonical_checksums_before
            )
        ),
        "canonical_store_unchanged": True,
    }



def historical_row_key(
    country: Any,
    supplier_name: Any,
    tax_identifier: Any,
    address: Any,
) -> str:
    """Build a stable historical-row comparison key."""
    return "|".join(
        [
            normalize_country(country),
            normalize_name(supplier_name),
            normalize_identifier(tax_identifier),
            normalize_name(address),
        ]
    )


def prepare_historical_matched(
    historical: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize the historical CBRE matched file."""
    required = {
        "0",
        "1",
        "2",
        "3",
        "4",
        "_supplier_raw_name",
        "supplier_country",
        "matched_register",
        "matched_entity_name",
        "match_type",
        "match_score",
    }
    require_columns(
        historical,
        required,
        "Historical matched file",
    )

    prepared = historical.copy().reset_index(drop=True)
    prepared["_historical_position"] = prepared.index.astype(int)

    country_disagreement = (
        prepared["0"].map(normalize_country)
        != prepared["supplier_country"].map(normalize_country)
    )
    name_disagreement = (
        prepared["1"].map(normalize_name)
        != prepared["_supplier_raw_name"].map(normalize_name)
    )

    if country_disagreement.any():
        raise ValueError(
            "Historical matched file has disagreement between "
            "column 0 and supplier_country."
        )

    if name_disagreement.any():
        raise ValueError(
            "Historical matched file has disagreement between "
            "column 1 and _supplier_raw_name."
        )

    prepared["_country_norm"] = prepared[
        "supplier_country"
    ].map(normalize_country)
    prepared["_supplier_name_norm"] = prepared[
        "_supplier_raw_name"
    ].map(normalize_name)
    prepared["_identifier_norm"] = prepared["2"].map(
        normalize_identifier
    )
    prepared["_row_key"] = prepared.apply(
        lambda row: historical_row_key(
            row.get("supplier_country"),
            row.get("_supplier_raw_name"),
            row.get("2"),
            row.get("3"),
        ),
        axis=1,
    )
    prepared["_has_match"] = prepared["match_type"].map(
        clean_text
    ).ne("")

    return prepared



def attach_historical_rows(
    resolved_source_rows: pd.DataFrame,
    historical_matched: pd.DataFrame,
    *,
    source_row_offset: int | None = None,
) -> pd.DataFrame:
    """Attach historical evidence by exact country-aware source names.

    The aggregate workbook's source_row_number is local to the combined
    workbook and is not a positional key into the independently ordered
    historical CBRE file. Historical evidence is therefore retrieved using
    exact normalized country plus any preserved source supplier name.

    Zero historical matches are permitted. Multiple historical rows are
    retained as separate evidence observations and must not be treated as
    accepted identity.
    """
    del source_row_offset

    required = {
        "supplier_baseline_id",
        "source_record_id",
        "source_row_number",
        "supplier_name",
        "original_supplier_name",
        "erp_supplier_name",
        "country_code",
    }
    require_columns(
        resolved_source_rows,
        required,
        "Resolved source rows",
    )

    historical = prepare_historical_matched(
        historical_matched
    )

    source = resolved_source_rows.copy()
    source["_country_norm"] = source[
        "country_code"
    ].map(normalize_country)

    source_name_rows: list[dict[str, Any]] = []

    for row in source.to_dict("records"):
        names = {
            normalize_name(row.get("supplier_name")),
            normalize_name(row.get("original_supplier_name")),
            normalize_name(row.get("erp_supplier_name")),
        } - {""}

        if not names:
            source_name_rows.append(
                {
                    "source_record_id": clean_text(
                        row.get("source_record_id")
                    ),
                    "_source_candidate_name_norm": "",
                }
            )
            continue

        for name in sorted(names):
            source_name_rows.append(
                {
                    "source_record_id": clean_text(
                        row.get("source_record_id")
                    ),
                    "_source_candidate_name_norm": name,
                }
            )

    source_names = pd.DataFrame(source_name_rows)

    source = source.merge(
        source_names,
        on="source_record_id",
        how="left",
        validate="one_to_many",
    )

    historical_columns = [
        "_historical_position",
        "0",
        "1",
        "2",
        "3",
        "4",
        "matched_register",
        "matched_entity_name",
        "match_type",
        "match_score",
        "match_country",
        "match_region",
        "_country_norm",
        "_supplier_name_norm",
        "_identifier_norm",
        "_row_key",
        "_has_match",
    ]

    attached = source.merge(
        historical[historical_columns],
        how="left",
        left_on=[
            "_country_norm",
            "_source_candidate_name_norm",
        ],
        right_on=[
            "_country_norm",
            "_supplier_name_norm",
        ],
        validate="many_to_many",
        suffixes=("_source", "_historical"),
    )

    # A source observation may expose the same historical row through more
    # than one preserved source-name field. Retain it only once.
    matched = attached.loc[
        attached["_historical_position"].notna()
    ].copy()

    matched = matched.drop_duplicates(
        subset=[
            "source_record_id",
            "_historical_position",
        ],
        keep="first",
    )

    matched["_historical_candidate_found"] = True
    matched["_has_match"] = (
        matched["_has_match"].fillna(False).astype(bool)
    )

    matched_source_ids = set(
        matched["source_record_id"].map(clean_text)
    )

    unmatched = resolved_source_rows.loc[
        ~resolved_source_rows["source_record_id"]
        .map(clean_text)
        .isin(matched_source_ids)
    ].copy()

    if not unmatched.empty:
        unmatched["_country_norm"] = unmatched[
            "country_code"
        ].map(normalize_country)
        unmatched["_source_candidate_name_norm"] = ""
        unmatched["_historical_position"] = pd.NA
        unmatched["0"] = ""
        unmatched["1"] = ""
        unmatched["2"] = ""
        unmatched["3"] = ""
        unmatched["4"] = ""
        unmatched["matched_register"] = ""
        unmatched["matched_entity_name"] = ""
        unmatched["match_type"] = ""
        unmatched["match_score"] = ""
        unmatched["match_country"] = ""
        unmatched["match_region"] = ""
        unmatched["_supplier_name_norm"] = ""
        unmatched["_identifier_norm"] = ""
        unmatched["_row_key"] = ""
        unmatched["_has_match"] = False
        unmatched["_historical_candidate_found"] = False

    combined = pd.concat(
        [matched, unmatched],
        ignore_index=True,
        sort=False,
    )

    expected_source_ids = set(
        resolved_source_rows[
            "source_record_id"
        ].map(clean_text)
    )
    returned_source_ids = set(
        combined["source_record_id"].map(clean_text)
    )

    if expected_source_ids != returned_source_ids:
        missing = sorted(
            expected_source_ids - returned_source_ids
        )
        extra = sorted(
            returned_source_ids - expected_source_ids
        )
        raise ValueError(
            "Historical evidence attachment changed the resolved "
            f"source population; missing={missing}, extra={extra}"
        )

    return combined.sort_values(
        [
            "supplier_baseline_id",
            "source_record_id",
            "_historical_position",
        ],
        na_position="last",
    ).reset_index(drop=True)

def prepare_published_historical(
    frame: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Normalize a historical positive or client-safe publication."""
    required = {
        "Supplier name",
        "Supplier TAX ID (TIN)",
        "Supplier Address1",
        "supplier_country",
    }
    require_columns(frame, required, label)

    prepared = frame.copy()
    prepared["_row_key"] = prepared.apply(
        lambda row: historical_row_key(
            row.get("supplier_country"),
            row.get("Supplier name"),
            row.get("Supplier TAX ID (TIN)"),
            row.get("Supplier Address1"),
        ),
        axis=1,
    )
    return prepared


def accepted_source_record_candidates(
    source_records: pd.DataFrame,
    source_values: Iterable[Any],
) -> set[str]:
    """Resolve exact accepted canonical source-record references."""
    if source_records.empty:
        return set()

    required = {
        "source_record_id",
        "source_record_key",
        "continuity_key",
        "entity_id",
        "resolution_status",
    }
    require_columns(
        source_records,
        required,
        "source_records",
    )

    candidate_values = {
        clean_text(value)
        for value in source_values
        if clean_text(value)
    }
    if not candidate_values:
        return set()

    accepted = source_records.loc[
        source_records["entity_id"].map(clean_text).ne("")
        & ~source_records["resolution_status"]
        .map(clean_text)
        .str.casefold()
        .isin({"review", "blocked", "rejected", "quarantined"})
    ].copy()

    match = (
        accepted["source_record_id"].map(clean_text).isin(
            candidate_values
        )
        | accepted["source_record_key"].map(clean_text).isin(
            candidate_values
        )
        | accepted["continuity_key"].map(clean_text).isin(
            candidate_values
        )
    )

    return {
        clean_text(value)
        for value in accepted.loc[match, "entity_id"]
        if clean_text(value)
    }


def accepted_supplier_link_candidates(
    links: pd.DataFrame,
    *,
    supplier_keys: Iterable[Any],
    names: Iterable[Any],
    country: Any,
) -> tuple[set[str], set[str]]:
    """Return exact-key and reviewed cross-client name candidates."""
    accepted = accepted_link_rows(links)
    if accepted.empty:
        return set(), set()

    keys = {
        clean_text(value)
        for value in supplier_keys
        if clean_text(value)
    }

    exact = accepted.loc[
        accepted["supplier_record_key"]
        .map(clean_text)
        .isin(keys),
        "matched_entity_id",
    ]

    exact_candidates = {
        clean_text(value)
        for value in exact
        if clean_text(value)
    }

    reviewed_name_candidates: set[str] = set()

    if {
        "supplier_name_original",
        "supplier_country",
    }.issubset(accepted.columns):
        name_norms = {
            normalize_name(value)
            for value in names
            if normalize_name(value)
        }
        country_norm = normalize_country(country)

        name_match = accepted.loc[
            accepted["supplier_name_original"]
            .map(normalize_name)
            .isin(name_norms)
            & accepted["supplier_country"]
            .map(normalize_country)
            .eq(country_norm),
            "matched_entity_id",
        ]

        reviewed_name_candidates = {
            clean_text(value)
            for value in name_match
            if clean_text(value)
        }

    return exact_candidates, reviewed_name_candidates


def identifier_candidates_for_values(
    identifier_lookup: dict[str, set[str]],
    values: Iterable[Any],
) -> set[str]:
    """Resolve all exact accepted canonical identifier candidates."""
    candidates: set[str] = set()

    for value in values:
        normalized = normalize_identifier(value)
        if normalized:
            candidates.update(
                identifier_lookup.get(normalized, set())
            )

    return candidates


def candidates_from_name_lookup(
    lookup: dict[str, set[str]],
    names: Iterable[Any],
    country: Any,
) -> set[str]:
    """Resolve country-aware name candidates from a prepared lookup."""
    country_norm = normalize_country(country)
    candidates: set[str] = set()

    for value in names:
        name_norm = normalize_name(value)
        if name_norm and country_norm:
            candidates.update(
                lookup.get(
                    f"{country_norm}|{name_norm}",
                    set(),
                )
            )

    return candidates


def build_live_reconciliation(
    *,
    aggregate_rows: pd.DataFrame,
    attached_source_rows: pd.DataFrame,
    historical_positive: pd.DataFrame,
    historical_safe: pd.DataFrame,
    canonical_tables: dict[str, pd.DataFrame],
    source_workbook: Path,
    aggregate_sheet: str,
    row_level_sheet: str,
) -> pd.DataFrame:
    """Build one governed reconciliation row per aggregate supplier."""
    entities = canonical_tables["canonical_entities"]
    links = canonical_tables["supplier_entity_links"]
    aliases = canonical_tables["entity_aliases"]
    identifiers = canonical_tables["entity_identifiers"]
    source_records = canonical_tables["source_records"]

    entity_metadata = build_entity_metadata(entities)
    identifier_lookup = build_identifier_lookup(identifiers)

    entity_name_lookup = build_name_country_lookup(
        entities,
        "entity_id",
        "canonical_name",
        "country",
    )

    reviewed_alias_lookup = build_name_country_lookup(
        reviewed_alias_rows(aliases),
        "entity_id",
        "alias_name",
        "country",
    )

    positive = prepare_published_historical(
        historical_positive,
        "Historical positive file",
    )
    safe = prepare_published_historical(
        historical_safe,
        "Historical client-safe file",
    )

    positive_key_counts = Counter(positive["_row_key"])
    safe_key_counts = Counter(safe["_row_key"])

    output_rows: list[dict[str, Any]] = []

    ranked = aggregate_rows.sort_values(
        ["spend_total_eur", "supplier_baseline_id"],
        ascending=[False, True],
    ).copy()
    materiality_ranks = {
        supplier_id: rank
        for rank, supplier_id in enumerate(
            ranked["supplier_baseline_id"],
            start=1,
        )
    }

    for aggregate in aggregate_rows.to_dict("records"):
        supplier_id = clean_text(
            aggregate.get("supplier_baseline_id")
        )
        country = normalize_country(
            aggregate.get("country_code")
        )

        source_group = attached_source_rows.loc[
            attached_source_rows[
                "supplier_baseline_id"
            ].eq(supplier_id)
        ].copy()

        source_ids = source_group[
            "source_record_id"
        ].map(clean_text).tolist()

        supplier_keys = [
            supplier_id,
            *source_ids,
        ]

        names = [
            aggregate.get("supplier_name"),
            *split_semicolon_values(
                aggregate.get("supplier_name_variants")
            ),
            *source_group["supplier_name"].tolist(),
            *source_group["original_supplier_name"].tolist(),
            *source_group["erp_supplier_name"].tolist(),
        ]

        historical_identifiers = [
            value
            for value in source_group["2"].tolist()
            if normalize_identifier(value)
        ]

        exact_link_candidates, cross_client_candidates = (
            accepted_supplier_link_candidates(
                links,
                supplier_keys=supplier_keys,
                names=names,
                country=country,
            )
        )

        identifier_candidates = (
            identifier_candidates_for_values(
                identifier_lookup,
                historical_identifiers,
            )
        )

        source_candidates = accepted_source_record_candidates(
            source_records,
            [
                *source_ids,
                *source_group["matched_register"].tolist(),
            ],
        )

        alias_candidates = candidates_from_name_lookup(
            reviewed_alias_lookup,
            names,
            country,
        )

        canonical_name_candidates = (
            candidates_from_name_lookup(
                entity_name_lookup,
                names,
                country,
            )
        )

        strong_candidates = (
            exact_link_candidates
            | identifier_candidates
            | source_candidates
        )
        reviewed_candidates = (
            alias_candidates
            | cross_client_candidates
        )

        historical_keys = source_group["_row_key"].tolist()

        positive_count = sum(
            positive_key_counts.get(key, 0)
            for key in historical_keys
        )
        safe_count = sum(
            safe_key_counts.get(key, 0)
            for key in historical_keys
        )

        matched_count = int(
            source_group["_has_match"].sum()
        )

        category, all_candidates, evidence_basis = (
            decide_reconciliation_category(
                identifier_or_source_candidates=(
                    strong_candidates
                ),
                reviewed_evidence_candidates=(
                    reviewed_candidates
                ),
                name_only_candidates=(
                    canonical_name_candidates
                    - strong_candidates
                    - reviewed_candidates
                ),
                has_historical_positive_evidence=(
                    matched_count > 0
                    or positive_count > 0
                    or safe_count > 0
                ),
                is_blocked=False,
            )
        )

        candidate_ids = sorted(all_candidates)
        candidate_entity_id = (
            candidate_ids[0]
            if len(candidate_ids) == 1
            else ""
        )

        candidate_metadata = entity_metadata.get(
            candidate_entity_id,
            {},
        )

        row = {
            "supplier_baseline_id": supplier_id,
            "baseline_id": clean_text(
                aggregate.get("baseline_id")
            ),
            "supplier_name": clean_text(
                aggregate.get("supplier_name")
            ),
            "supplier_name_norm": clean_text(
                aggregate.get("supplier_name_norm")
            ),
            "country_code": country,
            "country_name": clean_text(
                aggregate.get("country_name")
            ),
            "spend_total_eur": float(
                aggregate.get("spend_total_eur")
            ),
            "source_record_ids_merged": clean_text(
                aggregate.get("source_record_ids_merged")
            ),
            "supplier_name_variants": clean_text(
                aggregate.get("supplier_name_variants")
            ),
            "source_workbook": str(source_workbook),
            "aggregate_sheet": aggregate_sheet,
            "row_level_sheet": row_level_sheet,
            "expanded_source_row_count": int(
                source_group["source_record_id"].nunique()
            ),
            "expanded_source_record_ids": join_sorted(
                source_ids
            ),
            "historical_matched_row_count": matched_count,
            "historical_positive_row_count": int(
                positive_count
            ),
            "historical_safe_row_count": int(safe_count),
            "historical_match_methods": join_sorted(
                source_group["match_type"]
            ),
            "historical_matched_registers": join_sorted(
                source_group["matched_register"]
            ),
            "historical_matched_entity_names": join_sorted(
                source_group["matched_entity_name"]
            ),
            "historical_identifiers": join_sorted(
                historical_identifiers
            ),
            "canonical_supplier_link_candidate_ids": (
                join_sorted(exact_link_candidates)
            ),
            "canonical_identifier_candidate_ids": (
                join_sorted(identifier_candidates)
            ),
            "canonical_source_record_candidate_ids": (
                join_sorted(source_candidates)
            ),
            "canonical_reviewed_alias_candidate_ids": (
                join_sorted(
                    alias_candidates
                    | cross_client_candidates
                )
            ),
            "canonical_name_country_candidate_ids": (
                join_sorted(canonical_name_candidates)
            ),
            "canonical_candidate_entity_ids": (
                join_sorted(candidate_ids)
            ),
            "canonical_candidate_count": len(
                candidate_ids
            ),
            "candidate_entity_id": candidate_entity_id,
            "candidate_canonical_name": (
                candidate_metadata.get(
                    "canonical_name",
                    "",
                )
            ),
            "candidate_canonical_country": (
                candidate_metadata.get("country", "")
            ),
            "evidence_basis": evidence_basis,
            "identity_conflict": (
                len(candidate_ids) > 1
                or category
                == "ambiguous_identity_owner_review"
            ),
            "reconciliation_category": category,
            "materiality_rank": int(
                materiality_ranks[supplier_id]
            ),
            "materiality_at_least_100k": bool(
                float(aggregate.get("spend_total_eur"))
                >= 100000
            ),
            "owner_decision": "",
            "owner_approved_entity_id": "",
            "owner_review_notes": "",
        }

        output_rows.append(row)

    output = pd.DataFrame(output_rows)

    for column in OUTPUT_COLUMNS:
        if column not in output.columns:
            output[column] = ""

    return output[OUTPUT_COLUMNS].sort_values(
        ["materiality_rank", "supplier_baseline_id"]
    ).reset_index(drop=True)


def write_reconciliation_outputs(
    reconciliation: pd.DataFrame,
    output_dir: Path,
    manifest: dict[str, Any],
    before_checksums: dict[str, str],
    after_checksums: dict[str, str],
) -> None:
    """Write deterministic review outputs outside the canonical store."""
    output_dir.mkdir(parents=True, exist_ok=True)

    reconciliation.to_csv(
        output_dir / "cbre_aggregate_reconciliation.csv",
        index=False,
    )

    for category in sorted(ALLOWED_CATEGORIES):
        subset = reconciliation.loc[
            reconciliation[
                "reconciliation_category"
            ].eq(category)
        ].copy()

        subset.to_csv(
            output_dir / f"cbre_{category}.csv",
            index=False,
        )

    reconciliation.sort_values(
        [
            "materiality_rank",
            "reconciliation_category",
            "supplier_baseline_id",
        ]
    ).to_csv(
        output_dir / "cbre_owner_review.csv",
        index=False,
    )

    (
        output_dir / "operational_store_before.sha256.json"
    ).write_text(
        json.dumps(
            before_checksums,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        output_dir / "operational_store_after.sha256.json"
    ).write_text(
        json.dumps(
            after_checksums,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        output_dir / "cbre_reconciliation_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only aggregate supplier reconciliation "
            "against the canonical entity store."
        )
    )
    parser.add_argument(
        "--aggregate-workbook",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--aggregate-sheet",
        default="Supplier by baseline",
    )
    parser.add_argument(
        "--row-level-sheet",
        default="Row-level combined",
    )
    parser.add_argument(
        "--baseline-id",
        required=True,
    )
    parser.add_argument(
        "--client-code",
        required=True,
    )
    parser.add_argument(
        "--canonical-store",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--historical-matched",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--historical-positive",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--historical-safe",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--historical-source-row-offset",
        type=int,
        default=2,
        help=(
            "Spreadsheet row-number offset used to map source rows "
            "to the zero-indexed historical matched CSV."
        ),
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--expected-spend-eur",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    """Run the governed read-only reconciliation."""
    args = parse_args()

    before_checksums = checksum_directory(
        args.canonical_store
    )

    aggregate = pd.read_excel(
        args.aggregate_workbook,
        sheet_name=args.aggregate_sheet,
    )
    row_level = pd.read_excel(
        args.aggregate_workbook,
        sheet_name=args.row_level_sheet,
    )

    selected = select_baseline_rows(
        aggregate,
        args.baseline_id,
    )
    resolved = expand_source_rows(
        selected,
        row_level,
    )

    historical_matched = pd.read_csv(
        args.historical_matched,
        low_memory=False,
    )
    historical_positive = pd.read_csv(
        args.historical_positive,
        low_memory=False,
    )
    historical_safe = pd.read_csv(
        args.historical_safe,
        low_memory=False,
    )

    attached = attach_historical_rows(
        resolved,
        historical_matched,
        source_row_offset=args.historical_source_row_offset,
    )

    canonical_tables = load_canonical_tables(
        args.canonical_store
    )

    reconciliation = build_live_reconciliation(
        aggregate_rows=selected,
        attached_source_rows=attached,
        historical_positive=historical_positive,
        historical_safe=historical_safe,
        canonical_tables=canonical_tables,
        source_workbook=args.aggregate_workbook,
        aggregate_sheet=args.aggregate_sheet,
        row_level_sheet=args.row_level_sheet,
    )

    validate_reconciliation(
        reconciliation,
        expected_rows=args.expected_rows,
        expected_spend_eur=args.expected_spend_eur,
    )

    after_checksums = checksum_directory(
        args.canonical_store
    )

    manifest = build_manifest_summary(
        reconciliation,
        expanded_source_rows=resolved[
            "source_record_id"
        ].nunique(),
        unique_source_record_ids=resolved[
            "source_record_id"
        ].nunique(),
        expected_rows=args.expected_rows,
        expected_spend_eur=args.expected_spend_eur,
        canonical_checksums_before=before_checksums,
        canonical_checksums_after=after_checksums,
    )

    manifest.update(
        {
            "status": "read_only_reconciliation_complete",
            "client_code": args.client_code,
            "baseline_id": args.baseline_id,
            "aggregate_workbook": str(
                args.aggregate_workbook
            ),
            "aggregate_sheet": args.aggregate_sheet,
            "row_level_sheet": args.row_level_sheet,
            "historical_source_row_offset": (
                args.historical_source_row_offset
            ),
            "historical_matched_file": str(
                args.historical_matched
            ),
            "historical_positive_file": str(
                args.historical_positive
            ),
            "historical_safe_file": str(
                args.historical_safe
            ),
            "canonical_store": str(
                args.canonical_store
            ),
            "output_dir": str(args.output_dir),
            "owner_decisions_applied": 0,
            "canonical_links_created": 0,
            "canonical_entities_allocated": 0,
            "canonical_materialisation_performed": False,
        }
    )

    write_reconciliation_outputs(
        reconciliation,
        args.output_dir,
        manifest,
        before_checksums,
        after_checksums,
    )

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
