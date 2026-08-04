#!/usr/bin/env python3
"""Build the E1.6 canonical directory-candidate CSV export.

Task: SKO-011 / E1.6

The export:

- uses entity_id as the stable directory key;
- includes one row per eligible active canonical entity;
- aggregates accepted supplier-link provenance;
- preserves source recognition evidence separately from classification;
- creates conservative directory-review defaults;
- aligns editable profile fields with the existing Airtable directory;
- exposes internal commercial-materiality placeholders;
- does not publish to Airtable or Softr;
- does not treat canonical identity as proof of directory eligibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from canonical_entity_linkage import ACCEPTED_STATUSES


EXPORT_SCHEMA_VERSION = "1.0.0"

OUTPUT_COLUMNS = [
    # Stable canonical identity.
    "entity_id",
    "canonical_name",
    "canonical_name_norm",
    "country",
    "entity_status",
    "canonical_record_status",
    "primary_identifier_type",
    "primary_identifier_value",
    "primary_source_record_id",
    "identity_confidence",
    "identity_review_status",

    # Existing operational linkage and provenance.
    "accepted_supplier_link_count",
    "source_client_count",
    "source_clients",
    "source_register_name",
    "source_url",
    "source_recognition_type",
    "source_recognition_name",
    "source_recognition_evidence",
    "source_legal_form_family",
    "identity_resolution_method",
    "candidate_source",
    "last_link_reviewed_at",

    # Governed classification fields.
    "classification_id",
    "classification_scheme",
    "classification_status",
    "social_economy_category",
    "classification_confidence",
    "classification_reason",
    "classification_reviewed_at",

    # Directory-candidate workflow.
    "directory_candidate_status",
    "directory_inclusion_decision",
    "procurement_relevant",
    "readiness_reason",
    "review_priority",
    "last_reviewed_date",

    # Airtable-aligned directory profile fields.
    "Organisation",
    "Country HQ",
    "Countries Served",
    "Website",
    "Business Summary",
    "Sector",
    "Social Mission",
    "Corporate Clients",
    "Identified Clients",
    "Clients Publicly Referenced",
    "Data Status",
    "Verified",
    "Source",
    "Source Type",
    "Inclusion Reason",
    "Exclusion Reason",

    # Derived directory QA.
    "missing_website",
    "missing_country_hq",
    "missing_business_summary",
    "missing_sector",
    "missing_social_mission",
    "possible_existing_directory_record",

    # Internal commercial-materiality indicators.
    "known_client_relationship_count",
    "known_spend_eur",
    "highest_known_annual_spend_eur",
    "has_100k_plus_relationship",
    "commercial_materiality_tier",
    "spend_data_status",

    # Export controls.
    "export_schema_version",
]


def clean_text(value: Any) -> str:
    """Return a trimmed string, treating null values as blank."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def clean_boolean(value: Any) -> bool:
    """Convert common values into a conservative boolean."""
    if isinstance(value, bool):
        return value

    text = clean_text(value).lower()
    return text in {"1", "true", "yes", "y"}


def require_table(path: Path) -> pd.DataFrame:
    """Read a required Parquet table."""
    if not path.exists():
        raise FileNotFoundError(f"Required canonical table is missing: {path}")

    return pd.read_parquet(path)


def ensure_columns(
    frame: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """Raise a clear error when required source fields are absent."""
    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: "
            + ", ".join(missing)
        )


def parse_raw_payload(value: Any) -> dict[str, Any]:
    """Parse one source-record raw payload without failing the export."""
    text = clean_text(value)

    if not text:
        return {}

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}

    if not isinstance(payload, dict):
        return {}

    return payload


def sorted_unique(values: pd.Series | list[Any]) -> list[str]:
    """Return sorted, non-empty unique text values."""
    cleaned = {
        clean_text(value)
        for value in values
        if clean_text(value)
    }
    return sorted(cleaned, key=str.casefold)


def join_values(values: pd.Series | list[Any]) -> str:
    """Join sorted unique values for transparent CSV interchange."""
    return "; ".join(sorted_unique(values))


def select_primary_identifier(
    entity_identifiers: pd.DataFrame,
) -> tuple[str, str]:
    """Return the preferred active identifier type and normalized value."""
    if entity_identifiers.empty:
        return "", ""

    active = entity_identifiers.copy()

    if "verification_status" in active.columns:
        excluded_statuses = {
            "rejected",
            "conflicted",
            "superseded",
        }
        active = active.loc[
            ~active["verification_status"]
            .astype(str)
            .str.lower()
            .isin(excluded_statuses)
        ].copy()

    if active.empty:
        return "", ""

    if "is_primary" not in active.columns:
        active["is_primary"] = False

    active["_is_primary_sort"] = (
        active["is_primary"]
        .map(clean_boolean)
        .astype(int)
    )

    for column in [
        "identifier_type",
        "identifier_value_normalized",
        "identifier_id",
    ]:
        if column not in active.columns:
            active[column] = ""

    active = active.sort_values(
        by=[
            "_is_primary_sort",
            "identifier_type",
            "identifier_value_normalized",
            "identifier_id",
        ],
        ascending=[False, True, True, True],
        kind="stable",
    )

    row = active.iloc[0]

    return (
        clean_text(row.get("identifier_type")),
        clean_text(row.get("identifier_value_normalized")),
    )


def source_evidence_for_entity(
    entity_source_records: pd.DataFrame,
    primary_source_record_id: str,
) -> dict[str, str]:
    """Select source-level evidence without creating classification claims."""
    if entity_source_records.empty:
        return {
            "source_register_name": "",
            "source_url": "",
            "source_recognition_type": "",
            "source_recognition_name": "",
            "source_recognition_evidence": "",
            "source_legal_form_family": "",
            "identity_resolution_method": "",
        }

    records = entity_source_records.copy()

    records["_is_primary"] = (
        records["source_record_id"]
        .astype(str)
        .eq(primary_source_record_id)
        .astype(int)
    )

    records = records.sort_values(
        by=["_is_primary", "source_record_id"],
        ascending=[False, True],
        kind="stable",
    )

    selected = records.iloc[0]
    payload = parse_raw_payload(selected.get("raw_payload_json"))

    register_names = []
    recognition_types = []
    recognition_names = []
    recognition_evidence = []
    legal_form_families = []

    for row in records.to_dict("records"):
        row_payload = parse_raw_payload(row.get("raw_payload_json"))

        register_names.append(row_payload.get("ei_register_name"))
        recognition_types.append(
            row_payload.get("se_recognition_type")
        )
        recognition_names.append(
            row_payload.get("se_recognition_name")
        )
        recognition_evidence.append(
            row_payload.get("se_recognition_evidence")
        )
        legal_form_families.append(
            row_payload.get("base_legal_form_family")
        )

    return {
        "source_register_name": join_values(register_names),
        "source_url": clean_text(selected.get("source_url")),
        "source_recognition_type": join_values(recognition_types),
        "source_recognition_name": join_values(recognition_names),
        "source_recognition_evidence": join_values(
            recognition_evidence
        ),
        "source_legal_form_family": join_values(
            legal_form_families
        ),
        "identity_resolution_method": clean_text(
            selected.get("resolution_method")
        ),
    }


def accepted_links_only(links: pd.DataFrame) -> pd.DataFrame:
    """Return accepted, resolved, non-review supplier links."""
    accepted = links.copy()

    accepted_statuses = {
        clean_text(value).lower()
        for value in ACCEPTED_STATUSES
    }

    accepted = accepted.loc[
        accepted["acceptance_status"]
        .astype(str)
        .str.lower()
        .isin(accepted_statuses)
    ].copy()

    accepted = accepted.loc[
        accepted["matched_entity_id"]
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    accepted = accepted.loc[
        ~accepted["review_required"].map(clean_boolean)
    ].copy()

    return accepted


def latest_non_empty(values: pd.Series) -> str:
    """Return the latest non-empty timestamp-like text value."""
    candidates = sorted_unique(values)

    if not candidates:
        return ""

    return max(candidates)


def build_directory_candidate_export(
    canonical_dir: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the E1.6 directory-candidate export."""
    entities = require_table(
        canonical_dir / "canonical_entities.parquet"
    )
    identifiers = require_table(
        canonical_dir / "entity_identifiers.parquet"
    )
    source_records = require_table(
        canonical_dir / "source_records.parquet"
    )
    links = require_table(
        canonical_dir / "supplier_entity_links.parquet"
    )

    ensure_columns(
        entities,
        [
            "entity_id",
            "canonical_name",
            "canonical_name_norm",
            "country",
            "entity_status",
            "record_status",
            "primary_source_record_id",
            "identity_confidence",
            "identity_review_status",
        ],
        "canonical_entities.parquet",
    )
    ensure_columns(
        identifiers,
        [
            "entity_id",
            "identifier_type",
            "identifier_value_normalized",
        ],
        "entity_identifiers.parquet",
    )
    ensure_columns(
        source_records,
        [
            "source_record_id",
            "entity_id",
            "source_url",
            "resolution_method",
            "raw_payload_json",
        ],
        "source_records.parquet",
    )
    ensure_columns(
        links,
        [
            "matched_entity_id",
            "acceptance_status",
            "review_required",
            "source_client",
            "reviewed_at",
        ],
        "supplier_entity_links.parquet",
    )

    accepted_links = accepted_links_only(links)

    linked_entity_ids = set(
        accepted_links["matched_entity_id"]
        .astype(str)
        .str.strip()
    )

    eligible_entities = entities.loc[
        entities["entity_id"]
        .astype(str)
        .str.strip()
        .ne("")
        & entities["record_status"]
        .astype(str)
        .str.lower()
        .eq("active")
        & entities["entity_id"]
        .astype(str)
        .isin(linked_entity_ids)
    ].copy()

    eligible_entities = eligible_entities.sort_values(
        by=["entity_id"],
        kind="stable",
    )

    rows: list[dict[str, Any]] = []

    for entity in eligible_entities.to_dict("records"):
        entity_id = clean_text(entity.get("entity_id"))

        entity_links = accepted_links.loc[
            accepted_links["matched_entity_id"]
            .astype(str)
            .eq(entity_id)
        ].copy()

        entity_identifiers = identifiers.loc[
            identifiers["entity_id"]
            .astype(str)
            .eq(entity_id)
        ].copy()

        entity_sources = source_records.loc[
            source_records["entity_id"]
            .astype(str)
            .eq(entity_id)
        ].copy()

        (
            primary_identifier_type,
            primary_identifier_value,
        ) = select_primary_identifier(entity_identifiers)

        primary_source_record_id = clean_text(
            entity.get("primary_source_record_id")
        )

        source_evidence = source_evidence_for_entity(
            entity_sources,
            primary_source_record_id,
        )

        source_clients = sorted_unique(
            entity_links["source_client"]
        )
        source_client_text = "; ".join(source_clients)
        source_client_count = len(source_clients)

        country = clean_text(entity.get("country"))
        canonical_name = clean_text(
            entity.get("canonical_name")
        )

        row = {
            "entity_id": entity_id,
            "canonical_name": canonical_name,
            "canonical_name_norm": clean_text(
                entity.get("canonical_name_norm")
            ),
            "country": country,
            "entity_status": clean_text(
                entity.get("entity_status")
            ),
            "canonical_record_status": clean_text(
                entity.get("record_status")
            ),
            "primary_identifier_type": (
                primary_identifier_type
            ),
            "primary_identifier_value": (
                primary_identifier_value
            ),
            "primary_source_record_id": (
                primary_source_record_id
            ),
            "identity_confidence": clean_text(
                entity.get("identity_confidence")
            ),
            "identity_review_status": clean_text(
                entity.get("identity_review_status")
            ),

            "accepted_supplier_link_count": len(entity_links),
            "source_client_count": source_client_count,
            "source_clients": source_client_text,
            **source_evidence,
            "candidate_source": "accepted_supplier_link",
            "last_link_reviewed_at": latest_non_empty(
                entity_links["reviewed_at"]
            ),

            # No governed classification table currently exists.
            "classification_id": "",
            "classification_scheme": "",
            "classification_status": "",
            "social_economy_category": "",
            "classification_confidence": "",
            "classification_reason": "",
            "classification_reviewed_at": "",

            # Conservative review defaults.
            "directory_candidate_status": "research_needed",
            "directory_inclusion_decision": "review",
            "procurement_relevant": "",
            "readiness_reason": (
                "Classification and directory assessment "
                "not yet completed."
            ),
            "review_priority": "",
            "last_reviewed_date": "",

            # Airtable-aligned proposed profile fields.
            "Organisation": canonical_name,
            "Country HQ": country,
            "Countries Served": "",
            "Website": "",
            "Business Summary": "",
            "Sector": "",
            "Social Mission": "",
            "Corporate Clients": "",
            "Identified Clients": "",
            "Clients Publicly Referenced": "No",
            "Data Status": "Review",
            "Verified": "No",
            "Source": source_evidence[
                "source_register_name"
            ],
            "Source Type": "Matching System",
            "Inclusion Reason": "",
            "Exclusion Reason": "",

            # Derived QA.
            "missing_website": True,
            "missing_country_hq": not bool(country),
            "missing_business_summary": True,
            "missing_sector": True,
            "missing_social_mission": True,
            "possible_existing_directory_record": "",

            # Internal commercial-materiality fields.
            "known_client_relationship_count": (
                source_client_count
            ),
            "known_spend_eur": "",
            "highest_known_annual_spend_eur": "",
            "has_100k_plus_relationship": "",
            "commercial_materiality_tier": "",
            "spend_data_status": "not_available",

            "export_schema_version": EXPORT_SCHEMA_VERSION,
        }

        rows.append(row)

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    duplicate_entity_ids = (
        int(output["entity_id"].duplicated().sum())
        if not output.empty
        else 0
    )

    if duplicate_entity_ids:
        raise ValueError(
            "Directory candidate export contains duplicate entity_id "
            f"values: {duplicate_entity_ids}"
        )

    qa = {
        "canonical_entity_rows": len(entities),
        "accepted_supplier_link_rows": len(accepted_links),
        "accepted_linked_entity_count": len(linked_entity_ids),
        "eligible_export_rows": len(output),
        "duplicate_entity_id_count": duplicate_entity_ids,
        "research_needed_count": (
            int(
                output["directory_candidate_status"]
                .eq("research_needed")
                .sum()
            )
            if not output.empty
            else 0
        ),
        "missing_website_count": (
            int(output["missing_website"].sum())
            if not output.empty
            else 0
        ),
        "missing_country_hq_count": (
            int(output["missing_country_hq"].sum())
            if not output.empty
            else 0
        ),
    }

    return output, qa


def write_export(
    canonical_dir: Path,
    output_csv: Path,
) -> dict[str, int]:
    """Build and write the E1.6 CSV export."""
    output, qa = build_directory_candidate_export(
        canonical_dir
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(
        output_csv,
        index=False,
        encoding="utf-8",
    )

    return qa


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical directory-candidate CSV export."
        )
    )
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        required=True,
        help="Directory containing the canonical Parquet tables.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV path.",
    )

    args = parser.parse_args()

    qa = write_export(
        canonical_dir=args.canonical_dir,
        output_csv=args.output,
    )

    print("=== E1.6 DIRECTORY CANDIDATE EXPORT ===")
    print(f"output={args.output}")
    for key, value in qa.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
