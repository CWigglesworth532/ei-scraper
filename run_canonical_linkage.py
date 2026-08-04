#!/usr/bin/env python3
"""Link accepted supplier matches to persisted canonical entities.

E1.4 increment 2:

- consume accepted supplier-match records;
- resolve them through the pure canonical linkage decision engine;
- preserve prior supplier-to-entity decisions;
- write authoritative Parquet linkage and review outputs;
- provide DuckDB QA views;
- do not alter matching, classification or publication behaviour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from canonical_entity_linkage import (
    LinkageDecision,
    build_alias_lookup,
    build_canonical_name_lookup,
    build_identifier_lookup,
    build_source_record_lookup,
    clean_text,
    resolve_accepted_match,
)


LINK_COLUMNS = [
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
]

REQUIRED_INPUT_COLUMNS = {
    "supplier_record_key",
    "supplier_name_original",
    "supplier_country",
    "acceptance_status",
}


def read_csv_strings(path: Path) -> pd.DataFrame:
    """Read a CSV while preserving identifiers and original values."""
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


def read_parquet_or_empty(path: Path) -> pd.DataFrame:
    """Read a Parquet table or return an empty frame."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def ensure_input_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate required fields and add supported optional fields."""
    missing = REQUIRED_INPUT_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            "Accepted-match input is missing required columns: "
            + ", ".join(sorted(missing))
        )

    optional_columns = [
        "supplier_name_norm",
        "supplier_identifier_type",
        "supplier_identifier_value",
        "source_client",
        "source_client_file",
        "reviewed_by",
        "reviewed_at",
        "matched_source_record_id",
        "relationship_type",
        "related_entity_id",
    ]

    result = frame.copy()
    for column in optional_columns:
        if column not in result.columns:
            result[column] = ""

    return result


def build_prior_supplier_link_lookup(
    prior_links: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Return prior final decisions by stable supplier record key."""
    lookup: dict[str, dict[str, Any]] = {}

    if prior_links.empty:
        return lookup

    required = {
        "supplier_record_key",
        "matched_entity_id",
        "resolution_status",
    }
    if not required.issubset(prior_links.columns):
        return lookup

    for row in prior_links.to_dict("records"):
        supplier_record_key = clean_text(
            row.get("supplier_record_key")
        )
        matched_entity_id = clean_text(
            row.get("matched_entity_id")
        )
        resolution_status = clean_text(
            row.get("resolution_status")
        )

        if (
            supplier_record_key
            and matched_entity_id
            and resolution_status in {"reused", "new"}
        ):
            lookup[supplier_record_key] = row

    return lookup


def decision_from_prior_link(
    supplier: dict[str, Any],
    prior: dict[str, Any],
) -> LinkageDecision:
    """Reconstruct an accepted persisted decision."""
    return LinkageDecision(
        supplier_record_key=clean_text(
            supplier.get("supplier_record_key")
        ),
        resolution_status="reused",
        resolution_method="persisted_supplier_link",
        matched_entity_id=clean_text(
            prior.get("matched_entity_id")
        ),
        matched_source_record_id=(
            clean_text(
                supplier.get("matched_source_record_id")
            )
            or clean_text(
                prior.get("matched_source_record_id")
            )
        ),
        allocate_new_entity=False,
        review_required=False,
        candidate_entity_ids=clean_text(
            prior.get("matched_entity_id")
        ),
        resolution_reason=(
            "Reused the persisted supplier-record-to-entity decision."
        ),
        relationship_type=(
            clean_text(supplier.get("relationship_type")).lower()
            or clean_text(
                prior.get("relationship_type")
            ).lower()
        ),
        related_entity_id=(
            clean_text(supplier.get("related_entity_id"))
            or clean_text(prior.get("related_entity_id"))
        ),
    )


def create_empty_alias_frame() -> pd.DataFrame:
    """Return the minimum supported aliases schema."""
    return pd.DataFrame(
        columns=[
            "entity_id",
            "alias_name_norm",
            "country",
            "verification_status",
            "review_status",
        ]
    )


def run_linkage(
    accepted_matches_csv: Path,
    canonical_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve and persist accepted supplier-to-entity links."""
    if output_dir is None:
        output_dir = canonical_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_entities_path = (
        canonical_dir / "canonical_entities.parquet"
    )
    source_records_path = (
        canonical_dir / "source_records.parquet"
    )
    identifiers_path = (
        canonical_dir / "entity_identifiers.parquet"
    )
    aliases_path = canonical_dir / "entity_aliases.parquet"

    for required_path in [
        canonical_entities_path,
        source_records_path,
        identifiers_path,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required canonical table is missing: {required_path}"
            )

    canonical_entities = pd.read_parquet(
        canonical_entities_path
    )
    source_records = pd.read_parquet(source_records_path)
    identifiers = pd.read_parquet(identifiers_path)
    aliases = (
        pd.read_parquet(aliases_path)
        if aliases_path.exists()
        else create_empty_alias_frame()
    )

    accepted_matches = ensure_input_columns(
        read_csv_strings(accepted_matches_csv)
    )

    links_path = output_dir / "supplier_entity_links.parquet"
    review_path = output_dir / "identity_review_queue.parquet"
    review_csv_path = output_dir / "identity_review_queue.csv"

    prior_links = read_parquet_or_empty(links_path)
    prior_link_lookup = build_prior_supplier_link_lookup(
        prior_links
    )

    source_lookup = build_source_record_lookup(source_records)
    identifier_lookup = build_identifier_lookup(identifiers)
    alias_lookup = build_alias_lookup(aliases)
    canonical_name_lookup = build_canonical_name_lookup(
        canonical_entities
    )

    result_rows: list[dict[str, Any]] = []

    for supplier in accepted_matches.to_dict("records"):
        supplier_record_key = clean_text(
            supplier.get("supplier_record_key")
        )

        prior = prior_link_lookup.get(supplier_record_key)
        if prior is not None:
            decision = decision_from_prior_link(
                supplier,
                prior,
            )
        else:
            decision = resolve_accepted_match(
                supplier,
                source_record_lookup=source_lookup,
                identifier_lookup=identifier_lookup,
                alias_lookup=alias_lookup,
                canonical_name_lookup=canonical_name_lookup,
            )

        result_rows.append(
            {
                "supplier_record_key": supplier_record_key,
                "supplier_name_original": clean_text(
                    supplier.get("supplier_name_original")
                ),
                "supplier_name_norm": clean_text(
                    supplier.get("supplier_name_norm")
                ),
                "supplier_country": clean_text(
                    supplier.get("supplier_country")
                ).upper(),
                "supplier_identifier_type": clean_text(
                    supplier.get("supplier_identifier_type")
                ),
                "supplier_identifier_value": clean_text(
                    supplier.get("supplier_identifier_value")
                ),
                "acceptance_status": clean_text(
                    supplier.get("acceptance_status")
                ),
                "source_client": clean_text(
                    supplier.get("source_client")
                ),
                "source_client_file": clean_text(
                    supplier.get("source_client_file")
                ),
                "reviewed_by": clean_text(
                    supplier.get("reviewed_by")
                ),
                "reviewed_at": clean_text(
                    supplier.get("reviewed_at")
                ),
                **decision.to_dict(),
            }
        )

    current_links = pd.DataFrame(
        result_rows,
        columns=LINK_COLUMNS,
    )

    if prior_links.empty:
        persisted_links = current_links
    else:
        current_keys = set(
            current_links["supplier_record_key"].astype(str)
        )
        retained_prior = prior_links.loc[
            ~prior_links["supplier_record_key"]
            .astype(str)
            .isin(current_keys)
        ]

        persisted_links = pd.concat(
            [retained_prior, current_links],
            ignore_index=True,
        )

    persisted_links = (
        persisted_links
        .drop_duplicates(
            subset=["supplier_record_key"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    review_queue = persisted_links.loc[
        persisted_links["review_required"].astype(bool)
    ].copy()

    persisted_links.to_parquet(links_path, index=False)
    review_queue.to_parquet(review_path, index=False)
    review_queue.to_csv(review_csv_path, index=False)

    qa_path = output_dir / "canonical_linkage_qa.duckdb"
    if qa_path.exists():
        qa_path.unlink()

    connection = duckdb.connect(str(qa_path))
    try:
        connection.execute(
            f"""
            CREATE VIEW supplier_entity_links AS
            SELECT *
            FROM read_parquet('{links_path.as_posix()}')
            """
        )
        connection.execute(
            f"""
            CREATE VIEW identity_review_queue AS
            SELECT *
            FROM read_parquet('{review_path.as_posix()}')
            """
        )
        connection.execute(
            """
            CREATE VIEW qa_duplicate_supplier_record_keys AS
            SELECT supplier_record_key, COUNT(*) AS row_count
            FROM supplier_entity_links
            GROUP BY supplier_record_key
            HAVING COUNT(*) > 1
            """
        )
        connection.execute(
            """
            CREATE VIEW qa_links_without_entity_id AS
            SELECT *
            FROM supplier_entity_links
            WHERE resolution_status IN ('reused', 'new')
              AND (
                  matched_entity_id IS NULL
                  OR matched_entity_id = ''
              )
            """
        )
        connection.execute(
            """
            CREATE VIEW qa_linkage_status_counts AS
            SELECT resolution_status, COUNT(*) AS row_count
            FROM supplier_entity_links
            GROUP BY resolution_status
            ORDER BY resolution_status
            """
        )
        connection.execute(
            """
            CREATE VIEW qa_new_entities AS
            SELECT *
            FROM supplier_entity_links
            WHERE allocate_new_entity = TRUE
            """
        )
    finally:
        connection.close()

    summary = {
        "input_rows": int(len(accepted_matches)),
        "persisted_supplier_links": int(len(persisted_links)),
        "reused": int(
            current_links["resolution_status"].eq("reused").sum()
        ),
        "new": int(
            current_links["resolution_status"].eq("new").sum()
        ),
        "review": int(
            current_links["resolution_status"].eq("review").sum()
        ),
        "not_eligible": int(
            current_links["resolution_status"]
            .eq("not_eligible")
            .sum()
        ),
        "review_queue_rows": int(len(review_queue)),
    }

    manifest_path = output_dir / "linkage_manifest.json"
    manifest_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Link accepted supplier matches to canonical skopia entities."
        )
    )
    parser.add_argument(
        "--accepted-matches",
        required=True,
        type=Path,
        help="CSV containing accepted supplier-match records.",
    )
    parser.add_argument(
        "--canonical-dir",
        required=True,
        type=Path,
        help="Directory containing E1.3 canonical Parquet tables.",
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
    """Run the canonical supplier-linkage process."""
    args = parse_args()
    summary = run_linkage(
        accepted_matches_csv=args.accepted_matches,
        canonical_dir=args.canonical_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
