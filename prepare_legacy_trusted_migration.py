#!/usr/bin/env python3
"""Prepare the legacy trusted roster for canonical migration.

Every valid legacy name is retained in one of two outputs:

- uniquely entity-linked migration candidates;
- explicit unresolved or ambiguous review records.

No canonical entity is created and no safe-list term is approved by this
preparation step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from canonical_entity_linkage import (
    build_alias_lookup,
    build_canonical_name_lookup,
    clean_text,
    normalize_name,
)


COUNTRY_MAP = {
    "austria": "AT",
    "belgium": "BE",
    "denmark": "DK",
    "france": "FR",
    "germany": "DE",
    "ireland": "IE",
    "italy": "IT",
    "netherlands": "NL",
    "portugal": "PT",
    "spain": "ES",
    "sweden": "SE",
    "switzerland": "CH",
}


def normalize_country(value: Any) -> str:
    text = clean_text(value)
    if len(text) == 2:
        return text.upper()
    return COUNTRY_MAP.get(text.lower(), "")


def prepare_legacy_migration(
    legacy_csv: Path,
    canonical_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    entities = pd.read_parquet(
        canonical_dir / "canonical_entities.parquet"
    )

    aliases_path = canonical_dir / "entity_aliases.parquet"
    aliases = (
        pd.read_parquet(aliases_path)
        if aliases_path.exists()
        else pd.DataFrame(
            columns=[
                "entity_id",
                "alias_name_norm",
                "country",
                "verification_status",
                "review_status",
            ]
        )
    )

    legacy = pd.read_csv(
        legacy_csv,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    required = {
        "raw_brand_name",
        "country_hint",
    }
    missing = required - set(legacy.columns)
    if missing:
        raise ValueError(
            "Legacy roster is missing columns: "
            + ", ".join(sorted(missing))
        )

    canonical_lookup = build_canonical_name_lookup(entities)
    alias_lookup = build_alias_lookup(aliases)

    linked_rows = []
    review_rows = []

    for row_number, row in enumerate(
        legacy.to_dict("records"),
        start=2,
    ):
        term_raw = clean_text(row.get("raw_brand_name"))
        country = normalize_country(row.get("country_hint"))
        term_norm = normalize_name(term_raw)
        publish_status = clean_text(
            row.get("publish_status")
        ).lower()
        source = clean_text(row.get("source"))
        trusted_reason = clean_text(
            row.get("trusted_reason")
        )
        website_hint = clean_text(
            row.get("website_hint")
        )

        errors = []
        if not term_raw:
            errors.append("missing trusted name")
        if not country:
            errors.append("country could not be normalized")
        if publish_status not in {"published", "draft"}:
            errors.append(
                "publish_status is not published or draft"
            )

        candidates = set()
        if country and term_norm:
            key = f"{country}|{term_norm}"
            candidates.update(alias_lookup.get(key, set()))
            candidates.update(canonical_lookup.get(key, set()))

        base = {
            "legacy_row_number": row_number,
            "term_raw": term_raw,
            "term_normalized": term_norm,
            "term_type": (
                "brand"
                if source == "SOCIAL_BRANDS"
                else "reviewed_alias"
            ),
            "country": country,
            "website_hint": website_hint,
            "trusted_reason": trusted_reason,
            "publish_status": publish_status,
            "legacy_source": source,
        }

        if errors:
            review_rows.append(
                {
                    **base,
                    "candidate_entity_ids": "",
                    "review_reason": "; ".join(errors),
                }
            )
        elif len(candidates) == 1:
            linked_rows.append(
                {
                    **base,
                    "entity_id": next(iter(candidates)),
                    "migration_status": (
                        "unique_existing_entity_candidate"
                    ),
                }
            )
        elif len(candidates) > 1:
            review_rows.append(
                {
                    **base,
                    "candidate_entity_ids": "|".join(
                        sorted(candidates)
                    ),
                    "review_reason": (
                        "trusted name resolves to multiple entities"
                    ),
                }
            )
        else:
            review_rows.append(
                {
                    **base,
                    "candidate_entity_ids": "",
                    "review_reason": (
                        "no existing canonical entity link found"
                    ),
                }
            )

    linked = pd.DataFrame(linked_rows)
    review = pd.DataFrame(review_rows)

    linked_path = output_dir / "legacy_trusted_link_candidates.csv"
    review_path = output_dir / "legacy_trusted_migration_review.csv"

    linked.to_csv(linked_path, index=False)
    review.to_csv(review_path, index=False)

    summary = {
        "legacy_rows": int(len(legacy)),
        "unique_link_candidates": int(len(linked)),
        "review_rows": int(len(review)),
        "reconciled_rows": int(len(linked) + len(review)),
    }

    if summary["reconciled_rows"] != summary["legacy_rows"]:
        raise RuntimeError(
            "Legacy trusted roster reconciliation failed."
        )

    manifest = (
        output_dir / "legacy_trusted_migration_manifest.json"
    )
    manifest.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare legacy trusted roster migration."
    )
    parser.add_argument(
        "--legacy",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--canonical-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_legacy_migration(
        legacy_csv=args.legacy,
        canonical_dir=args.canonical_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
