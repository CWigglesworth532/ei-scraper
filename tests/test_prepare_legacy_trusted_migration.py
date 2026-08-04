from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from canonical_entity_layer import build_canonical_layer
from prepare_legacy_trusted_migration import (
    prepare_legacy_migration,
)


class PrepareLegacyTrustedMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.canonical_dir = self.temp_dir / "canonical"
        self.output_dir = self.temp_dir / "migration"

        build_canonical_layer(
            Path("tests/fixtures/canonical/source_records.csv"),
            self.canonical_dir,
            selection_csv=Path(
                "tests/fixtures/canonical/selection.csv"
            ),
        )

        entities = pd.read_parquet(
            self.canonical_dir / "canonical_entities.parquet"
        )
        self.ie_entity = entities.loc[
            entities["country"].eq("IE")
        ].iloc[0]

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def write_legacy(self, rows) -> Path:
        path = self.temp_dir / "legacy.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def legacy_row(self, **overrides):
        row = {
            "raw_brand_name": self.ie_entity["canonical_name"],
            "country_hint": "Ireland",
            "website_hint": "https://example.invalid/",
            "trusted_reason": "Synthetic legacy trusted term",
            "publish_status": "published",
            "source": "DIRECTORY_V22",
        }
        row.update(overrides)
        return row

    def test_unique_existing_name_becomes_link_candidate(self) -> None:
        summary = prepare_legacy_migration(
            self.write_legacy([self.legacy_row()]),
            self.canonical_dir,
            self.output_dir,
        )

        linked = pd.read_csv(
            self.output_dir
            / "legacy_trusted_link_candidates.csv",
            dtype=str,
            keep_default_na=False,
        )

        self.assertEqual(summary["legacy_rows"], 1)
        self.assertEqual(summary["unique_link_candidates"], 1)
        self.assertEqual(summary["review_rows"], 0)
        self.assertEqual(
            linked.iloc[0]["entity_id"],
            self.ie_entity["entity_id"],
        )

    def test_unlinked_name_enters_review(self) -> None:
        summary = prepare_legacy_migration(
            self.write_legacy(
                [
                    self.legacy_row(
                        raw_brand_name="Unlinked Synthetic Brand",
                    )
                ]
            ),
            self.canonical_dir,
            self.output_dir,
        )

        review = pd.read_csv(
            self.output_dir
            / "legacy_trusted_migration_review.csv",
            dtype=str,
            keep_default_na=False,
        )

        self.assertEqual(summary["unique_link_candidates"], 0)
        self.assertEqual(summary["review_rows"], 1)
        self.assertIn(
            "no existing canonical entity link",
            review.iloc[0]["review_reason"],
        )

    def test_invalid_country_is_retained_in_review(self) -> None:
        summary = prepare_legacy_migration(
            self.write_legacy(
                [
                    self.legacy_row(
                        country_hint="Synthetic Unknown Country",
                    )
                ]
            ),
            self.canonical_dir,
            self.output_dir,
        )

        review = pd.read_csv(
            self.output_dir
            / "legacy_trusted_migration_review.csv",
            dtype=str,
            keep_default_na=False,
        )

        self.assertEqual(summary["review_rows"], 1)
        self.assertIn(
            "country could not be normalized",
            review.iloc[0]["review_reason"],
        )

    def test_every_legacy_row_is_reconciled(self) -> None:
        input_rows = [
            self.legacy_row(),
            self.legacy_row(
                raw_brand_name="Unlinked Synthetic Brand",
            ),
            self.legacy_row(
                raw_brand_name="Draft Synthetic Brand",
                publish_status="draft",
            ),
            self.legacy_row(
                raw_brand_name="Excluded Synthetic Brand",
                publish_status="excluded",
            ),
        ]

        summary = prepare_legacy_migration(
            self.write_legacy(input_rows),
            self.canonical_dir,
            self.output_dir,
        )

        self.assertEqual(summary["legacy_rows"], 4)
        self.assertEqual(summary["reconciled_rows"], 4)
        self.assertEqual(
            summary["unique_link_candidates"]
            + summary["review_rows"],
            4,
        )

        manifest = json.loads(
            (
                self.output_dir
                / "legacy_trusted_migration_manifest.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["reconciled_rows"], 4)

    def test_social_brand_source_is_preserved_as_brand(self) -> None:
        prepare_legacy_migration(
            self.write_legacy(
                [
                    self.legacy_row(
                        source="SOCIAL_BRANDS",
                    )
                ]
            ),
            self.canonical_dir,
            self.output_dir,
        )

        linked = pd.read_csv(
            self.output_dir
            / "legacy_trusted_link_candidates.csv",
            dtype=str,
            keep_default_na=False,
        )

        self.assertEqual(linked.iloc[0]["term_type"], "brand")


if __name__ == "__main__":
    unittest.main()
