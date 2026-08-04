from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from canonical_entity_layer import build_canonical_layer
from ingest_manual_trusted_terms import ingest_manual_terms


class IngestManualTrustedTermsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.canonical_dir = self.temp_dir / "canonical"

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
        self.entity_id = entities.iloc[0]["entity_id"]

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def write_manual(self, rows) -> Path:
        path = self.temp_dir / "manual.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def valid_row(self, **overrides):
        row = {
            "entity_id": self.entity_id,
            "term_raw": "Synthetic Trading Name",
            "term_type": "reviewed_alias",
            "country": "IE",
            "identifier_type": "",
            "identifier_value": "",
            "approved_for_matching": "TRUE",
            "approved_by": "synthetic-reviewer",
            "approved_at": "2026-08-04T12:00:00Z",
            "reason": "Synthetic accepted alias",
            "evidence_reference": "synthetic-evidence",
        }
        row.update(overrides)
        return row

    def test_valid_manual_alias_is_added(self) -> None:
        summary = ingest_manual_terms(
            self.write_manual([self.valid_row()]),
            self.canonical_dir,
        )

        aliases = pd.read_parquet(
            self.canonical_dir / "entity_aliases.parquet"
        )

        self.assertEqual(summary["aliases_added"], 1)
        self.assertIn(
            "synthetictradingname",
            set(aliases["alias_name_norm"]),
        )

    def test_unknown_entity_enters_review(self) -> None:
        summary = ingest_manual_terms(
            self.write_manual(
                [
                    self.valid_row(
                        entity_id="sko_ent_missing",
                    )
                ]
            ),
            self.canonical_dir,
        )

        review = pd.read_parquet(
            self.canonical_dir
            / "manual_trusted_terms_review.parquet"
        )

        self.assertEqual(summary["review_rows"], 1)
        self.assertEqual(len(review), 1)

    def test_unapproved_term_enters_review(self) -> None:
        summary = ingest_manual_terms(
            self.write_manual(
                [
                    self.valid_row(
                        approved_for_matching="FALSE",
                    )
                ]
            ),
            self.canonical_dir,
        )

        self.assertEqual(summary["review_rows"], 1)

    def test_manual_identifier_is_added(self) -> None:
        summary = ingest_manual_terms(
            self.write_manual(
                [
                    self.valid_row(
                        term_raw="IE-CRO-778899",
                        term_type="identifier",
                        identifier_type="IE_CRO",
                        identifier_value="778899",
                    )
                ]
            ),
            self.canonical_dir,
        )

        identifiers = pd.read_parquet(
            self.canonical_dir / "entity_identifiers.parquet"
        )

        self.assertEqual(summary["identifiers_added"], 1)
        self.assertIn(
            "778899",
            set(
                identifiers[
                    "identifier_value_normalized"
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
