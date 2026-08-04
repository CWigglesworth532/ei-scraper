from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from canonical_entity_layer import build_canonical_layer
from run_canonical_linkage import run_linkage


class RunCanonicalLinkageTests(unittest.TestCase):
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

        identifiers = pd.read_parquet(
            self.canonical_dir / "entity_identifiers.parquet"
        )
        self.ie_identifier = identifiers.loc[
            identifiers["country"].eq("IE")
        ].iloc[0]

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def write_matches(
        self,
        rows: list[dict[str, str]],
        filename: str = "accepted_matches.csv",
    ) -> Path:
        path = self.temp_dir / filename
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def accepted_row(self, **overrides):
        row = {
            "supplier_record_key": "CLIENT-A|SUPPLIER-1",
            "supplier_name_original": "Accepted Supplier",
            "supplier_name_norm": "accepted supplier",
            "supplier_country": "IE",
            "supplier_identifier_type": "",
            "supplier_identifier_value": "",
            "acceptance_status": "accepted",
            "source_client": "SYNTHETIC_CLIENT",
            "source_client_file": "synthetic.csv",
            "reviewed_by": "synthetic-reviewer",
            "reviewed_at": "2026-08-04T12:00:00Z",
            "matched_source_record_id": "",
            "relationship_type": "legal_entity",
            "related_entity_id": "",
        }
        row.update(overrides)
        return row

    def test_identifier_link_is_persisted(self) -> None:
        input_path = self.write_matches(
            [
                self.accepted_row(
                    supplier_identifier_type=(
                        self.ie_identifier["identifier_type"]
                    ),
                    supplier_identifier_value=(
                        self.ie_identifier[
                            "identifier_value_normalized"
                        ]
                    ),
                )
            ]
        )

        summary = run_linkage(
            input_path,
            self.canonical_dir,
        )

        links = pd.read_parquet(
            self.canonical_dir
            / "supplier_entity_links.parquet"
        )

        self.assertEqual(summary["reused"], 1)
        self.assertEqual(len(links), 1)
        self.assertEqual(
            links.iloc[0]["matched_entity_id"],
            self.ie_identifier["entity_id"],
        )
        self.assertEqual(
            links.iloc[0]["resolution_method"],
            "accepted_identifier",
        )

    def test_repeated_run_preserves_new_entity_id(self) -> None:
        input_path = self.write_matches(
            [
                self.accepted_row(
                    supplier_record_key="CLIENT-A|NEW-1",
                    supplier_name_original=(
                        "Synthetic Newly Accepted Cooperative"
                    ),
                    supplier_name_norm=(
                        "synthetic newly accepted cooperative"
                    ),
                    supplier_identifier_type="IE_CRO",
                    supplier_identifier_value="999999",
                )
            ]
        )

        run_linkage(input_path, self.canonical_dir)
        first = pd.read_parquet(
            self.canonical_dir
            / "supplier_entity_links.parquet"
        )
        first_id = first.iloc[0]["matched_entity_id"]

        run_linkage(input_path, self.canonical_dir)
        second = pd.read_parquet(
            self.canonical_dir
            / "supplier_entity_links.parquet"
        )
        second_id = second.iloc[0]["matched_entity_id"]

        self.assertEqual(first_id, second_id)
        self.assertEqual(
            second.iloc[0]["resolution_method"],
            "persisted_supplier_link",
        )

    def test_name_only_candidate_enters_review(self) -> None:
        entities = pd.read_parquet(
            self.canonical_dir / "canonical_entities.parquet"
        )
        ireland = entities.loc[
            entities["country"].eq("IE")
        ].iloc[0]

        input_path = self.write_matches(
            [
                self.accepted_row(
                    supplier_record_key="CLIENT-A|REVIEW-1",
                    supplier_name_original=(
                        ireland["canonical_name"]
                    ),
                    supplier_name_norm=(
                        ireland["canonical_name_norm"]
                    ),
                )
            ]
        )

        summary = run_linkage(
            input_path,
            self.canonical_dir,
        )

        review = pd.read_parquet(
            self.canonical_dir
            / "identity_review_queue.parquet"
        )

        self.assertEqual(summary["review"], 1)
        self.assertEqual(len(review), 1)
        self.assertEqual(
            review.iloc[0]["resolution_method"],
            "name_candidate_only",
        )
        self.assertEqual(
            review.iloc[0]["matched_entity_id"],
            "",
        )

    def test_nonaccepted_row_receives_no_entity(self) -> None:
        input_path = self.write_matches(
            [
                self.accepted_row(
                    supplier_record_key="CLIENT-A|PROBABLE-1",
                    acceptance_status="probable",
                )
            ]
        )

        summary = run_linkage(
            input_path,
            self.canonical_dir,
        )

        links = pd.read_parquet(
            self.canonical_dir
            / "supplier_entity_links.parquet"
        )

        self.assertEqual(summary["not_eligible"], 1)
        self.assertEqual(
            links.iloc[0]["matched_entity_id"],
            "",
        )
        self.assertFalse(
            bool(links.iloc[0]["allocate_new_entity"])
        )

    def test_duckdb_linkage_qa_views_are_clean(self) -> None:
        input_path = self.write_matches(
            [
                self.accepted_row(
                    supplier_identifier_type=(
                        self.ie_identifier["identifier_type"]
                    ),
                    supplier_identifier_value=(
                        self.ie_identifier[
                            "identifier_value_normalized"
                        ]
                    ),
                )
            ]
        )

        run_linkage(input_path, self.canonical_dir)

        connection = duckdb.connect(
            str(
                self.canonical_dir
                / "canonical_linkage_qa.duckdb"
            ),
            read_only=True,
        )
        try:
            duplicate_keys = connection.execute(
                """
                SELECT COUNT(*)
                FROM qa_duplicate_supplier_record_keys
                """
            ).fetchone()[0]

            missing_ids = connection.execute(
                """
                SELECT COUNT(*)
                FROM qa_links_without_entity_id
                """
            ).fetchone()[0]
        finally:
            connection.close()

        self.assertEqual(duplicate_keys, 0)
        self.assertEqual(missing_ids, 0)


if __name__ == "__main__":
    unittest.main()
