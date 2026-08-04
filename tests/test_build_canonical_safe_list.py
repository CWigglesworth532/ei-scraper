from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from build_canonical_safe_list import build_safe_list
from canonical_entity_layer import build_canonical_layer
from materialise_canonical_links import materialise_links
from run_canonical_linkage import run_linkage


class BuildCanonicalSafeListTests(unittest.TestCase):
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

        accepted_path = self.temp_dir / "accepted.csv"
        pd.DataFrame(
            [
                {
                    "supplier_record_key": "CLIENT-A|SUPPLIER-1",
                    "supplier_name_original": (
                        "Synthetic Community Services"
                    ),
                    "supplier_name_norm": (
                        "synthetic community services"
                    ),
                    "supplier_country": "IE",
                    "supplier_identifier_type": "IE_CRO",
                    "supplier_identifier_value": "998877",
                    "acceptance_status": "accepted",
                    "source_client": "SYNTHETIC_CLIENT",
                    "source_client_file": "synthetic.csv",
                    "reviewed_by": "synthetic-reviewer",
                    "reviewed_at": "2026-08-04T12:00:00Z",
                    "matched_source_record_id": "",
                    "relationship_type": "legal_entity",
                    "related_entity_id": "",
                }
            ]
        ).to_csv(accepted_path, index=False)

        run_linkage(accepted_path, self.canonical_dir)
        materialise_links(self.canonical_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_safe_list_contains_entity_linked_terms(self) -> None:
        summary = build_safe_list(self.canonical_dir)

        trusted = pd.read_parquet(
            self.canonical_dir / "trusted_match_terms.parquet"
        )

        self.assertGreater(summary["trusted_terms"], 0)
        self.assertTrue(
            trusted["entity_id"].astype(str).str.strip().ne("").all()
        )
        self.assertTrue(
            trusted["approved_for_matching"].astype(bool).all()
        )
        self.assertIn(
            "client_variant",
            set(trusted["term_type"]),
        )
        self.assertIn(
            "identifier",
            set(trusted["term_type"]),
        )

    def test_unreviewed_alias_is_excluded(self) -> None:
        aliases_path = (
            self.canonical_dir / "entity_aliases.parquet"
        )
        aliases = pd.read_parquet(aliases_path)

        entity_id = aliases.iloc[0]["entity_id"]

        unreviewed = aliases.iloc[[0]].copy()
        unreviewed["alias_id"] = "sko_alias_unreviewed"
        unreviewed["alias_name"] = "Unreviewed Supplier Name"
        unreviewed["alias_name_norm"] = (
            "unreviewed supplier name"
        )
        unreviewed["verification_status"] = "asserted"
        unreviewed["review_status"] = "unreviewed"

        aliases = pd.concat(
            [aliases, unreviewed],
            ignore_index=True,
        )
        aliases.to_parquet(aliases_path, index=False)

        build_safe_list(self.canonical_dir)

        trusted = pd.read_parquet(
            self.canonical_dir / "trusted_match_terms.parquet"
        )

        excluded = trusted.loc[
            trusted["entity_id"].eq(entity_id)
            & trusted["term_normalized"].eq(
                "unreviewedsuppliername"
            )
        ]
        self.assertTrue(excluded.empty)

    def test_conflicted_identifier_is_excluded(self) -> None:
        identifiers_path = (
            self.canonical_dir / "entity_identifiers.parquet"
        )
        identifiers = pd.read_parquet(identifiers_path)

        conflicted = identifiers.iloc[[0]].copy()
        conflicted["identifier_id"] = "sko_id_conflicted"
        conflicted["identifier_value_raw"] = "CONFLICT-999"
        conflicted["identifier_value_normalized"] = "conflict999"
        conflicted["verification_status"] = "conflicted"

        identifiers = pd.concat(
            [identifiers, conflicted],
            ignore_index=True,
        )
        identifiers.to_parquet(identifiers_path, index=False)

        build_safe_list(self.canonical_dir)

        trusted = pd.read_parquet(
            self.canonical_dir / "trusted_match_terms.parquet"
        )

        self.assertNotIn(
            "conflict999",
            set(
                trusted[
                    "identifier_value_normalized"
                ].astype(str)
            ),
        )

    def test_safe_list_rerun_is_logically_stable(self) -> None:
        build_safe_list(self.canonical_dir)
        first = pd.read_parquet(
            self.canonical_dir / "trusted_match_terms.parquet"
        ).sort_values(
            [
                "entity_id",
                "country",
                "term_type",
                "term_normalized",
                "identifier_type",
            ]
        ).reset_index(drop=True)

        build_safe_list(self.canonical_dir)
        second = pd.read_parquet(
            self.canonical_dir / "trusted_match_terms.parquet"
        ).sort_values(
            [
                "entity_id",
                "country",
                "term_type",
                "term_normalized",
                "identifier_type",
            ]
        ).reset_index(drop=True)

        comparable_columns = [
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
            "source_client",
            "supplier_record_key",
            "schema_version",
        ]

        self.assertTrue(
            first[comparable_columns].equals(
                second[comparable_columns]
            )
        )

    def test_safe_list_qa_views_are_clean(self) -> None:
        build_safe_list(self.canonical_dir)

        connection = duckdb.connect(
            str(
                self.canonical_dir
                / "trusted_match_terms_qa.duckdb"
            ),
            read_only=True,
        )
        try:
            missing_entities = connection.execute(
                """
                SELECT COUNT(*)
                FROM qa_terms_without_entity_id
                """
            ).fetchone()[0]

            unapproved = connection.execute(
                """
                SELECT COUNT(*)
                FROM qa_unapproved_terms
                """
            ).fetchone()[0]

            duplicates = connection.execute(
                """
                SELECT COUNT(*)
                FROM qa_duplicate_trusted_terms
                """
            ).fetchone()[0]

            disallowed = connection.execute(
                """
                SELECT COUNT(*)
                FROM qa_disallowed_status_terms
                """
            ).fetchone()[0]
        finally:
            connection.close()

        self.assertEqual(missing_entities, 0)
        self.assertEqual(unapproved, 0)
        self.assertEqual(duplicates, 0)
        self.assertEqual(disallowed, 0)


if __name__ == "__main__":
    unittest.main()
