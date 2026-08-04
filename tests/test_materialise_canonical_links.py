from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from canonical_entity_layer import build_canonical_layer
from materialise_canonical_links import materialise_links
from run_canonical_linkage import run_linkage


class MaterialiseCanonicalLinksTests(unittest.TestCase):
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

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def write_matches(
        self,
        rows: list[dict[str, str]],
    ) -> Path:
        path = self.temp_dir / "accepted_matches.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def accepted_row(self, **overrides):
        row = {
            "supplier_record_key": "SYNTHETIC|NEW-1",
            "supplier_name_original": (
                "Synthetic Community Services Limited"
            ),
            "supplier_name_norm": (
                "synthetic community services limited"
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
        row.update(overrides)
        return row

    def test_new_entity_identifier_and_alias_are_materialised(
        self,
    ) -> None:
        input_path = self.write_matches(
            [self.accepted_row()]
        )

        run_linkage(input_path, self.canonical_dir)
        links = pd.read_parquet(
            self.canonical_dir
            / "supplier_entity_links.parquet"
        )
        entity_id = links.iloc[0]["matched_entity_id"]

        summary = materialise_links(self.canonical_dir)

        entities = pd.read_parquet(
            self.canonical_dir / "canonical_entities.parquet"
        )
        identifiers = pd.read_parquet(
            self.canonical_dir / "entity_identifiers.parquet"
        )
        aliases = pd.read_parquet(
            self.canonical_dir / "entity_aliases.parquet"
        )

        self.assertEqual(
            summary["new_entities_materialised"],
            1,
        )
        self.assertIn(entity_id, set(entities["entity_id"]))

        entity_identifiers = identifiers.loc[
            identifiers["entity_id"].eq(entity_id)
        ]
        self.assertEqual(len(entity_identifiers), 1)
        self.assertEqual(
            entity_identifiers.iloc[0]["identifier_type"],
            "IE_CRO",
        )

        entity_aliases = aliases.loc[
            aliases["entity_id"].eq(entity_id)
        ]
        self.assertEqual(len(entity_aliases), 1)
        self.assertEqual(
            entity_aliases.iloc[0]["alias_type"],
            "client_variant",
        )

    def test_second_supplier_reuses_materialised_identifier(
        self,
    ) -> None:
        first_input = self.write_matches(
            [
                self.accepted_row(
                    supplier_record_key="CLIENT-A|SUPPLIER-1",
                )
            ]
        )

        run_linkage(first_input, self.canonical_dir)
        materialise_links(self.canonical_dir)

        first_links = pd.read_parquet(
            self.canonical_dir
            / "supplier_entity_links.parquet"
        )
        first_entity_id = first_links.loc[
            first_links["supplier_record_key"].eq(
                "CLIENT-A|SUPPLIER-1"
            ),
            "matched_entity_id",
        ].iloc[0]

        second_input = self.write_matches(
            [
                self.accepted_row(
                    supplier_record_key="CLIENT-B|SUPPLIER-9",
                    supplier_name_original=(
                        "Synthetic Community Services"
                    ),
                    supplier_name_norm=(
                        "synthetic community services"
                    ),
                )
            ]
        )

        run_linkage(second_input, self.canonical_dir)

        second_links = pd.read_parquet(
            self.canonical_dir
            / "supplier_entity_links.parquet"
        )
        second_row = second_links.loc[
            second_links["supplier_record_key"].eq(
                "CLIENT-B|SUPPLIER-9"
            )
        ].iloc[0]

        self.assertEqual(
            second_row["matched_entity_id"],
            first_entity_id,
        )
        self.assertEqual(
            second_row["resolution_method"],
            "accepted_identifier",
        )

    def test_second_supplier_reuses_materialised_alias(
        self,
    ) -> None:
        first_input = self.write_matches(
            [
                self.accepted_row(
                    supplier_record_key="CLIENT-A|SUPPLIER-1",
                )
            ]
        )

        run_linkage(first_input, self.canonical_dir)
        materialise_links(self.canonical_dir)

        first_links = pd.read_parquet(
            self.canonical_dir
            / "supplier_entity_links.parquet"
        )
        first_entity_id = first_links.loc[
            first_links["supplier_record_key"].eq(
                "CLIENT-A|SUPPLIER-1"
            ),
            "matched_entity_id",
        ].iloc[0]

        second_input = self.write_matches(
            [
                self.accepted_row(
                    supplier_record_key="CLIENT-B|SUPPLIER-2",
                    supplier_identifier_type="",
                    supplier_identifier_value="",
                )
            ]
        )

        run_linkage(second_input, self.canonical_dir)

        links = pd.read_parquet(
            self.canonical_dir
            / "supplier_entity_links.parquet"
        )
        second_row = links.loc[
            links["supplier_record_key"].eq(
                "CLIENT-B|SUPPLIER-2"
            )
        ].iloc[0]

        self.assertEqual(
            second_row["matched_entity_id"],
            first_entity_id,
        )
        self.assertEqual(
            second_row["resolution_method"],
            "reviewed_alias",
        )

    def test_rerun_does_not_duplicate_materialised_rows(
        self,
    ) -> None:
        input_path = self.write_matches(
            [self.accepted_row()]
        )

        run_linkage(input_path, self.canonical_dir)
        materialise_links(self.canonical_dir)
        materialise_links(self.canonical_dir)

        links = pd.read_parquet(
            self.canonical_dir
            / "supplier_entity_links.parquet"
        )
        entity_id = links.iloc[0]["matched_entity_id"]

        entities = pd.read_parquet(
            self.canonical_dir / "canonical_entities.parquet"
        )
        identifiers = pd.read_parquet(
            self.canonical_dir / "entity_identifiers.parquet"
        )
        aliases = pd.read_parquet(
            self.canonical_dir / "entity_aliases.parquet"
        )

        self.assertEqual(
            entities["entity_id"].eq(entity_id).sum(),
            1,
        )
        self.assertEqual(
            identifiers["entity_id"].eq(entity_id).sum(),
            1,
        )
        self.assertEqual(
            aliases["entity_id"].eq(entity_id).sum(),
            1,
        )

    def test_relationship_does_not_collapse_entities(
        self,
    ) -> None:
        entities = pd.read_parquet(
            self.canonical_dir / "canonical_entities.parquet"
        )
        parent_id = entities.iloc[0]["entity_id"]

        input_path = self.write_matches(
            [
                self.accepted_row(
                    supplier_record_key="CLIENT-A|SUBSIDIARY-1",
                    supplier_name_original=(
                        "Synthetic Subsidiary Limited"
                    ),
                    supplier_name_norm=(
                        "synthetic subsidiary limited"
                    ),
                    supplier_identifier_value="112233",
                    relationship_type="subsidiary",
                    related_entity_id=parent_id,
                )
            ]
        )

        run_linkage(input_path, self.canonical_dir)
        materialise_links(self.canonical_dir)

        links = pd.read_parquet(
            self.canonical_dir
            / "supplier_entity_links.parquet"
        )
        subsidiary_id = links.iloc[0]["matched_entity_id"]

        relationships = pd.read_parquet(
            self.canonical_dir
            / "entity_relationships.parquet"
        )

        self.assertNotEqual(subsidiary_id, parent_id)
        self.assertEqual(len(relationships), 1)
        self.assertEqual(
            relationships.iloc[0]["subject_entity_id"],
            subsidiary_id,
        )
        self.assertEqual(
            relationships.iloc[0]["object_entity_id"],
            parent_id,
        )
        self.assertEqual(
            relationships.iloc[0]["relationship_type"],
            "subsidiary",
        )

    def test_materialisation_qa_views_are_clean(self) -> None:
        input_path = self.write_matches(
            [self.accepted_row()]
        )

        run_linkage(input_path, self.canonical_dir)
        materialise_links(self.canonical_dir)

        connection = duckdb.connect(
            str(
                self.canonical_dir
                / "canonical_materialisation_qa.duckdb"
            ),
            read_only=True,
        )
        try:
            duplicate_entities = connection.execute(
                """
                SELECT COUNT(*)
                FROM qa_duplicate_entity_ids
                """
            ).fetchone()[0]

            identifier_conflicts = connection.execute(
                """
                SELECT COUNT(*)
                FROM qa_identifier_conflicts
                """
            ).fetchone()[0]

            aliases_without_entities = connection.execute(
                """
                SELECT COUNT(*)
                FROM qa_aliases_without_entities
                """
            ).fetchone()[0]

            relationships_without_entities = connection.execute(
                """
                SELECT COUNT(*)
                FROM qa_relationships_without_entities
                """
            ).fetchone()[0]
        finally:
            connection.close()

        self.assertEqual(duplicate_entities, 0)
        self.assertEqual(identifier_conflicts, 0)
        self.assertEqual(aliases_without_entities, 0)
        self.assertEqual(relationships_without_entities, 0)


if __name__ == "__main__":
    unittest.main()
