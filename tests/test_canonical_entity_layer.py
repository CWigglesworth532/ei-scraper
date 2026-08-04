from __future__ import annotations

import re
import shutil
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from canonical_entity_layer import build_canonical_layer


class CanonicalEntityLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "canonical"
        self.fixture = Path(
            "tests/fixtures/canonical/source_records.csv"
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_build_creates_required_outputs(self) -> None:
        summary = build_canonical_layer(
            self.fixture,
            self.output_dir,
        )

        self.assertEqual(
            summary["input_rows_before_selection"],
            8,
        )
        self.assertEqual(summary["source_records"], 8)

        for filename in [
            "canonical_entities.parquet",
            "source_records.parquet",
            "entity_identifiers.parquet",
            "entity_events.parquet",
            "schema_manifest.json",
            "canonical_qa.duckdb",
        ]:
            self.assertTrue(
                (self.output_dir / filename).exists(),
                filename,
            )

    def test_entity_ids_are_opaque_uuid_values(self) -> None:
        build_canonical_layer(self.fixture, self.output_dir)

        entities = pd.read_parquet(
            self.output_dir / "canonical_entities.parquet"
        )

        pattern = re.compile(
            r"^sko_ent_[0-9a-f]{8}-"
            r"[0-9a-f]{4}-"
            r"[0-9a-f]{4}-"
            r"[0-9a-f]{4}-"
            r"[0-9a-f]{12}$"
        )

        self.assertTrue(
            entities["entity_id"].map(
                lambda value: bool(pattern.match(value))
            ).all()
        )

    def test_same_identifier_resolves_to_one_entity(self) -> None:
        build_canonical_layer(self.fixture, self.output_dir)

        records = pd.read_parquet(
            self.output_dir / "source_records.parquet"
        )

        ireland = records.loc[
            records["country"].eq("IE")
        ]

        self.assertEqual(ireland["entity_id"].nunique(), 1)
        self.assertEqual(len(ireland), 2)

    def test_french_sirets_share_siren_entity(self) -> None:
        build_canonical_layer(self.fixture, self.output_dir)

        records = pd.read_parquet(
            self.output_dir / "source_records.parquet"
        )

        france = records.loc[
            records["country"].eq("FR")
        ]

        self.assertEqual(france["entity_id"].nunique(), 1)
        self.assertEqual(len(france), 2)

    def test_same_name_different_identifiers_do_not_merge(self) -> None:
        build_canonical_layer(self.fixture, self.output_dir)

        records = pd.read_parquet(
            self.output_dir / "source_records.parquet"
        )

        netherlands = records.loc[
            records["country"].eq("NL")
        ]

        self.assertEqual(len(netherlands), 2)
        self.assertEqual(
            netherlands["entity_name_norm"].nunique(),
            1,
        )
        self.assertEqual(netherlands["entity_id"].nunique(), 2)

    def test_missing_country_is_quarantined_and_retained(self) -> None:
        build_canonical_layer(self.fixture, self.output_dir)

        records = pd.read_parquet(
            self.output_dir / "source_records.parquet"
        )

        quarantined = records.loc[
            records["resolution_status"].eq("quarantined")
        ]

        self.assertEqual(len(quarantined), 1)
        self.assertTrue(quarantined["entity_id"].isna().all())

    def test_rerun_reuses_source_and_entity_ids(self) -> None:
        build_canonical_layer(self.fixture, self.output_dir)

        first = pd.read_parquet(
            self.output_dir / "source_records.parquet"
        ).sort_values("continuity_key")

        build_canonical_layer(self.fixture, self.output_dir)

        second = pd.read_parquet(
            self.output_dir / "source_records.parquet"
        ).sort_values("continuity_key")

        self.assertListEqual(
            first["source_record_id"].tolist(),
            second["source_record_id"].tolist(),
        )
        self.assertListEqual(
            first["entity_id"].fillna("").tolist(),
            second["entity_id"].fillna("").tolist(),
        )

    def test_shuffled_input_preserves_ids(self) -> None:
        build_canonical_layer(self.fixture, self.output_dir)

        first = pd.read_parquet(
            self.output_dir / "source_records.parquet"
        )

        shuffled_input = self.temp_dir / "shuffled.csv"
        fixture_frame = pd.read_csv(
            self.fixture,
            dtype=str,
            keep_default_na=False,
        )
        fixture_frame.sample(
            frac=1,
            random_state=42,
        ).to_csv(shuffled_input, index=False)

        build_canonical_layer(
            shuffled_input,
            self.output_dir,
        )

        second = pd.read_parquet(
            self.output_dir / "source_records.parquet"
        )

        first_map = dict(
            zip(
                first["continuity_key"],
                first["source_record_id"],
            )
        )
        second_map = dict(
            zip(
                second["continuity_key"],
                second["source_record_id"],
            )
        )

        self.assertDictEqual(first_map, second_map)

    def test_duplicate_continuity_keys_keep_distinct_source_ids(
        self,
    ) -> None:
        duplicate_input = self.temp_dir / "duplicates.csv"

        fixture = pd.read_csv(
            self.fixture,
            dtype=str,
            keep_default_na=False,
        )

        duplicate_row = fixture.iloc[[0]].copy()
        duplicate_row["entity_name"] = (
            "Example Community Cooperative Ltd - duplicate observation"
        )

        combined = pd.concat(
            [fixture, duplicate_row],
            ignore_index=True,
        )
        combined.to_csv(duplicate_input, index=False)

        build_canonical_layer(
            duplicate_input,
            self.output_dir,
        )
        first = pd.read_parquet(
            self.output_dir / "source_records.parquet"
        )

        build_canonical_layer(
            duplicate_input,
            self.output_dir,
        )
        second = pd.read_parquet(
            self.output_dir / "source_records.parquet"
        )

        self.assertEqual(
            first["source_record_id"].nunique(),
            len(first),
        )
        self.assertEqual(
            second["source_record_id"].nunique(),
            len(second),
        )

        first_ids = sorted(first["source_record_id"].tolist())
        second_ids = sorted(second["source_record_id"].tolist())
        self.assertListEqual(first_ids, second_ids)

    def test_exact_duplicate_observations_keep_stable_ids(
        self,
    ) -> None:
        duplicate_input = self.temp_dir / "exact_duplicates.csv"

        fixture = pd.read_csv(
            self.fixture,
            dtype=str,
            keep_default_na=False,
        )

        combined = pd.concat(
            [fixture, fixture.iloc[[0]], fixture.iloc[[0]]],
            ignore_index=True,
        )
        combined.to_csv(duplicate_input, index=False)

        build_canonical_layer(
            duplicate_input,
            self.output_dir,
        )
        first = pd.read_parquet(
            self.output_dir / "source_records.parquet"
        ).sort_values("observation_key")

        build_canonical_layer(
            duplicate_input,
            self.output_dir,
        )
        second = pd.read_parquet(
            self.output_dir / "source_records.parquet"
        ).sort_values("observation_key")

        self.assertEqual(
            first["source_record_id"].nunique(),
            len(first),
        )
        self.assertListEqual(
            first["source_record_id"].tolist(),
            second["source_record_id"].tolist(),
        )
        self.assertListEqual(
            first["entity_id"].fillna("").tolist(),
            second["entity_id"].fillna("").tolist(),
        )

    def test_selection_limits_materialised_records(
        self,
    ) -> None:
        selection = Path(
            "tests/fixtures/canonical/selection.csv"
        )

        summary = build_canonical_layer(
            self.fixture,
            self.output_dir,
            selection_csv=selection,
        )

        records = pd.read_parquet(
            self.output_dir / "source_records.parquet"
        )

        self.assertTrue(summary["selection_applied"])
        self.assertEqual(
            summary["input_rows_before_selection"],
            8,
        )
        self.assertEqual(summary["selected_input_rows"], 3)
        self.assertEqual(len(records), 3)

        self.assertSetEqual(
            set(records["country"]),
            {"IE", "NL"},
        )

        ireland = records.loc[
            records["country"].eq("IE")
        ]
        self.assertEqual(len(ireland), 2)
        self.assertEqual(ireland["entity_id"].nunique(), 1)

    def test_incremental_selections_preserve_prior_records(
        self,
    ) -> None:
        first_selection = self.temp_dir / "first_selection.csv"
        second_selection = self.temp_dir / "second_selection.csv"

        first_selection.write_text(
            "country,ei_register_name,tax_id\n"
            "IE,Synthetic Cooperative Register,IECOOP12345\n"
            "IE,Synthetic Tax Register,IECOOP12345\n",
            encoding="utf-8",
        )

        second_selection.write_text(
            "country,ei_register_name,tax_id\n"
            "NL,Synthetic Foundation Register,NL1001\n",
            encoding="utf-8",
        )

        first_summary = build_canonical_layer(
            self.fixture,
            self.output_dir,
            selection_csv=first_selection,
        )

        first_identifiers = pd.read_parquet(
            self.output_dir / "entity_identifiers.parquet"
        )

        self.assertEqual(first_summary["source_records"], 2)
        self.assertEqual(first_summary["canonical_entities"], 1)
        self.assertEqual(len(first_identifiers), 1)

        second_summary = build_canonical_layer(
            self.fixture,
            self.output_dir,
            selection_csv=second_selection,
        )

        records = pd.read_parquet(
            self.output_dir / "source_records.parquet"
        )
        entities = pd.read_parquet(
            self.output_dir / "canonical_entities.parquet"
        )
        identifiers = pd.read_parquet(
            self.output_dir / "entity_identifiers.parquet"
        )

        self.assertEqual(
            second_summary["selected_source_records"],
            1,
        )
        self.assertEqual(second_summary["source_records"], 3)
        self.assertEqual(len(records), 3)
        self.assertEqual(len(entities), 2)
        self.assertEqual(len(identifiers), 2)

        self.assertSetEqual(
            set(records["country"]),
            {"IE", "NL"},
        )

    def test_identifiers_survive_unchanged_rerun(
        self,
    ) -> None:
        selection = Path(
            "tests/fixtures/canonical/selection.csv"
        )

        build_canonical_layer(
            self.fixture,
            self.output_dir,
            selection_csv=selection,
        )

        first = pd.read_parquet(
            self.output_dir / "entity_identifiers.parquet"
        ).sort_values(
            [
                "country",
                "identifier_type",
                "identifier_value_normalized",
            ]
        ).reset_index(drop=True)

        build_canonical_layer(
            self.fixture,
            self.output_dir,
            selection_csv=selection,
        )

        second = pd.read_parquet(
            self.output_dir / "entity_identifiers.parquet"
        ).sort_values(
            [
                "country",
                "identifier_type",
                "identifier_value_normalized",
            ]
        ).reset_index(drop=True)

        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)

        self.assertListEqual(
            first["identifier_id"].tolist(),
            second["identifier_id"].tolist(),
        )

        self.assertListEqual(
            first["entity_id"].tolist(),
            second["entity_id"].tolist(),
        )

    def test_duckdb_qa_views_are_clean(self) -> None:
        build_canonical_layer(self.fixture, self.output_dir)

        connection = duckdb.connect(
            str(self.output_dir / "canonical_qa.duckdb"),
            read_only=True,
        )
        try:
            duplicate_sources = connection.execute(
                """
                SELECT COUNT(*)
                FROM qa_duplicate_source_record_ids
                """
            ).fetchone()[0]

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
        finally:
            connection.close()

        self.assertEqual(duplicate_sources, 0)
        self.assertEqual(duplicate_entities, 0)
        self.assertEqual(identifier_conflicts, 0)


if __name__ == "__main__":
    unittest.main()
