from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from match_suppliers_v2 import match_suppliers


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "matcher"


class MatcherPipelineAssertionTests(unittest.TestCase):
    """E2.4 end-to-end matcher assertions on a controlled synthetic mini-file."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_path = self.temp_dir / "matched.csv"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def run_pipeline(self) -> pd.DataFrame:
        match_suppliers(
            suppliers_path=str(FIXTURE_DIR / "pipeline_suppliers.csv"),
            master_path=str(FIXTURE_DIR / "pipeline_master.csv"),
            suppliers_name_col="0",
            suppliers_tax_col="2",
            suppliers_country_col="1",
            master_name_col="entity_name",
            master_tax_col="tax_id",
            master_register_col="ei_register_name",
            master_country_col="country",
            master_region_col="ccaa",
            canonical_safe_list_path=str(FIXTURE_DIR / "pipeline_safe_list.csv"),
            out_path=str(self.output_path),
        )
        return pd.read_csv(
            self.output_path,
            dtype=str,
            keep_default_na=False,
        )

    def test_pipeline_summary_counts(self) -> None:
        output = self.run_pipeline()

        technical = output["match_type"].ne("")
        heuristic = (
            output["name_coop_candidate"].eq("YES")
            | output["name_marker_candidate"].eq("YES")
        )

        self.assertEqual(len(output), 8)
        self.assertEqual(int(technical.sum()), 4)
        self.assertEqual(int((heuristic & ~technical).sum()), 2)
        self.assertEqual(int((heuristic & technical).sum()), 1)
        self.assertEqual(int((technical | heuristic).sum()), 6)
        self.assertEqual(int((~technical & ~heuristic).sum()), 2)

    def test_pipeline_row_outcomes(self) -> None:
        output = self.run_pipeline()
        expected = pd.read_csv(
            FIXTURE_DIR / "pipeline_expected.csv",
            dtype=str,
            keep_default_na=False,
        )

        actual_by_key = {
            (row["0"], row["1"]): row
            for _, row in output.iterrows()
        }

        self.assertEqual(len(actual_by_key), len(expected))

        for _, case in expected.iterrows():
            key = (case["supplier_name"], case["country"])
            with self.subTest(supplier_name=key[0], country=key[1]):
                self.assertIn(key, actual_by_key)
                row = actual_by_key[key]

                self.assertEqual(
                    row["match_type"],
                    case["expected_match_type"],
                )
                self.assertEqual(
                    row["matched_entity_id"],
                    case["expected_entity_id"],
                )
                self.assertEqual(
                    row["name_coop_candidate"],
                    case["expected_coop"],
                )
                self.assertEqual(
                    row["name_marker_candidate"],
                    case["expected_marker"],
                )

                technical = row["match_type"] != ""
                heuristic = (
                    row["name_coop_candidate"] == "YES"
                    or row["name_marker_candidate"] == "YES"
                )
                if technical and heuristic:
                    bucket = "technical_with_heuristic"
                elif technical:
                    bucket = "technical"
                elif heuristic:
                    bucket = "heuristic_only"
                else:
                    bucket = "unflagged"

                self.assertEqual(bucket, case["expected_bucket"])


if __name__ == "__main__":
    unittest.main()
