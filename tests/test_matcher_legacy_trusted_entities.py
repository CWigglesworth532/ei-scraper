from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from match_suppliers_v2 import match_suppliers


class MatcherLegacyTrustedEntitiesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.master_path = self.temp_dir / "master.csv"
        self.trusted_path = self.temp_dir / "trusted.csv"
        self.supplier_path = self.temp_dir / "suppliers.csv"
        self.output_path = self.temp_dir / "matched.csv"

        pd.DataFrame(
            [
                {
                    "entity_name": "Synthetic Unrelated Entity",
                    "tax_id": "DE999999",
                    "ei_register_name": "Synthetic Register",
                    "country": "DE",
                    "ccaa": "",
                }
            ]
        ).to_csv(
            self.master_path,
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "raw_brand_name": "Legacy Synthetic Brand E15",
                    "country_hint": "Germany",
                    "publish_status": "published",
                    "source": "SOCIAL_BRANDS",
                }
            ]
        ).to_csv(
            self.trusted_path,
            index=False,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def run_matcher(
        self,
        supplier_name: str,
    ) -> pd.Series:
        pd.DataFrame(
            [[supplier_name, "DE", ""]]
        ).to_csv(
            self.supplier_path,
            index=False,
            header=False,
        )

        match_suppliers(
            suppliers_path=str(self.supplier_path),
            master_path=str(self.master_path),
            suppliers_name_col="0",
            suppliers_tax_col="2",
            suppliers_country_col="1",
            master_name_col="entity_name",
            master_tax_col="tax_id",
            master_register_col="ei_register_name",
            master_country_col="country",
            master_region_col="ccaa",
            trusted_entities_path=str(self.trusted_path),
            out_path=str(self.output_path),
        )

        output = pd.read_csv(
            self.output_path,
            dtype=str,
            keep_default_na=False,
        )
        return output.iloc[0]

    def test_legacy_trusted_brand_remains_positive(self) -> None:
        row = self.run_matcher(
            "Legacy Synthetic Brand E15 Services"
        )

        self.assertEqual(
            row["social_enterprise_supplier"],
            "YES",
        )
        self.assertEqual(
            row["match_type"],
            "known_social_brand",
        )
        self.assertEqual(
            row["matched_entity_name"],
            "Legacy Synthetic Brand E15",
        )
        self.assertEqual(
            row["matched_entity_id"],
            "",
        )
        self.assertEqual(
            row["matched_source_record_id"],
            "",
        )

    def test_unrelated_supplier_is_not_whitelisted(self) -> None:
        row = self.run_matcher(
            "Completely Unrelated Commercial Supplier"
        )

        self.assertNotEqual(
            row["match_type"],
            "known_social_brand",
        )


if __name__ == "__main__":
    unittest.main()
