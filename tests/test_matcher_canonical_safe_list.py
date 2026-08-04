from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from match_suppliers_v2 import match_suppliers


ENTITY_A = "sko_ent_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
SOURCE_A = "sko_src_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


class MatcherCanonicalSafeListTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.master_path = self.temp_dir / "master.csv"
        self.safe_path = self.temp_dir / "trusted_match_terms.csv"
        self.output_path = self.temp_dir / "matched.csv"

        pd.DataFrame(
            [
                {
                    "entity_name": "Unrelated Cooperative",
                    "tax_id": "DE999999",
                    "ei_register_name": "Synthetic Register",
                    "country": "DE",
                    "ccaa": "",
                }
            ]
        ).to_csv(self.master_path, index=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def write_supplier(self, name: str, country: str, tax: str = ""):
        path = self.temp_dir / "suppliers.csv"
        pd.DataFrame([[name, country, tax]]).to_csv(
            path,
            index=False,
            header=False,
        )
        return path

    def write_safe_term(self, **overrides):
        row = {
            "entity_id": ENTITY_A,
            "source_record_id": SOURCE_A,
            "term_type": "reviewed_alias",
            "term_raw": "Example Social Trading",
            "term_normalized": "examplesocialtrading",
            "country": "DE",
            "identifier_type": "",
            "identifier_value_normalized": "",
            "verification_status": "human_verified",
            "review_status": "accepted",
            "approved_for_matching": True,
        }
        row.update(overrides)
        pd.DataFrame([row]).to_csv(
            self.safe_path,
            index=False,
        )

    def run_matcher(self, supplier_path: Path) -> pd.DataFrame:
        match_suppliers(
            suppliers_path=str(supplier_path),
            master_path=str(self.master_path),
            suppliers_name_col="0",
            suppliers_tax_col="2",
            suppliers_country_col="1",
            master_name_col="entity_name",
            master_tax_col="tax_id",
            master_register_col="ei_register_name",
            master_country_col="country",
            master_region_col="ccaa",
            canonical_safe_list_path=str(self.safe_path),
            out_path=str(self.output_path),
        )
        return pd.read_csv(
            self.output_path,
            dtype=str,
            keep_default_na=False,
        )

    def test_exact_alias_outputs_persistent_ids(self) -> None:
        self.write_safe_term()

        output = self.run_matcher(
            self.write_supplier(
                "Example Social Trading",
                "DE",
            )
        )

        row = output.iloc[0]
        self.assertEqual(row["social_enterprise_supplier"], "YES")
        self.assertEqual(
            row["match_type"],
            "canonical_safe_name_exact",
        )
        self.assertEqual(row["matched_entity_id"], ENTITY_A)
        self.assertEqual(
            row["matched_source_record_id"],
            SOURCE_A,
        )

    def test_approved_brand_outputs_persistent_id(self) -> None:
        self.write_safe_term(
            term_type="brand",
            term_raw="Auticon",
            term_normalized="auticon",
        )

        output = self.run_matcher(
            self.write_supplier(
                "Auticon Deutschland GmbH",
                "DE",
            )
        )

        row = output.iloc[0]
        self.assertEqual(
            row["match_type"],
            "canonical_safe_brand",
        )
        self.assertEqual(row["matched_entity_id"], ENTITY_A)

    def test_exact_identifier_without_type_outputs_id(
        self,
    ) -> None:
        self.write_safe_term(
            term_type="identifier",
            term_raw="HRB 1001",
            term_normalized="hrb1001",
            identifier_type="DE_HRB",
            identifier_value_normalized="hrb1001",
        )

        output = self.run_matcher(
            self.write_supplier(
                "Different Client Supplier Name",
                "DE",
                "HRB 1001",
            )
        )

        row = output.iloc[0]
        self.assertEqual(
            row["match_type"],
            "canonical_safe_identifier",
        )
        self.assertEqual(row["matched_entity_id"], ENTITY_A)

    def test_cross_border_alias_does_not_match(self) -> None:
        self.write_safe_term()

        output = self.run_matcher(
            self.write_supplier(
                "Example Social Trading",
                "FR",
            )
        )

        row = output.iloc[0]
        self.assertEqual(row["matched_entity_id"], "")
        self.assertNotEqual(
            row["match_type"],
            "canonical_safe_name_exact",
        )


if __name__ == "__main__":
    unittest.main()
