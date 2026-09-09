"""SKO-036 targeted ONS M72 adapter behaviours."""
from __future__ import annotations

import unittest

import extract_ons_direct_economic as ons


class OnsDirectEconomicTests(unittest.TestCase):
    def test_contract_constants(self):
        self.assertEqual(ons.MODEL_SECTOR_CODE, "M72")
        self.assertEqual(ons.REFERENCE_YEAR, "2023")
        self.assertEqual(ons.MODEL_CLASSIFICATION_VERSION, "Rev. 2 A*64")

    def test_source_fields_match_eurostat_contract_shape(self):
        required = {
            "source_family", "source_organisation", "source_dataset_id", "country",
            "source_classification", "source_sector_code", "model_sector_code",
            "reference_year", "concept_code", "value", "normalized_unit", "currency",
        }
        self.assertTrue(required.issubset(set(ons.SOURCE_FIELDS)))

    def test_sut_and_bres_semantics_are_distinct_in_source_code(self):
        import inspect
        src = inspect.getsource(ons)
        self.assertIn('"source_family": "national_accounts"', src)
        self.assertIn('"source_family": "business_statistics"', src)
        self.assertIn('"concept_code": "EMP_PERSONS"', src)
        self.assertIn('"currency": "GBP"', src)

    def test_accounting_identity_guard_is_present(self):
        import inspect
        self.assertIn("P1=P2+GVA failed", inspect.getsource(ons.extract_sut_m72))


if __name__ == "__main__":
    unittest.main()
