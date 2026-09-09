"""SKO-036 Eurostat live extraction adapter behaviours."""
from __future__ import annotations

import unittest

import extract_eurostat_direct_economic as ex


class EurostatExtractionTests(unittest.TestCase):
    def test_flatten_jsonstat_decodes_dense_coordinates(self):
        payload = {
            "id": ["geo", "nace_r2", "time"],
            "size": [1, 2, 2],
            "dimension": {
                "geo": {"category": {"index": {"FR": 0}, "label": {"FR": "France"}}},
                "nace_r2": {"category": {"index": {"M72": 0, "G46": 1}, "label": {"M72": "R&D", "G46": "Wholesale"}}},
                "time": {"category": {"index": {"2022": 0, "2023": 1}}},
            },
            "value": {"0": 10, "1": 11, "2": 20, "3": 21},
            "status": {"3": "p"},
        }
        rows = ex.flatten_jsonstat(payload)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["nace_r2"], "M72")
        self.assertEqual(rows[0]["time"], "2022")
        self.assertEqual(rows[3]["nace_r2"], "G46")
        self.assertEqual(rows[3]["time"], "2023")
        self.assertEqual(rows[3]["status_flag"], "p")

    def test_flatten_jsonstat_skips_missing_values(self):
        payload = {
            "id": ["geo", "time"],
            "size": [1, 2],
            "dimension": {
                "geo": {"category": {"index": {"DE": 0}}},
                "time": {"category": {"index": {"2022": 0, "2023": 1}}},
            },
            "value": {"1": 7},
        }
        rows = ex.flatten_jsonstat(payload)
        self.assertEqual(rows, [{"geo": "DE", "time": "2023", "value": 7, "status_flag": ""}])

    def test_default_country_cohort_excludes_uk(self):
        self.assertIn("CH", ex.DEFAULT_COUNTRIES)
        self.assertIn("IE", ex.DEFAULT_COUNTRIES)
        self.assertNotIn("UK", ex.DEFAULT_COUNTRIES)

    def test_governed_source_contract_fields_are_complete(self):
        required = {
            "source_family", "source_organisation", "source_dataset_id", "country",
            "model_sector_code", "reference_year", "concept_code", "value",
            "normalized_unit", "price_basis", "status_flag",
        }
        self.assertTrue(required.issubset(set(ex.SOURCE_FIELDS)))

    def test_trade_divisions_are_governed_model_overrides(self):
        self.assertEqual(ex.GOVERNED_TRADE_DIVISIONS, {"G45", "G46", "G47"})

    def test_sbs_trade_division_retained_even_when_a64_has_only_aggregate(self):
        base = {
            "country": "ES",
            "source_dataset_id": "nama_10_a64",
            "model_sector_code": "G45-G47",
        }
        sbs_trade = {
            "country": "ES",
            "source_dataset_id": "sbs_ovw_act",
            "model_sector_code": "G46",
        }
        sbs_detail = {
            "country": "ES",
            "source_dataset_id": "sbs_ovw_act",
            "model_sector_code": "G466",
        }
        kept, dropped, overrides = ex.constrain_to_a64_model([base, sbs_trade, sbs_detail])
        self.assertIn(sbs_trade, kept)
        self.assertNotIn(sbs_detail, kept)
        self.assertEqual((dropped, overrides), (1, 1))


if __name__ == "__main__":
    unittest.main()
