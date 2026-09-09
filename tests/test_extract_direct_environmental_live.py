from __future__ import annotations

import unittest
from decimal import Decimal

import extract_direct_environmental_live as live


class DirectEnvironmentalLiveExtractionTests(unittest.TestCase):
    def test_eurostat_thousand_tonnes_normalise_to_tonnes(self):
        payload = {
            "id": ["freq", "airpol", "unit", "nace_r2", "geo", "time"],
            "size": [1, 1, 1, 1, 1, 1],
            "dimension": {
                "freq": {"category": {"index": {"A": 0}, "label": {"A": "Annual"}}},
                "airpol": {"category": {"index": {"GHG": 0}, "label": {"GHG": "GHG"}}},
                "unit": {"category": {"index": {"THS_T": 0}, "label": {"THS_T": "Thousand tonnes"}}},
                "nace_r2": {"category": {"index": {"M72": 0}, "label": {"M72": "Scientific research and development"}}},
                "geo": {"category": {"index": {"BE": 0}, "label": {"BE": "Belgium"}}},
                "time": {"category": {"index": {"2023": 0}, "label": {"2023": "2023"}}},
            },
            "value": {"0": 12.5},
            "updated": "2026-09-01T00:00:00Z",
        }
        rows = live.normalize_eurostat_ghg(
            payload=payload,
            url="https://example.invalid/eurostat",
            country="BE",
            required_sectors={"M72"},
            year="2023",
            retrieved_at="2026-09-09T00:00:00Z",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["normalized_unit"], "tonne_co2e")
        self.assertEqual(Decimal(rows[0]["value"]), Decimal("12500"))
        self.assertEqual(rows[0]["model_sector_code"], "M72")

    def test_ons_contract_asserts_unit_boundary_and_2023_sic72(self):
        cells = {
            "A1": "GHG total: Mass of air emissions per annum in thousand tonnes of carbon dioxide equivalent, 1990 to 2024",
            "A4": "UK residence basis",
            "EC7": "72",
            "EC8": "Scientific research and development services",
            "A42": "2023",
            "EC42": "215",
        }
        row = live.extract_ons_sic72_from_cells(
            cells,
            retrieved_at="2026-09-09T00:00:00Z",
            workbook_hash="abc123",
        )
        self.assertEqual(row["country"], "UK")
        self.assertEqual(row["source_sector_code"], "72")
        self.assertEqual(row["model_sector_code"], "M72")
        self.assertEqual(row["normalized_unit"], "tonne_co2e")
        self.assertEqual(Decimal(row["value"]), Decimal("215000"))
        self.assertEqual(row["source_release_version"], "sha256:abc123")

    def test_ons_wrong_unit_is_rejected(self):
        cells = {
            "A1": "GHG total: tonnes",
            "A4": "UK residence basis",
            "EC7": "72",
            "EC8": "Scientific research and development services",
            "A42": "2023",
            "EC42": "215",
        }
        with self.assertRaisesRegex(ValueError, "unit contract"):
            live.extract_ons_sic72_from_cells(cells, retrieved_at="x", workbook_hash="y")

    def test_ons_wrong_boundary_is_rejected(self):
        cells = {
            "A1": "Mass of air emissions per annum in thousand tonnes of carbon dioxide equivalent",
            "A4": "Territorial basis",
            "EC7": "72",
            "EC8": "Scientific research and development services",
            "A42": "2023",
            "EC42": "215",
        }
        with self.assertRaisesRegex(ValueError, "residence-basis"):
            live.extract_ons_sic72_from_cells(cells, retrieved_at="x", workbook_hash="y")


if __name__ == "__main__":
    unittest.main()
