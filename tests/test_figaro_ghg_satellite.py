from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import figaro_ghg_satellite as ghg


class FigaroGhgSatelliteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "env_ac_ghgfp.csv"
        self.outputs = self.root / "figaro_outputs_2023.csv"
        self.output = self.root / "figaro_ghg_2023.csv"
        self.diag = self.root / "diag.json"
        with self.outputs.open("w", newline="", encoding="utf-8") as handle:
            w = csv.writer(handle, lineterminator="\n")
            w.writerow(["country", "sector", "output_million_eur"])
            w.writerow(["DE", "C20", "200"])
            w.writerow(["FR", "M72", "100"])
            w.writerow(["FIGW1", "A01", "300"])

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, rows):
        with self.source.open("w", newline="", encoding="utf-8") as handle:
            w = csv.writer(handle, lineterminator="\n")
            w.writerow(["c_orig", "nace_r2", "c_dest", "na_item", "freq", "unit", "time_period", "obs_value"])
            w.writerows(rows)

    def _valid_rows(self):
        return [
            ["DE", "C20", "WORLD", "TOTAL", "A", "THS_T", "2023", "12.5"],
            ["FR", "M72", "WORLD", "TOTAL", "A", "THS_T", "2023", "2"],
            ["WRL_REST", "A01", "WORLD", "TOTAL", "A", "THS_T", "2023", "7"],
            ["DE", "TOTAL", "WORLD", "TOTAL", "A", "THS_T", "2023", "999"],
            ["DE", "C20", "FR", "TOTAL", "A", "THS_T", "2023", "99"],
            ["DE", "C20", "WORLD", "P3_S14", "A", "THS_T", "2023", "99"],
            ["DE", "C20", "WORLD", "TOTAL", "A", "THS_T", "2022", "99"],
        ]

    def test_builds_total_origin_sector_satellite_and_scales_to_tonnes(self):
        self._write(self._valid_rows())
        result = ghg.build_ghg_satellite(self.source, self.outputs, self.output, self.diag)
        self.assertEqual(result["coverage"]["covered_nodes"], 3)
        with self.output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        values = {(r["country"], r["sector"]): float(r["value"]) for r in rows}
        self.assertEqual(values[("DE", "C20")], 12500.0)
        self.assertEqual(values[("FR", "M72")], 2000.0)
        self.assertEqual(values[("FIGW1", "A01")], 7000.0)
        self.assertTrue(all(r["outcome"] == "GHG" and r["unit"] == "tCO2e" for r in rows))

    def test_rest_of_world_alias_maps_to_figw1(self):
        self._write([["WLR_REST", "A01", "WORLD", "TOTAL", "A", "THS_T", "2023", "1"]])
        result = ghg.build_ghg_satellite(self.source, self.outputs, self.output)
        self.assertEqual(result["coverage"]["covered_nodes"], 1)
        with self.output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["country"], "FIGW1")

    def test_missing_figaro_nodes_are_reported_not_zero_filled(self):
        self._write([["DE", "C20", "WORLD", "TOTAL", "A", "THS_T", "2023", "1"]])
        result = ghg.build_ghg_satellite(self.source, self.outputs, self.output)
        self.assertEqual(result["coverage"]["covered_nodes"], 1)
        self.assertEqual(result["coverage"]["missing_nodes"], 2)
        self.assertIn("FR_M72", result["coverage"]["missing_node_labels"])
        self.assertIn("FIGW1_A01", result["coverage"]["missing_node_labels"])
        with self.output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)

    def test_duplicate_selected_cell_is_rejected(self):
        row = ["DE", "C20", "WORLD", "TOTAL", "A", "THS_T", "2023", "1"]
        self._write([row, row])
        with self.assertRaisesRegex(ValueError, "duplicate selected"):
            ghg.build_ghg_satellite(self.source, self.outputs, self.output)

    def test_negative_value_is_rejected(self):
        self._write([["DE", "C20", "WORLD", "TOTAL", "A", "THS_T", "2023", "-1"]])
        with self.assertRaisesRegex(ValueError, "negative GHG"):
            ghg.build_ghg_satellite(self.source, self.outputs, self.output)

    def test_total_and_household_nace_are_excluded(self):
        self._write([
            ["DE", "TOTAL", "WORLD", "TOTAL", "A", "THS_T", "2023", "1"],
            ["DE", "TOTAL_HH", "WORLD", "TOTAL", "A", "THS_T", "2023", "1"],
            ["DE", "HH", "WORLD", "TOTAL", "A", "THS_T", "2023", "1"],
            ["DE", "C20", "WORLD", "TOTAL", "A", "THS_T", "2023", "1"],
        ])
        result = ghg.build_ghg_satellite(self.source, self.outputs, self.output)
        self.assertEqual(result["rows"]["selected_rows"], 1)

    def test_non_total_destination_and_na_item_are_excluded(self):
        self._write([
            ["DE", "C20", "FR", "TOTAL", "A", "THS_T", "2023", "1"],
            ["DE", "C20", "WORLD", "P3_S14", "A", "THS_T", "2023", "1"],
            ["DE", "C20", "WORLD", "TOTAL", "A", "THS_T", "2023", "2"],
        ])
        result = ghg.build_ghg_satellite(self.source, self.outputs, self.output)
        self.assertEqual(result["rows"]["selected_rows"], 1)

    def test_wrong_year_frequency_or_unit_are_excluded(self):
        self._write([
            ["DE", "C20", "WORLD", "TOTAL", "A", "THS_T", "2022", "1"],
            ["DE", "C20", "WORLD", "TOTAL", "Q", "THS_T", "2023", "1"],
            ["DE", "C20", "WORLD", "TOTAL", "A", "T", "2023", "1"],
            ["DE", "C20", "WORLD", "TOTAL", "A", "THS_T", "2023", "2"],
        ])
        result = ghg.build_ghg_satellite(self.source, self.outputs, self.output)
        self.assertEqual(result["rows"]["selected_rows"], 1)

    def test_source_cell_not_in_figaro_is_reported(self):
        self._write([
            ["IT", "C10", "WORLD", "TOTAL", "A", "THS_T", "2023", "3"],
            ["DE", "C20", "WORLD", "TOTAL", "A", "THS_T", "2023", "1"],
        ])
        result = ghg.build_ghg_satellite(self.source, self.outputs, self.output)
        self.assertIn("IT_C10", result["coverage"]["source_cells_not_in_figaro"])

    def test_missing_required_column_is_rejected(self):
        with self.source.open("w", newline="", encoding="utf-8") as handle:
            w = csv.writer(handle)
            w.writerow(["c_orig", "nace_r2"])
            w.writerow(["DE", "C20"])
        with self.assertRaisesRegex(ValueError, "missing required"):
            ghg.build_ghg_satellite(self.source, self.outputs, self.output)

    def test_no_matching_rows_is_rejected(self):
        self._write([["DE", "C20", "WORLD", "TOTAL", "A", "THS_T", "2022", "1"]])
        with self.assertRaisesRegex(ValueError, "no env_ac_ghgfp rows matched"):
            ghg.build_ghg_satellite(self.source, self.outputs, self.output)

    def test_deterministic_checksum_and_coverage(self):
        self._write(self._valid_rows())
        first = ghg.build_ghg_satellite(self.source, self.outputs, self.output)
        second = ghg.build_ghg_satellite(self.source, self.outputs, self.output)
        self.assertEqual(first["source"]["sha256"], second["source"]["sha256"])
        self.assertEqual(first["coverage"], second["coverage"])


if __name__ == "__main__":
    unittest.main()
