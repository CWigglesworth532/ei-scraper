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
            w.writerow(["GB", "P85", "250"])
            w.writerow(["FIGW1", "A01", "300"])

    def tearDown(self):
        self.tmp.cleanup()

    def _set_outputs(self, rows):
        with self.outputs.open("w", newline="", encoding="utf-8") as handle:
            w = csv.writer(handle, lineterminator="\n")
            w.writerow(["country", "sector", "output_million_eur"])
            w.writerows(rows)

    def _write(self, rows):
        with self.source.open("w", newline="", encoding="utf-8") as handle:
            w = csv.writer(handle, lineterminator="\n")
            w.writerow(["c_orig", "nace_r2", "c_dest", "na_item", "freq", "unit", "TIME_PERIOD", "OBS_VALUE"])
            w.writerows(rows)

    def _code_rows(self):
        return [
            ["DE", "C20", "WORLD", "TOTAL", "A", "THS_T", "2023", "12.5"],
            ["FR", "M72", "WORLD", "TOTAL", "A", "THS_T", "2023", "2"],
            ["WRL_REST", "A01", "WORLD", "TOTAL", "A", "THS_T", "2023", "7"],
        ]

    def _label_rows(self):
        return [
            ["Germany", "Manufacture of chemicals and chemical products", "All countries of the world", "Total", "Annual", "Thousand tonnes", "2023", "12.5"],
            ["France", "Scientific research and development", "All countries of the world", "Total", "Annual", "Thousand tonnes", "2023", "2"],
            ["United Kingdom", "Education", "All countries of the world", "Total", "Annual", "Thousand tonnes", "2023", "3"],
            ["Rest of the world", "Crop and animal production, hunting and related service activities", "All countries of the world", "Total", "Annual", "Thousand tonnes", "2023", "7"],
        ]

    def test_code_form_builds_and_scales_to_tonnes(self):
        self._write(self._code_rows())
        result = ghg.build_ghg_satellite(self.source, self.outputs, self.output, self.diag)
        self.assertEqual(result["coverage"]["covered_nodes"], 3)
        with self.output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        values = {(r["country"], r["sector"]): float(r["value"]) for r in rows}
        self.assertEqual(values[("DE", "C20")], 12500.0)
        self.assertEqual(values[("FIGW1", "A01")], 7000.0)

    def test_real_databrowser_label_form_translates(self):
        self._write(self._label_rows())
        result = ghg.build_ghg_satellite(self.source, self.outputs, self.output, self.diag)
        self.assertEqual(result["coverage"]["covered_nodes"], 4)
        with self.output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        keys = {(r["country"], r["sector"]) for r in rows}
        self.assertEqual(keys, {("DE", "C20"), ("FR", "M72"), ("GB", "P85"), ("FIGW1", "A01")})
        self.assertGreater(result["translation"]["origin_modes"].get("label_map", 0), 0)
        self.assertGreater(result["translation"]["nace_modes"].get("label_map", 0), 0)

    def test_label_totals_are_accepted(self):
        self._write([["Germany", "Manufacture of chemicals and chemical products", "All countries of the world", "Total", "Annual", "Thousand tonnes", "2023", "1"]])
        result = ghg.build_ghg_satellite(self.source, self.outputs, self.output)
        self.assertEqual(result["rows"]["selected_rows"], 1)

    def test_aggregate_origins_are_excluded_and_reported(self):
        self._write([
            ["European Union - 27 countries (from 2020)", "Manufacture of chemicals and chemical products", "All countries of the world", "Total", "Annual", "Thousand tonnes", "2023", "99"],
            ["Extra-EU27 (from 2020)", "Manufacture of chemicals and chemical products", "All countries of the world", "Total", "Annual", "Thousand tonnes", "2023", "99"],
            ["All countries of the world", "Manufacture of chemicals and chemical products", "All countries of the world", "Total", "Annual", "Thousand tonnes", "2023", "99"],
            ["Germany", "Manufacture of chemicals and chemical products", "All countries of the world", "Total", "Annual", "Thousand tonnes", "2023", "1"],
        ])
        result = ghg.build_ghg_satellite(self.source, self.outputs, self.output)
        self.assertEqual(result["rows"]["selected_rows"], 1)
        self.assertEqual(result["rows"]["excluded_aggregate_origin_rows"], 3)

    def test_aggregate_nace_labels_are_excluded(self):
        self._write([
            ["Germany", "Manufacturing", "All countries of the world", "Total", "Annual", "Thousand tonnes", "2023", "99"],
            ["Germany", "Total - all NACE activities", "All countries of the world", "Total", "Annual", "Thousand tonnes", "2023", "99"],
            ["Germany", "Manufacture of chemicals and chemical products", "All countries of the world", "Total", "Annual", "Thousand tonnes", "2023", "1"],
        ])
        result = ghg.build_ghg_satellite(self.source, self.outputs, self.output)
        self.assertEqual(result["rows"]["selected_rows"], 1)
        self.assertEqual(result["rows"]["excluded_aggregate_nace_rows"], 2)

    def test_figaro_native_sector_spellings_are_used(self):
        cases = {
            "Manufacture of furniture; other manufacturing": "C31_32",
            "Motion picture, video, television programme production; programming and broadcasting activities": "J59_60",
            "Computer programming, consultancy, and information service activities": "J62_63",
            "Legal and accounting activities; activities of head offices; management consultancy activities": "M69_70",
            "Education": "P85",
        }
        for label, code in cases.items():
            mapped, mode = ghg.map_nace(label)
            self.assertEqual(mapped, code)
            self.assertEqual(mode, "label_map")

    def test_rest_of_world_label_maps_to_figw1(self):
        mapped, mode = ghg.map_origin("Rest of the world")
        self.assertEqual(mapped, "FIGW1")
        self.assertEqual(mode, "label_map")

    def test_row_group_bridge_allocates_by_output_and_conserves_emissions(self):
        self._set_outputs([
            ["FIGW1", "A01", "300"],
            ["AL", "A01", "100"],
            ["ME", "A01", "100"],
            ["MK", "A01", "200"],
            ["RS", "A01", "300"],
        ])
        self._write([["WRL_REST", "A01", "WORLD", "TOTAL", "A", "THS_T", "2023", "10"]])
        result = ghg.build_ghg_satellite(self.source, self.outputs, self.output, self.diag)
        with self.output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        values = {(r["country"], r["sector"]): float(r["value"]) for r in rows}
        self.assertEqual(set(values), {("FIGW1", "A01"), ("AL", "A01"), ("ME", "A01"), ("MK", "A01"), ("RS", "A01")})
        self.assertAlmostEqual(sum(values.values()), 10000.0, places=9)
        self.assertAlmostEqual(values[("FIGW1", "A01")], 3000.0)
        self.assertAlmostEqual(values[("AL", "A01")], 1000.0)
        self.assertAlmostEqual(values[("MK", "A01")], 2000.0)
        bridge = result["source_geography_alignment"]
        self.assertEqual(bridge["aligned_sectors"], 1)
        self.assertAlmostEqual(bridge["allocation_difference_tco2e"], 0.0, places=9)

    def test_row_group_bridge_produces_common_sector_intensity(self):
        outputs = [
            ["FIGW1", "A01", "300"],
            ["AL", "A01", "100"],
            ["ME", "A01", "200"],
        ]
        self._set_outputs(outputs)
        self._write([["WRL_REST", "A01", "WORLD", "TOTAL", "A", "THS_T", "2023", "6"]])
        ghg.build_ghg_satellite(self.source, self.outputs, self.output)
        with self.output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        output_by_country = {r[0]: float(r[2]) for r in outputs}
        intensities = [float(r["value"]) / output_by_country[r["country"]] for r in rows]
        self.assertTrue(all(abs(x - intensities[0]) < 1e-12 for x in intensities))

    def test_bridge_does_not_fill_genuinely_absent_source_geography(self):
        self._set_outputs([
            ["FIGW1", "A01", "300"],
            ["AL", "A01", "100"],
            ["XK", "C20", "50"],
        ])
        self._write([["WRL_REST", "A01", "WORLD", "TOTAL", "A", "THS_T", "2023", "4"]])
        result = ghg.build_ghg_satellite(self.source, self.outputs, self.output)
        self.assertIn("XK_C20", result["coverage"]["missing_node_labels"])
        self.assertEqual(result["coverage"]["missing_nodes"], 1)

    def test_missing_figaro_nodes_are_reported_not_zero_filled(self):
        self._write([["DE", "C20", "WORLD", "TOTAL", "A", "THS_T", "2023", "1"]])
        result = ghg.build_ghg_satellite(self.source, self.outputs, self.output)
        self.assertEqual(result["coverage"]["covered_nodes"], 1)
        self.assertEqual(result["coverage"]["missing_nodes"], 3)

    def test_duplicate_selected_cell_is_rejected(self):
        row = ["DE", "C20", "WORLD", "TOTAL", "A", "THS_T", "2023", "1"]
        self._write([row, row])
        with self.assertRaisesRegex(ValueError, "duplicate selected"):
            ghg.build_ghg_satellite(self.source, self.outputs, self.output)

    def test_negative_value_is_rejected(self):
        self._write([["DE", "C20", "WORLD", "TOTAL", "A", "THS_T", "2023", "-1"]])
        with self.assertRaisesRegex(ValueError, "negative GHG"):
            ghg.build_ghg_satellite(self.source, self.outputs, self.output)

    def test_non_total_destination_is_excluded(self):
        self._write([
            ["DE", "C20", "FR", "TOTAL", "A", "THS_T", "2023", "1"],
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
            ["IT", "C10T12", "WORLD", "TOTAL", "A", "THS_T", "2023", "3"],
            ["DE", "C20", "WORLD", "TOTAL", "A", "THS_T", "2023", "1"],
        ])
        result = ghg.build_ghg_satellite(self.source, self.outputs, self.output)
        self.assertIn("IT_C10T12", result["coverage"]["source_cells_not_in_figaro"])

    def test_unknown_origin_label_is_rejected(self):
        self._write([["Atlantis", "Manufacture of chemicals and chemical products", "All countries of the world", "Total", "Annual", "Thousand tonnes", "2023", "1"]])
        with self.assertRaisesRegex(ValueError, "unmapped Eurostat origin"):
            ghg.build_ghg_satellite(self.source, self.outputs, self.output)

    def test_missing_required_column_is_rejected(self):
        with self.source.open("w", newline="", encoding="utf-8") as handle:
            w = csv.writer(handle)
            w.writerow(["c_orig", "nace_r2"])
            w.writerow(["DE", "C20"])
        with self.assertRaisesRegex(ValueError, "missing required"):
            ghg.build_ghg_satellite(self.source, self.outputs, self.output)

    def test_deterministic_checksum_and_coverage(self):
        self._write(self._label_rows())
        first = ghg.build_ghg_satellite(self.source, self.outputs, self.output)
        second = ghg.build_ghg_satellite(self.source, self.outputs, self.output)
        self.assertEqual(first["source"]["sha256"], second["source"]["sha256"])
        self.assertEqual(first["coverage"], second["coverage"])
        self.assertEqual(first["source_geography_alignment"], second["source_geography_alignment"])


if __name__ == "__main__":
    unittest.main()
