"""SKO-036A live-source contract refinement behaviours."""
from __future__ import annotations

import csv
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config/direct_economic_coefficients.yaml"
FIX = ROOT / "tests/fixtures/direct_economic_coefficients/source_rows.csv"


class Sko036LiveSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CFG.read_text(encoding="utf-8"))
        with FIX.open(newline="", encoding="utf-8-sig") as handle:
            cls.fixture_rows = list(csv.DictReader(handle))

    def test_01_hybrid_denominator_architecture_unchanged(self):
        routes = self.config["denominator_routes"]
        self.assertEqual(routes["default"]["denominator_concept_code"], "P1")
        self.assertEqual(routes["default"]["method"], "output_basic_prices")
        self.assertEqual(routes["trade_turnover"]["model_sector_codes"], ["G45", "G46", "G47"])
        self.assertEqual(routes["trade_turnover"]["denominator_concept_code"], "TURNOVER")
        self.assertEqual(routes["trade_turnover"]["method"], "turnover_revenue_basis")

    def test_02_eurostat_employment_requires_na_item_and_unit(self):
        src = self.config["source_catalogue"]["eurostat_employment"]
        self.assertEqual(src["mapping_rule"], "na_item_and_unit_required")
        mapping = src["normalization_map"]
        self.assertEqual(mapping["EMP_DC|THS_PER"]["concept_code"], "EMP_PERSONS")
        self.assertEqual(mapping["EMP_DC|THS_PER"]["normalized_unit"], "persons")
        self.assertEqual(mapping["EMP_DC|THS_PER"]["scale"], "1000")
        self.assertEqual(mapping["EMP_DC|THS_HW"]["concept_code"], "EMP_HOURS")
        self.assertEqual(mapping["EMP_DC|THS_HW"]["normalized_unit"], "hours")
        self.assertEqual(mapping["EMP_DC|THS_HW"]["scale"], "1000")

    def test_03_trade_labour_costs_are_not_d1(self):
        outcomes = self.config["outcomes"]
        self.assertEqual(outcomes["EMPLOYEE_COMPENSATION"]["numerator_concept_code"], "D1")
        self.assertEqual(outcomes["LABOUR_COSTS"]["numerator_concept_code"], "LABOUR_COSTS")
        sbs = self.config["source_catalogue"]["eurostat_sbs"]
        self.assertEqual(sbs["source_measure_mapping"]["EXPN_SAL_BEN_MEUR"], "LABOUR_COSTS")
        self.assertNotEqual(sbs["source_measure_mapping"]["EXPN_SAL_BEN_MEUR"], "D1")

    def test_04_trade_fixture_preserves_semantic_distinction(self):
        trade_rows = [r for r in self.fixture_rows if r["source_family"] == "business_statistics"]
        concepts = {r["concept_code"] for r in trade_rows}
        self.assertIn("LABOUR_COSTS", concepts)
        self.assertNotIn("D1", concepts)
        labour = next(r for r in trade_rows if r["concept_code"] == "LABOUR_COSTS")
        self.assertEqual(labour["concept_label"], "Employee benefits expense")

    def test_05_uk_source_catalogue_expanded_and_hours_deferred(self):
        catalogue = self.config["source_catalogue"]
        for key in ("ons_supply_use", "ons_abs", "ons_bres", "ons_gfcf", "ons_capital_stocks", "ons_ashe_hours"):
            self.assertIn(key, catalogue)
        self.assertEqual(catalogue["ons_ashe_hours"]["governance_status"], "deferred_not_comparable_v1")
        self.assertEqual(catalogue["ons_ashe_hours"]["intended_concepts"], [])

    def test_06_year_fallback_remains_disabled_pending_coverage(self):
        policy = self.config["year_fallback_policy"]
        self.assertFalse(policy["automatic_fallback_enabled"])
        self.assertEqual(policy["status"], "deferred_pending_real_coverage_matrix")
        self.assertEqual(policy["required_evidence"], "country_x_sector_x_year_x_outcome_coverage_matrix")


if __name__ == "__main__":
    unittest.main()
