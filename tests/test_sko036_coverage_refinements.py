"""Focused SKO-036 live coverage refinements approved 2026-09-09."""
from __future__ import annotations

import inspect
import unittest
from decimal import Decimal
from pathlib import Path

import yaml

import extract_eurostat_direct_economic as ex

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config/direct_economic_coefficients.yaml"


class Sko036CoverageRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CFG.read_text(encoding="utf-8"))

    def test_accounting_identity_tolerance_matches_published_precision(self):
        identity = self.config["accounting_identity"]
        self.assertEqual(Decimal(identity["absolute_tolerance"]), Decimal("0.1"))
        self.assertEqual(identity["tolerance_basis"], "published_million_currency_rounding")

    def test_route_applicability_is_explicit(self):
        outcomes = self.config["outcomes"]
        self.assertEqual(outcomes["LABOUR_COSTS"]["applicable_routes"], ["trade_turnover"])
        self.assertEqual(outcomes["EMPLOYEE_COMPENSATION"]["applicable_routes"], ["default"])
        self.assertEqual(outcomes["GVA"]["applicable_routes"], ["default", "trade_turnover"])
        self.assertEqual(outcomes["EMPLOYMENT_PERSONS"]["applicable_routes"], ["default", "trade_turnover"])

    def test_operating_surplus_derivation_contract(self):
        outcome = self.config["outcomes"]["OPERATING_SURPLUS_MIXED_INCOME"]
        self.assertEqual(outcome["numerator_concept_code"], "B2A3G")
        self.assertEqual(outcome["derivation"]["operation"], "sum")
        self.assertEqual(outcome["derivation"]["source_concept_codes"], ["B2A3N", "P51C"])

    def test_live_extractor_requests_net_not_missing_gross_item(self):
        source = inspect.getsource(ex.extract_national_accounts)
        self.assertIn('"B2A3N"', source)
        self.assertNotIn('"B2A3G":', source)


if __name__ == "__main__":
    unittest.main()
