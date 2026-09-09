"""SKO-036 coefficient coverage matrix behaviours."""
from __future__ import annotations

import copy
import unittest
from pathlib import Path

import direct_economic_coefficients as de
import direct_economic_coverage as cov

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config/direct_economic_coefficients.yaml"
FIX = ROOT / "tests/fixtures/direct_economic_coefficients/source_rows.csv"
NOW = "2026-09-09T12:30:00Z"


class DirectEconomicCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = de.load_config(CFG)
        cls.rows = de.read_csv(FIX)
        extra = []
        for row in cls.rows:
            if row["country"] == "FR":
                item = copy.deepcopy(row)
                item["reference_year"] = "2022"
                item["source_release_version"] = "2025-08"
                extra.append(item)
        cls.result = cov.build_coverage(cls.rows + extra, config=cls.cfg, generated_at=NOW)

    def test_01_matrix_has_available_held_out_and_not_applicable(self):
        statuses = {row["availability_status"] for row in self.result["coverage_matrix"]}
        self.assertEqual(statuses, {"available", "held_out", "not_applicable"})

    def test_02_trade_route_is_explicit(self):
        rows = [r for r in self.result["coverage_matrix"] if r["country"] == "DE" and r["model_sector_code"] == "G46"]
        self.assertTrue(rows)
        self.assertTrue(all(r["denominator_route"] == "trade_turnover" for r in rows))

    def test_03_labour_cost_trade_available(self):
        row = next(r for r in self.result["coverage_matrix"] if r["country"] == "DE" and r["outcome_code"] == "LABOUR_COSTS")
        self.assertEqual(row["availability_status"], "available")

    def test_04_employee_compensation_trade_not_applicable(self):
        row = next(r for r in self.result["coverage_matrix"] if r["country"] == "DE" and r["outcome_code"] == "EMPLOYEE_COMPENSATION")
        self.assertEqual((row["availability_status"], row["availability_reason"]), ("not_applicable", "outcome_not_applicable_to_denominator_route"))

    def test_05_year_diagnostics_cover_two_years(self):
        row = next(r for r in self.result["year_diagnostics"] if r["country"] == "FR" and r["outcome_code"] == "GVA")
        self.assertEqual((row["available_years"], row["internal_gap_count"]), ("2022|2023", "0"))

    def test_06_fallback_stays_deferred(self):
        self.assertEqual(self.result["summary"]["fallback_policy_status"], "deferred_pending_real_coverage_matrix")

    def test_07_deterministic(self):
        first = cov.build_coverage(self.rows, config=self.cfg, generated_at=NOW)
        second = cov.build_coverage(list(reversed(self.rows)), config=self.cfg, generated_at=NOW)
        self.assertEqual(first, second)

    def test_08_applicable_rate_excludes_not_applicable(self):
        summary = self.result["summary"]
        self.assertEqual(summary["applicable_records"], summary["available_records"] + summary["held_out_records"])
        self.assertGreater(summary["not_applicable_records"], 0)


if __name__ == "__main__":
    unittest.main()
