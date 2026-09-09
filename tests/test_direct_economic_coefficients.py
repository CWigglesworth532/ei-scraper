"""SKO-036 direct economic coefficient behaviours."""
from __future__ import annotations

import copy
import inspect
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

import direct_economic_coefficients as de

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests/fixtures/direct_economic_coefficients"
CFG = ROOT / "config/direct_economic_coefficients.yaml"
NOW = "2026-09-09T11:30:00Z"


class DirectEconomicCoefficientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = de.load_config(CFG)
        cls.source = de.read_csv(FIX / "source_rows.csv")
        cls.result = de.build_coefficients(cls.source, config=cls.cfg, generated_at=NOW)

    def coef(self, country, sector, outcome):
        return next(r for r in self.result["coefficients"] if (r["country"], r["model_sector_code"], r["outcome_code"]) == (country, sector, outcome))

    def test_01_supplier_agnostic_contract(self):
        source = inspect.getsource(de).casefold()
        self.assertNotIn("social_economy", source)
        self.assertNotIn("canonical_entity", source)
        self.assertNotIn("airtable", source)

    def test_02_output_route_for_ordinary_sector(self):
        row = self.coef("FR", "M72", "GVA")
        self.assertEqual((row["denominator_method"], row["denominator_compatibility_class"]), ("output_basic_prices", "OUTPUT_PROXY_ACCEPTED"))

    def test_03_gva_per_output(self):
        self.assertEqual(Decimal(self.coef("FR", "M72", "GVA")["coefficient_value"]), Decimal("0.45"))

    def test_04_intermediate_consumption(self):
        self.assertEqual(Decimal(self.coef("FR", "M72", "INTERMEDIATE_CONSUMPTION")["coefficient_value"]), Decimal("0.55"))

    def test_05_employee_compensation(self):
        self.assertEqual(Decimal(self.coef("FR", "M72", "EMPLOYEE_COMPENSATION")["coefficient_value"]), Decimal("0.3"))

    def test_06_production_taxes(self):
        self.assertEqual(Decimal(self.coef("FR", "M72", "PRODUCTION_TAXES_NET")["coefficient_value"]), Decimal("0.02"))

    def test_07_operating_surplus(self):
        self.assertEqual(Decimal(self.coef("FR", "M72", "OPERATING_SURPLUS_MIXED_INCOME")["coefficient_value"]), Decimal("0.13"))

    def test_08_employment_persons(self):
        self.assertEqual(Decimal(self.coef("FR", "M72", "EMPLOYMENT_PERSONS")["coefficient_value"]), Decimal("8"))
        self.assertEqual(self.coef("FR", "M72", "EMPLOYMENT_PERSONS")["coefficient_unit"], "persons_per_million_currency_denominator")

    def test_09_hours(self):
        self.assertEqual(Decimal(self.coef("FR", "M72", "EMPLOYMENT_HOURS")["coefficient_value"]), Decimal("12000"))

    def test_10_capital_formation(self):
        self.assertEqual(Decimal(self.coef("FR", "M72", "GROSS_FIXED_CAPITAL_FORMATION")["coefficient_value"]), Decimal("0.08"))

    def test_11_trade_uses_turnover_not_output(self):
        row = self.coef("DE", "G46", "GVA")
        self.assertEqual(row["denominator_method"], "turnover_revenue_basis")
        self.assertEqual(row["denominator_concept_code"], "TURNOVER")
        self.assertEqual(Decimal(row["coefficient_value"]), Decimal("0.04"))

    def test_12_trade_missing_business_stat_outcome_is_held_out(self):
        row = self.coef("DE", "G46", "PRODUCTION_TAXES_NET")
        self.assertEqual((row["qa_status"], row["qa_reason"], row["coefficient_value"]), ("held_out", "numerator_missing", ""))

    def test_13_trade_na_output_never_substitutes_for_turnover(self):
        row = self.coef("DE", "G46", "GVA")
        self.assertNotEqual(row["denominator_value"], "10")
        self.assertEqual(row["denominator_value"], "100")

    def test_14_accounting_identity(self):
        self.assertEqual(self.result["qa"]["accounting_identity_checks"], 2)
        self.assertEqual(self.result["qa"]["accounting_identity_failures"], 0)

    def test_15_missing_denominator_holds_out(self):
        rows = [r for r in copy.deepcopy(self.source) if not (r["country"] == "DE" and r["source_family"] == "business_statistics" and r["concept_code"] == "TURNOVER")]
        result = de.build_coefficients(rows, config=self.cfg, generated_at=NOW)
        trade = next(r for r in result["coefficients"] if r["country"] == "DE" and r["outcome_code"] == "GVA")
        self.assertEqual((trade["qa_status"], trade["qa_reason"]), ("held_out", "denominator_missing"))

    def test_16_zero_denominator_rejected_for_calculation(self):
        rows = copy.deepcopy(self.source)
        next(r for r in rows if r["country"] == "FR" and r["concept_code"] == "P1")["value"] = "0"
        result = de.build_coefficients(rows, config=self.cfg, generated_at=NOW)
        self.assertEqual(self.coef_from(result, "FR", "M72", "GVA")["qa_reason"], "denominator_nonpositive")

    @staticmethod
    def coef_from(result, country, sector, outcome):
        return next(r for r in result["coefficients"] if (r["country"], r["model_sector_code"], r["outcome_code"]) == (country, sector, outcome))

    def test_17_conflicting_source_duplicate_rejected(self):
        rows = copy.deepcopy(self.source)
        conflict = copy.deepcopy(rows[0])
        conflict["value"] = "101"
        with self.assertRaisesRegex(ValueError, "Conflicting source observation"):
            de.build_coefficients(rows + [conflict], config=self.cfg, generated_at=NOW)

    def test_18_identical_source_duplicate_safe(self):
        result = de.build_coefficients(self.source + [copy.deepcopy(self.source[0])], config=self.cfg, generated_at=NOW)
        self.assertEqual(result["qa"]["duplicate_source_rows"], 1)

    def test_19_wrong_model_version_rejected(self):
        rows = copy.deepcopy(self.source)
        rows[0]["model_classification_version"] = "Rev. 2.1"
        with self.assertRaisesRegex(ValueError, "classification version"):
            de.build_coefficients(rows, config=self.cfg, generated_at=NOW)

    def test_20_negative_production_tax_allowed(self):
        rows = copy.deepcopy(self.source)
        next(r for r in rows if r["country"] == "FR" and r["concept_code"] == "D29X39")["value"] = "-1"
        result = de.build_coefficients(rows, config=self.cfg, generated_at=NOW)
        self.assertEqual(Decimal(self.coef_from(result, "FR", "M72", "PRODUCTION_TAXES_NET")["coefficient_value"]), Decimal("-0.01"))

    def test_21_deterministic(self):
        self.assertEqual(self.result, de.build_coefficients(self.source, config=self.cfg, generated_at=NOW))

    def test_22_shuffled_input_deterministic(self):
        self.assertEqual(self.result, de.build_coefficients(list(reversed(self.source)), config=self.cfg, generated_at=NOW))

    def test_23_schema_fields(self):
        self.assertTrue(all(set(r) == set(de.SOURCE_FIELDS) for r in self.result["sources"]))
        self.assertTrue(all(set(r) == set(de.COEFFICIENT_FIELDS) for r in self.result["coefficients"]))
        self.assertTrue(all(set(r) == set(de.COVERAGE_FIELDS) for r in self.result["coverage"]))

    def test_24_no_indirect_or_environmental_logic(self):
        source = inspect.getsource(de).casefold()
        self.assertNotIn("leontief", source)
        self.assertNotIn("ghg", source)
        self.assertNotIn("co2", source)

    def test_25_cli_deterministic(self):
        outputs = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as td:
                p = Path(td)
                args = [
                    sys.executable,
                    str(ROOT / "direct_economic_coefficients.py"),
                    "--config", str(CFG),
                    "--source-input", str(FIX / "source_rows.csv"),
                    "--source-output", str(p / "source.csv"),
                    "--coefficient-output", str(p / "coeff.csv"),
                    "--coverage-output", str(p / "coverage.csv"),
                    "--qa-output", str(p / "qa.json"),
                    "--generated-at", NOW,
                ]
                subprocess.run(args, check=True)
                outputs.append(tuple((p / name).read_bytes() for name in ("source.csv", "coeff.csv", "coverage.csv", "qa.json")))
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
