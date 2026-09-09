"""SKO-036 cross-family employment numerator and FX normalisation behaviours."""
from __future__ import annotations

import copy
import unittest
from decimal import Decimal
from pathlib import Path

import apply_direct_economic_coefficients as app
import direct_economic_coefficients as de
import direct_economic_coverage as cov

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config/direct_economic_coefficients.yaml"
NOW = "2026-09-09T15:30:00Z"


def source(family: str, concept: str, value: str, unit: str, currency: str = "GBP") -> dict[str, str]:
    return {
        "source_family": family,
        "source_organisation": "Office for National Statistics",
        "source_dataset_id": "sut" if family == "national_accounts" else "bres",
        "source_release_version": "2023",
        "source_release_date": "2025-10-31",
        "retrieved_at": NOW,
        "source_url": "local-test",
        "licence": "Open Government Licence",
        "country": "UK",
        "source_classification": "UK SIC 2007",
        "source_classification_version": "SIC 2007",
        "source_sector_code": "M72" if family == "national_accounts" else "72",
        "source_sector_label": "Scientific research and development",
        "model_classification": "NACE",
        "model_classification_version": "Rev. 2 A*64",
        "model_sector_code": "M72",
        "model_sector_label": "Scientific research and development",
        "reference_year": "2023",
        "concept_code": concept,
        "concept_label": concept,
        "value": value,
        "normalized_unit": unit,
        "currency": currency if unit == "million_currency" else "",
        "price_basis": "basic_prices" if concept in {"P1", "B1G"} else ("purchasers_prices" if concept == "P2" else "not_applicable"),
        "status_flag": "",
    }


class FxAndCrossFamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = de.load_config(CFG)
        cls.rows = [
            source("national_accounts", "P1", "45979", "million_currency"),
            source("national_accounts", "P2", "18798", "million_currency"),
            source("national_accounts", "B1G", "27181", "million_currency"),
            source("business_statistics", "EMP_PERSONS", "175400", "persons", currency=""),
        ]

    def test_default_route_can_use_governed_bres_employment_numerator(self):
        result = de.build_coefficients(self.rows, config=self.cfg, generated_at=NOW)
        row = next(r for r in result["coefficients"] if r["outcome_code"] == "EMPLOYMENT_PERSONS")
        self.assertEqual(row["qa_status"], "calculated")
        self.assertEqual(row["denominator_currency"], "GBP")
        self.assertEqual(row["source_family"], "business_statistics|national_accounts")
        self.assertEqual(Decimal(row["coefficient_value"]), Decimal("175400") / Decimal("45979"))
        self.assertIn("bres", row["source_dataset_ids"])
        self.assertIn("sut", row["source_dataset_ids"])

    def test_coverage_preserves_denominator_currency(self):
        result = cov.build_coverage(self.rows, config=self.cfg, generated_at=NOW)
        row = next(r for r in result["coverage_matrix"] if r["outcome_code"] == "EMPLOYMENT_PERSONS")
        self.assertEqual((row["availability_status"], row["denominator_currency"]), ("available", "GBP"))

    def test_missing_fx_rate_holds_absolute_outcome_out_attribution(self):
        coverage = cov.build_coverage(self.rows, config=self.cfg, generated_at=NOW)["coverage_matrix"]
        cfg = copy.deepcopy(self.cfg)
        del cfg["currency_normalisation"]["rates"]["GBP"]
        cohort = [{
            "selection_id": "UK1", "supplier": "University", "client": "Bayer", "country": "UK",
            "spend_eur": "1000000", "proposed_nace_rev2_code": "72.1", "nace_level": "group",
            "nace_description": "R&D", "treatment": "single",
        }]
        result = app.apply_coefficients(cohort, coverage, config=cfg)
        emp = next(r for r in result["outcomes"] if r["outcome_code"] == "EMPLOYMENT_PERSONS")
        gva = next(r for r in result["outcomes"] if r["outcome_code"] == "GVA")
        self.assertEqual((emp["attribution_status"], emp["attribution_reason"]), ("held_out", "missing_fx_rate"))
        self.assertEqual(gva["attribution_status"], "available")


if __name__ == "__main__":
    unittest.main()
