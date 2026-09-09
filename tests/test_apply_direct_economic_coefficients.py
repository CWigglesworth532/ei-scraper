"""SKO-036 procurement attribution pilot behaviours."""
from __future__ import annotations

import unittest
from pathlib import Path

import direct_economic_coefficients as de
import apply_direct_economic_coefficients as app

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config/direct_economic_coefficients.yaml"


def cell(country, sector, outcome, status="available", value="0.5", unit="currency_gva_per_currency_denominator", route="default"):
    return {
        "country": country,
        "model_sector_code": sector,
        "model_sector_label": sector,
        "reference_year": "2023",
        "denominator_route": route,
        "denominator_concept_code": "P1" if route == "default" else "TURNOVER",
        "outcome_code": outcome,
        "outcome_label": outcome,
        "numerator_concept_code": outcome,
        "availability_status": status,
        "availability_reason": "" if status == "available" else ("outcome_not_applicable_to_denominator_route" if status == "not_applicable" else "numerator_missing"),
        "coefficient_value": value if status == "available" else "",
        "coefficient_unit": unit,
        "coefficient_id": f"id-{country}-{sector}-{outcome}",
        "source_family": "national_accounts" if route == "default" else "business_statistics",
        "source_dataset_ids": "synthetic",
        "source_release_versions": "v1",
        "source_release_dates": "2026-09-09",
    }


class ApplyDirectEconomicCoefficientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = de.load_config(CFG)
        cls.matrix = [
            cell("BE", "S94", "GVA", value="0.4"),
            cell("BE", "S94", "EMPLOYMENT_PERSONS", value="5", unit="persons_per_million_currency_denominator"),
            cell("BE", "S94", "LABOUR_COSTS", status="not_applicable"),
            cell("DE", "G46", "GVA", value="0.04", route="trade_turnover"),
            cell("DE", "G46", "EMPLOYMENT_PERSONS", value="2", unit="persons_per_million_currency_denominator", route="trade_turnover"),
            cell("DE", "G46", "LABOUR_COSTS", value="0.025", route="trade_turnover"),
        ]

    def test_class_code_maps_to_a64_division_sector(self):
        sector, _, status, reason = app.map_nace_to_model_sector("94.11", {"S94": "Membership organisations"})
        self.assertEqual((sector, status, reason), ("S94", "mapped", "division_to_a64_range"))

    def test_group_code_maps_to_a64_division_sector(self):
        sector, _, status, _ = app.map_nace_to_model_sector("72.1", {"M72": "Research and development"})
        self.assertEqual((sector, status), ("M72", "mapped"))

    def test_range_sector_mapping(self):
        sector, _, status, _ = app.map_nace_to_model_sector("10.82", {"C10-C12": "Food etc"})
        self.assertEqual((sector, status), ("C10-C12", "mapped"))

    def test_more_specific_sector_beats_broader_aggregate(self):
        sector, _, status, reason = app.map_nace_to_model_sector(
            "18.12",
            {"C16-C18": "Wood paper printing", "C18": "Printing"},
        )
        self.assertEqual((sector, status, reason), ("C18", "mapped", "division_to_most_specific_a64_sector"))

    def test_more_specific_partial_range_beats_broader_aggregate(self):
        sector, _, status, reason = app.map_nace_to_model_sector(
            "70.22",
            {"M69-M71": "Professional services", "M69_M70": "Legal accounting management consultancy"},
        )
        self.assertEqual((sector, status, reason), ("M69_M70", "mapped", "division_to_most_specific_a64_sector"))

    def test_equal_specificity_stays_unresolved(self):
        sector, _, status, reason = app.map_nace_to_model_sector(
            "71.20",
            {"M71": "Technical services A", "X71": "Technical services B"},
        )
        self.assertEqual((sector, status, reason), ("", "unresolved", "ambiguous_equally_specific_a64_sectors"))

    def test_currency_and_persons_attribution(self):
        cohort = [{
            "selection_id": "1", "supplier": "X", "client": "Bayer", "country": "BE",
            "spend_eur": "2000000", "proposed_nace_rev2_code": "94.11", "nace_level": "class",
            "nace_description": "Membership", "treatment": "single",
        }]
        result = app.apply_coefficients(cohort, self.matrix, config=self.cfg)
        gva = next(r for r in result["outcomes"] if r["outcome_code"] == "GVA")
        emp = next(r for r in result["outcomes"] if r["outcome_code"] == "EMPLOYMENT_PERSONS")
        self.assertEqual((gva["modelled_value"], gva["modelled_unit"]), ("800000", "EUR"))
        self.assertEqual((emp["modelled_value"], emp["modelled_unit"]), ("10", "persons"))

    def test_not_applicable_not_counted_as_held_out(self):
        cohort = [{
            "selection_id": "1", "supplier": "X", "client": "Bayer", "country": "BE",
            "spend_eur": "100", "proposed_nace_rev2_code": "94.11", "nace_level": "class",
            "nace_description": "Membership", "treatment": "single",
        }]
        result = app.apply_coefficients(cohort, self.matrix, config=self.cfg)
        obs = result["observations"][0]
        self.assertEqual(obs["not_applicable_outcomes"], "1")
        self.assertEqual(obs["held_out_outcomes"], "0")

    def test_uk_is_explicitly_held_out_when_layer_missing(self):
        cohort = [{
            "selection_id": "2", "supplier": "University", "client": "Bayer", "country": "UK",
            "spend_eur": "1000", "proposed_nace_rev2_code": "94.11", "nace_level": "class",
            "nace_description": "Membership", "treatment": "single",
        }]
        result = app.apply_coefficients(cohort, self.matrix, config=self.cfg)
        self.assertEqual(result["observations"][0]["observation_status"], "held_out_country_source_missing")
        self.assertTrue(all(r["attribution_reason"] == "country_not_in_coefficient_layer" for r in result["outcomes"]))

    def test_missing_nace_is_explicitly_held_out(self):
        cohort = [{
            "selection_id": "3", "supplier": "Unknown", "client": "Bayer", "country": "BE",
            "spend_eur": "1000", "proposed_nace_rev2_code": "", "nace_level": "",
            "nace_description": "", "treatment": "unresolved",
        }]
        result = app.apply_coefficients(cohort, self.matrix, config=self.cfg)
        self.assertEqual(result["observations"][0]["model_mapping_reason"], "nace_code_missing")
        self.assertEqual(result["observations"][0]["observation_status"], "held_out_nace_unresolved")

    def test_fixed_year_policy_is_used(self):
        cohort = [{
            "selection_id": "1", "supplier": "X", "client": "Bayer", "country": "BE",
            "spend_eur": "100", "proposed_nace_rev2_code": "94.11", "nace_level": "class",
            "nace_description": "Membership", "treatment": "single",
        }]
        result = app.apply_coefficients(cohort, self.matrix, config=self.cfg)
        self.assertEqual(result["summary"]["coefficient_reference_year"], "2023")
        self.assertFalse(result["summary"]["automatic_year_fallback_enabled"])
        self.assertEqual(result["summary"]["spend_year_present_observations"], 0)


if __name__ == "__main__":
    unittest.main()
