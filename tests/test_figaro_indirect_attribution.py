from __future__ import annotations

import copy
import unittest

import numpy as np

import figaro_indirect_attribution as figaro


class FigaroIndirectAttributionTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "figaro_source": {
                "product_id": "naio_10_fcp",
                "edition": "2026",
                "reference_year": 2023,
                "table_type": "industry_by_industry_icio",
            },
            "mapping_policy": {
                "version": "figaro-nace-a64-v1",
                "source": "synthetic A64 mapping",
                "explicit_country_mapping": {},
                "explicit_sector_mapping": {},
                "approved_fallbacks": [],
            },
            "valuation_policy": {"trade_sector_holdouts": ["G45", "G46", "G47"]},
            "outcomes": {
                "GVA": {"unit": "EUR", "coverage_scope": "all"},
                "GHG": {"unit": "tCO2e", "coverage_scope": "all"},
                "EMPLOYMENT_PERSONS": {"unit": "persons-equivalent", "coverage_scope": "qualified"},
            },
            "portfolio_policy": {
                "aggregation_boundary": "gross_observation_based_upstream_requirements",
                "double_counting_status": "not_network_deduplicated",
                "caveat": "synthetic caveat",
            },
            "sensitivity_case": "primary",
        }
        self.outputs = [
            {"country": "FR", "sector": "M72", "output_million_eur": "100"},
            {"country": "DE", "sector": "C20", "output_million_eur": "200"},
            {"country": "DE", "sector": "G46", "output_million_eur": "300"},
        ]
        self.transactions = [
            {"origin_country": "DE", "origin_sector": "C20", "destination_country": "FR", "destination_sector": "M72", "value_million_eur": "20"},
            {"origin_country": "FR", "origin_sector": "M72", "destination_country": "DE", "destination_sector": "C20", "value_million_eur": "10"},
            {"origin_country": "DE", "origin_sector": "C20", "destination_country": "DE", "destination_sector": "G46", "value_million_eur": "30"},
        ]
        self.satellites = []
        values = {
            ("FR", "M72"): {"GVA": (50, "EUR"), "GHG": (10, "tCO2e"), "EMPLOYMENT_PERSONS": (100, "persons-equivalent")},
            ("DE", "C20"): {"GVA": (80, "EUR"), "GHG": (40, "tCO2e"), "EMPLOYMENT_PERSONS": (200, "persons-equivalent")},
            ("DE", "G46"): {"GVA": (120, "EUR"), "GHG": (30, "tCO2e"), "EMPLOYMENT_PERSONS": (300, "persons-equivalent")},
        }
        for (country, sector), outcomes in values.items():
            for outcome, (value, unit) in outcomes.items():
                self.satellites.append({"country": country, "sector": sector, "outcome": outcome, "value": str(value), "unit": unit})
        self.cohort = [
            {"selection_id": "OBS-1", "country": "FR", "spend_eur": "1000000", "proposed_nace_rev2_code": "M72", "model_sector_code": "M72"},
            {"selection_id": "OBS-2", "country": "DE", "spend_eur": "2000000", "proposed_nace_rev2_code": "G46", "model_sector_code": "G46"},
            {"selection_id": "OBS-3", "country": "IT", "spend_eur": "500000", "proposed_nace_rev2_code": "", "model_sector_code": ""},
        ]
        self.direct = [
            {"selection_id": "OBS-1", "outcome_code": "B1G", "attribution_status": "available", "modelled_value": "500000", "modelled_unit": "EUR"},
            {"selection_id": "OBS-1", "outcome_code": "GHG", "attribution_status": "available", "modelled_value": "10", "modelled_unit": "tCO2e"},
            {"selection_id": "OBS-1", "outcome_code": "EMPLOYMENT_PERSONS", "attribution_status": "available", "modelled_value": "2", "modelled_unit": "persons-equivalent"},
        ]

    def test_matrix_builds_A_L_and_L_minus_I(self):
        model = figaro.build_figaro_model(self.transactions, self.outputs)
        self.assertLess(model["max_inverse_error"], 1e-9)
        ident = np.eye(len(model["nodes"]))
        self.assertTrue(np.allclose(model["u"], model["l"] - ident))
        fr = model["index"][("FR", "M72")]
        de = model["index"][("DE", "C20")]
        self.assertAlmostEqual(model["a"][de, fr], 0.2)

    def test_toy_indirect_calculation_uses_recursive_upstream_not_direct_identity(self):
        result = figaro.compose_figaro_attribution(self.cohort, self.direct, self.transactions, self.outputs, self.satellites, self.config)
        row = next(r for r in result["outcomes"] if r["selection_id"] == "OBS-1" and r["outcome"] == "GVA")
        self.assertEqual(row["status"], "available")
        self.assertGreater(float(row["indirect_value"]), 0)
        self.assertAlmostEqual(float(row["combined_value"]), 500000 + float(row["indirect_value"]))
        self.assertLess(float(row["indirect_value"]), 500000)

    def test_trade_is_mapped_but_primary_case_held_out(self):
        result = figaro.compose_figaro_attribution(self.cohort, self.direct, self.transactions, self.outputs, self.satellites, self.config)
        mapping = next(r for r in result["mapping"] if r["selection_id"] == "OBS-2")
        self.assertEqual(mapping["mapping_status"], "mapped_exact")
        self.assertEqual(mapping["valuation_case"], "trade_purchase_value_not_comparable")
        self.assertEqual(mapping["calculation_eligibility"], "held_out")
        out = next(r for r in result["outcomes"] if r["selection_id"] == "OBS-2" and r["outcome"] == "GHG")
        self.assertEqual(out["status"], "held_out")
        self.assertEqual(out["holdout_reason"], "trade_sector_primary_case_valuation_holdout")

    def test_missing_nace_survives_as_explicit_holdout(self):
        result = figaro.compose_figaro_attribution(self.cohort, self.direct, self.transactions, self.outputs, self.satellites, self.config)
        mapping = next(r for r in result["mapping"] if r["selection_id"] == "OBS-3")
        self.assertEqual(mapping["mapping_status"], "unmapped")
        self.assertEqual(mapping["mapping_reason"], "nace_code_missing")

    def test_explicit_country_equivalence_is_governed_exact_mapping(self):
        cfg = copy.deepcopy(self.config)
        cfg["mapping_policy"]["explicit_country_mapping"] = {"UK": "GB"}
        outputs = self.outputs + [{"country": "GB", "sector": "M72", "output_million_eur": "50"}]
        model = figaro.build_figaro_model([], outputs)
        cohort = [{"selection_id": "OBS-UK", "country": "UK", "spend_eur": "1000", "proposed_nace_rev2_code": "72.1", "model_sector_code": "M72"}]
        mapping = figaro.map_observations(cohort, model, cfg)[0]
        self.assertEqual(mapping["country"], "UK")
        self.assertEqual(mapping["figaro_country"], "GB")
        self.assertEqual(mapping["figaro_sector"], "M72")
        self.assertEqual(mapping["mapping_status"], "mapped_governed_exact")
        self.assertEqual(mapping["mapping_reason"], "explicit_governed_exact_equivalence")
        self.assertEqual(mapping["calculation_eligibility"], "eligible")

    def test_explicit_sector_notation_equivalence_is_governed_exact_mapping(self):
        cfg = copy.deepcopy(self.config)
        cfg["mapping_policy"]["explicit_sector_mapping"] = {"C10-C12": "C10T12"}
        outputs = self.outputs + [{"country": "FR", "sector": "C10T12", "output_million_eur": "50"}]
        model = figaro.build_figaro_model([], outputs)
        cohort = [{"selection_id": "OBS-SECTOR", "country": "FR", "spend_eur": "1000", "proposed_nace_rev2_code": "10", "model_sector_code": "C10-C12"}]
        mapping = figaro.map_observations(cohort, model, cfg)[0]
        self.assertEqual(mapping["model_sector"], "C10-C12")
        self.assertEqual(mapping["figaro_sector"], "C10T12")
        self.assertEqual(mapping["mapping_status"], "mapped_governed_exact")
        self.assertEqual(mapping["mapping_reason"], "explicit_governed_exact_equivalence")
        self.assertEqual(mapping["calculation_eligibility"], "eligible")

    def test_no_parent_fallback_is_permitted(self):
        cfg = copy.deepcopy(self.config)
        cfg["mapping_policy"]["approved_fallbacks"] = [{"from": "P85", "to": "P"}]
        model = figaro.build_figaro_model(self.transactions, self.outputs)
        with self.assertRaisesRegex(ValueError, "does not permit"):
            figaro.map_observations(self.cohort, model, cfg)

    def test_missing_satellite_upstream_node_holds_out_outcome_not_zero(self):
        satellites = [r for r in self.satellites if not (r["country"] == "DE" and r["sector"] == "C20" and r["outcome"] == "GHG")]
        result = figaro.compose_figaro_attribution(self.cohort, self.direct, self.transactions, self.outputs, satellites, self.config)
        row = next(r for r in result["outcomes"] if r["selection_id"] == "OBS-1" and r["outcome"] == "GHG")
        self.assertEqual(row["status"], "held_out")
        self.assertIn("satellite_coverage_missing_for_upstream_nodes", row["holdout_reason"])
        self.assertEqual(row["indirect_value"], "")

    def test_country_sector_contributions_reconcile_to_indirect_total(self):
        result = figaro.compose_figaro_attribution(self.cohort, self.direct, self.transactions, self.outputs, self.satellites, self.config)
        out = next(r for r in result["outcomes"] if r["selection_id"] == "OBS-1" and r["outcome"] == "GHG")
        contrib = [r for r in result["contributions"] if r["selection_id"] == "OBS-1" and r["outcome"] == "GHG"]
        self.assertAlmostEqual(sum(float(r["indirect_value"]) for r in contrib), float(out["indirect_value"]), places=10)
        self.assertTrue(any(r["origin_country"] == "DE" for r in contrib))

    def test_portfolio_is_explicitly_not_network_deduplicated(self):
        result = figaro.compose_figaro_attribution(self.cohort, self.direct, self.transactions, self.outputs, self.satellites, self.config)
        gva = next(r for r in result["portfolio"] if r["outcome"] == "GVA")
        self.assertEqual(gva["aggregation_boundary"], "gross_observation_based_upstream_requirements")
        self.assertEqual(gva["double_counting_status"], "not_network_deduplicated")
        self.assertEqual(gva["mapped_spend"], "1000000")
        self.assertAlmostEqual(float(gva["coverage_pct"]), 1000000 / 3500000 * 100)

    def test_direct_values_are_passed_through_not_recalculated(self):
        direct = copy.deepcopy(self.direct)
        direct[0]["modelled_value"] = "123456.789"
        result = figaro.compose_figaro_attribution(self.cohort, direct, self.transactions, self.outputs, self.satellites, self.config)
        row = next(r for r in result["outcomes"] if r["selection_id"] == "OBS-1" and r["outcome"] == "GVA")
        self.assertEqual(row["direct_value"], "123456.789")
        self.assertFalse(result["summary"]["direct_recalculated"])
        self.assertEqual(result["summary"]["direct_source"], "accepted_SKO_038")

    def test_social_economy_fields_do_not_affect_results(self):
        a = copy.deepcopy(self.cohort)
        b = copy.deepcopy(self.cohort)
        a[0]["social_economy_status"] = "confirmed"
        a[1]["social_economy_status"] = "excluded"
        first = figaro.compose_figaro_attribution(a, self.direct, self.transactions, self.outputs, self.satellites, self.config)
        second = figaro.compose_figaro_attribution(b, self.direct, self.transactions, self.outputs, self.satellites, self.config)
        self.assertEqual(first, second)

    def test_identical_inputs_are_deterministic(self):
        first = figaro.compose_figaro_attribution(self.cohort, self.direct, self.transactions, self.outputs, self.satellites, self.config)
        second = figaro.compose_figaro_attribution(self.cohort, self.direct, self.transactions, self.outputs, self.satellites, self.config)
        self.assertEqual(first, second)
        self.assertEqual(first["summary"]["determinism_fingerprint"], second["summary"]["determinism_fingerprint"])

    def test_duplicate_selection_id_rejected(self):
        cohort = self.cohort + [copy.deepcopy(self.cohort[0])]
        with self.assertRaisesRegex(ValueError, "duplicate selection_id"):
            figaro.compose_figaro_attribution(cohort, self.direct, self.transactions, self.outputs, self.satellites, self.config)

    def test_source_lineage_is_copied_to_outcomes(self):
        result = figaro.compose_figaro_attribution(self.cohort, self.direct, self.transactions, self.outputs, self.satellites, self.config)
        row = next(r for r in result["outcomes"] if r["selection_id"] == "OBS-1" and r["outcome"] == "GVA")
        self.assertEqual(row["source_product"], "naio_10_fcp")
        self.assertEqual(row["source_edition"], "2026")
        self.assertEqual(row["reference_year"], "2023")
        self.assertEqual(row["table_type"], "industry_by_industry_icio")

    def test_qa_contains_governed_dimensions(self):
        result = figaro.compose_figaro_attribution(self.cohort, self.direct, self.transactions, self.outputs, self.satellites, self.config)
        self.assertEqual(
            {r["breakdown_dimension"] for r in result["qa"]},
            {"country", "figaro_sector", "mapping_status", "holdout_reason", "valuation_case"},
        )


if __name__ == "__main__":
    unittest.main()
