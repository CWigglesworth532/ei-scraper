from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import whole_spend_attribution as whole


class WholeSpendAttributionTests(unittest.TestCase):
    def setUp(self):
        self.cohort = [
            {
                "selection_id": "OBS-001", "supplier": "Ordinary Supplier", "client": "Synthetic Client",
                "country": "FR", "spend_eur": "1000000", "spend_year": "2026",
                "proposed_nace_rev2_code": "M72", "nace_level": "division",
                "nace_description": "Research", "treatment": "accepted",
            },
            {
                "selection_id": "OBS-002", "supplier": "Trade Supplier", "client": "Synthetic Client",
                "country": "DE", "spend_eur": "2000000", "spend_year": "2026",
                "proposed_nace_rev2_code": "G46", "nace_level": "division",
                "nace_description": "Wholesale", "treatment": "accepted",
            },
            {
                "selection_id": "OBS-003", "supplier": "Unresolved Supplier", "client": "Synthetic Client",
                "country": "IT", "spend_eur": "500000", "spend_year": "2026",
                "proposed_nace_rev2_code": "", "nace_level": "", "nace_description": "",
                "treatment": "research_holdout",
            },
        ]
        self.economic = self._economic_result()
        self.environmental = self._environmental_result()

    @staticmethod
    def _obs(selection_id, sector, status, reason, available="1", held="0", na="0"):
        return {
            "selection_id": selection_id,
            "model_sector_code": sector,
            "model_sector_label": sector,
            "model_mapping_status": "mapped" if sector else "unresolved",
            "model_mapping_reason": reason,
            "observation_status": status,
            "available_outcomes": available,
            "held_out_outcomes": held,
            "not_applicable_outcomes": na,
        }

    @staticmethod
    def _make_outcome(
        selection_id, country, sector, code, *, status="available", reason="coefficient_applied",
        route="default", value="1", unit="EUR", coeff="0.5", coeff_unit="currency_per_currency_denominator",
        coeff_id=None, source_sector=None, specificity=None, source="synthetic_dataset",
    ):
        return {
            "selection_id": selection_id,
            "supplier": "Synthetic",
            "client": "Synthetic Client",
            "country": country,
            "spend_eur": "1000000" if selection_id == "OBS-001" else ("2000000" if selection_id == "OBS-002" else "500000"),
            "spend_year": "2026",
            "spend_currency": "EUR",
            "proposed_nace_rev2_code": sector,
            "nace_level": "division" if sector else "",
            "treatment": "accepted" if sector else "research_holdout",
            "model_sector_code": sector,
            "model_sector_label": sector,
            "coefficient_reference_year": "2023",
            "coefficient_source_sector_code": source_sector if source_sector is not None else sector,
            "coefficient_source_sector_label": source_sector if source_sector is not None else sector,
            "coefficient_specificity_status": specificity or ("exact" if sector else ""),
            "coefficient_specificity_reason": "synthetic_specificity",
            "outcome_code": code,
            "outcome_label": code,
            "denominator_route": route,
            "denominator_currency": "EUR",
            "attribution_status": status,
            "attribution_reason": reason,
            "coefficient_value": coeff if status == "available" else "",
            "coefficient_unit": coeff_unit if status == "available" else "",
            "coefficient_id": coeff_id or (f"coeff-{selection_id}-{code}" if status == "available" else ""),
            "modelled_value": value if status == "available" else "",
            "modelled_unit": unit if status == "available" else "",
            "fx_status": "not_required_dimensionless_ratio" if status == "available" else "",
            "fx_rate": "1" if status == "available" else "",
            "fx_reference_year": "2023" if status == "available" else "",
            "fx_source_organisation": "",
            "fx_source_series_key": "",
            "source_family": "synthetic" if status == "available" else "",
            "source_dataset_ids": source if status == "available" else "",
            "source_release_versions": "v1" if status == "available" else "",
            "source_release_dates": "2026-01-01" if status == "available" else "",
        }

    def _economic_result(self):
        observations = [
            self._obs("OBS-001", "M72", "modelled", "exact_model_sector_code"),
            self._obs("OBS-002", "G46", "modelled", "exact_model_sector_code"),
            self._obs("OBS-003", "", "held_out_nace_unresolved", "nace_code_missing", available="0", held="1"),
        ]
        outcomes = [
            self._make_outcome("OBS-001", "FR", "M72", "B1G", value="500000", unit="EUR"),
            self._make_outcome("OBS-002", "DE", "G46", "B1G", route="trade_turnover", value="400000", unit="EUR"),
            self._make_outcome("OBS-003", "IT", "", "B1G", status="held_out", reason="nace_code_missing", value=""),
        ]
        return {
            "observations": observations,
            "outcomes": outcomes,
            "summary": {"coefficient_reference_year": "2023", "observations": 3},
        }

    def _environmental_result(self):
        observations = [
            self._obs("OBS-001", "M72", "modelled", "exact_model_sector_code"),
            self._obs("OBS-002", "G46", "modelled", "exact_model_sector_code"),
            self._obs("OBS-003", "", "held_out_unresolved_activity", "nace_code_missing", available="0", held="1"),
        ]
        outcomes = [
            self._make_outcome(
                "OBS-001", "FR", "M72", "GHG", value="10", unit="tCO2e", coeff="10",
                coeff_unit="tco2e_per_million_currency_denominator",
            ),
            self._make_outcome(
                "OBS-002", "DE", "G46", "GHG", route="trade_turnover", value="20", unit="tCO2e", coeff="10",
                coeff_unit="tco2e_per_million_currency_denominator",
            ),
            self._make_outcome("OBS-003", "IT", "", "GHG", status="held_out", reason="model_mapping_nace_code_missing", value=""),
        ]
        return {
            "observations": observations,
            "outcomes": outcomes,
            "summary": {"coefficient_reference_year": "2023", "observations": 3},
        }

    def _compose(self, economic=None, environmental=None, cohort=None):
        with patch.object(whole.econ_apply, "apply_coefficients", return_value=economic or self.economic), patch.object(
            whole.env_apply, "apply_coefficients", return_value=environmental or self.environmental
        ):
            return whole.compose_attribution(
                cohort or self.cohort,
                [],
                [],
                economic_config={},
                environmental_config={},
            )

    def test_composes_economic_and_environmental_outcomes_in_one_ledger(self):
        result = self._compose()
        self.assertEqual(len(result["outcomes"]), 6)
        self.assertEqual({row["outcome_domain"] for row in result["outcomes"]}, {"economic", "environmental"})
        self.assertEqual(result["summary"]["observations_with_any_modelled_outcome"], 2)
        self.assertEqual(result["summary"]["held_out_observations"], 1)
        self.assertEqual(result["summary"]["modelled_spend"], "3000000")

    def test_unresolved_nace_survives_as_explicit_upstream_holdout(self):
        result = self._compose()
        row = next(row for row in result["observations"] if row["selection_id"] == "OBS-003")
        self.assertEqual(row["observation_status"], "held_out")
        self.assertEqual(row["mapping_status"], "unresolved")
        self.assertEqual(row["holdout_reason"], "nace_code_missing")

    def test_trade_available_rows_must_use_turnover(self):
        environmental = copy.deepcopy(self.environmental)
        environmental["outcomes"][1]["denominator_route"] = "default"
        with self.assertRaisesRegex(ValueError, "trade outcomes using non-turnover route"):
            self._compose(environmental=environmental)

    def test_approved_environmental_parent_fallback_is_exposed(self):
        environmental = copy.deepcopy(self.environmental)
        row = environmental["outcomes"][0]
        row["model_sector_code"] = "P85"
        row["coefficient_source_sector_code"] = "P"
        row["coefficient_source_sector_label"] = "Education"
        row["coefficient_specificity_status"] = "approved_parent_fallback"
        row["coefficient_specificity_reason"] = "approved synthetic fallback"
        environmental["observations"][0]["model_sector_code"] = "P85"
        result = self._compose(environmental=environmental)
        ghg = next(r for r in result["outcomes"] if r["selection_id"] == "OBS-001" and r["outcome_domain"] == "environmental")
        self.assertEqual(ghg["model_sector_code"], "P85")
        self.assertEqual(ghg["coefficient_source_sector_code"], "P")
        self.assertEqual(ghg["coefficient_specificity_status"], "approved_parent_fallback")

    def test_unapproved_parent_fallback_is_rejected(self):
        environmental = copy.deepcopy(self.environmental)
        row = environmental["outcomes"][0]
        row["coefficient_source_sector_code"] = "M"
        row["coefficient_specificity_status"] = "exact"
        with self.assertRaisesRegex(ValueError, "unapproved coefficient-sector fallback"):
            self._compose(environmental=environmental)

    def test_available_outcome_requires_exact_coefficient_lineage(self):
        economic = copy.deepcopy(self.economic)
        economic["outcomes"][0]["coefficient_id"] = ""
        with self.assertRaisesRegex(ValueError, "missing coefficient/source lineage"):
            self._compose(economic=economic)

    def test_economic_rows_are_normalised_to_exact_specificity(self):
        result = self._compose()
        row = next(r for r in result["outcomes"] if r["selection_id"] == "OBS-001" and r["outcome_domain"] == "economic")
        self.assertEqual(row["coefficient_source_sector_code"], "M72")
        self.assertEqual(row["coefficient_specificity_status"], "exact")
        self.assertEqual(row["coefficient_specificity_reason"], "exact_model_sector_coefficient")

    def test_reference_year_mismatch_is_rejected(self):
        environmental = copy.deepcopy(self.environmental)
        environmental["summary"]["coefficient_reference_year"] = "2022"
        with self.assertRaisesRegex(ValueError, "accepted coefficient release mismatch"):
            self._compose(environmental=environmental)

    def test_duplicate_procurement_observation_id_is_rejected_before_application(self):
        cohort = copy.deepcopy(self.cohort)
        cohort.append(copy.deepcopy(cohort[0]))
        with self.assertRaisesRegex(ValueError, "duplicate selection_id"):
            whole.compose_attribution(cohort, [], [], economic_config={}, environmental_config={})

    def test_portfolio_totals_reconcile_to_ledger(self):
        result = self._compose()
        gva = next(r for r in result["portfolio"] if r["outcome_domain"] == "economic" and r["outcome_code"] == "B1G")
        ghg = next(r for r in result["portfolio"] if r["outcome_domain"] == "environmental" and r["outcome_code"] == "GHG")
        self.assertEqual(gva["modelled_outcome_total"], "900000")
        self.assertEqual(ghg["modelled_outcome_total"], "30")
        self.assertTrue(result["summary"]["portfolio_total_reconciliation_passed"])

    def test_portfolio_coverage_uses_full_input_spend_denominator(self):
        result = self._compose()
        ghg = next(r for r in result["portfolio"] if r["outcome_domain"] == "environmental" and r["outcome_code"] == "GHG")
        self.assertEqual(ghg["total_spend"], "3500000")
        self.assertEqual(ghg["modelled_spend"], "3000000")
        self.assertAlmostEqual(float(ghg["spend_coverage_pct"]), 85.71428571428571)

    def test_missing_fx_remains_explicit_holdout(self):
        environmental = copy.deepcopy(self.environmental)
        row = environmental["outcomes"][0]
        row.update({
            "attribution_status": "held_out", "attribution_reason": "missing_fx_rate",
            "modelled_value": "", "modelled_unit": "", "coefficient_id": "",
            "source_dataset_ids": "", "fx_status": "missing_fx_rate", "fx_rate": "",
        })
        result = self._compose(environmental=environmental)
        out = next(r for r in result["outcomes"] if r["selection_id"] == "OBS-001" and r["outcome_domain"] == "environmental")
        self.assertEqual(out["attribution_status"], "held_out")
        self.assertEqual(out["attribution_reason"], "missing_fx_rate")
        self.assertEqual(out["fx_status"], "missing_fx_rate")

    def test_not_applicable_is_distinct_from_held_out(self):
        economic = copy.deepcopy(self.economic)
        row = economic["outcomes"][0]
        row.update({
            "attribution_status": "not_applicable", "attribution_reason": "route_not_applicable",
            "modelled_value": "", "modelled_unit": "", "coefficient_id": "", "source_dataset_ids": "",
        })
        result = self._compose(economic=economic)
        gva = next(r for r in result["portfolio"] if r["outcome_domain"] == "economic" and r["outcome_code"] == "B1G")
        self.assertEqual(gva["not_applicable_observations"], "1")
        self.assertEqual(gva["held_out_observations"], "1")

    def test_social_economy_field_does_not_change_attribution(self):
        with_status = copy.deepcopy(self.cohort)
        without_status = copy.deepcopy(self.cohort)
        with_status[0]["social_economy_status"] = "confirmed"
        with_status[1]["social_economy_status"] = "excluded"
        first = self._compose(cohort=with_status)
        second = self._compose(cohort=without_status)
        self.assertEqual(first["outcomes"], second["outcomes"])
        self.assertEqual(first["portfolio"], second["portfolio"])

    def test_outputs_are_deterministic_for_identical_inputs(self):
        first = self._compose()
        second = self._compose()
        self.assertEqual(first, second)

    def test_qa_breakdowns_include_governed_dimensions(self):
        result = self._compose()
        self.assertEqual(
            {row["breakdown_dimension"] for row in result["qa_breakdowns"]},
            {"country", "model_sector", "denominator_route", "coefficient_specificity", "holdout_reason"},
        )


if __name__ == "__main__":
    unittest.main()
