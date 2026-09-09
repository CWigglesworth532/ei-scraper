from __future__ import annotations

import copy
import unittest
from decimal import Decimal
from pathlib import Path

import apply_direct_environmental_coefficients as apply_env
import direct_environmental_coefficients as env

ROOT = Path(__file__).resolve().parents[1]
CONFIG = env.load_config(ROOT / "config" / "direct_environmental_coefficients.yaml")


def source_row(
    *, country="BE", sector="M72", family="environmental_accounts", concept="GHG",
    value="100", unit="tonne_co2e", currency="", status="published",
    dataset="synthetic", label="Synthetic sector",
):
    return {
        "source_family": family,
        "source_organisation": "Synthetic test source",
        "source_dataset_id": dataset,
        "source_release_version": "test-v1",
        "source_release_date": "2026-09-09",
        "retrieved_at": "2026-09-09T00:00:00Z",
        "source_url": "https://example.invalid/test",
        "licence": "test-only",
        "country": country,
        "source_classification": "NACE",
        "source_classification_version": "Rev. 2 A*64",
        "source_sector_code": sector,
        "source_sector_label": label,
        "model_classification": "NACE",
        "model_classification_version": "Rev. 2 A*64",
        "model_sector_code": sector,
        "model_sector_label": label,
        "reference_year": "2023",
        "concept_code": concept,
        "concept_label": concept,
        "value": value,
        "normalized_unit": unit,
        "currency": currency,
        "price_basis": "current_prices" if currency else "",
        "status_flag": status,
    }


def denominator(
    *, country="BE", sector="M72", value="50", currency="EUR", concept="P1",
    family="national_accounts", label="Synthetic sector", status="published",
):
    return source_row(
        country=country, sector=sector, family=family, concept=concept, value=value,
        unit="million_currency", currency=currency, dataset="synthetic-denominator",
        label=label, status=status,
    )


def cohort_row(*, country="BE", nace="72.1", spend="1000000", selection_id="T-001"):
    return {
        "selection_id": selection_id,
        "supplier": "Synthetic supplier",
        "client": "Synthetic client",
        "country": country,
        "spend_eur": spend,
        "spend_year": "2025",
        "proposed_nace_rev2_code": nace,
        "nace_level": "class",
        "nace_description": "Synthetic purchased activity",
        "treatment": "accepted",
        # Deliberately irrelevant: application must not depend on this field.
        "social_economy_status": "not_social_economy",
    }


class DirectEnvironmentalCoefficientTests(unittest.TestCase):
    def test_ordinary_sector_uses_p1_output(self):
        result = env.build_coefficients(
            [source_row(value="100"), denominator(value="50")],
            config=CONFIG, generated_at="2026-09-09T00:00:00Z",
        )
        rec = result["coefficients"][0]
        self.assertEqual(rec["qa_status"], "calculated")
        self.assertEqual(rec["denominator_route"], "default")
        self.assertEqual(rec["denominator_concept_code"], "P1")
        self.assertEqual(Decimal(rec["coefficient_value"]), Decimal("2"))
        self.assertEqual(rec["coefficient_unit"], "tco2e_per_million_currency_denominator")

    def test_trade_sector_requires_turnover_and_does_not_fall_back_to_p1(self):
        rows = [
            source_row(country="DE", sector="G46", value="200"),
            denominator(country="DE", sector="G46", value="100", concept="P1", family="national_accounts"),
            denominator(country="DE", sector="G46", value="40", concept="TURNOVER", family="business_statistics"),
        ]
        result = env.build_coefficients(rows, config=CONFIG, generated_at="2026-09-09T00:00:00Z")
        rec = result["coefficients"][0]
        self.assertEqual(rec["denominator_route"], "trade_turnover")
        self.assertEqual(rec["denominator_concept_code"], "TURNOVER")
        self.assertEqual(Decimal(rec["coefficient_value"]), Decimal("5"))

        without_turnover = env.build_coefficients(
            rows[:2], config=CONFIG, generated_at="2026-09-09T00:00:00Z"
        )["coefficients"][0]
        self.assertEqual(without_turnover["qa_status"], "held_out")
        self.assertEqual(without_turnover["qa_reason"], "missing_required_denominator")

    def test_suppressed_or_nonusable_source_is_held_out(self):
        result = env.build_coefficients(
            [source_row(status="confidential"), denominator()],
            config=CONFIG, generated_at="2026-09-09T00:00:00Z",
        )
        rec = result["coefficients"][0]
        self.assertEqual(rec["qa_status"], "held_out")
        self.assertEqual(rec["qa_reason"], "source_status_not_usable")

    def test_provisional_and_estimated_numeric_flags_are_usable_and_preserved(self):
        for flag in ("p", "e"):
            with self.subTest(flag=flag):
                result = env.build_coefficients(
                    [source_row(value="100"), denominator(value="50", status=flag)],
                    config=CONFIG, generated_at="2026-09-09T00:00:00Z",
                )
                rec = result["coefficients"][0]
                self.assertEqual(rec["qa_status"], "calculated")
                self.assertEqual(Decimal(rec["coefficient_value"]), Decimal("2"))
                self.assertEqual(rec["source_status_flags"], f"{flag}|published")

    def test_wrong_environmental_unit_is_held_out(self):
        result = env.build_coefficients(
            [source_row(unit="thousand_tonnes"), denominator()],
            config=CONFIG, generated_at="2026-09-09T00:00:00Z",
        )
        rec = result["coefficients"][0]
        self.assertEqual(rec["qa_status"], "held_out")
        self.assertEqual(rec["qa_reason"], "numerator_unit_mismatch")

    def test_application_converts_eur_spend_to_gbp_for_uk_absolute_outcome(self):
        result = env.build_coefficients(
            [
                source_row(country="UK", sector="M72", value="215000"),
                denominator(country="UK", sector="M72", value="45979", currency="GBP"),
            ],
            config=CONFIG, generated_at="2026-09-09T00:00:00Z",
        )
        applied = apply_env.apply_coefficients(
            [cohort_row(country="UK", nace="72.1", spend="3235649.16")],
            result["coverage_matrix"], config=CONFIG,
        )
        outcome = applied["outcomes"][0]
        expected = Decimal("3235649.16") * Decimal("0.86979") / Decimal("1000000") * (Decimal("215000") / Decimal("45979"))
        self.assertEqual(outcome["attribution_status"], "available")
        self.assertEqual(outcome["fx_status"], "converted_annual_average")
        self.assertAlmostEqual(float(outcome["modelled_value"]), float(expected), places=10)
        self.assertEqual(outcome["modelled_unit"], "tCO2e")

    def test_fr_p85_uses_explicit_approved_parent_p_coefficient(self):
        result = env.build_coefficients(
            [
                source_row(country="FR", sector="P", value="300", label="Education"),
                denominator(country="FR", sector="P", value="100", label="Education"),
            ],
            config=CONFIG, generated_at="2026-09-09T00:00:00Z",
        )
        applied = apply_env.apply_coefficients(
            [cohort_row(country="FR", nace="85.42", spend="200000", selection_id="FR-P85")],
            result["coverage_matrix"], config=CONFIG,
        )
        obs = applied["observations"][0]
        outcome = applied["outcomes"][0]
        self.assertEqual(obs["model_sector_code"], "P85")
        self.assertEqual(obs["coefficient_source_sector_code"], "P")
        self.assertEqual(obs["coefficient_specificity_status"], "approved_parent_fallback")
        self.assertEqual(outcome["attribution_status"], "available")
        self.assertEqual(Decimal(outcome["modelled_value"]), Decimal("0.6"))

    def test_unapproved_missing_sector_does_not_receive_parent_fallback(self):
        result = env.build_coefficients(
            [source_row(country="FR", sector="P", value="300"), denominator(country="FR", sector="P", value="100")],
            config=CONFIG, generated_at="2026-09-09T00:00:00Z",
        )
        applied = apply_env.apply_coefficients(
            [cohort_row(country="FR", nace="86.10", spend="200000", selection_id="FR-Q86")],
            result["coverage_matrix"], config=CONFIG,
        )
        self.assertEqual(applied["observations"][0]["observation_status"], "held_out_unresolved_activity")
        self.assertEqual(applied["outcomes"][0]["attribution_status"], "held_out")

    def test_no_automatic_year_fallback(self):
        matrix = [{
            "country": "BE", "coefficient_source_sector_code": "M72",
            "coefficient_source_sector_label": "Research", "reference_year": "2022",
            "outcome_code": "GHG", "outcome_label": "Direct greenhouse gas emissions",
            "denominator_route": "default", "denominator_currency": "EUR",
            "availability_status": "available", "availability_reason": "",
            "coefficient_value": "2", "coefficient_unit": "tco2e_per_million_currency_denominator",
            "coefficient_id": "test", "source_family": "environmental_accounts|national_accounts",
            "source_dataset_ids": "synthetic", "source_release_versions": "v1",
            "source_release_dates": "2026-09-09",
        }]
        with self.assertRaisesRegex(ValueError, "no rows for reference year 2023"):
            apply_env.apply_coefficients([cohort_row()], matrix, config=CONFIG)

    def test_social_economy_field_does_not_change_result(self):
        result = env.build_coefficients(
            [source_row(value="100"), denominator(value="50")],
            config=CONFIG, generated_at="2026-09-09T00:00:00Z",
        )
        a = cohort_row(spend="100000")
        b = copy.deepcopy(a)
        b["social_economy_status"] = "confirmed_social_economy"
        out_a = apply_env.apply_coefficients([a], result["coverage_matrix"], config=CONFIG)["outcomes"][0]
        out_b = apply_env.apply_coefficients([b], result["coverage_matrix"], config=CONFIG)["outcomes"][0]
        self.assertEqual(out_a["modelled_value"], out_b["modelled_value"])


if __name__ == "__main__":
    unittest.main()
