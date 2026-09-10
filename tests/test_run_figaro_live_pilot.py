from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_figaro_live_pilot as runner


class RunFigaroLivePilotTests(unittest.TestCase):
    def test_adapt_sko038_cohort_copies_spend_without_mutating_source(self):
        source = [{
            "selection_id": "S1",
            "country": "BE",
            "spend": "123.45",
            "spend_currency": "EUR",
            "proposed_nace_rev2_code": "72.1",
        }]
        adapted = runner.adapt_sko038_cohort(source)
        self.assertEqual(adapted[0]["spend_eur"], "123.45")
        self.assertNotIn("spend_eur", source[0])
        self.assertEqual(adapted[0]["country"], "BE")

    def test_adapt_sko038_cohort_rejects_non_eur_spend(self):
        with self.assertRaisesRegex(ValueError, "requires EUR procurement spend"):
            runner.adapt_sko038_cohort([{
                "selection_id": "S1",
                "country": "UK",
                "spend": "100",
                "spend_currency": "GBP",
                "proposed_nace_rev2_code": "72.1",
            }])

    def test_source_gate_requires_validated_permitted_package(self):
        fp = "a" * 64
        self.assertEqual(
            runner.validate_source_gate({
                "status": "source_package_validated",
                "pilot_execution_permitted": True,
                "package_fingerprint": fp,
            }),
            fp,
        )
        with self.assertRaises(ValueError):
            runner.validate_source_gate({
                "status": "source_package_validated",
                "pilot_execution_permitted": False,
                "package_fingerprint": fp,
            })

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_live_runner_preserves_direct_input_and_writes_deterministic_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cohort = root / "cohort.csv"
            direct = root / "direct.csv"
            model = root / "model.npz"
            gva = root / "gva.csv"
            ghg = root / "ghg.csv"
            config = root / "config.yaml"
            gate = root / "gate.json"
            out = root / "out"

            self._write_csv(cohort, [{
                "selection_id": "S1",
                "supplier": "Example",
                "client": "Client",
                "country": "BE",
                "spend": "1000000",
                "spend_currency": "EUR",
                "spend_year": "",
                "proposed_nace_rev2_code": "72.1",
                "nace_level": "group",
                "nace_description": "Research",
                "treatment": "broader",
                "model_sector_code": "M72",
                "model_sector_label": "Scientific research and development",
                "mapping_status": "mapped",
                "mapping_reason": "division_to_a64_range",
                "coefficient_reference_year": "2023",
                "economic_status": "modelled",
                "economic_available_outcomes": "1",
                "economic_held_out_outcomes": "0",
                "economic_not_applicable_outcomes": "0",
                "environmental_status": "modelled",
                "environmental_available_outcomes": "1",
                "environmental_held_out_outcomes": "0",
                "any_outcome_modelled": "true",
                "observation_status": "modelled",
                "holdout_reason": "",
            }])
            self._write_csv(direct, [{
                "selection_id": "S1",
                "outcome_code": "GVA",
                "attribution_status": "available",
                "modelled_value": "400000",
                "modelled_unit": "EUR",
            }])
            self._write_csv(gva, [{"country": "BE", "sector": "M72", "outcome": "GVA", "value": "500000", "unit": "EUR"}])
            self._write_csv(ghg, [{"country": "BE", "sector": "M72", "outcome": "GHG", "value": "20", "unit": "tCO2e"}])
            model.write_bytes(b"placeholder")
            config.write_text("outcomes:\n  GVA: {status: primary}\n  GHG: {status: primary}\n", encoding="utf-8")
            fingerprint = "b" * 64
            gate.write_text(json.dumps({
                "status": "source_package_validated",
                "pilot_execution_permitted": True,
                "package_fingerprint": fingerprint,
            }), encoding="utf-8")

            fake_result = {
                "mapping": [{"selection_id": "S1", "calculation_eligibility": "eligible"}],
                "outcomes": [{
                    "selection_id": "S1", "outcome": "GVA", "status": "available",
                    "direct_value": "400000", "indirect_value": "100000",
                    "combined_value": "500000", "unit": "EUR",
                }],
                "contributions": [{"selection_id": "S1", "outcome": "GVA", "value": "100000"}],
                "portfolio": [{"outcome": "GVA", "direct_total": "400000", "indirect_total": "100000", "combined_total": "500000"}],
                "qa": [{"breakdown_dimension": "country", "breakdown_value": "BE", "outcome": "GVA"}],
                "summary": {
                    "mapped_eligible_observations": 1,
                    "held_out_observations": 0,
                    "trade_holdouts": 0,
                    "unmapped_observations": 0,
                    "determinism_fingerprint": "c" * 64,
                },
            }

            with mock.patch.object(runner, "build_figaro_model_from_npz", return_value={"mock": True}), \
                 mock.patch.object(runner, "compose_figaro_attribution_with_model", return_value=fake_result) as compose:
                result1 = runner.run_live_pilot(
                    cohort_path=cohort,
                    direct_outcomes_path=direct,
                    model_path=model,
                    gva_satellite_path=gva,
                    ghg_satellite_path=ghg,
                    config_path=config,
                    source_validation_summary_path=gate,
                    output_dir=out,
                )
                first_bytes = {p.name: p.read_bytes() for p in sorted(out.iterdir())}
                result2 = runner.run_live_pilot(
                    cohort_path=cohort,
                    direct_outcomes_path=direct,
                    model_path=model,
                    gva_satellite_path=gva,
                    ghg_satellite_path=ghg,
                    config_path=config,
                    source_validation_summary_path=gate,
                    output_dir=out,
                )
                second_bytes = {p.name: p.read_bytes() for p in sorted(out.iterdir())}

            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(result1["summary"]["source_package_fingerprint"], fingerprint)
            self.assertFalse(result1["summary"]["direct_input_recalculated"])
            self.assertEqual(result1["summary"]["cohort_rows"], 1)
            self.assertEqual(result2["outcomes"][0]["direct_value"], "400000")

            args, _ = compose.call_args
            adapted_cohort = args[0]
            passed_direct = args[1]
            self.assertEqual(adapted_cohort[0]["spend_eur"], "1000000")
            self.assertEqual(adapted_cohort[0]["country"], "BE")
            self.assertEqual(passed_direct[0]["modelled_value"], "400000")

            expected_files = {
                "figaro_live_mapping.csv",
                "figaro_live_outcomes.csv",
                "figaro_live_contributions.csv",
                "figaro_live_portfolio_summary.csv",
                "figaro_live_qa.csv",
                "figaro_live_pilot_summary.json",
            }
            self.assertEqual(set(first_bytes), expected_files)


if __name__ == "__main__":
    unittest.main()
