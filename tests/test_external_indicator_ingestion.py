"""SKO-028 generic external-indicator ingestion behaviours (T01-T22)."""

from __future__ import annotations

import copy
import csv
import hashlib
import inspect
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

import external_indicator_ingestion as ingestion


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "external_indicators.yaml"
FIXTURES = ROOT / "tests" / "fixtures" / "external_indicators"
RETRIEVED_AT = "2026-08-18T12:00:00Z"


class ExternalIndicatorIngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ingestion.load_config(CONFIG)
        cls.result = ingestion.ingest_extracts(
            cls.config, input_dir=FIXTURES, retrieved_at=RETRIEVED_AT,
        )
        cls.rows = cls.result["observations"]

    def selected(self, **values):
        return [row for row in self.rows
                if all(row[key] == value for key, value in values.items())]

    def copy_fixtures(self, directory: Path) -> None:
        for path in FIXTURES.glob("*.csv"):
            shutil.copyfile(path, directory / path.name)

    def test_t01_valid_eurostat_observations_normalize(self) -> None:
        rows = self.selected(source_id="eurostat", indicator_id="regional_unemployment_rate")
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["value"] for row in rows}, {"5.4", "5.7"})

    def test_t02_two_eurostat_indicators_share_one_contract(self) -> None:
        rows = self.selected(source_id="eurostat")
        self.assertEqual({row["indicator_id"] for row in rows},
                         {"regional_unemployment_rate", "regional_gdp"})
        self.assertTrue(all(list(row) == ingestion.OUTPUT_FIELDS for row in rows))

    def test_t03_edgar_has_no_source_specific_output_columns(self) -> None:
        row = self.selected(source_id="jrc_edgar")[0]
        self.assertEqual((row["indicator_theme"], row["value"], row["unit"]),
                         ("environmental", "987.6", "kt_co2e"))
        self.assertFalse({"series", "region_code", "amount", "flag"} & set(row))

    def test_t04_lightweight_ons_itl_normalizes(self) -> None:
        row = self.selected(source_id="ons")[0]
        self.assertEqual((row["geography_scheme"], row["geography_version"],
                          row["geography_level"], row["geography_code"]),
                         ("ITL", "2021", "2", "TLI3"))

    def test_t05_nuts_and_itl_coexist_without_conversion(self) -> None:
        self.assertEqual({row["geography_scheme"] for row in self.rows}, {"NUTS", "ITL"})
        self.assertTrue(any(row["geography_code"] == "BE21" for row in self.rows))
        self.assertTrue(any(row["geography_code"] == "TLI3" for row in self.rows))

    def test_t06_geography_linkage_boundary_is_exact(self) -> None:
        row = self.selected(source_id="eurostat", indicator_id="regional_gdp")[0]
        boundary = ("geography_scheme", "geography_version",
                    "geography_level", "geography_code")
        self.assertEqual(tuple(row[field] for field in boundary),
                         ("NUTS", "2024", "2", "BE21"))

    def test_t07_invalid_geography_is_rejected_not_resolved(self) -> None:
        self.assertIn("unsupported_geography_metadata",
                      {row["reason"] for row in self.result["rejections"]})
        self.assertFalse(any(row["geography_version"] == "2016" for row in self.rows))
        ons_expected = self.config["extracts"]["ons_regional_gva"]["expected_geography"]
        false_ons_nuts = {
            "geography_scheme": "NUTS", "geography_version": "2024",
            "geography_level": "2", "geography_code": "BE21",
        }
        self.assertFalse(ingestion._geography_supported(
            false_ons_nuts, self.config["geography_governance"], ons_expected,
        ))

    def test_t08_period_unit_and_source_status_are_retained(self) -> None:
        unemployment = self.selected(indicator_id="regional_unemployment_rate")
        self.assertEqual({row["period"] for row in unemployment}, {"2023", "2024"})
        self.assertEqual({row["unit"] for row in unemployment}, {"percent_labour_force"})
        self.assertEqual({row["value_status"] for row in unemployment},
                         {"final", "provisional"})

    def test_t09_source_dataset_and_version_provenance_is_retained(self) -> None:
        row = self.selected(source_id="eurostat")[0]
        for field in ("source_name", "dataset_id", "dataset_title", "dataset_version",
                      "source_version", "source_reference"):
            self.assertTrue(row[field])
        self.assertTrue(row["source_reference"].startswith("synthetic://"))

    def test_t10_input_sha256_is_retained(self) -> None:
        expected = ingestion.file_sha256(FIXTURES / "edgar_regional_emissions.csv")
        self.assertEqual(self.selected(source_id="jrc_edgar")[0]["input_file_sha256"],
                         expected)

    def test_t11_transformation_and_ingestion_versions_are_retained(self) -> None:
        row = self.rows[0]
        self.assertEqual(row["transformation_method"], "configured_local_csv_extract")
        self.assertEqual(row["transformation_version"], "configured-csv-v1")
        self.assertEqual(row["ingestion_version"], "sko-028-external-indicators-v1")

    def test_t12_ids_fingerprints_and_reruns_are_deterministic(self) -> None:
        self.assertEqual(len({row["observation_id"] for row in self.rows}), len(self.rows))
        self.assertTrue(all(len(row["observation_fingerprint"]) == 64 for row in self.rows))
        second = ingestion.ingest_extracts(
            ingestion.load_config(CONFIG), input_dir=FIXTURES, retrieved_at=RETRIEVED_AT,
        )
        self.assertEqual(self.result, second)

    def test_t13_shuffled_input_has_same_stable_logical_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.copy_fixtures(directory)
            path = directory / "eurostat_regional_context.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields, rows = list(reader.fieldnames or []), list(reader)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(reversed(rows))
            shuffled = ingestion.ingest_extracts(
                self.config, input_dir=directory, retrieved_at=RETRIEVED_AT,
            )["observations"]
        dependent = {"input_file_sha256", "observation_fingerprint"}
        logical = lambda rows: [
            {key: value for key, value in row.items() if key not in dependent}
            for row in rows
        ]
        self.assertEqual(logical(self.rows), logical(shuffled))

    def test_t14_period_unit_and_version_observations_stay_distinct(self) -> None:
        unemployment = self.selected(indicator_id="regional_unemployment_rate")
        self.assertEqual(len({row["observation_id"] for row in unemployment}), 2)
        changed = copy.deepcopy(self.config)
        changed["extracts"]["eurostat_regional_context"]["dataset_version"] = "2025-fixture-v1"
        versioned = ingestion.ingest_extracts(
            changed, input_dir=FIXTURES, retrieved_at=RETRIEVED_AT,
        )["observations"]
        old = {row["observation_id"] for row in self.selected(source_id="eurostat")}
        new = {row["observation_id"] for row in versioned if row["source_id"] == "eurostat"}
        self.assertFalse(old & new)
        key = ingestion._observation_key(self.selected(indicator_id="regional_gdp")[0])
        self.assertNotEqual(ingestion._canonical_json(key),
                            ingestion._canonical_json({**key, "unit": "index"}))

    def test_t15_exact_duplicates_deduplicate(self) -> None:
        self.assertEqual(len(self.selected(
            indicator_id="regional_unemployment_rate", period="2024")), 1)

    def test_t16_conflicting_duplicates_raise_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.copy_fixtures(directory)
            path = directory / "eurostat_regional_context.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields, rows = list(reader.fieldnames or []), list(reader)
            conflict = dict(rows[0])
            conflict["obs_value"] = "6.1"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows([*rows, conflict])
            with self.assertRaises(ingestion.ConflictingObservationError):
                ingestion.ingest_extracts(
                    self.config, input_dir=directory, retrieved_at=RETRIEVED_AT,
                )

    def test_t17_only_configured_indicators_ingest(self) -> None:
        self.assertFalse(self.selected(indicator_id="not_configured"))
        self.assertIn("unconfigured_indicator",
                      {row["reason"] for row in self.result["rejections"]})

    def test_t18_contract_has_no_supplier_or_entity_fields(self) -> None:
        forbidden = {"entity_id", "supplier_id", "supplier_name",
                     "classification_id", "client_spend"}
        self.assertFalse(forbidden & set(ingestion.OUTPUT_FIELDS))

    def test_t19_no_network_api_linkage_or_causal_path(self) -> None:
        source = inspect.getsource(ingestion).casefold()
        forbidden = ["import requests", "import urllib", "httpx", "socket",
                     "canonical_entity", "supplier_id", "client spend caused"]
        self.assertFalse([token for token in forbidden if token in source])

    def test_t20_retrieved_at_is_explicit(self) -> None:
        self.assertTrue(all(row["retrieved_at"] == RETRIEVED_AT for row in self.rows))
        with self.assertRaisesRegex(ValueError, "retrieved_at"):
            ingestion.ingest_extracts(self.config, input_dir=FIXTURES, retrieved_at="")

    def test_t21_schema_and_config_validate(self) -> None:
        schema = json.loads(
            (ROOT / "schemas" / "external_indicator_observation.schema.json").read_text()
        )
        self.assertEqual(set(schema["required"]), set(ingestion.OUTPUT_FIELDS))
        self.assertEqual(yaml.safe_load(CONFIG.read_text())["ingestion_version"],
                         "sko-028-external-indicators-v1")

    def test_t22_serialized_output_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_path, second_path = Path(first) / "out.csv", Path(second) / "out.csv"
            ingestion.write_csv(self.rows, first_path, ingestion.OUTPUT_FIELDS)
            ingestion.write_csv(self.rows, second_path, ingestion.OUTPUT_FIELDS)
            self.assertEqual(hashlib.sha256(first_path.read_bytes()).hexdigest(),
                             hashlib.sha256(second_path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
