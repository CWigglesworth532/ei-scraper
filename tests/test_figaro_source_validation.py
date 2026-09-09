from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

import figaro_source_validation as source


class FigaroSourceValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config_path = self.root / "config.yaml"
        config = {
            "figaro_source": {
                "organisation": "Eurostat",
                "product_id": "naio_10_fcp",
                "edition": "2025",
                "reference_year": 2023,
                "table_type": "industry_by_industry_icio",
                "classification": "NACE Rev. 2 A*64",
                "valuation": "basic_prices",
                "currency": "EUR",
                "unit": "million_eur",
            }
        }
        self.config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        self.transactions = self.root / "figaro_transactions.csv"
        self.outputs = self.root / "figaro_outputs.csv"
        self.satellites = self.root / "figaro_satellites.csv"
        self._write_csv(
            self.transactions,
            ["origin_country", "origin_sector", "destination_country", "destination_sector", "value_million_eur"],
            [
                ["DE", "C20", "FR", "M72", "20"],
                ["FR", "M72", "DE", "C20", "10"],
            ],
        )
        self._write_csv(
            self.outputs,
            ["country", "sector", "output_million_eur"],
            [["DE", "C20", "200"], ["FR", "M72", "100"]],
        )
        self._write_csv(
            self.satellites,
            ["country", "sector", "outcome", "value", "unit"],
            [
                ["DE", "C20", "GVA", "80", "EUR"],
                ["FR", "M72", "GVA", "50", "EUR"],
                ["DE", "C20", "GHG", "40", "tCO2e"],
                ["FR", "M72", "GHG", "10", "tCO2e"],
                ["DE", "C20", "EMPLOYMENT_PERSONS", "200", "persons-equivalent"],
                ["FR", "M72", "EMPLOYMENT_PERSONS", "100", "persons-equivalent"],
            ],
        )
        self._write_manifest()

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _write_csv(path: Path, fields: list[str], rows: list[list[str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(fields)
            writer.writerows(rows)

    def _sha(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_manifest(self, **overrides):
        manifest = {
            "organisation": "Eurostat",
            "product_id": "naio_10_fcp",
            "edition": "2025",
            "reference_year": 2023,
            "table_type": "industry_by_industry_icio",
            "classification": "NACE Rev. 2 A*64",
            "valuation": "basic_prices",
            "currency": "EUR",
            "unit": "million_eur",
            "files": [
                {"filename": self.transactions.name, "role": "transactions", "sha256": self._sha(self.transactions)},
                {"filename": self.outputs.name, "role": "outputs", "sha256": self._sha(self.outputs)},
                {"filename": self.satellites.name, "role": "satellites", "sha256": self._sha(self.satellites)},
            ],
        }
        manifest.update(overrides)
        (self.root / "figaro_source_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def test_valid_package_passes_and_is_fingerprinted(self):
        result = source.validate_package(self.root, self.config_path)
        self.assertEqual(result["status"], "source_package_validated")
        self.assertTrue(result["manifest"]["checksums_verified"])
        self.assertTrue(result["contract"]["normalized_contract_valid"])
        self.assertEqual(result["contract"]["output_nodes"], 2)
        self.assertTrue(result["pilot_execution_permitted"])
        self.assertEqual(len(result["package_fingerprint"]), 64)

    def test_manifest_year_mismatch_is_rejected(self):
        self._write_manifest(reference_year=2022)
        with self.assertRaisesRegex(ValueError, "manifest/config mismatch"):
            source.validate_package(self.root, self.config_path)

    def test_manifest_edition_mismatch_is_rejected(self):
        self._write_manifest(edition="2026")
        with self.assertRaisesRegex(ValueError, "manifest/config mismatch"):
            source.validate_package(self.root, self.config_path)

    def test_checksum_mismatch_is_rejected(self):
        manifest = json.loads((self.root / "figaro_source_manifest.json").read_text(encoding="utf-8"))
        manifest["files"][0]["sha256"] = "0" * 64
        (self.root / "figaro_source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            source.validate_package(self.root, self.config_path)

    def test_unknown_transaction_node_is_rejected(self):
        self._write_csv(
            self.transactions,
            ["origin_country", "origin_sector", "destination_country", "destination_sector", "value_million_eur"],
            [["IT", "C10", "FR", "M72", "20"]],
        )
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "transaction references unknown node"):
            source.validate_package(self.root, self.config_path)

    def test_duplicate_output_node_is_rejected(self):
        self._write_csv(
            self.outputs,
            ["country", "sector", "output_million_eur"],
            [["DE", "C20", "200"], ["DE", "C20", "201"], ["FR", "M72", "100"]],
        )
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate output node"):
            source.validate_package(self.root, self.config_path)

    def test_negative_transaction_is_rejected(self):
        self._write_csv(
            self.transactions,
            ["origin_country", "origin_sector", "destination_country", "destination_sector", "value_million_eur"],
            [["DE", "C20", "FR", "M72", "-1"]],
        )
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            source.validate_package(self.root, self.config_path)

    def test_unsupported_satellite_outcome_is_rejected(self):
        with self.satellites.open("a", encoding="utf-8") as handle:
            handle.write("DE,C20,UNKNOWN,1,unit\n")
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "unsupported satellite outcome"):
            source.validate_package(self.root, self.config_path)

    def test_missing_required_satellite_outcome_is_rejected(self):
        rows = [
            ["DE", "C20", "GVA", "80", "EUR"],
            ["FR", "M72", "GVA", "50", "EUR"],
            ["DE", "C20", "GHG", "40", "tCO2e"],
            ["FR", "M72", "GHG", "10", "tCO2e"],
        ]
        self._write_csv(self.satellites, ["country", "sector", "outcome", "value", "unit"], rows)
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "missing satellite outcomes"):
            source.validate_package(self.root, self.config_path)

    def test_partial_employment_coverage_is_reported_not_imputed(self):
        rows = [
            ["DE", "C20", "GVA", "80", "EUR"],
            ["FR", "M72", "GVA", "50", "EUR"],
            ["DE", "C20", "GHG", "40", "tCO2e"],
            ["FR", "M72", "GHG", "10", "tCO2e"],
            ["DE", "C20", "EMPLOYMENT_PERSONS", "200", "persons-equivalent"],
        ]
        self._write_csv(self.satellites, ["country", "sector", "outcome", "value", "unit"], rows)
        self._write_manifest()
        result = source.validate_package(self.root, self.config_path)
        employment = result["contract"]["satellite_coverage"]["EMPLOYMENT_PERSONS"]
        self.assertEqual(employment["covered_nodes"], 1)
        self.assertEqual(employment["total_nodes"], 2)
        self.assertEqual(employment["missing_nodes"], ["FR-M72"])

    def test_identical_package_has_identical_fingerprint(self):
        first = source.validate_package(self.root, self.config_path)
        second = source.validate_package(self.root, self.config_path)
        self.assertEqual(first["package_fingerprint"], second["package_fingerprint"])


if __name__ == "__main__":
    unittest.main()
