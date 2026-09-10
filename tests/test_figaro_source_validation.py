from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
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
                "edition": "2026",
                "reference_year": 2023,
                "table_type": "industry_by_industry_icio",
                "classification": "NACE Rev. 2 A*64",
                "valuation": "basic_prices",
                "currency": "EUR",
                "unit": "million_eur",
            },
            "outcomes": {
                "GVA": {"status": "primary"},
                "GHG": {"status": "primary"},
                "EMPLOYMENT_PERSONS": {"status": "secondary"},
            },
        }
        self.config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        self.transactions = self.root / "figaro_transactions.csv"
        self.outputs = self.root / "figaro_outputs.csv"
        self.satellites = self.root / "figaro_satellites.csv"
        self.model = self.root / "figaro_model_26ed_2023.npz"
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
        self._write_model()
        self._write_manifest()

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _write_csv(path: Path, fields: list[str], rows: list[list[str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(fields)
            writer.writerows(rows)

    def _write_model(self, *, output_vector=None, schema="sko-039-figaro-compact-model-v1"):
        x = np.asarray(output_vector if output_vector is not None else [200.0, 100.0], dtype=float)
        np.savez_compressed(
            self.model,
            schema_version=np.asarray([schema]),
            source_filename=np.asarray(["matrix_eu-ic-io_ind-by-ind_26ed_2023.csv"]),
            source_sha256=np.asarray(["0" * 64]),
            countries=np.asarray(["DE", "FR"]),
            sectors=np.asarray(["C20", "M72"]),
            z_million_eur=np.asarray([[0.0, 20.0], [10.0, 0.0]]),
            output_million_eur=x,
            gva_eur=np.asarray([80.0, 50.0]),
        )

    def _sha(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_manifest(self, *, compact=False, files=None, **overrides):
        if files is None:
            files = [
                {"filename": self.outputs.name, "role": "outputs", "sha256": self._sha(self.outputs)},
                {"filename": self.satellites.name, "role": "satellites", "sha256": self._sha(self.satellites)},
            ]
            if compact:
                files.insert(0, {"filename": self.model.name, "role": "compact_model", "sha256": self._sha(self.model)})
            else:
                files.insert(0, {"filename": self.transactions.name, "role": "transactions", "sha256": self._sha(self.transactions)})
        manifest = {
            "organisation": "Eurostat",
            "product_id": "naio_10_fcp",
            "edition": "2026",
            "reference_year": 2023,
            "table_type": "industry_by_industry_icio",
            "classification": "NACE Rev. 2 A*64",
            "valuation": "basic_prices",
            "currency": "EUR",
            "unit": "million_eur",
            "files": files,
        }
        manifest.update(overrides)
        (self.root / "figaro_source_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def test_valid_legacy_package_passes_and_is_fingerprinted(self):
        result = source.validate_package(self.root, self.config_path)
        self.assertEqual(result["status"], "source_package_validated")
        self.assertTrue(result["manifest"]["checksums_verified"])
        self.assertTrue(result["contract"]["normalized_contract_valid"])
        self.assertEqual(result["contract"]["output_nodes"], 2)
        self.assertTrue(result["pilot_execution_permitted"])
        self.assertEqual(len(result["package_fingerprint"]), 64)

    def test_compact_model_package_passes_without_transaction_csv(self):
        self._write_manifest(compact=True)
        result = source.validate_package(self.root, self.config_path)
        self.assertTrue(result["compact_model"]["compact_model_valid"])
        self.assertEqual(result["compact_model"]["nodes"], 2)
        self.assertEqual(result["compact_model"]["nonzero_transactions"], 2)
        self.assertFalse(result["contract"]["transaction_contract_checked"])
        self.assertIsNone(result["contract"]["transactions"])
        self.assertTrue(result["pilot_execution_permitted"])

    def test_compact_model_output_mismatch_is_rejected(self):
        self._write_model(output_vector=[201.0, 100.0])
        self._write_manifest(compact=True)
        with self.assertRaisesRegex(ValueError, "output vector does not match"):
            source.validate_package(self.root, self.config_path)

    def test_compact_model_schema_mismatch_is_rejected(self):
        self._write_model(schema="wrong-schema")
        self._write_manifest(compact=True)
        with self.assertRaisesRegex(ValueError, "unsupported compact FIGARO schema"):
            source.validate_package(self.root, self.config_path)

    def test_manifest_requires_transactions_or_compact_model(self):
        files = [
            {"filename": self.outputs.name, "role": "outputs", "sha256": self._sha(self.outputs)},
            {"filename": self.satellites.name, "role": "satellites", "sha256": self._sha(self.satellites)},
        ]
        self._write_manifest(files=files)
        with self.assertRaisesRegex(ValueError, "requires transactions or compact_model"):
            source.validate_package(self.root, self.config_path)

    def test_manifest_year_mismatch_is_rejected(self):
        self._write_manifest(reference_year=2022)
        with self.assertRaisesRegex(ValueError, "manifest/config mismatch"):
            source.validate_package(self.root, self.config_path)

    def test_manifest_edition_mismatch_is_rejected(self):
        self._write_manifest(edition="2025")
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

    def test_missing_primary_satellite_outcome_is_rejected(self):
        rows = [
            ["DE", "C20", "GVA", "80", "EUR"],
            ["FR", "M72", "GVA", "50", "EUR"],
        ]
        self._write_csv(self.satellites, ["country", "sector", "outcome", "value", "unit"], rows)
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "missing required satellite outcomes.*GHG"):
            source.validate_package(self.root, self.config_path)

    def test_absent_secondary_employment_does_not_block_primary_package(self):
        rows = [
            ["DE", "C20", "GVA", "80", "EUR"],
            ["FR", "M72", "GVA", "50", "EUR"],
            ["DE", "C20", "GHG", "40", "tCO2e"],
            ["FR", "M72", "GHG", "10", "tCO2e"],
        ]
        self._write_csv(self.satellites, ["country", "sector", "outcome", "value", "unit"], rows)
        self._write_manifest(compact=True)
        result = source.validate_package(self.root, self.config_path)
        employment = result["contract"]["satellite_coverage"]["EMPLOYMENT_PERSONS"]
        self.assertFalse(employment["supplied"])
        self.assertFalse(employment["required"])
        self.assertEqual(employment["covered_nodes"], 0)
        self.assertEqual(employment["missing_nodes"], ["DE-C20", "FR-M72"])
        self.assertTrue(result["pilot_execution_permitted"])

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
        self.assertTrue(employment["supplied"])
        self.assertFalse(employment["required"])
        self.assertEqual(employment["covered_nodes"], 1)
        self.assertEqual(employment["total_nodes"], 2)
        self.assertEqual(employment["missing_nodes"], ["FR-M72"])

    def test_multiple_satellite_files_are_combined(self):
        gva = self.root / "figaro_gva.csv"
        ghg = self.root / "figaro_ghg.csv"
        self._write_csv(
            gva,
            ["country", "sector", "outcome", "value", "unit"],
            [["DE", "C20", "GVA", "80", "EUR"], ["FR", "M72", "GVA", "50", "EUR"]],
        )
        self._write_csv(
            ghg,
            ["country", "sector", "outcome", "value", "unit"],
            [["DE", "C20", "GHG", "40", "tCO2e"], ["FR", "M72", "GHG", "10", "tCO2e"]],
        )
        files = [
            {"filename": self.model.name, "role": "compact_model", "sha256": self._sha(self.model)},
            {"filename": self.outputs.name, "role": "outputs", "sha256": self._sha(self.outputs)},
            {"filename": gva.name, "role": "satellites", "sha256": self._sha(gva)},
            {"filename": ghg.name, "role": "satellites", "sha256": self._sha(ghg)},
        ]
        self._write_manifest(files=files)
        result = source.validate_package(self.root, self.config_path)
        self.assertEqual(result["contract"]["outcome_counts"], {"GHG": 2, "GVA": 2})
        self.assertTrue(result["pilot_execution_permitted"])

    def test_identical_package_has_identical_fingerprint(self):
        self._write_manifest(compact=True)
        first = source.validate_package(self.root, self.config_path)
        second = source.validate_package(self.root, self.config_path)
        self.assertEqual(first["package_fingerprint"], second["package_fingerprint"])


if __name__ == "__main__":
    unittest.main()
