"""SKO-029 selected-only governed activity evidence behaviours (T01-T28)."""

from __future__ import annotations

import copy
import csv
import hashlib
import inspect
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import activity_evidence as activity


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "activity_evidence.yaml"
FIXTURES = ROOT / "tests" / "fixtures" / "activity_evidence"
RETRIEVED_AT = "2026-08-18T10:00:00Z"
EXTRACTED_AT = "2026-08-18T10:05:00Z"


class ActivityEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = activity.load_config(CONFIG)
        cls.selected_rows = activity.read_csv(FIXTURES / "selected_subjects.csv")
        cls.result = activity.capture_activity_evidence(
            cls.selected_rows, config=cls.config, input_dir=FIXTURES,
            retrieved_at=RETRIEVED_AT, extracted_at=EXTRACTED_AT,
        )
        cls.rows = cls.result["evidence"]

    def selected(self, **values):
        return [row for row in self.rows if all(row[key] == value for key, value in values.items())]

    def copy_fixtures(self, directory: Path) -> None:
        for path in FIXTURES.glob("*.csv"):
            shutil.copyfile(path, directory / path.name)

    def test_t01_only_explicitly_selected_subjects_are_processed(self) -> None:
        self.assertFalse(self.selected(subject_id="ce-unselected"))
        self.assertEqual(self.result["qa"]["selected_subjects"], 7)

    def test_t02_canonical_entity_subject_and_pass_through_id_work(self) -> None:
        rows = self.selected(subject_id="ce-official")
        self.assertTrue(rows)
        self.assertEqual({row["entity_id"] for row in rows}, {"entity-001"})

    def test_t03_supplier_observation_without_entity_id_works(self) -> None:
        row = self.selected(subject_id="obs-unresolved")[0]
        self.assertEqual((row["subject_type"], row["entity_id"]), ("supplier_observation", ""))

    def test_t04_no_entity_is_created_from_evidence(self) -> None:
        observation_rows = [row for row in self.rows if row["subject_type"] == "supplier_observation"]
        self.assertTrue(observation_rows)
        self.assertTrue(all(row["entity_id"] == "" for row in observation_rows))

    def test_t05_official_code_and_description_are_retained_together(self) -> None:
        row = self.selected(subject_id="ce-official", evidence_type="official_activity_code")[0]
        self.assertEqual((row["activity_code_raw"], row["activity_description_raw"]),
                         ("01.20", "Cultivo de frutas"))

    def test_t06_raw_code_formatting_is_preserved_exactly(self) -> None:
        row = self.selected(subject_id="obs-unresolved")[0]
        self.assertEqual(row["activity_code_raw"], " 07.10")

    def test_t07_unresolved_version_is_not_inferred(self) -> None:
        row = self.selected(subject_id="obs-unresolved")[0]
        self.assertEqual((row["activity_scheme"], row["activity_version"], row["evidence_status"]),
                         ("CCAE", "", "review_required"))

    def test_t08_scheme_is_only_retained_when_explicitly_configured(self) -> None:
        official = self.selected(subject_id="ce-official", evidence_type="official_activity_code")[0]
        directory = self.selected(subject_id="ce-directory", evidence_type="directory_sector")[0]
        self.assertEqual(official["activity_scheme"], "CNAE")
        self.assertEqual(directory["activity_scheme"], "")

    def test_t09_principal_activity_is_tri_state_and_not_single_code_inference(self) -> None:
        self.assertEqual(self.selected(subject_id="ce-official", evidence_type="official_activity_code")[0]["is_principal_activity"], "true")
        self.assertEqual(self.selected(subject_id="obs-description")[0]["is_principal_activity"], "false")
        self.assertEqual(self.selected(subject_id="obs-unresolved")[0]["is_principal_activity"], "unknown")

    def test_t10_description_only_official_evidence_is_retained(self) -> None:
        row = self.selected(subject_id="obs-description")[0]
        self.assertEqual(row["activity_code_raw"], "")
        self.assertEqual(row["activity_description_raw"], "Synthetic repair and reuse services")

    def test_t11_directory_evidence_stays_distinct_and_lower_authority(self) -> None:
        sector = self.selected(subject_id="ce-directory", evidence_type="directory_sector")[0]
        summary = self.selected(subject_id="ce-directory", evidence_type="directory_business_summary")[0]
        self.assertEqual((sector["evidence_authority"], summary["evidence_authority"]), ("contextual", "weak"))

    def test_t12_procurement_category_is_context_not_supplier_classification(self) -> None:
        row = self.selected(subject_id="obs-procurement")[0]
        self.assertEqual((row["evidence_type"], row["evidence_authority"], row["evidence_status"]),
                         ("procurement_category", "contextual", "review_required"))

    def test_t13_one_subject_retains_multiple_evidence_items(self) -> None:
        self.assertEqual(len(self.selected(subject_id="ce-official")), 2)
        self.assertEqual(len(self.selected(subject_id="ce-directory")), 2)

    def test_t14_multiple_evidence_items_do_not_overwrite(self) -> None:
        rows = self.selected(subject_id="ce-official")
        self.assertEqual({row["evidence_type"] for row in rows},
                         {"official_activity_code", "official_company_description"})

    def test_t15_provenance_versions_sha_and_timestamps_are_retained(self) -> None:
        row = self.rows[0]
        for field in ("source_name", "source_record_id", "source_reference", "source_version", "source_file_sha256"):
            self.assertTrue(row[field])
        self.assertEqual((row["retrieved_at"], row["extracted_at"]), (RETRIEVED_AT, EXTRACTED_AT))

    def test_t16_deterministic_ids_fingerprints_order_and_rerun(self) -> None:
        self.assertEqual(len({row["activity_evidence_id"] for row in self.rows}), len(self.rows))
        self.assertTrue(all(len(row["evidence_fingerprint"]) == 64 for row in self.rows))
        rerun = activity.capture_activity_evidence(
            self.selected_rows, config=activity.load_config(CONFIG), input_dir=FIXTURES,
            retrieved_at=RETRIEVED_AT, extracted_at=EXTRACTED_AT,
        )
        self.assertEqual(self.result, rerun)

    def test_t17_shuffled_source_input_has_same_logical_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); self.copy_fixtures(directory)
            path = directory / "official_activity_records.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle); fields, rows = list(reader.fieldnames or []), list(reader)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader(); writer.writerows(reversed(rows))
            shuffled = activity.capture_activity_evidence(
                list(reversed(self.selected_rows)), config=self.config, input_dir=directory,
                retrieved_at=RETRIEVED_AT, extracted_at=EXTRACTED_AT,
            )["evidence"]
        dependent = {"source_file_sha256", "evidence_fingerprint"}
        logical = lambda values: [{k: v for k, v in row.items() if k not in dependent} for row in values]
        self.assertEqual(logical(self.rows), logical(shuffled))

    def test_t18_identical_duplicate_evidence_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); self.copy_fixtures(directory)
            path = directory / "procurement_activity_records.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle); fields, rows = list(reader.fieldnames or []), list(reader)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader(); writer.writerows([*rows, rows[0]])
            result = activity.capture_activity_evidence(
                self.selected_rows, config=self.config, input_dir=directory,
                retrieved_at=RETRIEVED_AT, extracted_at=EXTRACTED_AT,
            )["evidence"]
        self.assertEqual(sum(row["subject_id"] == "obs-procurement" for row in result), 1)

    def test_t19_material_conflicts_are_preserved_and_flagged(self) -> None:
        rows = self.selected(subject_id="ce-conflict")
        self.assertEqual({row["activity_code_raw"] for row in rows}, {"10.10", "20.20"})
        self.assertEqual({row["evidence_status"] for row in rows}, {"ambiguous"})

    def test_t20_schema_contract_matches_every_output_row(self) -> None:
        schema = json.loads((ROOT / "schemas" / "activity_evidence.schema.json").read_text())
        self.assertEqual(schema["required"], activity.OUTPUT_FIELDS)
        for row in self.rows:
            self.assertEqual(set(row), set(schema["properties"]))
            self.assertRegex(row["activity_evidence_id"], r"^actev_[0-9a-f]{24}$")
            self.assertRegex(row["source_file_sha256"], r"^[0-9a-f]{64}$")

    def test_t21_production_config_loader_enforces_governance(self) -> None:
        self.assertEqual(self.config["schema_version"], "sko-029-activity-evidence-v1")
        changed = copy.deepcopy(self.config)
        changed["authority_status"]["directory_sector"]["authority"] = "authoritative"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.yaml"
            path.write_text(json.dumps(changed), encoding="utf-8")
            loaded = activity.load_config(path)
        self.assertEqual(loaded["authority_status"]["directory_sector"]["authority"], "authoritative")

    def test_t22_contract_excludes_classification_geography_impact_and_publication(self) -> None:
        forbidden = {"nace_code", "nace_version", "classification_confidence", "client_spend",
                     "impact_value", "geography_code", "geography_scheme", "social_economy_category",
                     "directory_publication_status"}
        self.assertFalse(forbidden & set(activity.OUTPUT_FIELDS))

    def test_t23_no_network_crosswalk_or_canonical_mutation_path(self) -> None:
        source = inspect.getsource(activity).casefold()
        forbidden = ["import requests", "import urllib", "httpx", "socket", "crosswalk(",
                     "create_entity", "update_entity", "client_spend"]
        self.assertFalse([token for token in forbidden if token in source])

    def test_t24_pilot_qa_counts_are_inspectable(self) -> None:
        self.assertEqual(self.result["qa"], {
            "selected_subjects": 7,
            "subjects_with_any_activity_evidence": 6,
            "subjects_with_authoritative_or_strong_evidence": 4,
            "subjects_with_contextual_or_weak_evidence_only": 2,
            "subjects_with_multiple_evidence_items": 3,
            "subjects_with_no_activity_evidence": 1,
            "review_required": 3,
            "ambiguous": 2,
        })

    def test_t25_output_field_order_is_exact(self) -> None:
        self.assertTrue(all(list(row) == activity.OUTPUT_FIELDS for row in self.rows))

    def test_t26_explicit_timestamps_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "retrieved_at and extracted_at"):
            activity.capture_activity_evidence(
                self.selected_rows, config=self.config, input_dir=FIXTURES,
                retrieved_at="", extracted_at=EXTRACTED_AT,
            )

    def test_t27_canonical_entity_without_existing_id_is_rejected(self) -> None:
        rows = [{"selected": "yes", "subject_type": "canonical_entity", "subject_id": "x", "entity_id": ""}]
        with self.assertRaisesRegex(ValueError, "existing entity_id"):
            activity.capture_activity_evidence(
                rows, config=self.config, input_dir=FIXTURES,
                retrieved_at=RETRIEVED_AT, extracted_at=EXTRACTED_AT,
            )

    def test_t28_cli_outputs_are_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            hashes = []
            for directory in (Path(first), Path(second)):
                output, qa = directory / "evidence.csv", directory / "qa.json"
                subprocess.run([
                    sys.executable, str(ROOT / "activity_evidence.py"), "--config", str(CONFIG),
                    "--selected-subjects", str(FIXTURES / "selected_subjects.csv"),
                    "--input-dir", str(FIXTURES), "--output", str(output), "--qa-output", str(qa),
                    "--retrieved-at", RETRIEVED_AT, "--extracted-at", EXTRACTED_AT,
                ], check=True)
                hashes.append((hashlib.sha256(output.read_bytes()).hexdigest(), hashlib.sha256(qa.read_bytes()).hexdigest()))
        self.assertEqual(hashes[0], hashes[1])


if __name__ == "__main__":
    unittest.main()
