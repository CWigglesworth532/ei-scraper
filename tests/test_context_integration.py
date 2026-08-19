import ast
import csv
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema
import yaml

import context_integration as module


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "context_integration"
CONFIG = ROOT / "config" / "context_integration.yaml"
SCHEMA = ROOT / "schemas" / "context_integration_record.schema.json"
INTEGRATED_AT = "2026-08-18T09:00:00Z"
EXPECTED_QA = {
    "selected_subjects": 10,
    "canonical_selected_subjects": 6,
    "non_canonical_selected_subjects": 4,
    "geography_classification_rows": 10,
    "resolved_geography_assertions": 8,
    "unresolved_geography_assertions": 2,
    "subjects_with_resolved_geography": 8,
    "subjects_with_unresolved_geography": 2,
    "subjects_with_compatible_indicators": 6,
    "subjects_with_resolved_geography_no_compatible_indicator": 2,
    "subjects_with_any_external_indicator": 6,
    "subjects_with_no_compatible_external_indicator": 4,
    "subjects_with_all_three_layers": 5,
    "subjects_with_geography_and_indicators_only": 1,
    "subjects_with_geography_and_activity_only": 0,
    "subjects_with_activity_only": 2,
    "distinct_indicator_ids": 4,
    "distinct_geography_schemes": 2,
    "distinct_geography_versions": 2,
    "blocked_geography_version_mismatch_count": 1,
    "compatible_indicator_observations": 11,
    "enriched_rows": 11,
    "staging_rows": 4,
    "output_rows": 15,
    "subjects_with_any_activity_evidence": 7,
    "subjects_with_authoritative_or_strong_activity_evidence": 4,
    "subjects_with_contextual_or_weak_activity_evidence_only": 2,
    "subjects_with_multiple_activity_evidence_items": 3,
    "subjects_with_no_activity_evidence": 3,
    "activity_evidence_rows": 11,
    "activity_review_required_rows": 3,
    "activity_ambiguous_rows": 2,
    "exact_geography_join_matches": 11,
    "entity_id_only_join_matches": 0,
    "case_j_multiple_resolved_levels": 0,
}


def fixture(name):
    return module.read_csv(FIXTURES / name)


def integrate(**overrides):
    inputs = {
        "selected_rows": fixture("selected_subjects.fixture"),
        "classification_rows": fixture("geographic_classifications.fixture"),
        "observation_rows": fixture("external_indicators.fixture"),
        "activity_rows": fixture("activity_evidence.fixture"),
    }
    inputs.update(overrides)
    return module.integrate_context(
        **inputs, config=module.load_config(CONFIG), integrated_at=INTEGRATED_AT
    )


class ContextIntegrationTests(unittest.TestCase):
    def test_production_shaped_cohort_and_qa_contract(self):
        result = integrate()
        self.assertEqual(EXPECTED_QA, result["qa"])
        self.assertEqual(15, len(result["records"]))
        by_subject = {}
        for row in result["records"]:
            by_subject.setdefault(row["subject_id"], []).append(row)
        self.assertEqual(4, len(by_subject["canonical-01"]))
        self.assertEqual(2, len(by_subject["canonical-02"]))
        self.assertEqual("", by_subject["supplier-03"][0]["entity_id"])
        self.assertEqual("no_compatible_indicator", by_subject["canonical-04"][0]["missingness_reason"])
        self.assertEqual("unresolved_geography", by_subject["supplier-05"][0]["missingness_reason"])
        self.assertEqual("1", by_subject["supplier-05"][0]["activity_evidence_count"])
        self.assertEqual("2", by_subject["canonical-06"][0]["activity_evidence_count"])
        self.assertEqual("2", by_subject["supplier-07"][0]["activity_ambiguous_count"])
        self.assertEqual(2, len(by_subject["canonical-08"]))
        self.assertEqual("3", by_subject["canonical-08"][0]["activity_evidence_count"])
        self.assertEqual("no_compatible_indicator", by_subject["canonical-09"][0]["missingness_reason"])
        self.assertEqual("2021", by_subject["canonical-09"][0]["geography_version"])
        self.assertEqual("unresolved_geography", by_subject["supplier-10"][0]["missingness_reason"])

    def test_subject_level_layer_combination_definitions(self):
        result = integrate()
        records = result["records"]
        subjects = {(row["subject_type"], row["subject_id"]) for row in records}
        with_indicator = {(row["subject_type"], row["subject_id"]) for row in records if row["record_type"] == "enriched"}
        resolved = {(row["subject_type"], row["subject_id"]) for row in records if row["mapping_status"] == "resolved"}
        with_activity = {(row["subject_type"], row["subject_id"]) for row in records if int(row["activity_evidence_count"]) > 0}
        qa = result["qa"]
        self.assertEqual(len(with_indicator), qa["subjects_with_any_external_indicator"])
        self.assertEqual(len(subjects - with_indicator), qa["subjects_with_no_compatible_external_indicator"])
        self.assertEqual(len(resolved & with_indicator & with_activity), qa["subjects_with_all_three_layers"])
        self.assertEqual(len((resolved & with_indicator) - with_activity), qa["subjects_with_geography_and_indicators_only"])
        self.assertEqual(len((resolved & with_activity) - with_indicator), qa["subjects_with_geography_and_activity_only"])
        self.assertEqual(len((subjects - resolved) & with_activity), qa["subjects_with_activity_only"])
        self.assertEqual(4, qa["distinct_indicator_ids"])
        self.assertEqual(2, qa["distinct_geography_schemes"])
        self.assertEqual(2, qa["distinct_geography_versions"])
        self.assertEqual(1, qa["blocked_geography_version_mismatch_count"])

    def test_exact_four_field_geography_join_no_scheme_or_version_conversion(self):
        records = integrate()["records"]
        mismatch = next(row for row in records if row["subject_id"] == "canonical-09")
        self.assertEqual("staging", mismatch["record_type"])
        self.assertEqual("", mismatch["observation_id"])
        gb_rows = [row for row in records if row["subject_id"] == "canonical-02"]
        self.assertTrue(all(row["geography_scheme"] == "ITL" for row in gb_rows))

    def test_subject_join_uses_type_and_id_and_entity_is_context_only(self):
        subjects = fixture("selected_subjects.fixture")
        subjects.append({"selected": "true", "subject_type": "supplier_observation", "subject_id": "canonical-01", "entity_id": "entity-01"})
        selected = module._selected_subjects(subjects, module.load_config(CONFIG))
        self.assertIn(("canonical_entity", "canonical-01"), selected)
        self.assertIn(("supplier_observation", "canonical-01"), selected)
        self.assertEqual(11, len(selected))

    def test_conflicting_entity_context_is_rejected_not_used_as_join(self):
        activity = fixture("activity_evidence.fixture")
        activity[0]["entity_id"] = "different-entity"
        with self.assertRaisesRegex(ValueError, "entity_id conflicts"):
            integrate(activity_rows=activity)

    def test_activity_is_summarized_not_multiplied_or_inferred(self):
        result = integrate()
        rows = [row for row in result["records"] if row["subject_id"] == "canonical-08"]
        self.assertEqual(2, len(rows))
        self.assertTrue(all(row["activity_evidence_count"] == "3" for row in rows))
        self.assertFalse(any("industry" in field for field in module.OUTPUT_FIELDS))

    def test_shuffle_equivalence(self):
        forbidden = {"industry", "impact", "causal", "causality", "contribution", "attribution"}
        self.assertFalse(any(token in field for field in module.OUTPUT_FIELDS for token in forbidden))
        baseline = integrate()
        rng = random.Random(30)
        shuffled = {}
        for argument, name in (
            ("selected_rows", "selected_subjects.fixture"),
            ("classification_rows", "geographic_classifications.fixture"),
            ("observation_rows", "external_indicators.fixture"),
            ("activity_rows", "activity_evidence.fixture"),
        ):
            rows = fixture(name)
            rng.shuffle(rows)
            shuffled[argument] = rows
        self.assertEqual(baseline, integrate(**shuffled))

    def test_schema_and_config_validate(self):
        config = module.load_config(CONFIG)
        self.assertEqual("sko-030-context-integration-v1", config["integration_version"])
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
        for row in integrate()["records"]:
            validator.validate(row)

    def test_cli_is_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            outputs = []
            for directory in (first, second):
                output = Path(directory) / "records.csv"
                qa = Path(directory) / "qa.json"
                command = [
                    sys.executable, str(ROOT / "context_integration.py"),
                    "--config", str(CONFIG), "--selected-subjects", str(FIXTURES / "selected_subjects.fixture"),
                    "--geographic-classifications", str(FIXTURES / "geographic_classifications.fixture"),
                    "--external-indicators", str(FIXTURES / "external_indicators.fixture"),
                    "--activity-evidence", str(FIXTURES / "activity_evidence.fixture"),
                    "--output", str(output), "--qa-output", str(qa), "--integrated-at", INTEGRATED_AT,
                ]
                subprocess.run(command, check=True, cwd=ROOT)
                outputs.append((output.read_bytes(), qa.read_bytes()))
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(EXPECTED_QA, json.loads(outputs[0][1]))

    def test_source_has_no_network_or_api_dependencies(self):
        tree = ast.parse((ROOT / "context_integration.py").read_text(encoding="utf-8"))
        forbidden = {"requests", "urllib", "httpx", "aiohttp", "socket", "boto3"}
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertFalse(imported & forbidden)

    def test_case_j_is_not_manufactured_under_sko027_v1(self):
        result = integrate()
        self.assertEqual(0, result["qa"]["case_j_multiple_resolved_levels"])
        per_subject = {}
        for row in fixture("geographic_classifications.fixture"):
            if row["subject_id"] == "not-selected":
                continue
            per_subject.setdefault((row["subject_type"], row["subject_id"]), []).append(row)
        self.assertTrue(all(len(rows) == 1 for rows in per_subject.values()))

    def test_multilevel_subject_qa_does_not_treat_mixed_resolution_as_activity_only(self):
        subjects = fixture("selected_subjects.fixture")
        classifications = fixture("geographic_classifications.fixture")
        observations = fixture("external_indicators.fixture")
        activity = fixture("activity_evidence.fixture")

        target = next(
            row for row in classifications
            if row["subject_type"] == "canonical_entity"
            and row["subject_id"] == "canonical-01"
        )

        extra = dict(target)
        extra["classification_id"] = "geo-multilevel-regression"
        extra["geography_level"] = "2"
        extra["geography_code"] = ""
        extra["geography_name"] = ""
        extra["mapping_status"] = "insufficient_evidence"
        classifications.append(extra)

        result = module.integrate_context(
            subjects,
            classifications,
            observations,
            activity,
            config=module.load_config(CONFIG),
            integrated_at=INTEGRATED_AT,
        )

        self.assertEqual(
            3,
            result["qa"]["subjects_with_unresolved_geography"],
        )

        # canonical-01 has a resolved geography and therefore must not become
        # activity-only merely because it also has an unresolved level.
        records = [
            row for row in result["records"]
            if row["subject_type"] == "canonical_entity"
            and row["subject_id"] == "canonical-01"
        ]
        self.assertTrue(
            any(row["mapping_status"] == "resolved" for row in records)
        )

        resolved_subjects = {
            (row["subject_type"], row["subject_id"])
            for row in result["records"]
            if row["mapping_status"] == "resolved"
        }
        with_activity = {
            (row["subject_type"], row["subject_id"])
            for row in result["records"]
            if int(row["activity_evidence_count"]) > 0
        }
        all_subjects = {
            (row["subject_type"], row["subject_id"])
            for row in result["records"]
        }

        self.assertEqual(
            len((all_subjects - resolved_subjects) & with_activity),
            result["qa"]["subjects_with_activity_only"],
        )

    def test_case_j_counts_subjects_with_multiple_resolved_levels(self):
        subjects = fixture("selected_subjects.fixture")
        classifications = fixture("geographic_classifications.fixture")
        observations = fixture("external_indicators.fixture")
        activity = fixture("activity_evidence.fixture")

        target = next(
            row for row in classifications
            if row["subject_type"] == "canonical_entity"
            and row["subject_id"] == "canonical-01"
        )

        extra = dict(target)
        extra["classification_id"] = "geo-case-j-regression"
        extra["geography_level"] = "2"
        extra["geography_code"] = "TEST-L2"
        extra["geography_name"] = "Synthetic level 2"
        extra["mapping_status"] = "resolved"
        classifications.append(extra)

        result = module.integrate_context(
            subjects,
            classifications,
            observations,
            activity,
            config=module.load_config(CONFIG),
            integrated_at=INTEGRATED_AT,
        )

        self.assertEqual(
            1,
            result["qa"]["case_j_multiple_resolved_levels"],
        )


if __name__ == "__main__":
    unittest.main()
