import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema

import pilot_preparation as module


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "pilot_preparation"
CONFIG = ROOT / "config" / "pilot_preparation.yaml"
PREPARED_AT = "2026-08-19T09:00:00Z"
COMMIT = "7f355d4426c116d6a6ffefac783a91ec62dac0fa"
RUNTIME_METRICS = [*module.SKO030_QA_MAP.values(), "subjects_with_activity_review_required"]


def prepared():
    rows = module.read_csv(FIXTURES / "selected_subjects.fixture")
    return module.validate_selected_subjects(rows, module.load_config(CONFIG))


def upstream_qa():
    return json.loads((FIXTURES / "context_qa.json").read_text())


class RuntimeAvailabilityTests(unittest.TestCase):
    def test_absent_runtime_counts_are_null_and_flags_false(self):
        rows, validation = prepared()
        qa = module.build_qa(rows, validation, {}, tracked_live_files=0)
        self.assertTrue(all(qa[field] is None for field in RUNTIME_METRICS))
        for field in ("sko030_runtime_available", "geography_runtime_available", "indicator_runtime_available", "activity_runtime_available"):
            self.assertIs(qa[field], False)
        self.assertEqual(3, qa["selected_subjects"])
        self.assertEqual(2, qa["subjects_with_country_and_postcode"])

    def test_available_runtime_zero_remains_zero(self):
        rows, validation = prepared()
        runtime = upstream_qa()
        for field in module.SKO030_QA_MAP:
            runtime[field] = 0
        runtime["activity_review_required_rows"] = 0
        qa = module.build_qa(rows, validation, runtime, tracked_live_files=0)
        self.assertTrue(all(qa[field] == 0 for field in RUNTIME_METRICS))
        self.assertTrue(all(qa[field] is True for field in ("sko030_runtime_available", "geography_runtime_available", "indicator_runtime_available", "activity_runtime_available")))

    def test_available_positive_counts_are_retained_exactly(self):
        rows, validation = prepared()
        runtime = upstream_qa()
        qa = module.build_qa(rows, validation, runtime, tracked_live_files=0)
        for upstream, local in module.SKO030_QA_MAP.items():
            self.assertEqual(runtime[upstream], qa[local])
        self.assertEqual(runtime["activity_review_required_rows"], qa["subjects_with_activity_review_required"])

    def test_missing_runtime_is_ready_with_known_gaps(self):
        rows, validation = prepared()
        qa = module.build_qa(rows, validation, {}, tracked_live_files=0)
        decision = module.readiness_decision(qa, dependencies_present=False, context_row_count=len(rows))
        self.assertEqual("ready_with_known_gaps", decision["status"])
        self.assertIn("contextual_runtime_not_available", decision["known_gaps"])
        self.assertEqual([], decision["structural_blockers"])

    def test_preparation_only_rows_do_not_imply_assessed_missingness(self):
        rows, _ = prepared()
        review = module.build_context_review(rows, [])
        self.assertEqual(len(rows), len(review))
        for row in review:
            self.assertEqual("preparation_only", row["record_type"])
            self.assertEqual("contextual_runtime_not_available", row["missingness_reason"])
            self.assertEqual("false", row["review_required"])
            self.assertEqual("", row["mapping_status"])
            self.assertEqual("", row["activity_evidence_count"])
            self.assertNotIn(row["missingness_reason"], {"no_activity_evidence", "unresolved_geography", "no_compatible_indicator"})

    def test_unavailable_qa_and_preparation_rows_validate_strict_schemas(self):
        rows, validation = prepared()
        qa = module.build_qa(rows, validation, {}, tracked_live_files=0)
        review = module.build_context_review(rows, [])
        qa_schema = json.loads((ROOT / "schemas" / "pilot_pre_pilot_qa.schema.json").read_text())
        review_schema = json.loads((ROOT / "schemas" / "pilot_context_review.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(qa_schema)
        jsonschema.Draft202012Validator.check_schema(review_schema)
        jsonschema.validate(qa, qa_schema)
        for row in review:
            jsonschema.validate(row, review_schema)
        invalid = copy.deepcopy(qa)
        invalid["selected_subjects"] = None
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(invalid, qa_schema)

    def test_absent_runtime_cli_is_byte_deterministic(self):
        outputs = []
        with tempfile.TemporaryDirectory() as temp:
            for run_number in (1, 2):
                base = Path(temp) / str(run_number)
                command = [sys.executable, str(ROOT / "pilot_preparation.py"), "--config", str(CONFIG), "--pilot-subjects", str(FIXTURES / "selected_subjects.fixture"), "--pilot-id", "synthetic-pilot", "--prepared-at", PREPARED_AT, "--repository-commit", COMMIT, "--prepared-subjects-output", str(base / "inputs.csv"), "--manifest-output", str(base / "manifest.json"), "--qa-output", str(base / "qa.json"), "--context-review-output", str(base / "review.csv"), "--readiness-output", str(base / "readiness.json")]
                subprocess.run(command, check=True, cwd=ROOT)
                outputs.append(tuple((base / name).read_bytes() for name in ("inputs.csv", "manifest.json", "qa.json", "review.csv", "readiness.json")))
        self.assertEqual(outputs[0], outputs[1])
        qa = json.loads(outputs[0][2])
        self.assertIsNone(qa["subjects_with_no_activity_evidence"])


if __name__ == "__main__":
    unittest.main()
