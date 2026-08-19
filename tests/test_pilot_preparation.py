import ast
import copy
import json
import random
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


def subjects():
    return module.read_csv(FIXTURES / "selected_subjects.fixture")


def context():
    return module.read_csv(FIXTURES / "context_integration.fixture")


def config():
    return module.load_config(CONFIG)


def validated(rows=None, **kwargs):
    return module.validate_selected_subjects(rows or subjects(), config(), **kwargs)


def upstream_qa():
    return json.loads((FIXTURES / "context_qa.json").read_text())


class PilotPreparationTests(unittest.TestCase):
    def test_subject_grain_and_identity_contract(self):
        rows, _ = validated()
        self.assertEqual(3, len(rows))
        self.assertEqual("entity-01", rows[0]["entity_id"])
        self.assertEqual("entity-01", rows[1]["entity_id"])
        self.assertEqual("", rows[2]["entity_id"])

    def test_invalid_subject_type_and_location_role_rejected(self):
        for field, value, message in (("subject_type", "site", "Unsupported subject_type"), ("location_role", "branch", "Unsupported location_role")):
            rows = subjects(); rows[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message): validated(rows)

    def test_canonical_blank_entity_rejected(self):
        rows = subjects(); rows[0]["entity_id"] = ""
        with self.assertRaisesRegex(ValueError, "existing entity_id"): validated(rows)

    def test_duplicate_policy_is_explicit_and_deterministic(self):
        rows = subjects(); rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate selected subject"): validated(rows)
        accepted, qa = validated(rows, allow_identical_duplicates=True)
        self.assertEqual(3, len(accepted)); self.assertEqual(1, qa["duplicate_subject_keys"])

    def test_conflicting_key_rejected(self):
        rows = subjects(); conflict = copy.deepcopy(rows[0]); conflict["country"] = "NL"; rows.append(conflict)
        with self.assertRaisesRegex(ValueError, "Conflicting selected subject"): validated(rows)

    def test_missing_postcode_retained_and_country_postcode_counted(self):
        rows, validation = validated()
        qa = module.build_qa(rows, validation, upstream_qa(), tracked_live_files=0)
        self.assertEqual(3, qa["selected_subjects"])
        self.assertEqual(2, qa["subjects_with_country_and_postcode"])
        self.assertEqual(1, qa["subjects_lacking_sufficient_geography_input"])

    def test_traceability_preserved(self):
        rows, _ = validated()
        self.assertEqual("Client A", rows[0]["source_client"])
        review = module.build_context_review(rows, context())
        self.assertEqual("A-001", review[0]["source_supplier_record_key"])

    def test_sko030_metrics_reflected_without_recomputation(self):
        rows, validation = validated()
        qa = module.build_qa(rows, validation, upstream_qa(), tracked_live_files=0)
        self.assertEqual(2, qa["subjects_with_resolved_geography"])
        self.assertEqual(1, qa["subjects_with_contextual_or_weak_only_activity_evidence"])

    def test_context_review_missingness_and_activity_not_multiplied(self):
        rows, _ = validated(); review = module.build_context_review(rows, context())
        self.assertEqual(3, len(review))
        unresolved = next(row for row in review if row["subject_id"] == "observation-02")
        self.assertEqual("unresolved_geography", unresolved["missingness_reason"])
        self.assertEqual("true", unresolved["review_required"])
        self.assertEqual("2", next(row for row in review if row["subject_id"] == "observation-01")["activity_evidence_count"])

    def test_no_compatible_indicator_explicit(self):
        rows, _ = validated(); ctx = context(); ctx[0]["record_type"] = "staging"; ctx[0]["indicator_id"] = ""; ctx[0]["missingness_reason"] = "no_compatible_indicator"
        review = module.build_context_review(rows, ctx)
        self.assertEqual("no_compatible_indicator", review[0]["missingness_reason"])

    def test_readiness_states(self):
        rows, validation = validated(); qa = module.build_qa(rows, validation, upstream_qa(), tracked_live_files=0)
        self.assertEqual("ready_with_known_gaps", module.readiness_decision(qa, dependencies_present=False, context_row_count=3)["status"])
        complete = dict(qa)
        for key in ("subjects_lacking_sufficient_geography_input", "subjects_with_unresolved_geography", "subjects_with_resolved_geography_no_compatible_indicator", "subjects_with_no_activity_evidence"): complete[key] = 0
        self.assertEqual("ready_for_sko_032", module.readiness_decision(complete, dependencies_present=True, context_row_count=3)["status"])
        complete["tracked_live_files"] = 1
        self.assertEqual("blocked", module.readiness_decision(complete, dependencies_present=True, context_row_count=3)["status"])

    def test_manifest_has_hash_counts_and_no_rows(self):
        rows, _ = validated()
        entry = {"logical_role": "pilot_subjects", "path_reference": "local/input.fixture", "sha256": module.file_sha256(FIXTURES / "selected_subjects.fixture"), "row_count": 3}
        manifest = module.build_manifest(rows, pilot_id="synthetic-pilot", config=config(), prepared_at=PREPARED_AT, repository_commit=COMMIT, input_files=[entry], expected_local_root="data/pilots/sko-031/", tracked_live_files=0)
        self.assertEqual(3, manifest["selected_subject_count"])
        self.assertRegex(manifest["input_files"][0]["sha256"], "^[0-9a-f]{64}$")
        rendered = json.dumps(manifest)
        self.assertNotIn("Synthetic Alpha", rendered); self.assertNotIn("A-001", rendered)

    def test_schemas_validate_outputs(self):
        rows, validation = validated(); review = module.build_context_review(rows, context())
        qa = module.build_qa(rows, validation, upstream_qa(), tracked_live_files=0)
        manifest = module.build_manifest(rows, pilot_id="synthetic-pilot", config=config(), prepared_at=PREPARED_AT, repository_commit=COMMIT, input_files=[], expected_local_root="data/pilots/sko-031/", tracked_live_files=0)
        readiness = module.readiness_decision(qa, dependencies_present=False, context_row_count=len(review)); readiness.update({"pilot_id": "synthetic-pilot", "preparation_version": config()["preparation_version"], "prepared_at": PREPARED_AT})
        for name, values in (("pilot_manifest.schema.json", [manifest]), ("pilot_readiness.schema.json", [readiness]), ("pilot_context_review.schema.json", review)):
            schema = json.loads((ROOT / "schemas" / name).read_text())
            jsonschema.Draft202012Validator.check_schema(schema)
            validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
            for value in values: validator.validate(value)

    def test_shuffled_input_has_identical_sorted_output(self):
        original, _ = validated(); shuffled = subjects(); random.Random(31).shuffle(shuffled); changed, _ = validated(shuffled)
        self.assertEqual(original, changed)
        self.assertEqual(module.build_context_review(original, context()), module.build_context_review(changed, list(reversed(context()))))

    def test_cli_byte_deterministic(self):
        outputs = []
        with tempfile.TemporaryDirectory() as temp:
            for run_number in (1, 2):
                base = Path(temp) / str(run_number)
                command = [sys.executable, str(ROOT / "pilot_preparation.py"), "--config", str(CONFIG), "--pilot-subjects", str(FIXTURES / "selected_subjects.fixture"), "--context-integration", str(FIXTURES / "context_integration.fixture"), "--context-qa", str(FIXTURES / "context_qa.json"), "--pilot-id", "synthetic-pilot", "--prepared-at", PREPARED_AT, "--repository-commit", COMMIT, "--prepared-subjects-output", str(base / "inputs.csv"), "--manifest-output", str(base / "manifest.json"), "--qa-output", str(base / "qa.json"), "--context-review-output", str(base / "review.csv"), "--readiness-output", str(base / "readiness.json")]
                subprocess.run(command, check=True, cwd=ROOT)
                outputs.append(tuple((base / name).read_bytes() for name in ("inputs.csv", "manifest.json", "qa.json", "review.csv", "readiness.json")))
        self.assertEqual(outputs[0], outputs[1])

    def test_source_boundaries(self):
        source = (ROOT / "pilot_preparation.py").read_text()
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module: imported.add(node.module.split(".")[0])
        self.assertFalse(imported & {"requests", "urllib", "httpx", "aiohttp", "socket", "boto3"})
        self.assertNotRegex(source, r"def .*?(create|materiali[sz]e|import).*canonical")
        forbidden = {"supplier_impact", "supplier_emissions", "jobs_supported", "mission_value", "spend_derived_impact", "supplier_reported_impact"}
        self.assertFalse(forbidden & set(module.CONTEXT_REVIEW_FIELDS))


if __name__ == "__main__":
    unittest.main()
