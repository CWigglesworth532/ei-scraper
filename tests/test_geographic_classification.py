"""SKO-027 governed geographic classification behavioural tests (T01-T24)."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import geographic_classification as geo


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "geographic_classification"
CONFIG = ROOT / "config" / "geographic_classification.yaml"
CLASSIFIED_AT = "2026-08-18T09:30:00Z"
SOURCE = "synthetic-postcode-correspondence"
SOURCE_VERSION = "fixture-v1"


class GeographicClassificationBehaviourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = geo.load_config(CONFIG)
        cls.suppliers = geo.read_csv(FIXTURES / "selected_suppliers.csv")
        cls.correspondence_path = FIXTURES / "postcode_correspondence.csv"
        cls.correspondence = geo.read_csv(cls.correspondence_path)
        cls.source_hash = geo.file_sha256(cls.correspondence_path)
        cls.results = geo.classify_rows(
            cls.suppliers, cls.correspondence, config=cls.config,
            classified_at=CLASSIFIED_AT, mapping_source=SOURCE,
            mapping_source_version=SOURCE_VERSION,
            mapping_source_sha256=cls.source_hash,
        )
        cls.by_subject = {row["subject_id"]: row for row in cls.results}

    def classify(self, suppliers=None, correspondence=None, **kwargs):
        parameters = {
            "config": self.config, "classified_at": CLASSIFIED_AT,
            "mapping_source": SOURCE, "mapping_source_version": SOURCE_VERSION,
            "mapping_source_sha256": self.source_hash,
        }
        parameters.update(kwargs)
        return geo.classify_rows(
            copy.deepcopy(self.suppliers if suppliers is None else suppliers),
            copy.deepcopy(self.correspondence if correspondence is None else correspondence),
            **parameters,
        )

    def test_t01_exact_postcode_resolution(self) -> None:
        row = self.by_subject["ent_eu"]
        self.assertEqual((row["mapping_status"], row["geography_code"]), ("resolved", "BE211"))
        self.assertEqual(row["mapping_confidence"], "strong")
        self.assertEqual(
            (row["geography_scheme"], row["geography_version"], row["geography_level"],
             row["geography_name"]),
            ("NUTS", "2024", "3", "Arr. Antwerpen"),
        )

    def test_t02_config_country_normalization(self) -> None:
        row = self.by_subject["obs_activity"]
        self.assertEqual((row["country"], row["geography_code"]), ("GB", "TLM50"))
        self.assertEqual(
            (row["geography_scheme"], row["geography_version"], row["geography_level"]),
            ("ITL", "2021", "3"),
        )

    def test_t03_ambiguous_mapping(self) -> None:
        row = self.by_subject["obs_ambiguous"]
        evidence = json.loads(row["mapping_evidence"])
        self.assertEqual(row["mapping_status"], "ambiguous")
        self.assertEqual(evidence["distinct_supported_result_count"], 2)
        self.assertEqual(row["geography_code"], "")

    def test_t04_unsupported_country(self) -> None:
        row = self.by_subject["obs_unsupported"]
        self.assertEqual(row["mapping_status"], "unsupported_country")
        self.assertEqual(row["geography_code"], "")
        invalid_iso2 = copy.deepcopy(self.suppliers[0])
        invalid_iso2["country"] = "ZZ"
        self.assertEqual(self.classify(suppliers=[invalid_iso2])[0]["mapping_status"],
                         "unsupported_country")

    def test_t05_missing_postcode_is_insufficient(self) -> None:
        row = self.by_subject["obs_missing_postcode"]
        self.assertEqual(row["mapping_status"], "insufficient_evidence")

    def test_t06_only_explicitly_selected_rows_execute(self) -> None:
        self.assertNotIn("ent_not_selected", self.by_subject)
        self.assertEqual(len(self.results), 7)

    def test_t07_canonical_entity_id_is_unchanged(self) -> None:
        self.assertEqual(self.by_subject["ent_alpha"]["entity_id"], "ent_alpha")
        self.assertEqual(self.by_subject["ent_eu"]["entity_id"], "ent_eu")

    def test_t08_supplier_observation_is_supported(self) -> None:
        row = self.by_subject["obs_activity"]
        self.assertEqual(row["subject_type"], "supplier_observation")

    def test_t09_supplier_observation_gets_no_manufactured_entity_id(self) -> None:
        observations = [row for row in self.results if row["subject_type"] == "supplier_observation"]
        self.assertTrue(observations)
        self.assertTrue(all(row["entity_id"] == "" for row in observations))

    def test_t10_hq_role_is_preserved_exactly(self) -> None:
        self.assertEqual(self.by_subject["ent_alpha"]["location_role"], "registered_or_hq")

    def test_t11_contracted_activity_role_is_preserved_exactly(self) -> None:
        self.assertEqual(self.by_subject["obs_activity"]["location_role"], "contracted_activity")
        self.assertNotEqual(self.by_subject["ent_alpha"]["location_role"], "contracted_activity")

    def test_t12_provenance_and_versions_are_retained(self) -> None:
        row = self.by_subject["ent_alpha"]
        self.assertEqual(row["geography_scheme"], "ITL")
        self.assertEqual(row["geography_version"], "2021")
        self.assertEqual(row["geography_code"], "TLI32")
        self.assertEqual(row["mapping_source"], SOURCE)
        self.assertEqual(row["mapping_source_version"], SOURCE_VERSION)
        self.assertEqual(row["classifier_version"], "sko-027-statistical-geography-v1")
        evidence = json.loads(row["mapping_evidence"])
        self.assertEqual({result["geography_scheme"]
                          for result in evidence["distinct_supported_results"]}, {"ITL"})
        self.assertTrue(any(source["country"] == "GB" and source["geography_scheme"] == "NUTS"
                            for source in self.correspondence))
        invalid_nuts = [
            source for source in self.correspondence
            if source["country"] == "GB" and source["geography_scheme"] == "NUTS"
        ]
        gb_only = [supplier for supplier in self.suppliers if supplier["subject_id"] == "ent_alpha"]
        self.assertEqual(self.classify(gb_only, invalid_nuts)[0]["mapping_status"], "insufficient_evidence")

    def test_t13_correspondence_sha_is_retained(self) -> None:
        self.assertTrue(all(row["mapping_source_sha256"] == self.source_hash for row in self.results))
        self.assertEqual(len(self.source_hash), 64)

    def test_t14_deterministic_rerun(self) -> None:
        self.assertEqual(self.results, self.classify())

    def test_t15_shuffled_supplier_input_is_logically_identical(self) -> None:
        self.assertEqual(self.results, self.classify(suppliers=list(reversed(self.suppliers))))

    def test_t16_shuffled_correspondence_is_logically_identical(self) -> None:
        self.assertEqual(self.results, self.classify(correspondence=list(reversed(self.correspondence))))

    def test_t17_duplicate_identical_correspondence_is_not_ambiguous(self) -> None:
        row = self.by_subject["ent_alpha"]
        evidence = json.loads(row["mapping_evidence"])
        self.assertEqual(row["mapping_status"], "resolved")
        self.assertEqual(evidence["distinct_supported_result_count"], 1)
        self.assertEqual(evidence["distinct_supported_results"][0]["geography_scheme"], "ITL")

    def test_t18_city_does_not_infer_geography(self) -> None:
        self.assertEqual(self.by_subject["obs_no_match"]["mapping_status"], "insufficient_evidence")
        source = inspect.getsource(geo._build_index)
        self.assertNotIn('get("city")', source)

    def test_t19_address_does_not_infer_geography(self) -> None:
        self.assertEqual(self.by_subject["obs_missing_postcode"]["geography_code"], "")
        source = inspect.getsource(geo._build_index)
        self.assertNotIn('get("address")', source)

    def test_t20_postcode_normalization_is_conservative(self) -> None:
        self.assertEqual(geo.normalize_postcode("  sw1a   1aa "), "SW1A 1AA")
        self.assertEqual(geo.normalize_postcode("12-34"), "12-34")

    def test_t21_classified_at_is_explicit_and_retained(self) -> None:
        self.assertTrue(all(row["classified_at"] == CLASSIFIED_AT for row in self.results))
        with self.assertRaisesRegex(ValueError, "classified_at"):
            self.classify(classified_at="")

    def test_t22_invalid_subject_type_is_rejected(self) -> None:
        row = copy.deepcopy(self.suppliers[0])
        row["subject_type"] = "new_canonical_entity"
        with self.assertRaisesRegex(ValueError, "subject_type"):
            self.classify(suppliers=[row])

    def test_t23_invalid_location_role_is_rejected_not_inferred(self) -> None:
        row = copy.deepcopy(self.suppliers[0])
        row["location_role"] = "headquarters_activity"
        with self.assertRaisesRegex(ValueError, "location_role"):
            self.classify(suppliers=[row])

    def test_t24_cli_output_is_stable_and_contains_no_indicator_fields(self) -> None:
        forbidden = {"income", "deprivation", "emissions", "population", "indicator"}
        self.assertFalse(forbidden & set(geo.OUTPUT_FIELDS))
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_path, second_path = Path(first) / "out.csv", Path(second) / "out.csv"
            geo.write_csv(self.results, first_path)
            geo.write_csv(self.classify(), second_path)
            self.assertEqual(hashlib.sha256(first_path.read_bytes()).hexdigest(),
                             hashlib.sha256(second_path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
