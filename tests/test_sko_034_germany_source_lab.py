from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.source_lab.sko_034_germany import (
    ContractError,
    normalize_ch_uid,
    normalize_german_legal_form,
    normalize_identifier,
    parse_bag_if_html,
    parse_zer_fixture,
    validate_candidate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/source_lab/sko_034_candidates.yaml"
FIXTURES = ROOT / "tests/fixtures/source_lab"


class CandidateContractTests(unittest.TestCase):
    def test_candidate_contract_is_valid_and_all_disabled(self) -> None:
        payload = validate_candidate_config(CONFIG)
        candidates = payload["candidate_sources"]
        self.assertTrue(candidates)
        self.assertTrue(all(candidate["enabled"] is False for candidate in candidates))
        self.assertTrue(all(candidate["policy_classification"] == "none" for candidate in candidates))

    def test_enabled_candidate_is_rejected(self) -> None:
        payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        payload["candidate_sources"][0]["enabled"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaises(ContractError):
                validate_candidate_config(path)

    def test_contract_accepts_every_approved_source_format(self) -> None:
        payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        formats = ("CSV", "XLSX", "HTML", "PDF", "XML", "ZIP", "local-build")
        for source_format in formats:
            with self.subTest(source_format=source_format), tempfile.TemporaryDirectory() as tmp:
                candidate = payload["candidate_sources"][0].copy()
                candidate["source_format"] = source_format
                path = Path(tmp) / "candidate.yaml"
                path.write_text(yaml.safe_dump({**payload, "candidate_sources": [candidate]}), encoding="utf-8")
                validate_candidate_config(path)


class GermanIdentifierTests(unittest.TestCase):
    def test_supported_vocabulary(self) -> None:
        examples = {
            "DE_HRB": ("HRB 100", "AG BERLIN"),
            "DE_HRA": ("HRA 101", "AG BERLIN"),
            "DE_GNR": ("GnR 102", "AG BONN"),
            "DE_VR": ("VR 103", "AG KOLN"),
            "DE_PR": ("PR 104", "AG ESSEN"),
            "DE_GSR": ("GsR 105", "AG MUNICH"),
        }
        for kind, (value, court) in examples.items():
            with self.subTest(kind=kind):
                result = normalize_identifier(kind, value, court=court)
                self.assertEqual(result["identifier_type"], kind)
                self.assertEqual(result["identifier_scope"], "register_entry")
        self.assertEqual(normalize_identifier("DE_VAT", "123 456 789")["identifier_value_normalized"], "DE123456789")
        self.assertEqual(normalize_identifier("DE_EUID", "DE.R1101.HRB100")["identifier_scope"], "legal_entity")

    def test_same_register_number_at_different_courts_does_not_collide(self) -> None:
        berlin = normalize_identifier("DE_HRB", "HRB 100", court="AG Berlin")
        bonn = normalize_identifier("DE_HRB", "HRB 100", court="AG Bonn")
        self.assertNotEqual(berlin["identifier_value_normalized"], bonn["identifier_value_normalized"])

    def test_register_identifier_requires_verified_court(self) -> None:
        with self.assertRaises(ValueError):
            normalize_identifier("DE_GNR", "GnR 25")

    def test_zer_identifier_is_source_local_and_not_reusable(self) -> None:
        result = normalize_identifier("DE_ZER_INTERNAL", "ZER-SYN-001")
        self.assertEqual(result["identifier_scope"], "internal_source")
        self.assertFalse(result["reusable_for_identity_matching"])


class GermanLegalFormTests(unittest.TestCase):
    def test_forms_and_variants(self) -> None:
        examples = {
            "Synthetic Hilfe e.V.": "EINGETRAGENER_VEREIN",
            "Synthetic Hilfe E. V.": "EINGETRAGENER_VEREIN",
            "Synthetic Coop eG": "EINGETRAGENE_GENOSSENSCHAFT",
            "Synthetic Coop E.G.": "EINGETRAGENE_GENOSSENSCHAFT",
            "Synthetic Work gGmbH": "GGMBH",
            "Synthetic Trade GMBH": "GMBH",
            "Synthetic Stiftung": "STIFTUNG",
        }
        for name, expected in examples.items():
            with self.subTest(name=name):
                result = normalize_german_legal_form(name)
                self.assertIsNotNone(result)
                self.assertEqual(result["base_legal_form_code"], expected)
                self.assertEqual(result["policy_classification"], "none")

    def test_heg_is_not_eg(self) -> None:
        self.assertIsNone(normalize_german_legal_form("Synthetic H.E.G. Services"))
        self.assertIsNone(normalize_german_legal_form("Strategiegestaltung"))


class BagIfParserTests(unittest.TestCase):
    def test_offline_parser_preserves_provenance_rejections_and_duplicates(self) -> None:
        result = parse_bag_if_html(
            (FIXTURES / "de_bag_if_sample.html").read_text(encoding="utf-8"),
            source_url="https://example.test/bag-if-fixture",
            retrieved_at="2026-08-19T12:00:00+00:00",
        )
        self.assertEqual(result.metrics["accepted_rows"], 3)
        self.assertEqual(result.metrics["duplicate_rows"], 1)
        self.assertGreaterEqual(result.metrics["rejected_rows"], 2)
        self.assertTrue(all(row["source_url"] for row in result.accepted))
        self.assertTrue(all(row["policy_classification"] == "none" for row in result.accepted))
        forms = {row["legal_form"]["base_legal_form_code"] for row in result.accepted if row["legal_form"]}
        self.assertIn("GGMBH", forms)
        self.assertIn("EINGETRAGENER_VEREIN", forms)

    def test_parser_is_deterministic(self) -> None:
        html = (FIXTURES / "de_bag_if_sample.html").read_text(encoding="utf-8")
        kwargs = {"source_url": "https://example.test/bag", "retrieved_at": "2026-08-19T12:00:00+00:00"}
        first = parse_bag_if_html(html, **kwargs)
        second = parse_bag_if_html(html, **kwargs)
        self.assertEqual(first, second)


class ZerParserTests(unittest.TestCase):
    def test_zer_parser_keeps_internal_ids_out_of_tax_id(self) -> None:
        payload = json.loads((FIXTURES / "de_zer_sample.json").read_text(encoding="utf-8"))
        result = parse_zer_fixture(payload, source_url="https://zer.example.test", retrieved_at="2026-08-19T12:00:00+00:00")
        self.assertEqual(result.metrics, {
            "input_rows": 5,
            "accepted_rows": 2,
            "rejected_rows": 3,
            "duplicate_rows": 1,
            "missing_identifier_rows": 1,
            "null_or_invalid_name_rows": 1,
        })
        self.assertTrue(all(row["tax_id"] == "" for row in result.accepted))
        self.assertTrue(all(row["source_local_identifier"]["identifier_type"] == "DE_ZER_INTERNAL" for row in result.accepted))
        self.assertTrue(all(not row["source_local_identifier"]["reusable_for_identity_matching"] for row in result.accepted))
        self.assertTrue(all(row["se_recognition_type"] == "tax_designation" for row in result.accepted))
        self.assertTrue(all(row["policy_classification"] == "none" for row in result.accepted))
        self.assertTrue(all("rejection_reason" in row for row in result.rejected))


class SwitzerlandComparatorTests(unittest.TestCase):
    def test_uid_normalization(self) -> None:
        self.assertEqual(normalize_ch_uid("CHE123456789"), "CHE-123.456.789")
        self.assertEqual(normalize_ch_uid("123.456.789"), "CHE-123.456.789")


if __name__ == "__main__":
    unittest.main()
