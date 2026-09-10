import csv
import unittest
from pathlib import Path

from match_suppliers_v2 import classify_name_candidates


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "matcher" / "expected_cases.csv"


class MatcherLegalFormFixtureTests(unittest.TestCase):
    """Run the governed synthetic legal-form regression catalogue."""

    @classmethod
    def setUpClass(cls):
        with FIXTURE_PATH.open(newline="", encoding="utf-8") as handle:
            cls.cases = list(csv.DictReader(handle))

    def test_all_fixture_cases(self):
        self.assertGreater(len(self.cases), 0)

        for case in self.cases:
            with self.subTest(case_id=case["case_id"]):
                result = classify_name_candidates(
                    case["country"],
                    case["supplier_name"],
                )

                self.assertEqual(
                    result["name_coop_candidate"],
                    case["expected_coop"],
                )
                self.assertEqual(
                    result["name_marker_candidate"],
                    case["expected_marker"],
                )
                self.assertEqual(
                    result["name_candidate_reason"],
                    case["expected_reason"],
                )


if __name__ == "__main__":
    unittest.main()
