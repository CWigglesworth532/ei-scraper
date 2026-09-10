import csv
import unittest
from pathlib import Path


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "matcher" / "expected_cases.csv"
REQUIRED_COLUMNS = {
    "case_id",
    "country",
    "supplier_name",
    "expected_coop",
    "expected_marker",
    "expected_reason",
    "notes",
}
ALLOWED_YES_NO = {"YES", "NO"}


class MatcherRegressionFixtureContractTests(unittest.TestCase):
    """Validate the committed synthetic matcher fixture contract."""

    @classmethod
    def setUpClass(cls):
        with FIXTURE_PATH.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            cls.fieldnames = set(reader.fieldnames or [])
            cls.rows = list(reader)

    def test_required_columns_exist(self):
        self.assertTrue(REQUIRED_COLUMNS.issubset(self.fieldnames))

    def test_case_ids_are_present_and_unique(self):
        case_ids = [row["case_id"].strip() for row in self.rows]
        self.assertTrue(all(case_ids))
        self.assertEqual(len(case_ids), len(set(case_ids)))

    def test_expected_flags_use_governed_values(self):
        for row in self.rows:
            with self.subTest(case_id=row["case_id"]):
                self.assertIn(row["expected_coop"], ALLOWED_YES_NO)
                self.assertIn(row["expected_marker"], ALLOWED_YES_NO)

    def test_fixture_is_nonempty_and_reviewable(self):
        self.assertGreaterEqual(len(self.rows), 28)
        for row in self.rows:
            with self.subTest(case_id=row["case_id"]):
                self.assertTrue(row["supplier_name"].strip())
                self.assertTrue(row["notes"].strip())


if __name__ == "__main__":
    unittest.main()
