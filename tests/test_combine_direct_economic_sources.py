"""SKO-036 source combiner behaviours."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import combine_direct_economic_sources as comb
import extract_eurostat_direct_economic as euro


def row(country: str, dataset: str, concept: str) -> dict[str, str]:
    return {field: "" for field in euro.SOURCE_FIELDS} | {
        "source_family": "national_accounts",
        "source_organisation": "Test",
        "source_dataset_id": dataset,
        "source_release_version": "v1",
        "country": country,
        "source_classification": "NACE",
        "source_classification_version": "Rev. 2",
        "source_sector_code": "M72",
        "source_sector_label": "R&D",
        "model_classification": "NACE",
        "model_classification_version": "Rev. 2 A*64",
        "model_sector_code": "M72",
        "model_sector_label": "R&D",
        "reference_year": "2023",
        "concept_code": concept,
        "concept_label": concept,
        "value": "1",
        "normalized_unit": "million_currency",
    }


class CombineDirectEconomicSourcesTests(unittest.TestCase):
    def test_combines_and_sorts_without_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = []
            for name, rows in [
                ("a.csv", [row("UK", "ons", "P1")]),
                ("b.csv", [row("FR", "eurostat", "P1")]),
            ]:
                path = root / name
                with path.open("w", newline="", encoding="utf-8") as h:
                    w = csv.DictWriter(h, fieldnames=euro.SOURCE_FIELDS)
                    w.writeheader(); w.writerows(rows)
                paths.append(path)
            result = comb.combine(paths)
            self.assertEqual([r["country"] for r in result], ["FR", "UK"])
            self.assertEqual(result[1]["source_dataset_id"], "ons")


if __name__ == "__main__":
    unittest.main()
