from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

import figaro_indirect_attribution as figaro
import figaro_matrix_normalizer as normalizer


class FigaroMatrixNormalizerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "matrix_eu-ic-io_ind-by-ind_26ed_2023.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def _write_matrix(self, rows, labels=None):
        labels = labels or ["DE_C20", "FR_M72", "DE_U"]
        with self.source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["rowLabels", *labels, "DE_P3_S13", "FR_P3_S13"])
            writer.writerows(rows)

    def _valid_rows(self):
        return [
            ["DE_C20", "20", "10", "0", "70", "0"],
            ["FR_M72", "5", "30", "0", "0", "75"],
            ["DE_U", "0", "0", "0", "0", "0"],
            ["W2_D21X31", "25", "0", "0", "0", "0"],
            ["W2_D1", "25", "40", "0", "0", "0"],
            ["W2_D29X39", "5", "10", "0", "0", "0"],
            ["W2_B2A3G", "20", "20", "0", "0", "0"],
        ]

    def _normalize(self, model=False, transactions=True):
        tx = self.root / "transactions.csv" if transactions else None
        out = self.root / "outputs.csv"
        gva = self.root / "gva.csv"
        diag = self.root / "diag.json"
        model_path = self.root / "model.npz" if model else None
        result = normalizer.normalize_matrix(self.source, tx, out, gva, diag, model_path)
        return result, tx, out, gva, diag, model_path

    def test_discovers_industry_block_and_final_demand(self):
        self._write_matrix(self._valid_rows())
        s = normalizer.discover_structure(self.source)
        self.assertEqual(s["industry_node_count_native"], 3)
        self.assertEqual(s["final_demand_column_count"], 2)
        self.assertEqual(s["data_row_count"], 7)

    def test_figw1_rest_of_world_node_is_accepted(self):
        node = normalizer.split_node("FIGW1_C20")
        self.assertEqual((node.country, node.sector), ("FIGW1", "C20"))

    def test_normalizes_sparse_transactions_and_outputs(self):
        self._write_matrix(self._valid_rows())
        result, tx, out, _, _, _ = self._normalize()
        self.assertEqual(result["normalized"]["included_output_nodes"], 2)
        self.assertEqual(result["normalized"]["excluded_zero_output_nodes"], 1)
        with out.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            [(r["country"], r["sector"], float(r["output_million_eur"])) for r in rows],
            [("DE", "C20", 100.0), ("FR", "M72", 110.0)],
        )
        with tx.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(float(r["value_million_eur"]) for r in rows), 65.0)

    def test_gva_is_d1_plus_d29x39_plus_b2a3g_scaled_to_eur(self):
        self._write_matrix(self._valid_rows())
        _, _, _, gva, _, _ = self._normalize()
        with gva.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        values = {(r["country"], r["sector"]): float(r["value"]) for r in rows}
        self.assertEqual(values[("DE", "C20")], 50_000_000.0)
        self.assertEqual(values[("FR", "M72")], 70_000_000.0)
        self.assertTrue(all(r["unit"] == "EUR" and r["outcome"] == "GVA" for r in rows))

    def test_accounting_identity_reconciles(self):
        self._write_matrix(self._valid_rows())
        result, *_ = self._normalize()
        self.assertLessEqual(
            result["accounting"]["max_abs_reconciliation_difference_million_eur"],
            1e-12,
        )
        self.assertEqual(result["accounting"]["product_tax_reconciliation_component"], "W2_D21X31")

    def test_accounting_identity_failure_is_rejected(self):
        rows = self._valid_rows()
        rows[3][1] = "24"
        self._write_matrix(rows)
        with self.assertRaisesRegex(ValueError, "accounting reconciliation failed"):
            self._normalize()

    def test_zero_output_node_must_be_structurally_zero(self):
        rows = self._valid_rows()
        rows[0][3] = "1"
        self._write_matrix(rows)
        with self.assertRaisesRegex(ValueError, "non-positive output node"):
            self._normalize()

    def test_negative_intermediate_value_rejected(self):
        rows = self._valid_rows()
        rows[0][1] = "-1"
        self._write_matrix(rows)
        with self.assertRaisesRegex(ValueError, "negative intermediate transaction"):
            self._normalize()

    def test_native_row_order_must_match_industry_columns(self):
        rows = self._valid_rows()
        rows[0], rows[1] = rows[1], rows[0]
        self._write_matrix(rows)
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            normalizer.discover_structure(self.source)

    def test_required_accounting_rows_are_mandatory(self):
        rows = [r for r in self._valid_rows() if r[0] != "W2_B2A3G"]
        self._write_matrix(rows)
        with self.assertRaisesRegex(ValueError, "missing required FIGARO accounting rows"):
            normalizer.discover_structure(self.source)

    def test_source_checksum_and_diagnostics_are_deterministic(self):
        self._write_matrix(self._valid_rows())
        first, *_ = self._normalize()
        second, *_ = self._normalize()
        self.assertEqual(first["source"]["sha256"], second["source"]["sha256"])
        self.assertEqual(first["normalized"], second["normalized"])
        self.assertEqual(first["accounting"], second["accounting"])

    def test_compact_model_round_trip_matches_row_model(self):
        self._write_matrix(self._valid_rows())
        result, tx, out, _, _, model_path = self._normalize(model=True)
        self.assertEqual(result["model_contract"]["compact_schema_version"], "sko-039-figaro-compact-model-v1")
        compact = figaro.build_figaro_model_from_npz(model_path)
        with tx.open(newline="", encoding="utf-8") as handle:
            transactions = list(csv.DictReader(handle))
        with out.open(newline="", encoding="utf-8") as handle:
            outputs = list(csv.DictReader(handle))
        row_model = figaro.build_figaro_model(transactions, outputs)
        self.assertEqual(compact["nodes"], row_model["nodes"])
        self.assertTrue(np.allclose(compact["x"], row_model["x"]))
        self.assertTrue(np.allclose(compact["a"], row_model["a"]))
        self.assertTrue(np.allclose(compact["u"], row_model["u"]))
        self.assertEqual(compact["compact_source"]["filename"], self.source.name)
        self.assertEqual(compact["compact_source"]["sha256"], normalizer.sha256_file(self.source))

    def test_compact_model_can_be_emitted_without_transaction_csv(self):
        self._write_matrix(self._valid_rows())
        result, tx, _, _, _, model_path = self._normalize(model=True, transactions=False)
        self.assertIsNone(tx)
        self.assertTrue(model_path.exists())
        self.assertIsNone(result["model_contract"]["transactions"])


if __name__ == "__main__":
    unittest.main()
