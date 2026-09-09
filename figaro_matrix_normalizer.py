#!/usr/bin/env python3
"""Normalize native Eurostat FIGARO matrix CSVs for SKO-039.

The native matrix contains an industry-by-industry intermediate block, final
demand columns, and accounting rows. This module can retain the historical CSV
normalization outputs for inspection while also emitting a compact NumPy model
package for live execution.

The compact package stores the governed square Z matrix, output vector, GVA
vector, node countries/sectors and frozen-source checksum. It avoids expanding
millions of transactions into Python dictionaries for model execution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GVA_COMPONENT_ROWS = ("W2_D1", "W2_D29X39", "W2_B2A3G")
PRODUCT_TAX_ROW = "W2_D21X31"
REQUIRED_ACCOUNTING_ROWS = (PRODUCT_TAX_ROW, *GVA_COMPONENT_ROWS)


@dataclass(frozen=True)
class Node:
    label: str
    country: str
    sector: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_node(label: str) -> Node:
    value = (label or "").strip()
    if "_" not in value:
        raise ValueError(f"invalid FIGARO industry node label: {label!r}")
    country, sector = value.split("_", 1)
    country_ok = (len(country) == 2 and country.isalpha()) or country.upper() == "FIGW1"
    if not country_ok or not sector:
        raise ValueError(f"invalid FIGARO industry node label: {label!r}")
    return Node(value, country.upper(), sector.upper())


def _num(value: str, context: str) -> float:
    text = (value or "").strip()
    if text == "":
        return 0.0
    try:
        result = float(text)
    except ValueError as exc:
        raise ValueError(f"invalid numeric value at {context}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value at {context}")
    return result


def discover_structure(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("FIGARO matrix is empty") from exc
        if not header or header[0] != "rowLabels":
            raise ValueError("FIGARO matrix first column must be rowLabels")

        rows = []
        accounting_found = {}
        for row_index, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"row {row_index} has {len(row)} columns; expected {len(header)}"
                )
            label = row[0].strip()
            rows.append(label)
            if label in REQUIRED_ACCOUNTING_ROWS:
                accounting_found[label] = row_index

        industry_labels = []
        for label in rows:
            if label.startswith("W2_"):
                break
            split_node(label)
            industry_labels.append(label)

        if not industry_labels:
            raise ValueError("no FIGARO industry rows detected")
        if header[1 : 1 + len(industry_labels)] != industry_labels:
            raise ValueError(
                "native industry row block does not exactly match industry column ordering"
            )
        missing = [x for x in REQUIRED_ACCOUNTING_ROWS if x not in accounting_found]
        if missing:
            raise ValueError(f"missing required FIGARO accounting rows: {missing}")

    return {
        "column_count": len(header),
        "data_row_count": len(rows),
        "industry_node_count_native": len(industry_labels),
        "industry_labels": industry_labels,
        "final_demand_column_count": len(header) - 1 - len(industry_labels),
        "accounting_rows": accounting_found,
    }


def normalize_matrix(
    source_path: Path,
    transactions_path: Path | None,
    outputs_path: Path | None,
    gva_path: Path | None,
    diagnostics_path: Path | None = None,
    model_path: Path | None = None,
) -> dict:
    structure = discover_structure(source_path)
    industry_labels = structure["industry_labels"]
    n = len(industry_labels)
    nodes = [split_node(x) for x in industry_labels]

    outputs: dict[str, float] = {}
    full_row_abs: dict[str, float] = {}
    z_col_abs = np.zeros(n, dtype=float)
    z_native = np.zeros((n, n), dtype=float)
    accounting: dict[str, np.ndarray] = {}

    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row_index, row in enumerate(reader, start=2):
            label = row[0].strip()
            if row_index <= n + 1:
                values = np.fromiter(
                    (_num(v, f"{label} column {j}") for j, v in enumerate(row[1:], start=2)),
                    dtype=float,
                    count=len(row) - 1,
                )
                z_values = values[:n]
                if np.any(z_values < 0):
                    raise ValueError(f"negative intermediate transaction in native row {label}")
                z_native[row_index - 2, :] = z_values
                outputs[label] = float(np.sum(values))
                full_row_abs[label] = float(np.sum(np.abs(values)))
                z_col_abs += np.abs(z_values)
            elif label in REQUIRED_ACCOUNTING_ROWS:
                accounting[label] = np.fromiter(
                    (_num(v, f"{label} column {j}") for j, v in enumerate(row[1:n+1], start=2)),
                    dtype=float,
                    count=n,
                )

    if len(outputs) != n:
        raise ValueError("failed to read complete industry output block")
    if any(name not in accounting for name in REQUIRED_ACCOUNTING_ROWS):
        raise ValueError("failed to read all required accounting vectors")

    output_native = np.array([outputs[node.label] for node in nodes], dtype=float)

    included_indices = []
    excluded_indices = []
    for j, node in enumerate(nodes):
        output = output_native[j]
        if output > 0:
            included_indices.append(j)
        elif abs(output) <= 1e-12 and full_row_abs[node.label] <= 1e-12 and z_col_abs[j] <= 1e-12:
            excluded_indices.append(j)
        else:
            raise ValueError(
                f"non-positive output node has non-zero activity and cannot be safely excluded: {node.label}"
            )

    intermediate_inputs = np.sum(z_native, axis=0)
    gva_native = sum(accounting[name] for name in GVA_COMPONENT_ROWS)
    product_taxes = accounting[PRODUCT_TAX_ROW]
    reconciliation = output_native - (intermediate_inputs + gva_native + product_taxes)
    if included_indices:
        included_reconciliation = np.abs(reconciliation[np.asarray(included_indices, dtype=int)])
        max_reconciliation_abs = float(np.max(included_reconciliation))
        if max_reconciliation_abs > 1e-6:
            local_idx = int(np.argmax(included_reconciliation))
            idx_bad = included_indices[local_idx]
            raise ValueError(
                "FIGARO accounting reconciliation failed: "
                f"{nodes[idx_bad].label} difference={reconciliation[idx_bad]:.15g} million_eur"
            )
    else:
        max_reconciliation_abs = 0.0

    included = [nodes[i] for i in included_indices]
    excluded = [nodes[i] for i in excluded_indices]
    idx = np.asarray(included_indices, dtype=int)
    z = z_native[np.ix_(idx, idx)]
    x = output_native[idx]
    gva_eur = gva_native[idx] * 1_000_000.0

    if outputs_path:
        outputs_path.parent.mkdir(parents=True, exist_ok=True)
        with outputs_path.open("w", encoding="utf-8", newline="") as out_handle:
            writer = csv.writer(out_handle, lineterminator="\n")
            writer.writerow(["country", "sector", "output_million_eur"])
            for node, output in zip(included, x):
                writer.writerow([node.country, node.sector, format(float(output), ".15g")])

    if gva_path:
        gva_path.parent.mkdir(parents=True, exist_ok=True)
        with gva_path.open("w", encoding="utf-8", newline="") as gva_handle:
            writer = csv.writer(gva_handle, lineterminator="\n")
            writer.writerow(["country", "sector", "outcome", "value", "unit"])
            for node, value in zip(included, gva_eur):
                writer.writerow([node.country, node.sector, "GVA", format(float(value), ".15g"), "EUR"])

    transaction_count = int(np.count_nonzero(z))
    transaction_sum = float(np.sum(z))
    if transactions_path:
        transactions_path.parent.mkdir(parents=True, exist_ok=True)
        with transactions_path.open("w", encoding="utf-8", newline="") as tx_handle:
            writer = csv.writer(tx_handle, lineterminator="\n")
            writer.writerow([
                "origin_country", "origin_sector", "destination_country",
                "destination_sector", "value_million_eur",
            ])
            rows, cols = np.nonzero(z)
            for i, j in zip(rows.tolist(), cols.tolist()):
                origin = included[i]
                destination = included[j]
                writer.writerow([
                    origin.country, origin.sector, destination.country, destination.sector,
                    format(float(z[i, j]), ".15g"),
                ])

    source_sha = sha256_file(source_path)
    if model_path:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            model_path,
            schema_version=np.asarray(["sko-039-figaro-compact-model-v1"]),
            source_filename=np.asarray([source_path.name]),
            source_sha256=np.asarray([source_sha]),
            countries=np.asarray([node.country for node in included], dtype="U8"),
            sectors=np.asarray([node.sector for node in included], dtype="U16"),
            z_million_eur=z,
            output_million_eur=x,
            gva_eur=gva_eur,
        )

    diagnostics = {
        "status": "normalized",
        "source": {"filename": source_path.name, "sha256": source_sha},
        "native": {
            "column_count": structure["column_count"],
            "data_row_count": structure["data_row_count"],
            "industry_node_count": n,
            "final_demand_column_count": structure["final_demand_column_count"],
        },
        "normalized": {
            "included_output_nodes": len(included),
            "excluded_zero_output_nodes": len(excluded),
            "excluded_zero_output_labels": [x.label for x in excluded],
            "nonzero_transactions": transaction_count,
            "intermediate_transaction_sum_million_eur": transaction_sum,
            "output_sum_million_eur": float(np.sum(x)),
            "gva_sum_eur": float(np.sum(gva_eur)),
        },
        "accounting": {
            "gva_components": list(GVA_COMPONENT_ROWS),
            "gva_formula": "D1 + D29X39 + B2A3G",
            "product_tax_reconciliation_component": PRODUCT_TAX_ROW,
            "identity": "output = intermediate_inputs + GVA + D21X31",
            "max_abs_reconciliation_difference_million_eur": max_reconciliation_abs,
            "native_accounting_unit": "million_eur",
            "normalized_gva_unit": "EUR",
            "gva_scale_factor": 1_000_000,
        },
        "model_contract": {
            "transactions": str(transactions_path) if transactions_path else None,
            "outputs": str(outputs_path) if outputs_path else None,
            "gva_satellite": str(gva_path) if gva_path else None,
            "compact_model": str(model_path) if model_path else None,
            "compact_schema_version": "sko-039-figaro-compact-model-v1" if model_path else None,
            "zero_output_policy": "exclude_only_if_full_row_and_intermediate_column_are_zero",
        },
    }
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--transactions-output", type=Path)
    parser.add_argument("--outputs-output", type=Path)
    parser.add_argument("--gva-output", type=Path)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--diagnostics-output", type=Path)
    args = parser.parse_args()
    if not any((args.transactions_output, args.outputs_output, args.gva_output, args.model_output)):
        parser.error("at least one normalized/model output must be requested")
    diagnostics = normalize_matrix(
        args.input,
        args.transactions_output,
        args.outputs_output,
        args.gva_output,
        args.diagnostics_output,
        args.model_output,
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
