#!/usr/bin/env python3
"""Normalize native Eurostat FIGARO matrix CSVs for SKO-039.

The native matrix contains:
- an industry-by-industry intermediate block in the first N industry columns;
- final-demand columns after that block;
- accounting rows including W2_D1, W2_D29X39 and W2_B2A3G.

This normalizer derives:
- sparse inter-industry transactions;
- total output by country x industry from row totals;
- GVA satellite values in EUR (native accounting values are EUR million);
- structural diagnostics and frozen-source lineage.

Zero-output industry nodes are excluded only when both their full native row and
their intermediate-use column are zero, preserving a valid square model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

ACCOUNTING_ROWS = ("W2_D1", "W2_D29X39", "W2_B2A3G")


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
    if len(country) != 2 or not country.isalpha() or not sector:
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
            if label in ACCOUNTING_ROWS:
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
        missing = [x for x in ACCOUNTING_ROWS if x not in accounting_found]
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
    transactions_path: Path,
    outputs_path: Path,
    gva_path: Path,
    diagnostics_path: Path | None = None,
) -> dict:
    structure = discover_structure(source_path)
    industry_labels = structure["industry_labels"]
    n = len(industry_labels)
    nodes = [split_node(x) for x in industry_labels]

    outputs = {}
    full_row_abs = {}
    z_col_abs = [0.0] * n
    accounting = {}

    # First pass: outputs, intermediate column activity, and accounting vectors.
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row_index, row in enumerate(reader, start=2):
            label = row[0].strip()
            if row_index <= n + 1:
                values = [_num(v, f"{label} column {j}") for j, v in enumerate(row[1:], start=2)]
                z_values = values[:n]
                if any(v < 0 for v in z_values):
                    raise ValueError(f"negative intermediate transaction in native row {label}")
                outputs[label] = sum(values)
                full_row_abs[label] = sum(abs(v) for v in values)
                for j, value in enumerate(z_values):
                    z_col_abs[j] += abs(value)
            elif label in ACCOUNTING_ROWS:
                accounting[label] = [
                    _num(v, f"{label} column {j}") for j, v in enumerate(row[1:n+1], start=2)
                ]

    if len(outputs) != n:
        raise ValueError("failed to read complete industry output block")
    if any(name not in accounting for name in ACCOUNTING_ROWS):
        raise ValueError("failed to read all required accounting vectors")

    included = []
    excluded = []
    for j, node in enumerate(nodes):
        output = outputs[node.label]
        if output > 0:
            included.append(node)
        elif abs(output) <= 1e-12 and full_row_abs[node.label] <= 1e-12 and z_col_abs[j] <= 1e-12:
            excluded.append(node)
        else:
            raise ValueError(
                f"non-positive output node has non-zero activity and cannot be safely excluded: {node.label}"
            )

    included_labels = {n.label for n in included}
    label_position = {label: i for i, label in enumerate(industry_labels)}

    outputs_path.parent.mkdir(parents=True, exist_ok=True)
    transactions_path.parent.mkdir(parents=True, exist_ok=True)
    gva_path.parent.mkdir(parents=True, exist_ok=True)

    with outputs_path.open("w", encoding="utf-8", newline="") as out_handle:
        writer = csv.writer(out_handle, lineterminator="\n")
        writer.writerow(["country", "sector", "output_million_eur"])
        for node in included:
            writer.writerow([node.country, node.sector, format(outputs[node.label], ".15g")])

    with gva_path.open("w", encoding="utf-8", newline="") as gva_handle:
        writer = csv.writer(gva_handle, lineterminator="\n")
        writer.writerow(["country", "sector", "outcome", "value", "unit"])
        for node in included:
            j = label_position[node.label]
            gva_million = sum(accounting[name][j] for name in ACCOUNTING_ROWS)
            writer.writerow([
                node.country,
                node.sector,
                "GVA",
                format(gva_million * 1_000_000.0, ".15g"),
                "EUR",
            ])

    transaction_count = 0
    transaction_sum = 0.0
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle, transactions_path.open(
        "w", encoding="utf-8", newline=""
    ) as tx_handle:
        reader = csv.reader(handle)
        next(reader)
        writer = csv.writer(tx_handle, lineterminator="\n")
        writer.writerow([
            "origin_country",
            "origin_sector",
            "destination_country",
            "destination_sector",
            "value_million_eur",
        ])
        for row_index, row in enumerate(reader, start=2):
            if row_index > n + 1:
                break
            origin_label = row[0].strip()
            if origin_label not in included_labels:
                continue
            origin = split_node(origin_label)
            for j, raw in enumerate(row[1:n+1]):
                destination_label = industry_labels[j]
                if destination_label not in included_labels:
                    continue
                value = _num(raw, f"{origin_label}->{destination_label}")
                if value < 0:
                    raise ValueError(
                        f"negative intermediate transaction {origin_label}->{destination_label}"
                    )
                if value == 0:
                    continue
                destination = split_node(destination_label)
                writer.writerow([
                    origin.country,
                    origin.sector,
                    destination.country,
                    destination.sector,
                    format(value, ".15g"),
                ])
                transaction_count += 1
                transaction_sum += value

    gva_total_eur = 0.0
    for node in included:
        j = label_position[node.label]
        gva_total_eur += sum(accounting[name][j] for name in ACCOUNTING_ROWS) * 1_000_000.0

    diagnostics = {
        "status": "normalized",
        "source": {
            "filename": source_path.name,
            "sha256": sha256_file(source_path),
        },
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
            "output_sum_million_eur": sum(outputs[x.label] for x in included),
            "gva_sum_eur": gva_total_eur,
        },
        "accounting": {
            "gva_components": list(ACCOUNTING_ROWS),
            "gva_formula": "D1 + D29X39 + B2A3G",
            "native_accounting_unit": "million_eur",
            "normalized_gva_unit": "EUR",
            "gva_scale_factor": 1_000_000,
        },
        "model_contract": {
            "transactions": str(transactions_path),
            "outputs": str(outputs_path),
            "gva_satellite": str(gva_path),
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
    parser.add_argument("--transactions-output", required=True, type=Path)
    parser.add_argument("--outputs-output", required=True, type=Path)
    parser.add_argument("--gva-output", required=True, type=Path)
    parser.add_argument("--diagnostics-output", type=Path)
    args = parser.parse_args()
    diagnostics = normalize_matrix(
        args.input,
        args.transactions_output,
        args.outputs_output,
        args.gva_output,
        args.diagnostics_output,
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
