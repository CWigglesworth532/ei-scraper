#!/usr/bin/env python3
"""Governed SKO-039 live-pilot runner.

This wrapper binds the accepted SKO-038 observation/direct-output files to the
validated SKO-039 compact FIGARO source package. It does not recalculate direct
outcomes and does not mutate accepted SKO-038 inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from figaro_indirect_attribution import (
    build_figaro_model_from_npz,
    compose_figaro_attribution_with_model,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return value


def adapt_sko038_cohort(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Adapt accepted SKO-038 spend field to SKO-039 in memory only."""
    if not rows:
        raise ValueError("SKO-038 observation cohort is empty")
    required = {"selection_id", "country", "spend", "spend_currency", "proposed_nace_rev2_code"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"SKO-038 observation cohort missing fields: {sorted(missing)}")

    adapted: list[dict[str, str]] = []
    for row in rows:
        currency = (row.get("spend_currency") or "").strip().upper()
        if currency != "EUR":
            raise ValueError(
                f"SKO-039 live pilot requires EUR procurement spend; "
                f"{row.get('selection_id', '')} has {currency or 'BLANK'}"
            )
        copy = dict(row)
        copy["spend_eur"] = row.get("spend", "")
        adapted.append(copy)
    return adapted


def validate_source_gate(summary: dict[str, Any]) -> str:
    if summary.get("status") != "source_package_validated":
        raise ValueError("SKO-039 source package is not validated")
    if summary.get("pilot_execution_permitted") is not True:
        raise ValueError("SKO-039 source package does not permit pilot execution")
    fingerprint = str(summary.get("package_fingerprint", "")).strip().lower()
    if len(fingerprint) != 64 or any(ch not in "0123456789abcdef" for ch in fingerprint):
        raise ValueError("SKO-039 source-package fingerprint is missing or invalid")
    return fingerprint


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_live_pilot(
    *,
    cohort_path: Path,
    direct_outcomes_path: Path,
    model_path: Path,
    gva_satellite_path: Path,
    ghg_satellite_path: Path,
    config_path: Path,
    source_validation_summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_gate = read_json(source_validation_summary_path)
    package_fingerprint = validate_source_gate(source_gate)

    cohort_source = read_csv(cohort_path)
    cohort = adapt_sko038_cohort(cohort_source)
    direct_rows = read_csv(direct_outcomes_path)
    satellite_rows = read_csv(gva_satellite_path) + read_csv(ghg_satellite_path)
    config = read_config(config_path)
    model = build_figaro_model_from_npz(model_path)

    result = compose_figaro_attribution_with_model(
        cohort,
        direct_rows,
        model,
        satellite_rows,
        config,
    )

    run_summary = dict(result["summary"])
    run_summary.update(
        {
            "status": "live_pilot_executed",
            "source_package_fingerprint": package_fingerprint,
            "source_validation_summary_sha256": sha256_file(source_validation_summary_path),
            "sko038_observations_sha256": sha256_file(cohort_path),
            "sko038_direct_outcomes_sha256": sha256_file(direct_outcomes_path),
            "figaro_compact_model_sha256": sha256_file(model_path),
            "figaro_gva_satellite_sha256": sha256_file(gva_satellite_path),
            "figaro_ghg_satellite_sha256": sha256_file(ghg_satellite_path),
            "accepted_direct_rows_supplied": len(direct_rows),
            "cohort_rows": len(cohort),
            "direct_input_recalculated": False,
        }
    )

    write_csv(output_dir / "figaro_live_mapping.csv", result["mapping"])
    write_csv(output_dir / "figaro_live_outcomes.csv", result["outcomes"])
    write_csv(output_dir / "figaro_live_contributions.csv", result["contributions"])
    write_csv(output_dir / "figaro_live_portfolio_summary.csv", result["portfolio"])
    write_csv(output_dir / "figaro_live_qa.csv", result["qa"])
    write_json(output_dir / "figaro_live_pilot_summary.json", run_summary)

    return {
        "mapping": result["mapping"],
        "outcomes": result["outcomes"],
        "contributions": result["contributions"],
        "portfolio": result["portfolio"],
        "qa": result["qa"],
        "summary": run_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("data/pilots/sko-038/whole_spend_observations.csv"))
    parser.add_argument("--direct-outcomes", type=Path, default=Path("data/pilots/sko-038/whole_spend_outcomes.csv"))
    parser.add_argument("--model", type=Path, default=Path("data/pilots/sko-039/source/normalized/figaro_model_26ed_2023.npz"))
    parser.add_argument("--gva-satellite", type=Path, default=Path("data/pilots/sko-039/source/normalized/figaro_gva_2023.csv"))
    parser.add_argument("--ghg-satellite", type=Path, default=Path("data/pilots/sko-039/source/normalized/figaro_ghg_2023.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/figaro_indirect_attribution.yaml"))
    parser.add_argument(
        "--source-validation-summary",
        type=Path,
        default=Path("data/pilots/sko-039/output/figaro_source_validation_summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/pilots/sko-039/output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_live_pilot(
        cohort_path=args.cohort,
        direct_outcomes_path=args.direct_outcomes,
        model_path=args.model,
        gva_satellite_path=args.gva_satellite,
        ghg_satellite_path=args.ghg_satellite,
        config_path=args.config,
        source_validation_summary_path=args.source_validation_summary,
        output_dir=args.output_dir,
    )
    summary = result["summary"]
    print(json.dumps({
        "status": summary["status"],
        "cohort_rows": summary["cohort_rows"],
        "mapped_eligible_observations": summary.get("mapped_eligible_observations"),
        "held_out_observations": summary.get("held_out_observations"),
        "trade_holdouts": summary.get("trade_holdouts"),
        "unmapped_observations": summary.get("unmapped_observations"),
        "source_package_fingerprint": summary["source_package_fingerprint"],
        "determinism_fingerprint": summary.get("determinism_fingerprint"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
