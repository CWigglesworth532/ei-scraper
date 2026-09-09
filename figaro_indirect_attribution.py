#!/usr/bin/env python3
"""SKO-039 governed FIGARO indirect supply-chain attribution prototype.

Accepted SKO-038 remains authoritative for direct effects. FIGARO supplies only
upstream indirect requirements, calculated as (L - I) @ spend_shock.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_REQUIRED = {"selection_id", "country", "spend_eur", "proposed_nace_rev2_code"}


def _f(value: Any, label: str) -> float:
    try:
        x = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {label}: {value!r}") from exc
    if not np.isfinite(x):
        raise ValueError(f"non-finite numeric value for {label}")
    return x


def _fmt(x: float) -> str:
    return format(0.0 if abs(x) < 5e-15 else x, ".15g")


def _node(country: str, sector: str) -> tuple[str, str]:
    return country.strip().upper(), sector.strip().upper()


def _require_cohort(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("cohort is empty")
    missing = _REQUIRED - set(rows[0])
    if missing:
        raise ValueError(f"cohort missing columns: {sorted(missing)}")
    ids = [(r.get("selection_id") or "").strip() for r in rows]
    if any(not i for i in ids):
        raise ValueError("selection_id must be non-blank")
    dup = sorted(i for i, n in Counter(ids).items() if n > 1)
    if dup:
        raise ValueError(f"duplicate selection_id values: {dup}")


def _source_meta(config: Mapping[str, Any]) -> dict[str, str]:
    s = config.get("figaro_source", {})
    required = ["product_id", "edition", "reference_year", "table_type"]
    missing = [k for k in required if str(s.get(k, "")).strip() == ""]
    if missing:
        raise ValueError(f"FIGARO config missing source metadata: {missing}")
    return {
        "source_product": str(s["product_id"]),
        "source_edition": str(s["edition"]),
        "reference_year": str(s["reference_year"]),
        "table_type": str(s["table_type"]),
    }


def _finalize_model(nodes: list[tuple[str, str]], z: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    if not nodes:
        raise ValueError("FIGARO output vector is empty")
    z = np.asarray(z, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(nodes)
    if z.shape != (n, n):
        raise ValueError(f"FIGARO Z matrix shape {z.shape} does not match {n} nodes")
    if x.shape != (n,):
        raise ValueError(f"FIGARO output vector shape {x.shape} does not match {n} nodes")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(x)):
        raise ValueError("FIGARO compact model contains non-finite values")
    if np.any(z < 0):
        raise ValueError("negative FIGARO intermediate transaction is not supported in v1")
    if np.any(x <= 0):
        raise ValueError("FIGARO output must be positive for every model node")
    if len(set(nodes)) != n:
        raise ValueError("duplicate FIGARO model node")
    index = {node: i for i, node in enumerate(nodes)}
    a = z / x[np.newaxis, :]
    ident = np.eye(n)
    try:
        l = np.linalg.inv(ident - a)
    except np.linalg.LinAlgError as exc:
        raise ValueError("FIGARO Leontief system is singular") from exc
    err = float(np.max(np.abs((ident - a) @ l - ident)))
    if err > 1e-9:
        raise ValueError(f"Leontief inversion reconciliation failed: {err}")
    return {
        "nodes": nodes, "index": index, "x": x, "a": a, "l": l,
        "u": l - ident, "max_inverse_error": err,
    }


def build_figaro_model(transactions: list[dict[str, str]], outputs: list[dict[str, str]]) -> dict[str, Any]:
    """Build A, L and U=L-I from normalized FIGARO rows."""
    output_by_node: dict[tuple[str, str], float] = {}
    for r in outputs:
        n = _node(r.get("country", ""), r.get("sector", ""))
        if not all(n):
            raise ValueError("FIGARO output row has blank country/sector")
        if n in output_by_node:
            raise ValueError(f"duplicate FIGARO output node: {n}")
        value = _f(r.get("output_million_eur", ""), f"output {n}")
        if value <= 0:
            raise ValueError(f"FIGARO output must be positive for {n}")
        output_by_node[n] = value
    nodes = sorted(output_by_node)
    index = {n: i for i, n in enumerate(nodes)}
    z = np.zeros((len(nodes), len(nodes)), dtype=float)
    for r in transactions:
        origin = _node(r.get("origin_country", ""), r.get("origin_sector", ""))
        dest = _node(r.get("destination_country", ""), r.get("destination_sector", ""))
        if origin not in index or dest not in index:
            raise ValueError(f"transaction references unknown FIGARO node: {origin} -> {dest}")
        value = _f(r.get("value_million_eur", ""), f"transaction {origin}->{dest}")
        if value < 0:
            raise ValueError("negative FIGARO intermediate transaction is not supported in v1")
        z[index[origin], index[dest]] += value
    x = np.array([output_by_node[n] for n in nodes], dtype=float)
    return _finalize_model(nodes, z, x)


def build_figaro_model_from_npz(path: str | Path) -> dict[str, Any]:
    """Load the governed compact model package emitted by figaro_matrix_normalizer."""
    package_path = Path(path)
    with np.load(package_path, allow_pickle=False) as pkg:
        required = {
            "schema_version", "source_filename", "source_sha256",
            "countries", "sectors", "z_million_eur", "output_million_eur",
        }
        missing = required - set(pkg.files)
        if missing:
            raise ValueError(f"compact FIGARO package missing arrays: {sorted(missing)}")
        schema = str(np.asarray(pkg["schema_version"]).reshape(-1)[0])
        if schema != "sko-039-figaro-compact-model-v1":
            raise ValueError(f"unsupported compact FIGARO schema: {schema}")
        countries = np.asarray(pkg["countries"]).astype(str)
        sectors = np.asarray(pkg["sectors"]).astype(str)
        if countries.shape != sectors.shape or countries.ndim != 1:
            raise ValueError("compact FIGARO country/sector arrays must be aligned 1-D vectors")
        nodes = [_node(c, s) for c, s in zip(countries.tolist(), sectors.tolist())]
        z = np.asarray(pkg["z_million_eur"], dtype=float)
        x = np.asarray(pkg["output_million_eur"], dtype=float)
        source = {
            "filename": str(np.asarray(pkg["source_filename"]).reshape(-1)[0]),
            "sha256": str(np.asarray(pkg["source_sha256"]).reshape(-1)[0]),
            "schema_version": schema,
        }
    model = _finalize_model(nodes, z, x)
    model["compact_source"] = source
    return model


def build_satellite_intensities(model: Mapping[str, Any], rows: list[dict[str, str]], config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return outcome intensity arrays per EURm output; missing cells stay NaN."""
    wanted = {str(k).upper(): v for k, v in config.get("outcomes", {}).items()}
    cells: dict[str, dict[tuple[str, str], tuple[float, str]]] = defaultdict(dict)
    for r in rows:
        outcome = (r.get("outcome") or "").strip().upper()
        if outcome not in wanted:
            continue
        n = _node(r.get("country", ""), r.get("sector", ""))
        if n not in model["index"]:
            raise ValueError(f"satellite references unknown FIGARO node: {n}")
        if n in cells[outcome]:
            raise ValueError(f"duplicate satellite cell for {outcome} {n}")
        cells[outcome][n] = (_f(r.get("value", ""), f"satellite {outcome} {n}"), r.get("unit", ""))
    result = {}
    for outcome, spec in wanted.items():
        intensity = np.full(len(model["nodes"]), np.nan)
        units = set()
        for i, n in enumerate(model["nodes"]):
            if n in cells[outcome]:
                value, unit = cells[outcome][n]
                intensity[i] = value / model["x"][i]
                if unit:
                    units.add(unit)
        if len(units) > 1:
            raise ValueError(f"mixed satellite units for {outcome}: {sorted(units)}")
        result[outcome] = {"intensity": intensity, "unit": next(iter(units), str(spec.get("unit", "")))}
    return result


def map_observations(cohort: list[dict[str, str]], model: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, str]]:
    _require_cohort(cohort)
    policy = config.get("mapping_policy", {})
    if policy.get("approved_fallbacks"):
        raise ValueError("SKO-039 v1 does not permit parent fallback; exact governed mapping only")
    explicit = {str(k).upper(): str(v).upper() for k, v in (policy.get("explicit_sector_mapping", {}) or {}).items()}
    trade = {str(v).upper() for v in config.get("valuation_policy", {}).get("trade_sector_holdouts", ["G45", "G46", "G47"])}
    nodes = set(model["nodes"])
    out = []
    for r in cohort:
        country = (r.get("country") or "").strip().upper()
        original = (r.get("proposed_nace_rev2_code") or "").strip().upper()
        model_sector = (r.get("model_sector_code") or original).strip().upper()
        mapped = explicit.get(model_sector, model_sector)
        if not original:
            status, reason, eligibility, valuation = "unmapped", "nace_code_missing", "held_out", "not_evaluated"
            fc, fs = "", ""
        elif not country:
            status, reason, eligibility, valuation = "unmapped", "country_missing", "held_out", "not_evaluated"
            fc, fs = "", ""
        elif (country, mapped) not in nodes:
            status, reason, eligibility, valuation = "unmapped", "figaro_country_sector_node_missing", "held_out", "not_evaluated"
            fc, fs = country, mapped
        else:
            status = "mapped_exact" if model_sector == mapped else "mapped_governed_exact"
            reason = "exact_a64_country_sector_node" if model_sector == mapped else "explicit_governed_exact_sector_mapping"
            eligibility, valuation, fc, fs = "eligible", "basic_price_spend_proxy", country, mapped
            if mapped in trade:
                reason, eligibility, valuation = "trade_sector_primary_case_valuation_holdout", "held_out", "trade_purchase_value_not_comparable"
        out.append({
            "selection_id": r["selection_id"].strip(), "country": country,
            "original_nace": original, "model_sector": model_sector,
            "figaro_country": fc, "figaro_sector": fs, "mapping_status": status,
            "mapping_reason": reason, "mapping_source": str(policy.get("source", "NACE Rev. 2 A*64 / FIGARO")),
            "mapping_version": str(policy.get("version", "figaro-nace-a64-v1")),
            "valuation_case": valuation, "calculation_eligibility": eligibility,
        })
    return out


def _direct_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], tuple[float, str]]:
    aliases = {"B1G": "GVA", "EMP_PERSONS": "EMPLOYMENT_PERSONS"}
    out = {}
    for r in rows:
        if (r.get("attribution_status") or "").strip() not in {"", "available", "modelled"}:
            continue
        sid = (r.get("selection_id") or "").strip()
        raw = (r.get("outcome_code") or r.get("outcome") or "").strip().upper()
        code = aliases.get(raw, raw)
        if not sid or not code:
            continue
        key = (sid, code)
        if key in out:
            raise ValueError(f"duplicate direct outcome row: {key}")
        out[key] = (_f(r.get("modelled_value", r.get("direct_value", "")), f"direct {key}"), r.get("modelled_unit", r.get("unit", "")))
    return out


def compose_figaro_attribution(cohort: list[dict[str, str]], direct_rows: list[dict[str, str]], transactions: list[dict[str, str]], outputs: list[dict[str, str]], satellite_rows: list[dict[str, str]], config: Mapping[str, Any]) -> dict[str, Any]:
    _require_cohort(cohort)
    model = build_figaro_model(transactions, outputs)
    return compose_figaro_attribution_with_model(cohort, direct_rows, model, satellite_rows, config)


def compose_figaro_attribution_with_model(cohort: list[dict[str, str]], direct_rows: list[dict[str, str]], model: Mapping[str, Any], satellite_rows: list[dict[str, str]], config: Mapping[str, Any]) -> dict[str, Any]:
    """Compose attribution from an already-built governed FIGARO model."""
    _require_cohort(cohort)
    mapping = map_observations(cohort, model, config)
    satellites = build_satellite_intensities(model, satellite_rows, config)
    by_id = {r["selection_id"]: r for r in cohort}
    map_by_id = {r["selection_id"]: r for r in mapping}
    direct = _direct_index(direct_rows)
    meta = _source_meta(config)
    boundary = str(config.get("portfolio_policy", {}).get("aggregation_boundary", "gross_observation_based_upstream_requirements"))
    double_counting = str(config.get("portfolio_policy", {}).get("double_counting_status", "not_network_deduplicated"))
    outcomes, contributions = [], []

    for sid in sorted(by_id):
        c, m = by_id[sid], map_by_id[sid]
        spend = _f(c["spend_eur"], f"spend {sid}")
        for outcome in sorted(satellites):
            satellite = satellites[outcome]
            direct_value, direct_unit = direct.get((sid, outcome), (np.nan, ""))
            unit = satellite["unit"] or direct_unit
            status, reason = "available", ""
            indirect = combined = multiplier = np.nan
            if m["calculation_eligibility"] != "eligible":
                status, reason = "held_out", m["mapping_reason"]
            else:
                shock = np.zeros(len(model["nodes"]))
                shock[model["index"][(m["figaro_country"], m["figaro_sector"])]] = spend / 1_000_000.0
                upstream = model["u"] @ shock
                intensity = np.asarray(satellite["intensity"])
                missing = (np.abs(upstream) > 1e-15) & np.isnan(intensity)
                if np.any(missing):
                    status = "held_out"
                    missing_nodes = [model["nodes"][i] for i in np.where(missing)[0][:5]]
                    reason = "satellite_coverage_missing_for_upstream_nodes:" + ",".join(f"{a}-{b}" for a, b in missing_nodes)
                else:
                    contrib = np.nan_to_num(intensity, nan=0.0) * upstream
                    indirect = float(np.sum(contrib))
                    multiplier = indirect / spend if spend else np.nan
                    combined = direct_value + indirect if np.isfinite(direct_value) else np.nan
                    denom = float(np.sum(np.abs(contrib)))
                    for i, value in enumerate(contrib):
                        if abs(value) <= 1e-15:
                            continue
                        country, sector = model["nodes"][i]
                        contributions.append({
                            "selection_id": sid, "outcome": outcome, "origin_country": country,
                            "origin_figaro_sector": sector, "indirect_value": _fmt(float(value)),
                            "unit": unit, "contribution_share": _fmt(abs(float(value)) / denom) if denom else "0",
                            **{k: meta[k] for k in ("source_product", "source_edition", "reference_year")},
                        })
            outcomes.append({
                "selection_id": sid, "spend_eur": _fmt(spend), "country": (c.get("country") or "").strip().upper(),
                "direct_model_sector": m["model_sector"], "figaro_source_country": m["figaro_country"],
                "figaro_source_sector": m["figaro_sector"], "outcome": outcome,
                "direct_value": "" if not np.isfinite(direct_value) else _fmt(direct_value),
                "indirect_value": "" if not np.isfinite(indirect) else _fmt(indirect),
                "combined_value": "" if not np.isfinite(combined) else _fmt(combined),
                "multiplier": "" if not np.isfinite(multiplier) else _fmt(multiplier),
                "unit": unit, **meta,
                "mapping_provenance": f"{m['mapping_version']}|{m['mapping_reason']}",
                "status": status, "holdout_reason": reason,
                "aggregation_boundary": boundary, "double_counting_status": double_counting,
            })

    outcomes.sort(key=lambda r: (r["selection_id"], r["outcome"]))
    contributions.sort(key=lambda r: (r["selection_id"], r["outcome"], r["origin_country"], r["origin_figaro_sector"]))
    portfolio = []
    total_spend = sum(_f(r["spend_eur"], "portfolio spend") for r in cohort)
    cohort_by_id = {r["selection_id"]: r for r in cohort}
    for outcome in sorted(satellites):
        rows = [r for r in outcomes if r["outcome"] == outcome]
        available = [r for r in rows if r["status"] == "available"]
        ids = {r["selection_id"] for r in available}
        mapped_spend = sum(_f(cohort_by_id[i]["spend_eur"], f"mapped spend {i}") for i in ids)
        direct_total = sum(_f(r["direct_value"], "direct") for r in available if r["direct_value"])
        indirect_total = sum(_f(r["indirect_value"], "indirect") for r in available if r["indirect_value"])
        portfolio.append({
            "outcome": outcome, "direct_total": _fmt(direct_total), "indirect_total": _fmt(indirect_total),
            "combined_total": _fmt(direct_total + indirect_total), "mapped_spend": _fmt(mapped_spend),
            "total_spend": _fmt(total_spend),
            "coverage_pct": _fmt(100 * mapped_spend / total_spend) if total_spend else "0",
            "unit": next((r["unit"] for r in available if r["unit"]), ""),
            "aggregation_boundary": boundary, "double_counting_status": double_counting,
            "caveats": str(config.get("portfolio_policy", {}).get("caveat", "gross upstream requirements may overlap across Tier-1 observations")),
        })

    qa = []
    dimensions = {
        "country": lambda sid: map_by_id[sid]["country"] or "BLANK",
        "figaro_sector": lambda sid: map_by_id[sid]["figaro_sector"] or "BLANK",
        "mapping_status": lambda sid: map_by_id[sid]["mapping_status"],
        "holdout_reason": lambda sid: map_by_id[sid]["mapping_reason"],
        "valuation_case": lambda sid: map_by_id[sid]["valuation_case"],
    }
    for outcome in sorted(satellites):
        rows_o = {r["selection_id"]: r for r in outcomes if r["outcome"] == outcome}
        for dim, getter in dimensions.items():
            groups = defaultdict(list)
            for sid in sorted(by_id):
                groups[getter(sid)].append(sid)
            for value in sorted(groups):
                ids = groups[value]
                spend = sum(_f(by_id[sid]["spend_eur"], "qa spend") for sid in ids)
                avail = [sid for sid in ids if rows_o[sid]["status"] == "available"]
                mapped = sum(_f(by_id[sid]["spend_eur"], "qa mapped") for sid in avail)
                indirect = sum(_f(rows_o[sid]["indirect_value"], "qa indirect") for sid in avail if rows_o[sid]["indirect_value"])
                qa.append({
                    "breakdown_dimension": dim, "breakdown_value": value, "outcome": outcome,
                    "observations": str(len(ids)), "mapped_observations": str(len(avail)),
                    "held_out_observations": str(len(ids)-len(avail)), "spend": _fmt(spend),
                    "mapped_spend": _fmt(mapped), "coverage_pct": _fmt(100*mapped/spend) if spend else "0",
                    "indirect_value": _fmt(indirect),
                    "unit": next((rows_o[s]["unit"] for s in avail if rows_o[s]["unit"]), ""),
                    "sensitivity_case": str(config.get("sensitivity_case", "primary_2023")),
                })

    eligible = {r["selection_id"] for r in mapping if r["calculation_eligibility"] == "eligible"}
    eligible_spend = sum(_f(r["spend_eur"], "eligible spend") for r in cohort if r["selection_id"] in eligible)
    summary = {
        "source": meta, "mapping_observations": len(mapping), "mapped_eligible_observations": len(eligible),
        "held_out_observations": len(mapping)-len(eligible), "total_spend_eur": _fmt(total_spend),
        "mapped_eligible_spend_eur": _fmt(eligible_spend),
        "mapped_eligible_spend_coverage_pct": _fmt(100*eligible_spend/total_spend) if total_spend else "0",
        "figaro_nodes": len(model["nodes"]),
        "leontief_inverse_max_reconciliation_error": _fmt(model["max_inverse_error"]),
        "upstream_operator": "L_minus_I", "direct_source": "accepted_SKO_038", "direct_recalculated": False,
        "aggregation_boundary": boundary, "double_counting_status": double_counting,
        "sensitivity_case": str(config.get("sensitivity_case", "primary_2023")),
        "trade_holdouts": sum(r["valuation_case"] == "trade_purchase_value_not_comparable" for r in mapping),
        "unmapped_observations": sum(r["mapping_status"] == "unmapped" for r in mapping),
    }
    summary["determinism_fingerprint"] = hashlib.sha256(
        json.dumps(
            {"mapping": mapping, "outcomes": outcomes, "contributions": contributions,
             "portfolio": portfolio, "qa": qa, "summary": summary},
            sort_keys=True, separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {"mapping": mapping, "outcomes": outcomes, "contributions": contributions, "portfolio": portfolio, "qa": qa, "summary": summary}
