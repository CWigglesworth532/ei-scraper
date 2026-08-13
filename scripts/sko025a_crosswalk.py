#!/usr/bin/env python3
"""SKO-025A offline crosswalk recovery and profile fingerprint governance."""
from __future__ import annotations

import argparse, hashlib, json, re, unicodedata
from pathlib import Path
from urllib.parse import urlsplit
import pandas as pd
import yaml

REQUIRED = ["Organisation", "Country", "Website", "Airtable Record ID"]

def clean(value):
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)): return ""
    return " ".join(unicodedata.normalize("NFC", str(value)).split())

def hostname(value):
    text=clean(value)
    if not text: return ""
    parsed=urlsplit(text if "://" in text else "//"+text)
    host=(parsed.hostname or "").encode("idna").decode("ascii").lower()
    if host.startswith("www."): host=host[4:]
    if not host or not re.fullmatch(r"[a-z0-9.-]+",host): raise ValueError("invalid Website hostname")
    return host

def normalize_country(value, mapping):
    key=clean(value).casefold()
    return mapping.get(key, "")

def fingerprint(row, config):
    country=normalize_country(row.get("Country"), config["country_mapping"])
    if clean(row.get("Country")) and not country: raise ValueError("unknown Country")
    payload={"fingerprint_version":config["fingerprint_version"],"fields":[
        ["Organisation",clean(row.get("Organisation")).casefold()],
        ["Country",country], ["Website hostname",hostname(row.get("Website"))]]}
    raw=json.dumps(payload,ensure_ascii=False,separators=(",",":"),sort_keys=False).encode()
    return "apf1:"+hashlib.sha256(raw).hexdigest()

def build(snapshot, canonical, config, snapshot_sha256):
    missing=[c for c in REQUIRED if c not in snapshot]
    if missing: raise ValueError(f"missing columns: {missing}")
    ids=snapshot["Airtable Record ID"].fillna("").astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any() or not ids.str.fullmatch(r"rec[A-Za-z0-9]+").all():
        raise ValueError("Airtable Record IDs must be nonblank, unique and structurally valid")
    canon=canonical.copy()
    canon["_name"]=canon["canonical_name"].map(clean).str.casefold()
    canon["_country"]=canon["country"].map(lambda v: normalize_country(v,config["country_mapping"]))
    rows=[]
    for source in snapshot.to_dict("records"):
        rid=clean(source["Airtable Record ID"]); name=clean(source["Organisation"]).casefold()
        country=normalize_country(source["Country"],config["country_mapping"])
        findings=[]
        try: fp=fingerprint(source,config)
        except ValueError as exc: fp=""; findings.append(str(exc))
        matches=canon.loc[canon["_name"].eq(name) & canon["_country"].eq(country)] if country else canon.iloc[0:0]
        entity=clean(matches.iloc[0]["entity_id"]) if len(matches)==1 else ""
        state="proposed_owner_review" if entity and fp else "unresolved"
        if len(matches)>1: findings.append("ambiguous exact canonical identity")
        elif not entity: findings.append("no unique accepted canonical identity")
        rows.append({"crosswalk_id":"xw_"+hashlib.sha256((snapshot_sha256+"|"+rid).encode()).hexdigest()[:24],
          "crosswalk_schema_version":config["crosswalk_schema_version"],"entity_id":entity,
          "airtable_record_id":rid,"profile_relationship_type":"","counting_entity_id":entity,
          "counting_mapping_source":"default_self" if entity else "unresolved","crosswalk_status":state,
          "fingerprint_version":config["fingerprint_version"],"approved_fingerprint":"",
          "proposed_fingerprint":fp,"airtable_snapshot_sha256":snapshot_sha256,
          "identity_decision_type":"deterministic_accepted_identity_reuse" if entity else "unresolved",
          "review_findings":"; ".join(findings),"reviewed_by":"","reviewed_at":""})
    return pd.DataFrame(rows).sort_values("airtable_record_id").reset_index(drop=True)

def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--snapshot",type=Path,required=True); p.add_argument("--canonical",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    cfg=yaml.safe_load(a.config.read_text()); snap=pd.read_csv(a.snapshot,dtype=str,keep_default_na=False); canonical=pd.read_parquet(a.canonical)
    out=build(snap,canonical,cfg,sha256(a.snapshot)); a.output_dir.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.output_dir/"proposed_crosswalk.csv",index=False)
    summary={"snapshot_sha256":sha256(a.snapshot),"snapshot_rows":len(snap),"proposed_owner_review":int(out.crosswalk_status.eq("proposed_owner_review").sum()),"unresolved":int(out.crosswalk_status.eq("unresolved").sum()),"approved":0,"counting_default_self":int(out.counting_mapping_source.eq("default_self").sum())}
    (a.output_dir/"aggregate_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True))
    print(json.dumps(summary,sort_keys=True))
if __name__=="__main__": main()
