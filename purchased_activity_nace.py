#!/usr/bin/env python3
"""SKO-035 governed purchased-activity/NACE decision materialisation."""
from __future__ import annotations

import argparse, csv, hashlib, json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping
import yaml

EVIDENCE_FIELDS = ["purchased_activity_evidence_id","subject_type","subject_id","evidence_type","evidence_reference","evidence_date","activity_scheme","activity_version","activity_code_raw","activity_description_raw","evidence_authority","evidence_fingerprint","schema_version"]
ASSERTION_FIELDS = ["activity_assignment_id","subject_type","subject_id","entity_id","client_reference","spend","spend_year","supplier_country","purchased_activity_description","nace_scheme","nace_version","nace_code","nace_level","nace_description","evidence_ids","evidence_dates","assignment_method","assignment_confidence","contract_specific","multi_activity","spend_split","allocation_share","allocated_spend","rationale","review_status","reviewer_decision_reference","materiality_state","review_escalation_state","conflicting_evidence","source_version","decision_version","generated_at","assertion_fingerprint","schema_version"]
QA_FIELDS = ["selected_observations","observations_with_activity_assignment","high_confidence","medium_confidence","low_confidence","unresolved","broader_level_assignment","contract_specific","multi_activity","spend_split","materiality_escalated","conflicting_evidence","procurement_category_only_evidence","authoritative_or_strong_activity_evidence","allocation_reconciliation_failures","invalid_nace_version_failures"]
TRUE = {"1","true","yes"}

def clean(v: Any) -> str: return "" if v is None else str(v).strip()
def canon(v: Any) -> str: return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",",":"))
def digest(v: Any) -> str: return hashlib.sha256(canon(v).encode()).hexdigest()
def read_csv(path: Path) -> list[dict[str,str]]:
    with path.open(newline="", encoding="utf-8-sig") as f: return [dict(r) for r in csv.DictReader(f)]
def _bool(v: Any) -> bool: return clean(v).casefold() in TRUE
def _decimal(v: Any, label: str) -> Decimal:
    try: result=Decimal(clean(v))
    except InvalidOperation as exc: raise ValueError(f"Invalid {label}") from exc
    return result
def _require(rows: list[dict[str,str]], fields: set[str], label: str) -> None:
    missing=fields-(set(rows[0]) if rows else set())
    if missing: raise ValueError(f"{label} missing columns: {sorted(missing)}")

def load_config(path: Path) -> dict[str,Any]:
    with path.open(encoding="utf-8") as f: cfg=yaml.safe_load(f)
    required={"schema_version","nace_scheme","nace_version","nace_catalogue","evidence_rules","confidence_rules","crosswalks","high_materiality_values"}
    if not isinstance(cfg,dict) or required-set(cfg): raise ValueError("Configuration contract incomplete")
    if cfg["nace_scheme"]!="NACE" or not clean(cfg["nace_version"]): raise ValueError("Explicit NACE scheme/version required")
    for code,item in cfg["nace_catalogue"].items():
        extras=set(item)-{"level","description"}
        if extras and all(item[key] is None for key in extras):
            item["description"] += ", " + ", ".join(sorted(extras))
            for key in extras: del item[key]
        if set(item)!={"level","description"}: raise ValueError(f"Invalid NACE catalogue entry {code}")
    for key,item in cfg["crosswalks"].items():
        if set(item)!={"source_scheme","source_version","source_code","target_scheme","target_version","target_code","correspondence_version"}: raise ValueError(f"Invalid crosswalk {key}")
        if item["target_scheme"]!=cfg["nace_scheme"] or item["target_version"]!=cfg["nace_version"]: raise ValueError("Crosswalk target mismatch")
    return cfg

def materialise(observations: list[dict[str,str]], evidence_rows: list[dict[str,str]], decisions: list[dict[str,str]], *, config: Mapping[str,Any], generated_at: str) -> dict[str,Any]:
    _require(observations,{"selected","subject_type","subject_id","entity_id","client_reference","spend","spend_year","supplier_country","materiality"},"observations")
    _require(evidence_rows,{"evidence_key","subject_id","evidence_type","evidence_reference","evidence_date","activity_scheme","activity_version","activity_code_raw","activity_description_raw"},"evidence")
    _require(decisions,{"decision_key","subject_id","purchased_activity_description","nace_scheme","nace_version","nace_code","nace_level","nace_description","evidence_keys","assignment_method","assignment_confidence","contract_specific","multi_activity","spend_split","allocation_share","rationale","review_status","reviewer_decision_reference","conflicting_evidence","source_version","decision_version","crosswalk_id"},"decisions")
    if not clean(generated_at): raise ValueError("generated_at required")
    selected={}
    for r in observations:
        if not _bool(r["selected"]): continue
        sid=clean(r["subject_id"]); st=clean(r["subject_type"])
        if st not in {"canonical_entity","supplier_observation"} or not sid: raise ValueError("Invalid selected subject")
        if st=="canonical_entity" and not clean(r["entity_id"]): raise ValueError("canonical_entity requires existing entity_id")
        normalized={k:clean(v) for k,v in r.items()}
        if sid in selected and selected[sid]!=normalized: raise ValueError(f"Conflicting observation {sid}")
        selected[sid]=normalized
    ev_by_key={}; evidence=[]
    for raw in evidence_rows:
        sid=clean(raw["subject_id"])
        if sid not in selected: continue
        key=clean(raw["evidence_key"]); et=clean(raw["evidence_type"]); rule=config["evidence_rules"].get(et)
        if not key or not rule: raise ValueError("Unknown evidence type or missing key")
        core={"subject_type":selected[sid]["subject_type"],"subject_id":sid,"evidence_type":et,"evidence_reference":clean(raw["evidence_reference"]),"evidence_date":clean(raw["evidence_date"]),"activity_scheme":clean(raw["activity_scheme"]),"activity_version":clean(raw["activity_version"]),"activity_code_raw":clean(raw["activity_code_raw"]),"activity_description_raw":clean(raw["activity_description_raw"]),"evidence_authority":rule["authority"],"schema_version":"sko-035-purchased-activity-evidence-v1"}
        fp=digest(core); row={"purchased_activity_evidence_id":"paev_"+fp[:24],**core,"evidence_fingerprint":fp}
        if key in ev_by_key and ev_by_key[key]!=row: raise ValueError(f"Conflicting evidence key {key}")
        ev_by_key[key]=row
    evidence=sorted({r["evidence_fingerprint"]:r for r in ev_by_key.values()}.values(),key=lambda r:r["purchased_activity_evidence_id"])
    ev_ids={k:v["purchased_activity_evidence_id"] for k,v in ev_by_key.items()}
    grouped={sid:[] for sid in selected}
    seen={}
    for d in decisions:
        sid=clean(d["subject_id"])
        if sid not in selected: continue
        key=clean(d["decision_key"]); normalized={k:clean(v) for k,v in d.items()}
        if key in seen:
            if seen[key]!=normalized: raise ValueError(f"Conflicting decision duplicate {key}")
            continue
        seen[key]=normalized; grouped[sid].append(normalized)
    assertions=[]; qa={k:0 for k in QA_FIELDS}; qa["selected_observations"]=len(selected)
    high_materiality={clean(v) for v in config["high_materiality_values"]}
    for sid, obs in sorted(selected.items()):
        ds=grouped[sid]
        if not ds: continue
        shares=[]
        for d in ds:
            share=_decimal(d["allocation_share"],"allocation_share")
            if share<=0 or share>1: raise ValueError("allocation_share must be > 0 and <= 1")
            shares.append(share)
        if sum(shares)!=Decimal("1"): raise ValueError(f"Allocation shares do not reconcile for {sid}")
        if len(ds)>1 and not all(_bool(d["spend_split"]) and _bool(d["multi_activity"]) for d in ds): raise ValueError("Multiple activities require evidenced spend split")
        qa["observations_with_activity_assignment"]+=1
        ev_for_subject=[ev_by_key[k] for d in ds for k in clean(d["evidence_keys"]).split("|") if k]
        authorities={e["evidence_authority"] for e in ev_for_subject}; types={e["evidence_type"] for e in ev_for_subject}
        if authorities & {"authoritative","strong"}: qa["authoritative_or_strong_activity_evidence"]+=1
        if types and types=={"procurement_category"}: qa["procurement_category_only_evidence"]+=1
        if any(_bool(d["contract_specific"]) for d in ds): qa["contract_specific"]+=1
        if any(_bool(d["multi_activity"]) for d in ds): qa["multi_activity"]+=1
        if any(_bool(d["spend_split"]) for d in ds): qa["spend_split"]+=1
        if any(_bool(d["conflicting_evidence"]) for d in ds): qa["conflicting_evidence"]+=1
        escalated=False
        for d,share in zip(ds,shares):
            keys=[k for k in d["evidence_keys"].split("|") if k]
            if any(k not in ev_by_key or ev_by_key[k]["subject_id"]!=sid for k in keys): raise ValueError("Unknown or cross-subject evidence key")
            types={ev_by_key[k]["evidence_type"] for k in keys}; auth={ev_by_key[k]["evidence_authority"] for k in keys}
            confidence=d["assignment_confidence"]
            if confidence not in {"High","Medium","Low"}: raise ValueError("Invalid confidence")
            if confidence=="High" and not (auth & {"authoritative","strong"}): raise ValueError("Weak evidence cannot be High confidence")
            if _bool(d["contract_specific"]) and "contract_service" not in types: raise ValueError("contract_specific requires contract evidence")
            code=d["nace_code"]
            if code:
                item=config["nace_catalogue"].get(code)
                if d["nace_scheme"]!=config["nace_scheme"] or d["nace_version"]!=config["nace_version"] or not item or str(item["level"])!=d["nace_level"] or item["description"]!=d["nace_description"]: raise ValueError("Invalid NACE/version/code/level/description")
                national=[ev_by_key[k] for k in keys if ev_by_key[k]["activity_scheme"] not in {"","NACE"} and ev_by_key[k]["activity_code_raw"]]
                if national and d["assignment_method"]=="governed_crosswalk":
                    cw=config["crosswalks"].get(d["crosswalk_id"])
                    if not cw or not any((e["activity_scheme"],e["activity_version"],e["activity_code_raw"])==(cw["source_scheme"],cw["source_version"],cw["source_code"]) for e in national) or cw["target_code"]!=code: raise ValueError("Explicit governed crosswalk required")
            elif any((d[x] for x in ("nace_scheme","nace_version","nace_level","nace_description"))): raise ValueError("Unresolved NACE fields must be blank")
            weak=not bool(auth & {"authoritative","strong"}) or confidence=="Low" or _bool(d["conflicting_evidence"])
            escalation="required" if obs["materiality"] in high_materiality and weak else "not_required"
            escalated |= escalation=="required"
            core={"subject_type":obs["subject_type"],"subject_id":sid,"entity_id":obs["entity_id"],"client_reference":obs["client_reference"],"spend":obs["spend"],"spend_year":obs["spend_year"],"supplier_country":obs["supplier_country"],"purchased_activity_description":d["purchased_activity_description"],"nace_scheme":d["nace_scheme"],"nace_version":d["nace_version"],"nace_code":code,"nace_level":d["nace_level"],"nace_description":d["nace_description"],"evidence_ids":"|".join(sorted(ev_ids[k] for k in keys)),"evidence_dates":"|".join(sorted({ev_by_key[k]["evidence_date"] for k in keys})),"assignment_method":d["assignment_method"],"assignment_confidence":confidence,"contract_specific":str(_bool(d["contract_specific"])).lower(),"multi_activity":str(_bool(d["multi_activity"])).lower(),"spend_split":str(_bool(d["spend_split"])).lower(),"allocation_share":format(share,"f"),"allocated_spend":format(_decimal(obs["spend"],"spend")*share,"f"),"rationale":d["rationale"],"review_status":d["review_status"],"reviewer_decision_reference":d["reviewer_decision_reference"],"materiality_state":obs["materiality"],"review_escalation_state":escalation,"conflicting_evidence":str(_bool(d["conflicting_evidence"])).lower(),"source_version":d["source_version"],"decision_version":d["decision_version"],"generated_at":generated_at,"schema_version":config["schema_version"]}
            fp=digest(core); assertions.append({"activity_assignment_id":"paas_"+fp[:24],**core,"assertion_fingerprint":fp})
            qa[confidence.casefold()+"_confidence"]+=1
            if not code: qa["unresolved"]+=1
            elif int(d["nace_level"])<4: qa["broader_level_assignment"]+=1
        if escalated: qa["materiality_escalated"]+=1
    return {"evidence":evidence,"assertions":sorted(assertions,key=lambda r:r["activity_assignment_id"]),"qa":qa}

def write_csv(rows:list[dict[str,Any]], path:Path, fields:list[str])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(rows)
def main()->int:
    p=argparse.ArgumentParser();
    for x in ("config","observations","evidence","decisions","evidence-output","assertion-output","qa-output"): p.add_argument("--"+x,type=Path,required=True)
    p.add_argument("--generated-at",required=True); a=p.parse_args()
    result=materialise(read_csv(a.observations),read_csv(a.evidence),read_csv(a.decisions),config=load_config(a.config),generated_at=a.generated_at)
    write_csv(result["evidence"],a.evidence_output,EVIDENCE_FIELDS); write_csv(result["assertions"],a.assertion_output,ASSERTION_FIELDS); a.qa_output.write_text(json.dumps(result["qa"],indent=2)+"\n",encoding="utf-8"); return 0
if __name__=="__main__": raise SystemExit(main())
