#!/usr/bin/env python3
"""Build a private, offline SKO-025A owner-review candidate pack."""
from __future__ import annotations
import argparse, hashlib, json, re
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd
import yaml
from sko025a_crosswalk import clean, hostname, normalize_country

def norm_name(value): return re.sub(r"[^a-z0-9]+", "", clean(value).casefold())
def load_tables(base): return {n:pd.read_parquet(base/f"{n}.parquet") for n in ["canonical_entities","entity_aliases","entity_identifiers","entity_relationships","source_records","supplier_entity_links"]}

def evidence_names(tables, country_map):
    rows=[]
    for r in tables["canonical_entities"].to_dict("records"):
        rows.append((clean(r["entity_id"]),clean(r["canonical_name"]),clean(r["country"]),"canonical_name"))
    for r in tables["entity_aliases"].to_dict("records"):
        if clean(r.get("verification_status")).casefold() in {"verified","accepted","approved"} or clean(r.get("review_status")).casefold() in {"reviewed","approved","accepted"}:
            rows.append((clean(r["entity_id"]),clean(r["alias_name"]),clean(r["country"]),"accepted_alias"))
    for r in tables["supplier_entity_links"].to_dict("records"):
        if clean(r.get("acceptance_status")).casefold() in {"reviewed_confirmed","accepted","verified","directory_approved","reporting_approved"} and clean(r.get("review_required")).casefold() not in {"true","1","yes"} and clean(r.get("matched_entity_id")):
            rows.append((clean(r["matched_entity_id"]),clean(r["supplier_name_original"]),clean(r["supplier_country"]),"accepted_supplier_link"))
    out=[]
    for entity,name,country,kind in rows:
        iso=normalize_country(country,country_map) or clean(country).upper()
        if entity and name: out.append({"entity_id":entity,"name":name,"norm":norm_name(name),"country":iso,"kind":kind})
    return out

def build(snapshot,tables,config):
    entities=tables["canonical_entities"].set_index("entity_id",drop=False)
    names=evidence_names(tables,config["country_mapping"])
    host_by_entity={}
    for r in tables["source_records"].to_dict("records"):
        try: host=hostname(r.get("source_url"))
        except ValueError: host=""
        if host and clean(r.get("entity_id")): host_by_entity.setdefault(clean(r["entity_id"]),set()).add(host)
    relationships={}
    approved_relationships=set(config["relationship_types"])
    for r in tables["entity_relationships"].to_dict("records"):
        if clean(r.get("relationship_status")).casefold() in {"accepted","approved","active"} and clean(r.get("relationship_type")) in approved_relationships:
            relationships.setdefault(clean(r["subject_entity_id"]),[]).append(r)
    output=[]
    for row in snapshot.to_dict("records"):
        rid=clean(row["Airtable Record ID"]); org=clean(row["Organisation"]); n=norm_name(org)
        country=normalize_country(row.get("Country"),config["country_mapping"])
        try: web=hostname(row.get("Website"))
        except ValueError: web=""
        exact={x["entity_id"] for x in names if x["norm"]==n and country and x["country"]==country}
        exact_signals={e:sorted({x["kind"] for x in names if x["entity_id"]==e and x["norm"]==n and x["country"]==country}) for e in exact}
        scored=[]
        for e in {x["entity_id"] for x in names if country and x["country"]==country}:
            sims=[SequenceMatcher(None,n,x["norm"]).ratio() for x in names if x["entity_id"]==e]
            sim=max(sims or [0]); host_match=bool(web and web in host_by_entity.get(e,set()))
            if sim>=0.86 or host_match: scored.append((e,sim,host_match))
        candidates=[]
        if len(exact)==1:
            e=next(iter(exact)); candidates=[(e,1.0,bool(web and web in host_by_entity.get(e,set())),"Tier A",exact_signals[e])]
        elif len(exact)>1:
            candidates=[(e,1.0,bool(web and web in host_by_entity.get(e,set())),"Tier C",exact_signals[e]) for e in sorted(exact)]
        else:
            strong=[x for x in scored if x[1]>=0.90 and x[2]]
            plausible=[x for x in scored if x[1]>=0.86]
            if len(strong)==1: candidates=[(*strong[0],"Tier B",["name_similarity","website_hostname","same_country"])]
            elif len(plausible)>1: candidates=[(*x,"Tier C",["name_similarity","same_country"]+( ["website_hostname"] if x[2] else [])) for x in plausible]
        if not candidates:
            candidates=[("",0.0,False,"Unresolved",["no_credible_existing_candidate"])]
        for e,score,host_match,tier,signals in candidates:
            canonical=entities.loc[e] if e and e in entities.index else {}
            rels=relationships.get(e,[]) if e else []
            relationship=clean(rels[0].get("relationship_type")) if len(rels)==1 else ""
            grouped=clean(rels[0].get("object_entity_id")) if len(rels)==1 else ""
            output.append({"Airtable Record ID":rid,"Organisation":org,"Country":clean(row.get("Country")),"Website hostname":web,
              "candidate_entity_id":e,"canonical_name":clean(canonical.get("canonical_name")) if e else "","candidate_tier":tier,
              "evidence_signals":";".join(signals),"name_similarity":round(score,4),"website_hostname_match":host_match,
              "review_rationale":"Candidate suggestion only; explicit owner identity and relationship decision required.",
              "proposed_profile_relationship_type":relationship,"proposed_counting_entity_id":grouped,
              "accepted_relationship_signal":bool(rels),"owner_decision":"","owner_rationale":""})
    return pd.DataFrame(output).sort_values(["Airtable Record ID","candidate_tier","candidate_entity_id"]).reset_index(drop=True)

def summary(review,snapshot_rows):
    profile=lambda tier: review.loc[review.candidate_tier.eq(tier),"Airtable Record ID"].nunique()
    per=review.groupby("Airtable Record ID").candidate_entity_id.apply(lambda s:sum(bool(clean(x)) for x in s))
    chosen=review.loc[review.candidate_tier.isin(["Tier A","Tier B"]) & review.candidate_entity_id.ne("")]
    standalone=int(chosen.loc[chosen.proposed_counting_entity_id.eq(""),"Airtable Record ID"].nunique())
    grouped=int(chosen.loc[chosen.proposed_counting_entity_id.ne(""),"Airtable Record ID"].nunique())
    return {"snapshot_profiles":snapshot_rows,"tier_a_profiles":profile("Tier A"),"tier_b_profiles":profile("Tier B"),"ambiguous_profiles":profile("Tier C"),"unresolved_profiles":profile("Unresolved"),"profiles_with_multiple_candidates":int((per>1).sum()),"accepted_relationship_signal_profiles":int(review.loc[review.accepted_relationship_signal,"Airtable Record ID"].nunique()),"proposed_standalone_self_count_mappings":standalone,"proposed_grouped_counting_mappings":grouped,"canonical_decision_required":profile("Unresolved"),"collision_or_contradiction_profiles":int((per>1).sum())}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--snapshot",type=Path,required=True); p.add_argument("--canonical-dir",type=Path,required=True); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    cfg=yaml.safe_load(a.config.read_text()); snap=pd.read_csv(a.snapshot,dtype=str,keep_default_na=False); review=build(snap,load_tables(a.canonical_dir),cfg); a.output_dir.mkdir(parents=True,exist_ok=True)
    review.to_csv(a.output_dir/"owner_review_candidates.csv",index=False); stats=summary(review,len(snap)); (a.output_dir/"owner_review_summary.json").write_text(json.dumps(stats,indent=2,sort_keys=True)); print(json.dumps(stats,sort_keys=True))
if __name__=="__main__": main()
