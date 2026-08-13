import sys,unittest
from pathlib import Path
import pandas as pd,yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
import sko025a_owner_review as r
CFG=yaml.safe_load((ROOT/"config/airtable_profile_fingerprint_v1.yaml").read_text())
def tables():
 return {"canonical_entities":pd.DataFrame([{"entity_id":"e1","canonical_name":"Alpha Co","country":"GB"},{"entity_id":"e2","canonical_name":"Beta Co","country":"GB"}]),"entity_aliases":pd.DataFrame([{"entity_id":"e1","alias_name":"Alpha Trading","country":"GB","verification_status":"approved","review_status":"approved"}]),"supplier_entity_links":pd.DataFrame([{"matched_entity_id":"e2","supplier_name_original":"Beta Supplier","supplier_country":"GB","resolution_status":"resolved"}]),"source_records":pd.DataFrame([{"entity_id":"e1","source_url":"https://alpha.example"}]),"entity_relationships":pd.DataFrame(columns=["subject_entity_id","object_entity_id","relationship_type","relationship_status"]),"entity_identifiers":pd.DataFrame()}
class Tests(unittest.TestCase):
 def test_exact_accepted_alias_is_tier_a_but_not_approved(self):
  snap=pd.DataFrame([{"Airtable Record ID":"recA1","Organisation":"Alpha Trading","Country":"United Kingdom","Website":""}]); out=r.build(snap,tables(),CFG)
  self.assertEqual(out.iloc[0].candidate_tier,"Tier A"); self.assertEqual(out.iloc[0].candidate_entity_id,"e1"); self.assertEqual(out.iloc[0].owner_decision,"")
 def test_similarity_alone_does_not_create_tier_b(self):
  snap=pd.DataFrame([{"Airtable Record ID":"recA1","Organisation":"Alpha Tradng","Country":"United Kingdom","Website":""}]); out=r.build(snap,tables(),CFG); self.assertEqual(out.iloc[0].candidate_tier,"Unresolved")
 def test_name_and_hostname_create_tier_b(self):
  snap=pd.DataFrame([{"Airtable Record ID":"recA1","Organisation":"Alpha C","Country":"United Kingdom","Website":"www.alpha.example/path"}]); out=r.build(snap,tables(),CFG); self.assertEqual(out.iloc[0].candidate_tier,"Tier B")
 def test_ambiguous_exact_candidates_are_all_retained(self):
  t=tables(); t["canonical_entities"].loc[1,"canonical_name"]="Alpha Co"; snap=pd.DataFrame([{"Airtable Record ID":"recA1","Organisation":"Alpha Co","Country":"United Kingdom","Website":""}]); out=r.build(snap,t,CFG); self.assertEqual(len(out),2); self.assertTrue(out.candidate_tier.eq("Tier C").all())
if __name__=="__main__":unittest.main()
