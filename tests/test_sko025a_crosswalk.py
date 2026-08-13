import sys, unittest
from pathlib import Path
import pandas as pd, yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
import sko025a_crosswalk as s
CFG=yaml.safe_load((ROOT/"config/airtable_profile_fingerprint_v1.yaml").read_text())
class Tests(unittest.TestCase):
 def row(self,**kw): return {"Organisation":" Example  Org ","Country":"United Kingdom","Website":"HTTPS://WWW.Example.COM/path?q=1","Airtable Record ID":"recABC123",**kw}
 def test_fingerprint_normalization_and_material_changes(self):
  a=s.fingerprint(self.row(),CFG); self.assertEqual(a,s.fingerprint(self.row(Website="example.com/other"),CFG))
  self.assertNotEqual(a,s.fingerprint(self.row(Organisation="Other"),CFG)); self.assertNotEqual(a,s.fingerprint(self.row(Country="France"),CFG))
 def test_unknown_country_and_bad_url_block(self):
  for row in (self.row(Country="Atlantis"),self.row(Website="not a host !!")):
   with self.assertRaises(ValueError): s.fingerprint(row,CFG)
 def test_proposals_never_auto_approve_and_default_counting(self):
  snap=pd.DataFrame([self.row(),self.row(**{"Airtable Record ID":"recDEF456","Organisation":"Unknown"})])
  canon=pd.DataFrame([{"entity_id":"ent1","canonical_name":"Example Org","country":"GB"}])
  out=s.build(snap,canon,CFG,"a"*64); self.assertFalse(out.crosswalk_status.eq("approved").any())
  self.assertEqual(out.iloc[0].counting_entity_id,"ent1"); self.assertEqual(out.iloc[0].counting_mapping_source,"default_self")
 def test_duplicate_record_ids_block(self):
  snap=pd.DataFrame([self.row(),self.row()]); canon=pd.DataFrame(columns=["entity_id","canonical_name","country"])
  with self.assertRaises(ValueError): s.build(snap,canon,CFG,"a"*64)
 def test_relationship_vocabulary(self):
  self.assertEqual(set(CFG["relationship_types"]),{"group","division","service_line","brand","establishment","operating_unit"})
if __name__=="__main__": unittest.main()
