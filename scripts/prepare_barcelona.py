"""Fail-fast preparation pipeline. Requires local verified source artifacts."""
import argparse, json
from pathlib import Path
from src.network.build_sumo_network import build_network
from src.network.qc_network import write_report
from src.network.district_edges import map_edges_to_districts
from src.network.validate_district_mapping import validate_mapping

p=argparse.ArgumentParser()
p.add_argument("--osm",required=True)
p.add_argument("--districts",required=True)
p.add_argument("--district-name-field",required=True)
p.add_argument("--out",default="outputs/network")
a=p.parse_args()
out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
net=out/"barcelona.passenger.net.xml"
build_network(a.osm,net,out/"build_manifest.json")
write_report(net,out/"network_qc.json")
m=map_edges_to_districts(net,a.districts,a.district_name_field)
m.to_csv(out/"district_edges.csv",index=False)
q=validate_mapping(m)
(out/"district_mapping_qc.json").write_text(json.dumps(q,indent=2,ensure_ascii=False),encoding="utf-8")
if not q["pass"]: raise SystemExit("District mapping QC failed.")
print("Barcelona network preparation QC passed.")
