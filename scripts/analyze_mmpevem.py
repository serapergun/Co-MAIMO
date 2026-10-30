import argparse, json
from pathlib import Path
from src.evaluation.parse_tripinfo import parse_tripinfo
from src.evaluation.energy_metrics import physical_energy_metrics

p=argparse.ArgumentParser()
p.add_argument("--tripinfo",required=True)
p.add_argument("--out",required=True)
a=p.parse_args()
metrics=physical_energy_metrics(parse_tripinfo(a.tripinfo))
out=Path(a.out)
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
print(json.dumps(metrics,indent=2))
