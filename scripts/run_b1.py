import argparse,json
from pathlib import Path
from src.simulation.traci_runner import run
from src.baselines.b1_dispatch import dispatch
from src.evaluation.parse_tripinfo import parse_tripinfo
from src.evaluation.operational_metrics import vehicle_metrics
p=argparse.ArgumentParser(); p.add_argument("--sumocfg",required=True); p.add_argument("--out",required=True); p.add_argument("--max-time",type=float)
a=p.parse_args(); art=run(a.sumocfg,a.out,dispatch,a.max_time)
df=parse_tripinfo(art["tripinfo"]); metrics=vehicle_metrics(df)
Path(a.out,"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8"); df.to_csv(Path(a.out,"vehicle_tripinfo.csv"),index=False)
print(json.dumps(metrics,indent=2))
