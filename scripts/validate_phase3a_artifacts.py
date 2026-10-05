"""Final artifact completeness gate for the Barcelona Phase3A pipeline."""
from pathlib import Path
import csv
import json

FILES = [
    "artifacts/barcelona_b4_mappo/checkpoints/b4_mappo_final.npz",
    "artifacts/barcelona_b4_mappo/heldout_metrics.json",
    "artifacts/barcelona_b4_mappo/experiment_manifest.json",
    "artifacts/barcelona_b5_dcf/heldout_metrics.json",
    "artifacts/barcelona_b5_dcf/experiment_manifest.json",
    "artifacts/barcelona_b6_shapley/checkpoints/b6_mappo_dcf_shapley_final.npz",
    "artifacts/barcelona_b6_shapley/heldout_metrics.json",
    "artifacts/barcelona_b6_shapley/training_summary.json",
    "artifacts/barcelona_b6_shapley/experiment_manifest.json",
    "artifacts/barcelona_b6_lambda_sensitivity/lambda_sensitivity_summary.json",
    "artifacts/barcelona_b6_lambda_sensitivity/experiment_manifest.json",
    "artifacts/barcelona_b6_lambda_sensitivity/publication/lambda_sensitivity_table.csv",
    "artifacts/barcelona_b6_lambda_sensitivity/publication/lambda_paired_differences.csv",
    "artifacts/barcelona_b6_lambda_sensitivity/publication/publication_summary.json",
    "artifacts/barcelona_b6_lambda_sensitivity/publication/fig_lambda_pickup_wait.png",
    "artifacts/barcelona_b6_lambda_sensitivity/publication/fig_lambda_pickup_wait.pdf",
    "artifacts/barcelona_b6_lambda_sensitivity/publication/fig_lambda_empty_distance.png",
    "artifacts/barcelona_b6_lambda_sensitivity/publication/fig_lambda_energy.png",
    "artifacts/barcelona_b6_lambda_sensitivity/publication/fig_lambda_credit_coverage.png",
    "artifacts/barcelona_coordination_sensitivity/sensitivity_rows.json",
    "artifacts/barcelona_coordination_sensitivity/sensitivity_summary.json",
    "artifacts/barcelona_coordination_sensitivity/experiment_manifest.json",
    "artifacts/barcelona_coordination_sensitivity/publication/coordination_sensitivity_table.csv",
    "artifacts/barcelona_coordination_sensitivity/publication/publication_summary.json",
    "artifacts/barcelona_coordination_sensitivity/publication/fig_coordination_acceptance.png",
    "artifacts/barcelona_coordination_sensitivity/publication/fig_coordination_pickup_wait.png",
    "artifacts/barcelona_coordination_sensitivity/publication/fig_coordination_energy.png",
    "artifacts/barcelona_coordination_sensitivity/publication/fig_coordination_break_even_distribution.png",
    "artifacts/phase3a_b456/paired_b456.json",
    "artifacts/phase3a_b456/paired_b456.csv",
    "artifacts/phase3a_b456/summary_b456.json",
]

def main():
    missing=[]; empty=[]
    for name in FILES:
        p=Path(name)
        if not p.is_file():
            missing.append(name)
        elif p.stat().st_size == 0:
            empty.append(name)
    if missing or empty:
        raise SystemExit(f"artifact completeness failure; missing={missing}, empty={empty}")

    for name in FILES:
        p=Path(name)
        if p.suffix == ".json":
            json.loads(p.read_text(encoding="utf-8"))
        elif p.suffix == ".csv":
            with p.open(newline="",encoding="utf-8") as fh:
                rows=list(csv.reader(fh))
            if len(rows) < 2:
                raise SystemExit(f"CSV has no data rows: {name}")
    print(json.dumps({"artifact_gate":"pass","required_file_count":len(FILES)},indent=2))

if __name__ == "__main__":
    main()
