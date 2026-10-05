#!/usr/bin/env python3
"""Analyze a complete Phase3A coalition dataset."""

import argparse
import json
from pathlib import Path

from src.coalition.counterfactual_adapter import characteristic_values_from_counterfactuals
from src.coalition.game import allocation_qc, nucleolus, shapley_value
from src.coalition.operational_valuation import OperationalEvidence, ValuationWeights


def _label(s):
    return "{}" if not s else "{" + ",".join(sorted(s)) + "}"


def analyze_dataset(dataset, reference, weights=ValuationWeights()):
    if not dataset.get("complete", False):
        raise ValueError("Phase3A dataset is not marked complete")
    players = tuple(dataset["players"])
    expected_count = (1 << len(players)) - 1
    if int(dataset.get("coalition_count", -1)) != expected_count:
        raise ValueError("Phase3A coalition count does not match player set")

    values = characteristic_values_from_counterfactuals(
        players, dataset["records"], reference, weights
    )
    shapley = shapley_value(players, values)
    shapley_qc = allocation_qc(players, values, shapley)

    result = {
        "players": list(players),
        "weights": {
            "pickup_wait": weights.normalized()[0],
            "empty_distance": weights.normalized()[1],
            "energy": weights.normalized()[2],
        },
        "characteristic_values": {
            _label(s): float(v)
            for s, v in sorted(values.items(), key=lambda kv: (len(kv[0]), sorted(kv[0])))
        },
        "shapley": shapley,
        "shapley_qc": shapley_qc,
    }

    try:
        nuc, nuc_qc = nucleolus(players, values)
        result["nucleolus"] = nuc
        result["nucleolus_qc"] = {
            **nuc_qc,
            "tight_coalitions": [_label(s) for s in nuc_qc["tight_coalitions"]],
        }
    except RuntimeError as exc:
        result["nucleolus"] = None
        result["nucleolus_qc"] = {"computed": False, "reason": str(exc)}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--w-pickup", type=float, default=1.0)
    p.add_argument("--w-empty", type=float, default=1.0)
    p.add_argument("--w-energy", type=float, default=1.0)
    a = p.parse_args()

    dataset = json.loads(a.dataset.read_text(encoding="utf-8"))
    ref = json.loads(a.reference.read_text(encoding="utf-8"))
    reference = OperationalEvidence(
        completed=int(ref["completed"]),
        requests=int(ref["requests"]),
        mean_pickup_wait_s=float(ref["mean_pickup_wait_s"]),
        empty_km=float(ref["empty_km"]),
        net_energy_kWh=float(ref["net_energy_kWh"]),
    )
    weights = ValuationWeights(a.w_pickup, a.w_empty, a.w_energy)
    result = analyze_dataset(dataset, reference, weights)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
