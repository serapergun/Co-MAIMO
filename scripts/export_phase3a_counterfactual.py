#!/usr/bin/env python3
"""Export Phase3A SUMO coalition runs into the strict counterfactual contract."""

import argparse
import json
from pathlib import Path


REQUIRED_SUMMARY_FIELDS = (
    "completed",
    "requests",
    "mean_pickup_wait_s",
    "empty_km",
    "net_energy_kWh",
)


def record_from_summary(coalition, summary):
    missing = [k for k in REQUIRED_SUMMARY_FIELDS if k not in summary]
    if missing:
        raise ValueError(f"Operational summary missing fields: {missing}")
    if int(summary.get("unresolved", 0)) != 0:
        raise ValueError("Coalition run has unresolved requests")
    if int(summary.get("teleports", 0)) != 0:
        raise ValueError("Coalition run contains taxi teleport recovery")
    if bool(summary.get("horizon_exceeded", False)):
        raise ValueError("Coalition run exceeded the simulation horizon")
    if int(summary["completed"]) != int(summary["requests"]):
        raise ValueError("Coalition run did not complete all requests")
    return {
        "coalition": list(coalition),
        "completed": int(summary["completed"]),
        "requests": int(summary["requests"]),
        "mean_pickup_wait_s": float(summary["mean_pickup_wait_s"]),
        "empty_km": float(summary["empty_km"]),
        "net_energy_kWh": float(summary["net_energy_kWh"]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coalition", nargs="+", required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    summary = json.loads(a.summary.read_text(encoding="utf-8"))
    record = record_from_summary(a.coalition, summary)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
