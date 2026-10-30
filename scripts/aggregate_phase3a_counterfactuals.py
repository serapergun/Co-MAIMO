#!/usr/bin/env python3
"""Aggregate validated Phase3A coalition records into a complete dataset."""

import argparse
import json
from itertools import combinations
from pathlib import Path


def expected_coalitions(players):
    players = tuple(str(p) for p in players)
    return {
        frozenset(c)
        for r in range(1, len(players) + 1)
        for c in combinations(players, r)
    }


def aggregate_records(players, records):
    players = tuple(str(p) for p in players)
    if len(set(players)) != len(players):
        raise ValueError("Player labels must be unique")
    expected = expected_coalitions(players)
    by_key = {}
    for record in records:
        if "coalition" not in record:
            raise ValueError("Record missing coalition")
        key = frozenset(str(p) for p in record["coalition"])
        if not key or key not in expected:
            raise ValueError(f"Unknown or invalid coalition: {sorted(key)}")
        if len(key) != len(record["coalition"]):
            raise ValueError(f"Duplicate player inside coalition: {record['coalition']}")
        if key in by_key:
            raise ValueError(f"Duplicate coalition record: {sorted(key)}")
        by_key[key] = record

    missing = expected - set(by_key)
    if missing:
        labels = [sorted(s) for s in sorted(missing, key=lambda x: (len(x), sorted(x)))]
        raise ValueError(f"Incomplete Phase3A coalition set; missing={labels}")

    ordered = sorted(
        by_key.values(),
        key=lambda r: (len(r["coalition"]), sorted(str(p) for p in r["coalition"])),
    )
    return {
        "players": list(players),
        "coalition_count": len(ordered),
        "expected_coalition_count": (1 << len(players)) - 1,
        "complete": True,
        "records": ordered,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--players", nargs="+", required=True)
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    records = [json.loads(path.read_text(encoding="utf-8")) for path in a.inputs]
    dataset = aggregate_records(a.players, records)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "players": dataset["players"],
        "coalition_count": dataset["coalition_count"],
        "complete": dataset["complete"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
