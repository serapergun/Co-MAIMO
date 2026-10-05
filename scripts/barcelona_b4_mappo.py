"""Real B4 MAPPO training/evaluation entry point for Barcelona SUMO.

This script is intentionally separate from barcelona_b4_traci.py, which is a
lifecycle/load fixture. It trains on repeated SUMO episodes, freezes a
checkpoint, then evaluates the frozen policy on a separate held-out episode.
"""
from pathlib import Path
import json
import shutil

import numpy as np
import sumolib

from src.marl.mappo import LinearMAPPO
from src.marl.traci_adapter import MAPPOTraciDispatcher
from src.marl.episode import train_completed_episode
from src.marl.checkpoint import save_checkpoint, load_checkpoint
from src.simulation.traci_runner import run


ROOT = Path("artifacts/barcelona_b4_mappo")
NET = Path("artifacts/barcelona_network/barcelona.net.xml")
N_TAXIS = 8
N_REQUESTS = 24
TRAIN_EPISODES = 8
SEED = 4204


def build_scenario(root):
    root.mkdir(parents=True, exist_ok=True)
    net = sumolib.net.readNet(str(NET))
    edges = [
        e for e in net.getEdges()
        if not e.getID().startswith(":")
        and any(l.allows("passenger") for l in e.getLanes())
    ]
    pairs = []
    used = set()
    for o in edges[:2000]:
        if o.getID() in used:
            continue
        for d in edges[-1:max(-2001, -len(edges)):-20]:
            if o.getID() == d.getID():
                continue
            route, _ = net.getOptimalPath(o, d, vClass="passenger")
            if not route:
                continue
            route_len = sum(e.getLength() for e in route)
            ol = [l for l in o.getLanes() if l.allows("passenger")]
            dl = [l for l in d.getLanes() if l.allows("passenger")]
            if (3000 <= route_len <= 6000
                    and max((l.getLength() for l in ol), default=0) >= 80
                    and max((l.getLength() for l in dl), default=0) >= 80):
                pairs.append((o.getID(), d.getID()))
                used.add(o.getID())
                break
        if len(pairs) == N_TAXIS:
            break
    if len(pairs) != N_TAXIS:
        raise SystemExit("Could not construct B4 MAPPO route set")

    lines = ['<routes>', '<vType id="evTaxi" vClass="passenger" emissionClass="MMPEVEM" personCapacity="4" accel="2.0" decel="4.5" maxSpeed="13.89"><param key="has.battery.device" value="true"/><param key="has.taxi.device" value="true"/></vType>']
    for i, (o, _) in enumerate(pairs):
        lines.append(f'<vehicle id="taxi{i}" type="evTaxi" depart="0" line="taxi"><route edges="{o}"/></vehicle>')
    for wave, depart in enumerate((10, 1800, 3600)):
        for i, (o, d) in enumerate(pairs):
            rid = wave * N_TAXIS + i
            lines.append(f'<person id="request{rid}" depart="{depart}"><ride from="{o}" to="{d}" lines="taxi"/></person>')
    lines.append('</routes>')
    rou = root / "b4_mappo.rou.xml"
    cfg = root / "b4_mappo.sumocfg"
    rou.write_text("\n".join(lines), encoding="utf-8")
    cfg.write_text(f'<configuration><input><net-file value="{NET.resolve()}"/><route-files value="{rou.resolve()}"/></input><time><begin value="0"/><end value="18000"/><step-length value="1"/></time></configuration>', encoding="utf-8")
    return cfg, pairs


def main():
    if not NET.exists():
        raise SystemExit("Barcelona preflight network missing")
    cfg, pairs = build_scenario(ROOT / "scenario")
    policy = LinearMAPPO(N_TAXIS, 3, N_TAXIS, seed=SEED)
    history = []

    for episode in range(TRAIN_EPISODES):
        out = ROOT / "train" / f"episode_{episode:03d}"
        dispatcher = MAPPOTraciDispatcher(policy, seed=SEED + episode)
        outputs = run(cfg, out, dispatcher, max_time_s=18000)
        metrics = train_completed_episode(
            dispatcher, outputs["request_qc"], policy,
            gamma=0.99, actor_lr=0.01, critic_lr=0.01, clip_eps=0.2,
        )
        history.append({"episode": episode, **metrics})

    checkpoint = ROOT / "checkpoints" / "b4_mappo_final.npz"
    save_checkpoint(policy, checkpoint, {
        "benchmark": "B4",
        "training_episodes": TRAIN_EPISODES,
        "base_seed": SEED,
        "training_role": "SUMO training episodes; not held-out evaluation",
    })

    frozen = load_checkpoint(checkpoint)
    eval_out = ROOT / "heldout"
    evaluator = MAPPOTraciDispatcher(frozen, seed=SEED + 10000)
    outputs = run(cfg, eval_out, evaluator, max_time_s=18000)

    (ROOT / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (ROOT / "experiment_manifest.json").write_text(json.dumps({
        "benchmark": "B4_MAPPO",
        "training_episodes": TRAIN_EPISODES,
        "training_seed_base": SEED,
        "heldout_policy_sampling_seed": SEED + 10000,
        "taxi_count": N_TAXIS,
        "request_count": N_REQUESTS,
        "pairs": pairs,
        "checkpoint": str(checkpoint),
        "heldout_outputs": outputs,
        "note": "Held-out policy sampling is separate from training, but demand OD is fixed; this is an integration experiment, not yet a multi-seed statistical benchmark.",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"history": history, "heldout": outputs}, indent=2))


if __name__ == "__main__":
    main()
