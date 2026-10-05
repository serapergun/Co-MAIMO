"""CRN-controlled B4 MAPPO training and held-out Barcelona evaluation."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import sumolib

from src.data.barcelona_crn import safe_od_pool, seeded_demand
from src.evaluation.phase3a_crn import DEFAULT_PHASE3A_SEEDS
from src.evaluation.parse_mmpevem_emissions import mmpevem_emission_energy
from src.marl.mappo import LinearMAPPO
from src.marl.traci_adapter import MAPPOTraciDispatcher
from src.marl.episode import train_completed_episode
from src.marl.checkpoint import save_checkpoint, load_checkpoint
from src.simulation.traci_runner import run


ROOT = Path("artifacts/barcelona_b4_mappo")
NET = Path("artifacts/barcelona_network/barcelona.net.xml")
N_TAXIS = 8
N_REQUESTS = 24
POLICY_SEED = 4200


def write_scenario(root, pairs, demand_seed):
    root.mkdir(parents=True, exist_ok=True)
    demand = seeded_demand(pairs, N_REQUESTS, demand_seed)
    # Taxi initial positions are fixed across methods/seeds; only demand varies.
    starts = [pairs[i][0] for i in range(N_TAXIS)]
    lines = ['<routes>', '<vType id="evTaxi" vClass="passenger" emissionClass="MMPEVEM" personCapacity="4" accel="2.0" decel="4.5" maxSpeed="13.89"><param key="has.battery.device" value="true"/><param key="has.taxi.device" value="true"/></vType>']
    for i, edge in enumerate(starts):
        lines.append(f'<vehicle id="taxi{i}" type="evTaxi" depart="0" line="taxi"><route edges="{edge}"/></vehicle>')
    for r in demand:
        lines.append(
            f'<person id="{r["request_id"]}" depart="{r["depart_s"]}">'
            f'<ride from="{r["origin_edge"]}" to="{r["destination_edge"]}" lines="taxi"/>'
            '</person>'
        )
    lines.append('</routes>')
    rou = root / "scenario.rou.xml"
    cfg = root / "scenario.sumocfg"
    rou.write_text("\n".join(lines), encoding="utf-8")
    cfg.write_text(f'<configuration><input><net-file value="{NET.resolve()}"/><route-files value="{rou.resolve()}"/></input><time><begin value="0"/><end value="18000"/><step-length value="1"/></time></configuration>', encoding="utf-8")
    (root / "demand.json").write_text(json.dumps(demand, indent=2), encoding="utf-8")
    return cfg


def validate_episode(out):
    q = json.loads((out / "request_qc.json").read_text())
    if len(q["assigned_reservations"]) != N_REQUESTS:
        raise RuntimeError("not all requests assigned")
    if q["duplicate_assignments"] or q["final_active_reservations"] or q["remaining_persons"]:
        raise RuntimeError("reservation lifecycle QC failed")
    if q["teleported_vehicle_ids"] or q["horizon_exceeded"]:
        raise RuntimeError("SUMO teleport/horizon QC failed")
    pickup = list(map(float, q["pickup_wait_s"].values()))
    if len(pickup) != N_REQUESTS:
        raise RuntimeError("pickup telemetry incomplete")
    trips = ET.parse(out / "tripinfo.xml").getroot().findall("tripinfo")
    km = sum(float(t.get("routeLength", "0")) for t in trips if t.get("id", "").startswith("taxi")) / 1000.0
    energy = mmpevem_emission_energy(out / "emissions.xml")
    if km <= 0 or energy["net_energy_kWh"] <= 0:
        raise RuntimeError("non-positive physical metric")
    occupied = sum(map(float, q["occupied_distance_m_by_taxi"].values())) / 1000.0
    empty = sum(map(float, q["empty_distance_m_by_taxi"].values())) / 1000.0
    return {
        "completed": N_REQUESTS,
        "unresolved": 0,
        "teleports": 0,
        "mean_pickup_wait_s": sum(pickup) / len(pickup),
        "max_pickup_wait_s": max(pickup),
        "occupied_km": occupied,
        "empty_km": empty,
        "fleet_vehicle_km": km,
        **energy,
        "kWh_per_vehicle_km": energy["net_energy_kWh"] / km,
    }


def main():
    if not NET.exists():
        raise SystemExit("Barcelona preflight network missing")
    net = sumolib.net.readNet(str(NET))
    pairs = safe_od_pool(net)
    if len(pairs) < 12:
        raise SystemExit("safe Barcelona OD pool too small")

    plan = DEFAULT_PHASE3A_SEEDS.validate()
    policy = LinearMAPPO(N_TAXIS, 3, N_TAXIS, seed=POLICY_SEED)
    history = []

    for i, demand_seed in enumerate(plan.training):
        scenario = ROOT / "scenarios" / f"train_{demand_seed}"
        cfg = write_scenario(scenario, pairs, demand_seed)
        out = ROOT / "train" / f"seed_{demand_seed}"
        dispatcher = MAPPOTraciDispatcher(policy, seed=POLICY_SEED + i + 1)
        outputs = run(cfg, out, dispatcher, max_time_s=18000)
        validate_episode(out)
        update = train_completed_episode(dispatcher, outputs["request_qc"], policy)
        history.append({"demand_seed": demand_seed, **update})

    checkpoint = ROOT / "checkpoints" / "b4_mappo_final.npz"
    save_checkpoint(policy, checkpoint, {
        "benchmark": "B4",
        "training_demand_seeds": list(plan.training),
        "heldout_demand_seeds": list(plan.heldout),
        "policy_seed": POLICY_SEED,
    })

    frozen = load_checkpoint(checkpoint)
    heldout = []
    for i, demand_seed in enumerate(plan.heldout):
        scenario = ROOT / "scenarios" / f"heldout_{demand_seed}"
        cfg = write_scenario(scenario, pairs, demand_seed)
        out = ROOT / "heldout" / f"seed_{demand_seed}"
        evaluator = MAPPOTraciDispatcher(frozen, seed=POLICY_SEED + 10000 + i)
        run(cfg, out, evaluator, max_time_s=18000)
        metrics = validate_episode(out)
        metrics["demand_seed"] = demand_seed
        heldout.append(metrics)
        (out / "validated_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    (ROOT / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (ROOT / "heldout_metrics.json").write_text(json.dumps(heldout, indent=2), encoding="utf-8")
    (ROOT / "experiment_manifest.json").write_text(json.dumps({
        "benchmark": "B4_MAPPO",
        "training_demand_seeds": list(plan.training),
        "heldout_demand_seeds": list(plan.heldout),
        "policy_seed": POLICY_SEED,
        "training_updates_on_heldout": False,
        "taxi_count": N_TAXIS,
        "request_count": N_REQUESTS,
        "sumo_energy_model": "MMPEVEM",
        "comparison_design": "paired CRN; reuse identical held-out demand seeds for B5/B6",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"training": history, "heldout": heldout}, indent=2))


if __name__ == "__main__":
    main()
