from pathlib import Path
import json, shutil, subprocess, sys, urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.network.qc_network import write_report

SNAPSHOT_URL = "https://download.geofabrik.de/europe/spain/cataluna-240101.osm.pbf"
# Barcelona municipality envelope used only for extraction; district polygon QC is a later gate.
BBOX = (2.0520, 41.3170, 2.2290, 41.4690)  # west,south,east,north

root = Path("artifacts/barcelona_network")
root.mkdir(parents=True, exist_ok=True)
pbf = root / "cataluna-240101.osm.pbf"
clip = root / "barcelona_bbox.osm.pbf"
osm = root / "barcelona_bbox.osm"
net = root / "barcelona.net.xml"
manifest = root / "network_manifest.json"
qc_report = root / "network_qc.json"

for exe in ("osmium", "netconvert"):
    if not shutil.which(exe):
        raise RuntimeError(f"{exe} not found on PATH")

if not pbf.exists():
    urllib.request.urlretrieve(SNAPSHOT_URL, pbf)

bbox = ",".join(map(str, BBOX))
subprocess.run(["osmium", "extract", "-b", bbox, "--strategy=complete_ways",
                "--overwrite", "-o", str(clip), str(pbf)], check=True)
subprocess.run(["osmium", "cat", str(clip), "-f", "osm", "--overwrite",
                "-o", str(osm)], check=True)

cmd = [
    "netconvert", "--osm-files", str(osm), "--output-file", str(net),
    "--keep-edges.by-vclass", "passenger", "--keep-edges.components", "1",
    "--geometry.remove", "true", "--roundabouts.guess", "true",
    "--ramps.guess", "true", "--junctions.join", "true",
    "--tls.guess-signals", "true", "--tls.discard-simple", "true",
    "--tls.join", "true",
]
subprocess.run(cmd, check=True)
manifest.write_text(json.dumps({
    "source": SNAPSHOT_URL, "snapshot_nominal_date": "2024-01-01",
    "bbox_wsen": BBOX, "clip_role": "preflight extraction envelope",
    "final_municipal_boundary_gate": "pending district-polygon clipping/validation",
    "netconvert_command": cmd,
}, indent=2), encoding="utf-8")

qc = write_report(net, qc_report)
if not qc["scc_gate_pass"]:
    raise RuntimeError(
        "Barcelona network QC gate failed: "
        f"largest_scc_share={qc['largest_scc_share']:.6f}; "
        f"required>={qc['scc_acceptance_threshold']:.2f}"
    )

print(net)
