"""Build a passenger-vehicle SUMO network from a local OSM extract.

The script never downloads or silently modifies source geography. It invokes
SUMO netconvert with a reproducible command and records that command.
"""
from __future__ import annotations
from pathlib import Path
import json, shutil, subprocess

def build_network(osm_file, output_net, manifest_file=None):
    exe = shutil.which("netconvert")
    if not exe:
        raise RuntimeError("netconvert not found. Install SUMO and expose it on PATH.")
    osm_file, output_net = Path(osm_file), Path(output_net)
    if not osm_file.exists():
        raise FileNotFoundError(osm_file)
    output_net.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe, "--osm-files", str(osm_file),
        "--output-file", str(output_net),
        "--keep-edges.by-vclass", "passenger",
        "--geometry.remove", "true",
        "--roundabouts.guess", "true",
        "--ramps.guess", "true",
        "--junctions.join", "true",
        "--tls.guess-signals", "true",
        "--tls.discard-simple", "true",
        "--tls.join", "true",
    ]
    cp = subprocess.run(cmd, check=True, text=True, capture_output=True)
    manifest = {"command": cmd, "stdout": cp.stdout, "stderr": cp.stderr}
    if manifest_file:
        Path(manifest_file).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
