"""B1 integration runner contract.

This module intentionally fails fast until a validated SUMO network, request
file and calibrated fleet configuration are supplied. No fabricated demand or
performance values are generated.
"""
from pathlib import Path

def validate_inputs(net_file, requests_file, fleet_size):
    for p in (net_file, requests_file):
        if not Path(p).exists():
            raise FileNotFoundError(p)
    if fleet_size is None or fleet_size <= 0:
        raise ValueError("fleet_size must be empirically/experimentally configured.")
    return True

def run_b1(net_file, requests_file, fleet_size):
    validate_inputs(net_file, requests_file, fleet_size)
    raise NotImplementedError(
        "TraCI execution is enabled only after network/request artifacts pass QC."
    )
