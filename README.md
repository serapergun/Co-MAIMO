# Co-MAIMO

**Cooperative Multi-Agent Intelligent Mobility Optimization**

Reproducible research code for a Barcelona on-demand mobility digital twin integrating EMEF-calibrated demand, SUMO, hierarchical MAPPO, dynamic coalition formation, counterfactual coalition evaluation (CCE), and Shapley-informed cooperative learning.

## Scientific integrity
- EMEF is used to calibrate mobility demand; generated ODM requests are simulation inputs, not observed ride-hailing transactions.
- Observed, derived, simulated, and model-estimated quantities remain explicitly separated.
- EMEF field names are never guessed: the official dictionary must pass schema audit first.
- Training, validation, and test realizations are disjoint.
- Test data never tune zones, normalization, thresholds, hyperparameters, checkpoints, or reward weights.
- B1-B6 use a common feasibility/sequential-insertion layer.
- Reported results are produced only from QC-passing held-out runs.

## Benchmark chain
B1 unrestricted sequential dispatch → B2 static heterogeneous allocation → B3 demand-threshold heuristic → B4 MAPPO → B5 MAPPO + DCF + CCE → B6 B5 + Shapley-informed reward.

## Status
Implementation scaffold. No performance numbers are claimed until the Barcelona network, EMEF calibration, SUMO environment, and held-out experiments have been executed and validated.
