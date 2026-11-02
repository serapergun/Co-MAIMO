# B4-S4 feature identifiability: validated finding and next experiment

## Evidence (real Barcelona SUMO, heldout seeds 9101–9105)
- [B4-S4 simulation #41416190780](https://github.com/serapergun/Co-MAIMO/actions/runs/41416190780): five heldout seeds, 24 completed requests each; no teleports.
- [Paired decisions #41432489470](https://github.com/serapergun/Co-MAIMO/actions/runs/41432489470): 14/120 assignments differ from B4-S3, concentrated in seeds 9102 (12) and 9105 (2). Difference alone does not establish feature causality.
- [Frozen-policy ablation #41432742885](https://github.com/serapergun/Co-MAIMO/actions/runs/41432742885): zero sampled-choice changes under common random uniforms when density and/or SoC is removed; offline only.
- [Weight sensitivity #41434258482](https://github.com/serapergun/Co-MAIMO/actions/runs/41434258482): among 65 multi-feasible decisions, removing SoC changes 22 argmax choices despite small probability shifts; 8× density weight changes one argmax.
- [Identifiability #41440004749](https://github.com/serapergun/Co-MAIMO/actions/runs/41440004749): within-decision nonzero feature spread: access 38/65, density 1/65, SoC 64/65. Median weighted logit span: access 1.05621487, density 0, SoC 0.000050803.

## Interpretation
Current density is measured on each idle taxi's **current edge** using `edge_traffic_density` in `src/marl/b4_s4_traci_adapter.py`. This measurement is present, but does not distinguish candidate taxis in 64/65 observed multi-candidate decisions. It is NOT evidence that traffic congestion is universally irrelevant. SoC has nonzero but typically tiny within-decision spread. The dominant observed signal is access time. These observations do not prove PPO cannot improve in other demand or fleet conditions.

## Proposed next controlled experiments (not yet executed)
1. Add a read-only, decision-time **candidate-specific route congestion** probe: aggregate observed per-edge traffic counts/lengths along the candidate's feasible pickup route, with explicit unavailable status. Avoid future-trajectory data, fabricated edge counts and hidden imputation. Validate TraCI route APIs and ensure measurements occur before dispatch.
2. Unit-test probe availability, zero-length edges, empty routes, internal edges, route infeasibility, candidate permutation and no future leakage.
3. Run a telemetry-only SUMO pilot (seed 9101 initially), compare within-decision feature spread against current-edge density. **Do not** claim performance gains from a telemetry pilot.
4. Only if route density is meaningfully candidate-discriminative, train separate models on 4201–4208 for: access-only, access+SoC, access+route-density, full. Freeze all policies and evaluate 9101–9105 with paired demand seeds, identical fleet/request settings and no heldout training.
5. Report per-seed pickup wait, empty km, MMPEVEM kWh, completion, teleport, 95% paired confidence intervals, and multiplicity-adjusted tests versus B3 and B4-S4. No improvement claim unless supported.

## Controls
No changes to `develop`; no merge. All outputs are simulation-based and must retain exact source run IDs, code SHA, SUMO version, and artifact provenance.
