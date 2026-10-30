# Empirical calibration checks

The official EMEF 2023 Barcelona report provides aggregate constraints useful
for validating the demand-calibration pipeline. These are calibration checks,
not simulated on-demand request counts.

For Barcelona residents' weekday mobility, the report gives approximately
5.946 million trips and classifies them as:
- 47.9% internal to the residential district,
- 39.1% connections between Barcelona districts,
- 10.5% connections with other municipalities,
- 2.5% external trips.

These aggregates can be used as external consistency checks after the exact
public microdata schema is verified. They do not prove that the public
displacement CSV exposes district IDs for both trip ends.

Generated ODM demand must additionally be checked against its EMEF-derived
target using temporal-distribution MAE, OD-distribution MAE, OD-vector
correlation, origin/destination marginals, and intra-district share.

Train, validation and test request realizations are independent stochastic
draws from the frozen calibration model. Benchmark methods B1-B6 receive the
same held-out demand seed within each paired test realization.
