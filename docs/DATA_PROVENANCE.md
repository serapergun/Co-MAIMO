# Data provenance contract

Co-MAIMO separates four evidence classes throughout preprocessing, simulation,
analysis and manuscript reporting.

1. **Observed**: source survey observations/weights and source geospatial data.
2. **Derived**: weighted OD/time distributions and Shapley values computed from a declared characteristic function.
3. **Simulated**: generated ODM requests and SUMO operational outputs.
4. **Model-estimated**: electrical-energy indicators and CCE counterfactual values.

EMEF does not constitute observed SAV/ride-hailing transactions. District-level
OD may only be called empirically observed/derived if the verified public
microdata dictionary exposes the required origin and destination geography.
Otherwise a separately documented spatial-disaggregation model is required and
the result must be labelled EMEF-calibrated, spatially disaggregated OD.
