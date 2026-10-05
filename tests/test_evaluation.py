import pandas as pd
from src.data.split_realizations import seed_plan
from src.evaluation.common_random_numbers import paired_test_matrix
from src.evaluation.demand_validation import validate_od,validate_temporal

def test_splits_disjoint_and_crn_complete():
    p=seed_plan(42,3,2,2)
    assert not (set(p["train"])&set(p["test"]))
    m=paired_test_matrix(p["test"])
    assert len(m)==12
    for s in p["test"]:
        assert sum(x["demand_seed"]==s for x in m)==6

def test_perfect_validation():
    od=pd.DataFrame([{"time_bin":1,"origin":"A","destination":"B","weighted_n":2},
                     {"time_bin":2,"origin":"B","destination":"A","weighted_n":1}])
    req=pd.DataFrame([{"time_bin":1,"origin_district":"A","destination_district":"B"},
                      {"time_bin":1,"origin_district":"A","destination_district":"B"},
                      {"time_bin":2,"origin_district":"B","destination_district":"A"}])
    assert validate_od(od,req)["od_distribution_mae"]==0
    assert validate_temporal(od,req)["temporal_distribution_mae"]==0


def test_phase3a_crn_is_paired_and_leakage_free():
    from src.evaluation.phase3a_crn import DEFAULT_PHASE3A_SEEDS, paired_evaluation_rows
    plan = DEFAULT_PHASE3A_SEEDS.validate()
    assert set(plan.training).isdisjoint(plan.heldout)
    rows = paired_evaluation_rows()
    assert len(rows) == len(plan.heldout) * 3
    for seed in plan.heldout:
        methods = {r["method"] for r in rows if r["demand_seed"] == seed}
        assert methods == {"B4", "B5", "B6"}


def test_paired_effects_direction_and_zero_denominator():
    from src.evaluation.paired_effects import paired_effects
    r=paired_effects([10,20,0],[8,18,1])
    assert r["improved_count"]==2
    assert r["worsened_count"]==1
    assert r["tied_count"]==0
    assert r["percent_change_n"]==2
    assert r["mean_difference"]==-1.0

def test_paired_effects_rejects_unpaired_samples():
    import pytest
    from src.evaluation.paired_effects import paired_effects
    with pytest.raises(ValueError):
        paired_effects([1,2],[1])
