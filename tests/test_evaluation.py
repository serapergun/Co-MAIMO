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
