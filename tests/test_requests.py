import pandas as pd
from src.data.generate_edge_requests import generate_requests
from src.network.validate_district_mapping import EXPECTED_BCN_DISTRICTS,validate_mapping

def test_joint_od_sampling_preserves_pairs():
    od=pd.DataFrame([
      {"time_bin":1,"origin":"A","destination":"B","weighted_n":2},
      {"time_bin":2,"origin":"B","destination":"A","weighted_n":1}])
    em=pd.DataFrame({"district":["A","A","B","B"],"edge_id":["a1","a2","b1","b2"]})
    r=generate_requests(od,em,50,7)
    assert set(zip(r.origin_district,r.destination_district)) <= {("A","B"),("B","A")}

def test_expected_district_set_is_ten():
    assert len(EXPECTED_BCN_DISTRICTS)==10


def test_seeded_barcelona_demand_is_reproducible_and_seed_sensitive():
    from src.data.barcelona_crn import seeded_demand
    pairs = [(f"o{i}", f"d{i}") for i in range(20)]
    a = seeded_demand(pairs, n_requests=24, seed=9101)
    b = seeded_demand(pairs, n_requests=24, seed=9101)
    c = seeded_demand(pairs, n_requests=24, seed=9102)
    assert a == b
    assert a != c
    assert len(a) == 24
    assert {r["depart_s"] for r in a[:12]} == {10}
    assert {r["depart_s"] for r in a[12:]} == {1810}
    assert len({(r["origin_edge"], r["destination_edge"]) for r in a[:12]}) == 12
