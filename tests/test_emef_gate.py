import pytest
from src.data.emef_gate import GeographicAudit,require_district_od
def test_unverified_district_od_is_blocked():
    a=GeographicAudit(None,None,"w","t","official dictionary")
    with pytest.raises(RuntimeError): require_district_od(a)
def test_verified_contract_passes():
    require_district_od(GeographicAudit("o","d","w","t","official dictionary"))
