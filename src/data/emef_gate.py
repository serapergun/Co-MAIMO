from dataclasses import dataclass
import json
from pathlib import Path
@dataclass(frozen=True)
class GeographicAudit:
    origin_district_field: str|None
    destination_district_field: str|None
    weight_field: str|None
    start_time_field: str|None
    source_dictionary: str
    @property
    def district_od_verified(self):
        return all([self.origin_district_field,self.destination_district_field,self.weight_field,self.start_time_field])
def require_district_od(audit):
    if not audit.district_od_verified:
        raise RuntimeError("District-level EMEF OD is not verified. Do not generate or label a 10x10 empirical district OD matrix.")
def save_audit(audit,path):
    d=audit.__dict__|{"district_od_verified":audit.district_od_verified}
    Path(path).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding="utf-8")
