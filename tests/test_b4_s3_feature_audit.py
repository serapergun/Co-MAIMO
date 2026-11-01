"""Focused regression tests for real-decision audit QC."""
import json
from pathlib import Path
import pytest
from scripts.audit_b4_s3_features import audit

def fixture(tmp_path, missing=False):
    root=tmp_path/"b4"
    root.mkdir()
    (root/"training_history.json").write_text(json.dumps([{} for _ in range(8)]))
    (root/"heldout_metrics.json").write_text(json.dumps([
        {"demand_seed":s,"completed":24,"unresolved":0,"teleports":0}
        for s in range(9101,9106)]))
    for seed in range(9101,9106):
        d=root/"heldout"/f"seed_{seed}"
        d.mkdir(parents=True)
        if missing and seed==9101:continue
        records=[]
        for i in range(24):
            if i%2:
                records.append({"observation":[.1,.5],"feasible_mask":[1,1],"action":1})
            else:
                records.append({"observation":[.1,.5],"feasible_mask":[1,0],"action":0})
        (d/"dispatch_decisions.json").write_text(json.dumps(records))
    return root

def test_counts_and_non_nearest_rate(tmp_path):
    result=audit(fixture(tmp_path))
    counts=result["action_diagnostics"]
    assert counts["decision_count"]==120
    assert counts["single_feasible"]==60
    assert counts["multi_feasible"]==60
    assert counts["non_nearest_selected"]==60
    assert result["multi_feasible_non_nearest_rate"]==1.0
    assert result["multi_feasible_mean_normalized_access_regret"]==pytest.approx(.4)

def test_missing_logs_not_misrepresented(tmp_path):
    result=audit(fixture(tmp_path,missing=True))
    assert result["action_diagnostics"]["missing_decision_logs"]==1
    assert result["action_diagnostics"]["decision_count"]==96

def test_reject_infeasible_selected_action(tmp_path):
    root=fixture(tmp_path)
    p=root/"heldout"/"seed_9101"/"dispatch_decisions.json"
    rows=json.loads(p.read_text())
    rows[0]["action"]=1
    p.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="Invalid decision"):
        audit(root)
