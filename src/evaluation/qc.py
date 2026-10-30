def qc_report(metrics):
    zero_checks = [
        "duplicate_service",
        "capacity_violations",
        "soc_violations",
        "invalid_routes",
        "partition_violations",
        "unexpected_termination",
    ]
    checks = {k: metrics.get(k, 0) == 0 for k in zero_checks}
    if {"generated", "served", "unserved", "remaining"} <= metrics.keys():
        checks["request_conservation"] = (
            metrics["generated"]
            == metrics["served"] + metrics["unserved"] + metrics["remaining"]
        )
    return {"pass": all(checks.values()), "checks": checks}
