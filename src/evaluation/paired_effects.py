"""Pure paired-effect helpers for small CRN evaluation sets."""
import statistics

def paired_effects(baseline,current):
    if len(baseline)!=len(current) or not baseline:
        raise ValueError("paired samples must be non-empty and equal length")
    b=[float(x) for x in baseline]; c=[float(x) for x in current]
    diffs=[y-x for x,y in zip(b,c)]
    mean_diff=sum(diffs)/len(diffs)
    sd=statistics.stdev(diffs) if len(diffs)>1 else 0.0
    pct=[100.0*(y-x)/x for x,y in zip(b,c) if x!=0.0]
    return {
        "mean_difference":mean_diff,
        "median_difference":statistics.median(diffs),
        "min_difference":min(diffs),
        "max_difference":max(diffs),
        "improved_count":sum(d<0 for d in diffs),
        "worsened_count":sum(d>0 for d in diffs),
        "tied_count":sum(d==0 for d in diffs),
        "paired_standardized_effect_dz":mean_diff/sd if sd>0 else None,
        "mean_paired_percent_change":sum(pct)/len(pct) if pct else None,
        "median_paired_percent_change":statistics.median(pct) if pct else None,
        "percent_change_n":len(pct),
    }
