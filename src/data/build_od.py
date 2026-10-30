import pandas as pd

def build_weighted_od(df, origin_col, destination_col, time_col, weight_col,
                      eligibility_col=None, time_bin_minutes=15):
    x = df.copy().dropna(subset=[origin_col, destination_col, time_col, weight_col])
    if eligibility_col:
        x = x[x[eligibility_col].astype(bool)]
    ts = pd.to_datetime(x[time_col], errors="coerce")
    x = x[ts.notna()].copy()
    ts = ts[ts.notna()]
    x["time_bin"] = ((ts.dt.hour * 60 + ts.dt.minute) // time_bin_minutes).astype(int)
    x["survey_weight"] = pd.to_numeric(x[weight_col], errors="coerce")
    x = x[x["survey_weight"].notna() & (x["survey_weight"] >= 0)]
    return (
        x.groupby(["time_bin", origin_col, destination_col])
         .agg(raw_n=(weight_col, "size"), weighted_n=("survey_weight", "sum"))
         .reset_index()
         .rename(columns={origin_col: "origin", destination_col: "destination"})
    )
