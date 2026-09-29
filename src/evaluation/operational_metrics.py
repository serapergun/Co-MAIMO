import pandas as pd
def vehicle_metrics(df):
 x=df.copy()
 for c in ["routeLength","occupiedDistance","energy_consumed_Wh","energy_regenerated_Wh","customers"]:
  if c in x: x[c]=pd.to_numeric(x[c],errors="coerce")
 total=x.get("routeLength",pd.Series(dtype=float)).sum(); occ=x.get("occupiedDistance",pd.Series(dtype=float)).sum()
 cons=x.get("energy_consumed_Wh",pd.Series(dtype=float)).sum(); regen=x.get("energy_regenerated_Wh",pd.Series(dtype=float)).sum()
 empty=max(0.0,total-occ); customers=x.get("customers",pd.Series(dtype=float)).sum()
 return {"vehicle_distance_m":float(total),"occupied_distance_m":float(occ),"empty_distance_m":float(empty),"empty_distance_ratio":float(empty/total) if total>0 else None,"net_energy_Wh":float(cons-regen),"customers":float(customers)}
