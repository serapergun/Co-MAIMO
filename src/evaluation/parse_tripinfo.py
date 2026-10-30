import xml.etree.ElementTree as ET
import pandas as pd
def parse_tripinfo(path):
 rows=[]
 for t in ET.parse(path).getroot().findall("tripinfo"):
  row={k:t.get(k) for k in ["id","depart","arrival","duration","routeLength","waitingTime","vaporized"]}
  taxi=t.find("taxi"); bat=t.find("battery")
  if taxi is not None: row.update({"customers":taxi.get("customers"),"occupiedDistance":taxi.get("occupiedDistance"),"occupiedTime":taxi.get("occupiedTime")})
  if bat is not None: row.update({"battery_depleted":bat.get("depleted"),"battery_capacity_Wh":bat.get("actualBatteryCapacity"),"energy_consumed_Wh":bat.get("totalEnergyConsumed"),"energy_regenerated_Wh":bat.get("totalEnergyRegenerated")})
  rows.append(row)
 return pd.DataFrame(rows)
