def dispatch(traci,reservations,fleet,now):
 idle=list(traci.vehicle.getTaxiFleet(0)); pending=[r for r in reservations if r.state==1]
 ans=[]; used=set()
 for r in sorted(pending,key=lambda x:(x.reservationTime,x.id)):
  best=None
  for v in idle:
   if v in used: continue
   edge=traci.vehicle.getRoadID(v)
   if not edge or edge.startswith(":"): continue
   try: route=traci.simulation.findRoute(edge,r.fromEdge,vType=traci.vehicle.getTypeID(v))
   except Exception: continue
   if not route.edges: continue
   cand=(route.travelTime,v)
   if best is None or cand<best: best=cand
  if best: ans.append((best[1],[r.id])); used.add(best[1])
 return ans
