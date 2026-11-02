"""SUMO TraCI dispatcher for the B4-S4 telemetry-aware shared actor.

Every stored training feature is sampled at decision time, never retrospectively.
"""
import numpy as np
from src.marl.traci_adapter import reservation_features, _route_time
from src.marl.telemetry_probes import edge_traffic_density,battery_soc
from src.marl.b4_s4_features import encode_candidate

class B4S4Dispatcher:
    def __init__(self,policy,seed=0):
        self.policy=policy
        self.rng=np.random.default_rng(seed)
        self.decisions=[]

    def __call__(self,traci,reservations,fleet,now):
        fleet=tuple(sorted(fleet))
        idle=set(traci.vehicle.getTaxiFleet(0))
        used=set()
        assignments=[]
        for reservation in sorted(reservations,key=lambda r:(r.reservationTime,r.id)):
            access,route_mask=reservation_features(traci,reservation,fleet)
            mask=np.asarray([bool(route_mask[i] and taxi in idle and taxi not in used)
                             for i,taxi in enumerate(fleet)],dtype=bool)
            if not mask.any():continue
            density=[edge_traffic_density(traci,taxi) if mask[i] else None for i,taxi in enumerate(fleet)]
            soc=[battery_soc(traci,taxi) if mask[i] else None for i,taxi in enumerate(fleet)]
            from src.marl.telemetry_probes import Probe
            missing=Probe(None,False,"infeasible")
            density=[x if x is not None else missing for x in density]
            soc=[x if x is not None else missing for x in soc]
            features=np.zeros((len(fleet),5),dtype=float)
            for i in np.flatnonzero(mask):
                features[i]=encode_candidate(access[i],density[i],soc[i]).values
            probs=self.policy.masked_action_probs(access,mask,density,soc)
            action=int(self.rng.choice(len(fleet),p=probs))
            taxi=fleet[action]
            selected_access=_route_time(traci,taxi,reservation)
            if selected_access is None:raise RuntimeError("selected taxi lost feasible route")
            assignments.append((taxi,[reservation.id]))
            used.add(taxi)
            feasible_times=[_route_time(traci,fleet[i],reservation) for i in np.flatnonzero(mask)]
            feasible_times=[t for t in feasible_times if t is not None]
            scale=max(feasible_times,default=1.0)
            state=[len(idle-used)/len(fleet),int(mask.sum())/len(fleet),
                   min(feasible_times,default=0.0)/max(scale,1.0)]
            self.decisions.append({
                "time":float(now),"reservation_id":str(reservation.id),
                "vehicle_id":taxi,"action":action,"log_prob":float(np.log(max(probs[action],1e-15))),
                "old_log_prob":float(np.log(max(probs[action],1e-15))),
                "observation":access.tolist(),"candidate_features":features.tolist(),
                "feasible_mask":mask.astype(int).tolist(),"global_state":state,
                "selected_access_time_s":float(selected_access),
                "edge_density_available":int(sum(d.available for d in density)),
                "battery_soc_available":int(sum(d.available for d in soc)),
            })
        return assignments
