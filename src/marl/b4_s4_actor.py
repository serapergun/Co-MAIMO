"""B4-S4 shared candidate scorer with feasibility masking and permutation symmetry.

This is an actor inference implementation; training and SUMO comparison remain separate.
"""
import numpy as np
from src.marl.b4_s4_features import encode_candidate

class B4S4Actor:
    def __init__(self, weights=None):
        self.weights=np.asarray([-2.0,-.5,.5,0.,0.] if weights is None else weights,dtype=float)
        if self.weights.shape!=(5,) or not np.isfinite(self.weights).all():
            raise ValueError("B4-S4 actor requires five finite shared weights")

    def masked_action_probs(self,access_times,feasible_mask,edge_density,soc):
        access=np.asarray(access_times,dtype=float)
        mask=np.asarray(feasible_mask,dtype=bool)
        if access.ndim!=1 or mask.shape!=access.shape or not mask.any():
            raise ValueError("invalid action candidates or mask")
        if len(edge_density)!=len(access) or len(soc)!=len(access):
            raise ValueError("telemetry/candidate size mismatch")
        features=np.array([encode_candidate(access[i],edge_density[i],soc[i]).values
                           for i in range(len(access)) if mask[i]],dtype=float)
        logits=features@self.weights
        logits-=logits.max()
        scores=np.exp(logits)
        probabilities=np.zeros(len(access),dtype=float)
        probabilities[mask]=scores/scores.sum()
        return probabilities
