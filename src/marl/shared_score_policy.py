"""Taxi-permutation equivariant shared-score dispatch policy prototype.

This is an architecture validation baseline, NOT a trained MAPPO result.
A shared monotonic negative access-time score is applied to every candidate.
"""
import numpy as np

class SharedAccessScorePolicy:
    def __init__(self,temperature=0.2):
        if not np.isfinite(temperature) or temperature<=0:raise ValueError("temperature must be positive")
        self.temperature=float(temperature)
    def masked_action_probs(self,observation,mask):
        x=np.asarray(observation,dtype=float)
        m=np.asarray(mask,dtype=bool)
        if x.ndim!=1 or x.shape!=m.shape or not m.any():raise ValueError("invalid candidate features/mask")
        if not np.all(np.isfinite(x[m])):raise ValueError("nonfinite feasible features")
        scores=-x[m]/self.temperature
        scores-=np.max(scores)
        e=np.exp(scores);out=np.zeros_like(x,dtype=float);out[m]=e/e.sum()
        return out
    def select(self,observation,mask):
        return int(np.argmax(self.masked_action_probs(observation,mask)))
