"""Deterministic, disjoint demand-realization seed allocation."""
def seed_plan(master_seed, n_train, n_validation, n_test):
    import numpy as np
    if min(n_train,n_validation,n_test)<=0: raise ValueError("All splits require positive counts.")
    ss=np.random.SeedSequence(master_seed)
    children=ss.spawn(n_train+n_validation+n_test)
    seeds=[int(x.generate_state(1)[0]) for x in children]
    a=n_train; b=a+n_validation
    out={"train":seeds[:a],"validation":seeds[a:b],"test":seeds[b:]}
    flat=sum(out.values(),[])
    if len(flat)!=len(set(flat)): raise RuntimeError("Seed collision.")
    return out
