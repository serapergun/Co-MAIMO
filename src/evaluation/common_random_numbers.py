"""Pair B1-B6 on identical held-out demand realizations."""
METHODS=("B1","B2","B3","B4","B5","B6")
def paired_test_matrix(test_seeds, methods=METHODS):
    return [{"method":m,"demand_seed":int(s)} for s in test_seeds for m in methods]
