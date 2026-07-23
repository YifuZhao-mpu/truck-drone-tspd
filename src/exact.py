"""Independent makespan checker + brute-force exact TSP-D oracle (small n)."""
import itertools
import numpy as np
from problem import tspd_split


def simulate_makespan(order, ops, Dt, alpha, endurance=np.inf, Ddr=None):
    """
    INDEPENDENT recomputation of makespan from decoded ops (does not reuse the DP's
    leg-decomposition algebra). Dt=truck dist, Ddr=drone dist (defaults to Dt).
    Returns (makespan, feasible, served_set).
    """
    if Ddr is None:
        Ddr = Dt
    seq = order
    total = 0.0
    served = set()
    feasible = True
    for op in ops:
        if op[0] == "truck":
            i, k = op[1], op[2]
            assert k == i + 1
            total += Dt[seq[i], seq[k]]
            if seq[k] != 0:
                served.add(seq[k])
        elif op[0] == "sortie":
            i, j, k = op[1], op[2], op[3]
            # truck path i..k skipping j, recomputed directly node-by-node
            kept = [p for p in range(i, k + 1) if p != j]
            tr = sum(Dt[seq[kept[a]], seq[kept[a + 1]]] for a in range(len(kept) - 1))
            fl = Ddr[seq[i], seq[j]] + Ddr[seq[j], seq[k]]
            if fl > endurance + 1e-9:
                feasible = False
            dr = fl / alpha
            total += max(tr, dr)
            served.add(seq[j])
            for p in range(i + 1, k + 1):
                if p != j and seq[p] != 0:
                    served.add(seq[p])
        elif op[0] == "msortie":
            i, S, k = op[1], op[2], op[3]
            kept = [p for p in range(i, k + 1) if p not in S]
            tr = sum(Dt[seq[kept[a]], seq[kept[a + 1]]] for a in range(len(kept) - 1))
            fls = [Dt[seq[i], seq[j]] + Dt[seq[j], seq[k]] for j in S]
            if any(f > endurance + 1e-9 for f in fls):
                feasible = False
            drs = [f / alpha for f in fls]
            total += max(tr, max(drs))
            for j in S:
                served.add(seq[j])
            for p in range(i + 1, k + 1):
                if p not in S and seq[p] != 0:
                    served.add(seq[p])
    return total, feasible, served


def brute_force_optimal(inst, Dt, alpha, endurance=np.inf):
    """Exact TSP-D optimum for small n: min over all customer orders of the DP split."""
    n = inst["n"]
    cust = list(range(1, n + 1))
    best = float("inf")
    best_order = None
    for perm in itertools.permutations(cust):
        order = [0] + list(perm) + [0]
        ms, ops = tspd_split(order, Dt, alpha, endurance)
        if ms < best:
            best = ms
            best_order = order
    return best, best_order
