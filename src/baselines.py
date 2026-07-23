"""Baselines: truck-only TSP (NN + 2-opt + Or-opt) and greedy TSP-LS decode."""
import numpy as np
from problem import tspd_split, truck_only_length


def nearest_neighbor(Dt, n):
    unvis = set(range(1, n + 1))
    tour = [0]
    cur = 0
    while unvis:
        nxt = min(unvis, key=lambda c: Dt[cur, c])
        tour.append(nxt)
        unvis.remove(nxt)
        cur = nxt
    return tour + [0]


def two_opt(order, Dt, max_pass=60):
    """Standard 2-opt on a closed tour [0,...,0]; first-improvement."""
    best = order[:]
    improved = True
    L = len(best)
    p = 0
    while improved and p < max_pass:
        improved = False
        p += 1
        for i in range(1, L - 2):
            a, b = best[i - 1], best[i]
            for k in range(i + 1, L - 1):
                c, d = best[k], best[k + 1]
                if a == c:
                    continue
                delta = (Dt[a, c] + Dt[b, d]) - (Dt[a, b] + Dt[c, d])
                if delta < -1e-10:
                    best[i:k + 1] = best[i:k + 1][::-1]
                    improved = True
                    break
            if improved:
                break
    return best


def or_opt(order, Dt, max_pass=40):
    """Move segments of length 1..3 to a better position; first-improvement."""
    best = order[:]
    L = len(best)
    improved = True
    p = 0
    while improved and p < max_pass:
        improved = False
        p += 1
        for seg in (1, 2, 3):
            for i in range(1, L - 1 - seg):
                segment = best[i:i + seg]
                a, b = best[i - 1], best[i + seg]
                removed = Dt[a, segment[0]] + Dt[segment[-1], b] - Dt[a, b]
                for j in range(1, L - 1):
                    if i - 1 <= j <= i + seg:
                        continue
                    c, d = best[j], best[j + 1] if j + 1 < L else best[0]
                    added = Dt[c, segment[0]] + Dt[segment[-1], d] - Dt[c, d]
                    if added - removed < -1e-10:
                        rest = best[:i] + best[i + seg:]
                        jj = rest.index(c)
                        new = rest[:jj + 1] + segment + rest[jj + 1:]
                        if new[0] == 0 and new[-1] == 0 and len(new) == L:
                            best = new
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return best


def truck_only_tsp(inst, Dt, restarts=4, seed=0):
    """Best truck-only TSP via NN + 2-opt + Or-opt with a few random restarts."""
    n = inst["n"]
    rng = np.random.default_rng(seed)
    best_order = nearest_neighbor(Dt, n)
    best_order = or_opt(two_opt(best_order, Dt), Dt)
    best_len = truck_only_length(best_order, Dt)
    for r in range(restarts):
        o = [0] + list(rng.permutation(range(1, n + 1))) + [0]
        o = or_opt(two_opt(o, Dt), Dt)
        Ln = truck_only_length(o, Dt)
        if Ln < best_len:
            best_len, best_order = Ln, o
    return best_order, best_len


def greedy_tspd(inst, Dt, alpha, endurance=np.inf, max_span=None, seed=0):
    """Greedy TSP-LS: build a good TSP order, then one-shot optimal split decode."""
    order, _ = truck_only_tsp(inst, Dt, restarts=2, seed=seed)
    ms, ops = tspd_split(order, Dt, alpha, endurance, max_span)
    return ms, order, ops
