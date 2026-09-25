"""
TSP-D (Traveling Salesman Problem with Drone), min-makespan.
Formulation: Agatz, Bouman & Schmidt (2018). One truck + (m) drone(s); each sortie
serves exactly ONE customer; launch/land at nodes only; first-arriver waits.

Core: an "operations" dynamic program that, GIVEN a fixed visiting order of customers,
computes the optimal makespan TSP-D decoding (truck subsequence + single-customer drone
sorties). This is exact for the given order; the metaheuristic searches over orders.

Node 0 = depot. Truck speed = 1. Drone speed = alpha (>= 1). Euclidean distances.
"""
import numpy as np
from numba import njit


# ----------------------------- JIT split kernels -----------------------------
@njit(cache=True, fastmath=True)
def _split_full(seq, Dtr, Ddr, alpha, endurance, max_span):
    """DP optimal min-makespan TSP-D split (single drone). Dtr=truck dist, Ddr=drone dist.
    Returns (makespan, argi, argj)."""
    L = seq.shape[0] - 1
    INF = 1e18
    pref = np.zeros(L + 1)
    for a in range(L):
        pref[a + 1] = pref[a] + Dtr[seq[a], seq[a + 1]]
    best = np.full(L + 1, INF)
    best[0] = 0.0
    argi = np.full(L + 1, -1, dtype=np.int64)
    argj = np.full(L + 1, -1, dtype=np.int64)
    for k in range(1, L + 1):
        lo = k - max_span
        if lo < 0:
            lo = 0
        for i in range(lo, k):
            bi = best[i]
            if bi >= INF:
                continue
            if k == i + 1:
                c = Dtr[seq[i], seq[k]]
                jj = -1
            else:
                full_path = pref[k] - pref[i]
                si = seq[i]
                sk = seq[k]
                c = INF
                jj = -1
                for j in range(i + 1, k):
                    sj = seq[j]
                    fl = Ddr[si, sj] + Ddr[sj, sk]        # drone flight DISTANCE
                    if fl > endurance:                    # endurance = max flight distance
                        continue
                    dr = fl / alpha
                    tr = full_path - Dtr[seq[j - 1], seq[j]] - Dtr[seq[j], seq[j + 1]] + Dtr[seq[j - 1], seq[j + 1]]
                    cc = tr if tr > dr else dr
                    if cc < c:
                        c = cc
                        jj = j
                if c >= INF:
                    continue
            v = bi + c
            if v < best[k]:
                best[k] = v
                argi[k] = i
                argj[k] = jj
    return best[L], argi, argj


@njit(cache=True, fastmath=True)
def _split_cost(seq, Dtr, Ddr, alpha, endurance, max_span):
    """Makespan-only fast path for the ALNS inner loop. Dtr=truck, Ddr=drone."""
    L = seq.shape[0] - 1
    INF = 1e18
    pref = np.zeros(L + 1)
    for a in range(L):
        pref[a + 1] = pref[a] + Dtr[seq[a], seq[a + 1]]
    best = np.full(L + 1, INF)
    best[0] = 0.0
    for k in range(1, L + 1):
        lo = k - max_span
        if lo < 0:
            lo = 0
        for i in range(lo, k):
            bi = best[i]
            if bi >= INF:
                continue
            if k == i + 1:
                c = Dtr[seq[i], seq[k]]
            else:
                full_path = pref[k] - pref[i]
                si = seq[i]
                sk = seq[k]
                c = INF
                for j in range(i + 1, k):
                    sj = seq[j]
                    fl = Ddr[si, sj] + Ddr[sj, sk]        # drone flight DISTANCE
                    if fl > endurance:                    # endurance = max flight distance
                        continue
                    dr = fl / alpha
                    tr = full_path - Dtr[seq[j - 1], seq[j]] - Dtr[seq[j], seq[j + 1]] + Dtr[seq[j - 1], seq[j + 1]]
                    cc = tr if tr > dr else dr
                    if cc < c:
                        c = cc
                if c >= INF:
                    continue
            if c >= INF:
                continue
            v = bi + c
            if v < best[k]:
                best[k] = v
    return best[L]


@njit(cache=True, fastmath=True)
def _split_full_friction(seq, Dtr, Ddr, alpha, endurance, max_span,
                         t_service, t_launch, t_recover, eligible):
    """Single-drone split with customer service, launch/recovery time and eligibility.

    Customer service is charged once, on the vehicle serving that customer.  Service
    at a truck-served rendezvous is included on the truck side of the synchronized
    operation; depot service is zero.  Endurance remains a flight-distance limit.
    """
    all_eligible = True
    for q in range(1, eligible.shape[0]):
        if eligible[q] == 0:
            all_eligible = False
            break
    if t_service == 0.0 and t_launch == 0.0 and t_recover == 0.0 and all_eligible:
        return _split_full(seq, Dtr, Ddr, alpha, endurance, max_span)

    L = seq.shape[0] - 1
    INF = 1e18
    pref = np.zeros(L + 1)
    for a in range(L):
        pref[a + 1] = pref[a] + Dtr[seq[a], seq[a + 1]]
    best = np.full(L + 1, INF)
    best[0] = 0.0
    argi = np.full(L + 1, -1, dtype=np.int64)
    argj = np.full(L + 1, -1, dtype=np.int64)
    for k in range(1, L + 1):
        lo = k - max_span
        if lo < 0:
            lo = 0
        for i in range(lo, k):
            bi = best[i]
            if bi >= INF:
                continue
            if k == i + 1:
                c = Dtr[seq[i], seq[k]]
                if seq[k] != 0:
                    c += t_service
                jj = -1
            else:
                full_path = pref[k] - pref[i]
                si = seq[i]
                sk = seq[k]
                c = INF
                jj = -1
                for j in range(i + 1, k):
                    sj = seq[j]
                    if eligible[sj] == 0:
                        continue
                    fl = Ddr[si, sj] + Ddr[sj, sk]
                    if fl > endurance:
                        continue
                    drone_side = fl / alpha + t_service
                    truck_side = (full_path - Dtr[seq[j - 1], seq[j]]
                                  - Dtr[seq[j], seq[j + 1]]
                                  + Dtr[seq[j - 1], seq[j + 1]])
                    served = 0
                    for p in range(i + 1, k + 1):
                        if p != j and seq[p] != 0:
                            served += 1
                    truck_side += served * t_service
                    sync = truck_side if truck_side > drone_side else drone_side
                    cc = t_launch + sync + t_recover
                    if cc < c:
                        c = cc
                        jj = j
                if c >= INF:
                    continue
            v = bi + c
            if v < best[k]:
                best[k] = v
                argi[k] = i
                argj[k] = jj
    return best[L], argi, argj


@njit(cache=True, fastmath=True)
def _split_cost_friction(seq, Dtr, Ddr, alpha, endurance, max_span,
                         t_service, t_launch, t_recover, eligible):
    """Makespan-only fast path corresponding exactly to ``_split_full_friction``."""
    all_eligible = True
    for q in range(1, eligible.shape[0]):
        if eligible[q] == 0:
            all_eligible = False
            break
    if t_service == 0.0 and t_launch == 0.0 and t_recover == 0.0 and all_eligible:
        return _split_cost(seq, Dtr, Ddr, alpha, endurance, max_span)

    L = seq.shape[0] - 1
    INF = 1e18
    pref = np.zeros(L + 1)
    for a in range(L):
        pref[a + 1] = pref[a] + Dtr[seq[a], seq[a + 1]]
    best = np.full(L + 1, INF)
    best[0] = 0.0
    for k in range(1, L + 1):
        lo = k - max_span
        if lo < 0:
            lo = 0
        for i in range(lo, k):
            bi = best[i]
            if bi >= INF:
                continue
            if k == i + 1:
                c = Dtr[seq[i], seq[k]]
                if seq[k] != 0:
                    c += t_service
            else:
                full_path = pref[k] - pref[i]
                si = seq[i]
                sk = seq[k]
                c = INF
                for j in range(i + 1, k):
                    sj = seq[j]
                    if eligible[sj] == 0:
                        continue
                    fl = Ddr[si, sj] + Ddr[sj, sk]
                    if fl > endurance:
                        continue
                    drone_side = fl / alpha + t_service
                    truck_side = (full_path - Dtr[seq[j - 1], seq[j]]
                                  - Dtr[seq[j], seq[j + 1]]
                                  + Dtr[seq[j - 1], seq[j + 1]])
                    served = 0
                    for p in range(i + 1, k + 1):
                        if p != j and seq[p] != 0:
                            served += 1
                    truck_side += served * t_service
                    sync = truck_side if truck_side > drone_side else drone_side
                    cc = t_launch + sync + t_recover
                    if cc < c:
                        c = cc
                if c >= INF:
                    continue
            v = bi + c
            if v < best[k]:
                best[k] = v
    return best[L]


@njit(cache=True, fastmath=True)
def _split_obj(seq, Dtr, Ddr, alpha, endurance, max_span, lam, cl, cb, te, want_full):
    """DP minimizing scalarized makespan + lam*energy. Sortie energy = cl*t_out + cb*(t_out+t_back)
    (Dorling affine: loaded outbound, empty return); truck energy = te*distance. Dtr=truck, Ddr=drone.
    Returns (scalar_obj, argi, argj)."""
    L = seq.shape[0] - 1
    INF = 1e18
    pref = np.zeros(L + 1)
    for a in range(L):
        pref[a + 1] = pref[a] + Dtr[seq[a], seq[a + 1]]
    best = np.full(L + 1, INF)
    best[0] = 0.0
    argi = np.full(L + 1, -1, dtype=np.int64)
    argj = np.full(L + 1, -1, dtype=np.int64)
    for k in range(1, L + 1):
        lo = k - max_span
        if lo < 0:
            lo = 0
        for i in range(lo, k):
            bi = best[i]
            if bi >= INF:
                continue
            if k == i + 1:
                d = Dtr[seq[i], seq[k]]
                c = d + lam * te * d                  # truck leg: time + lam*truck_energy
                jj = -1
            else:
                full_path = pref[k] - pref[i]
                si = seq[i]; sk = seq[k]
                c = INF; jj = -1
                for j in range(i + 1, k):
                    sj = seq[j]
                    fl = Ddr[si, sj] + Ddr[sj, sk]
                    if fl > endurance:
                        continue
                    t_out = Ddr[si, sj] / alpha
                    t_back = Ddr[sj, sk] / alpha
                    dr = t_out + t_back
                    tr = full_path - Dtr[seq[j - 1], seq[j]] - Dtr[seq[j], seq[j + 1]] + Dtr[seq[j - 1], seq[j + 1]]
                    time = tr if tr > dr else dr
                    energy = cl * t_out + cb * (t_out + t_back) + te * tr
                    cc = time + lam * energy
                    if cc < c:
                        c = cc
                        jj = j
                if c >= INF:
                    continue
            v = bi + c
            if v < best[k]:
                best[k] = v
                argi[k] = i
                argj[k] = jj
    return best[L], argi, argj


@njit(cache=True, fastmath=True)
def _split_cost_multi(seq, Dtr, Ddr, alpha, endurance, max_span, m):
    """Makespan-only DP with up to m parallel single-customer drones per operation.
    Dtr=truck, Ddr=drone. Intermediate-node subsets enumerated by bitmask."""
    L = seq.shape[0] - 1
    INF = 1e18
    pref = np.zeros(L + 1)
    for a in range(L):
        pref[a + 1] = pref[a] + Dtr[seq[a], seq[a + 1]]
    best = np.full(L + 1, INF)
    best[0] = 0.0
    for k in range(1, L + 1):
        lo = k - max_span
        if lo < 0:
            lo = 0
        for i in range(lo, k):
            bi = best[i]
            if bi >= INF:
                continue
            if k == i + 1:
                c = Dtr[seq[i], seq[k]]
            else:
                inter = k - i - 1                      # intermediate positions i+1..k-1
                si = seq[i]; sk = seq[k]
                full_path = pref[k] - pref[i]
                c = full_path                          # |S|=0 : truck serves all
                for mask in range(1, 1 << inter):
                    # popcount
                    pc = 0
                    mm = mask
                    while mm:
                        pc += mm & 1
                        mm >>= 1
                    if pc > m:
                        continue
                    maxdr = 0.0
                    feas = True
                    # truck path over kept nodes (skip selected), drone max
                    prev = i
                    tr = 0.0
                    for t in range(inter):
                        pos = i + 1 + t
                        if (mask >> t) & 1:
                            fl = Ddr[si, seq[pos]] + Ddr[seq[pos], sk]
                            if fl > endurance:
                                feas = False
                                break
                            dr = fl / alpha
                            if dr > maxdr:
                                maxdr = dr
                        else:
                            tr += Dtr[seq[prev], seq[pos]]
                            prev = pos
                    if not feas:
                        continue
                    tr += Dtr[seq[prev], sk]
                    cc = tr if tr > maxdr else maxdr
                    if cc < c:
                        c = cc
            if c >= INF:
                continue
            v = bi + c
            if v < best[k]:
                best[k] = v
    return best[L]


@njit(cache=True, fastmath=True)
def _split_cost_multi3(seq, Dtr, Ddr, alpha, endurance, max_span, m):
    """Makespan-only DP with up to m<=3 parallel single-customer drones per operation.
    Subsets of drone-served intermediates enumerated directly by size (1..m); the truck
    time of a skip set is computed O(1) by decomposing the set into maximal consecutive
    runs (bridge edge minus skipped leg sum via the prefix array), so the same span cap
    as the single-drone search stays affordable. Dtr=truck, Ddr=drone."""
    L = seq.shape[0] - 1
    INF = 1e18
    pref = np.zeros(L + 1)
    for a in range(L):
        pref[a + 1] = pref[a] + Dtr[seq[a], seq[a + 1]]
    best = np.full(L + 1, INF)
    best[0] = 0.0
    for k in range(1, L + 1):
        lo = k - max_span
        if lo < 0:
            lo = 0
        for i in range(lo, k):
            bi = best[i]
            if bi >= INF:
                continue
            if k == i + 1:
                c = Dtr[seq[i], seq[k]]
            else:
                inter = k - i - 1
                si = seq[i]
                sk = seq[k]
                full_path = pref[k] - pref[i]
                c = INF  # multi-leg op must use >=1 drone (pure truck = single legs)
                # drone sortie time (or -1 if infeasible) per intermediate offset,
                # and O(1) single-skip truck saving per offset
                drt = np.empty(inter)
                sav1 = np.empty(inter)                 # full - tr for skipping one pos
                for t in range(inter):
                    p = i + 1 + t
                    sj = seq[p]
                    fl = Ddr[si, sj] + Ddr[sj, sk]
                    drt[t] = fl / alpha if fl <= endurance else -1.0
                    sav1[t] = (pref[p + 1] - pref[p - 1]) - Dtr[seq[p - 1], seq[p + 1]]
                for t1 in range(inter):
                    if drt[t1] < 0.0:
                        continue
                    p1 = i + 1 + t1
                    tr1 = full_path - sav1[t1]
                    cc = tr1 if tr1 > drt[t1] else drt[t1]
                    if cc < c:
                        c = cc
                    if m >= 2:
                        for t2 in range(t1 + 1, inter):
                            if drt[t2] < 0.0:
                                continue
                            p2 = i + 1 + t2
                            md2 = drt[t1] if drt[t1] > drt[t2] else drt[t2]
                            if t2 == t1 + 1:           # adjacent: one run [p1,p2]
                                tr2 = full_path - (pref[p2 + 1] - pref[p1 - 1]) \
                                    + Dtr[seq[p1 - 1], seq[p2 + 1]]
                            else:                       # two independent runs
                                tr2 = full_path - sav1[t1] - sav1[t2]
                            cc2 = tr2 if tr2 > md2 else md2
                            if cc2 < c:
                                c = cc2
                            if m >= 3:
                                for t3 in range(t2 + 1, inter):
                                    if drt[t3] < 0.0:
                                        continue
                                    p3 = i + 1 + t3
                                    md3 = md2 if md2 > drt[t3] else drt[t3]
                                    a12 = t2 == t1 + 1
                                    a23 = t3 == t2 + 1
                                    if a12 and a23:     # one run [p1,p3]
                                        tr3 = full_path - (pref[p3 + 1] - pref[p1 - 1]) \
                                            + Dtr[seq[p1 - 1], seq[p3 + 1]]
                                    elif a12:           # runs [p1,p2] + [p3]
                                        tr3 = full_path - (pref[p2 + 1] - pref[p1 - 1]) \
                                            + Dtr[seq[p1 - 1], seq[p2 + 1]] - sav1[t3]
                                    elif a23:           # runs [p1] + [p2,p3]
                                        tr3 = full_path - sav1[t1] \
                                            - (pref[p3 + 1] - pref[p2 - 1]) \
                                            + Dtr[seq[p2 - 1], seq[p3 + 1]]
                                    else:               # three independent runs
                                        tr3 = full_path - sav1[t1] - sav1[t2] - sav1[t3]
                                    cc3 = tr3 if tr3 > md3 else md3
                                    if cc3 < c:
                                        c = cc3
            if c >= INF:
                continue
            v = bi + c
            if v < best[k]:
                best[k] = v
    return best[L]


@njit(cache=True, fastmath=True)
def _split_full_multi3(seq, Dtr, Ddr, alpha, endurance, max_span, m):
    """Arg-tracking variant of _split_cost_multi3 (same O(1) run-decomposition skip costs).
    Returns (makespan, argi, argjs) where argjs[k] holds up to 3 drone positions
    (-1 padded) of the operation ending at k."""
    L = seq.shape[0] - 1
    INF = 1e18
    pref = np.zeros(L + 1)
    for a in range(L):
        pref[a + 1] = pref[a] + Dtr[seq[a], seq[a + 1]]
    best = np.full(L + 1, INF)
    best[0] = 0.0
    argi = np.full(L + 1, -1, dtype=np.int64)
    argjs = np.full((L + 1, 3), -1, dtype=np.int64)
    for k in range(1, L + 1):
        lo = k - max_span
        if lo < 0:
            lo = 0
        for i in range(lo, k):
            bi = best[i]
            if bi >= INF:
                continue
            cj1 = -1
            cj2 = -1
            cj3 = -1
            if k == i + 1:
                c = Dtr[seq[i], seq[k]]
            else:
                inter = k - i - 1
                si = seq[i]
                sk = seq[k]
                full_path = pref[k] - pref[i]
                c = INF  # multi-leg op must use >=1 drone (pure truck = single legs)
                drt = np.empty(inter)
                sav1 = np.empty(inter)
                for t in range(inter):
                    p = i + 1 + t
                    sj = seq[p]
                    fl = Ddr[si, sj] + Ddr[sj, sk]
                    drt[t] = fl / alpha if fl <= endurance else -1.0
                    sav1[t] = (pref[p + 1] - pref[p - 1]) - Dtr[seq[p - 1], seq[p + 1]]
                for t1 in range(inter):
                    if drt[t1] < 0.0:
                        continue
                    p1 = i + 1 + t1
                    tr1 = full_path - sav1[t1]
                    cc = tr1 if tr1 > drt[t1] else drt[t1]
                    if cc < c:
                        c = cc
                        cj1 = p1
                        cj2 = -1
                        cj3 = -1
                    if m >= 2:
                        for t2 in range(t1 + 1, inter):
                            if drt[t2] < 0.0:
                                continue
                            p2 = i + 1 + t2
                            md2 = drt[t1] if drt[t1] > drt[t2] else drt[t2]
                            if t2 == t1 + 1:
                                tr2 = full_path - (pref[p2 + 1] - pref[p1 - 1]) \
                                    + Dtr[seq[p1 - 1], seq[p2 + 1]]
                            else:
                                tr2 = full_path - sav1[t1] - sav1[t2]
                            cc2 = tr2 if tr2 > md2 else md2
                            if cc2 < c:
                                c = cc2
                                cj1 = p1
                                cj2 = p2
                                cj3 = -1
                            if m >= 3:
                                for t3 in range(t2 + 1, inter):
                                    if drt[t3] < 0.0:
                                        continue
                                    p3 = i + 1 + t3
                                    md3 = md2 if md2 > drt[t3] else drt[t3]
                                    a12 = t2 == t1 + 1
                                    a23 = t3 == t2 + 1
                                    if a12 and a23:
                                        tr3 = full_path - (pref[p3 + 1] - pref[p1 - 1]) \
                                            + Dtr[seq[p1 - 1], seq[p3 + 1]]
                                    elif a12:
                                        tr3 = full_path - (pref[p2 + 1] - pref[p1 - 1]) \
                                            + Dtr[seq[p1 - 1], seq[p2 + 1]] - sav1[t3]
                                    elif a23:
                                        tr3 = full_path - sav1[t1] \
                                            - (pref[p3 + 1] - pref[p2 - 1]) \
                                            + Dtr[seq[p2 - 1], seq[p3 + 1]]
                                    else:
                                        tr3 = full_path - sav1[t1] - sav1[t2] - sav1[t3]
                                    cc3 = tr3 if tr3 > md3 else md3
                                    if cc3 < c:
                                        c = cc3
                                        cj1 = p1
                                        cj2 = p2
                                        cj3 = p3
            if c >= INF:
                continue
            v = bi + c
            if v < best[k]:
                best[k] = v
                argi[k] = i
                argjs[k, 0] = cj1
                argjs[k, 1] = cj2
                argjs[k, 2] = cj3
    return best[L], argi, argjs


# ----------------------------- instances -----------------------------
def gen_instance(n, seed, depot="center"):
    """n customers ~ U[0,1]^2; depot at center (default), corner, or random. Node 0 = depot.
    The customer stream depends only on the seed, so the same seed with different depot
    positions yields identical demand (used to deconfound depot location from demand)."""
    rng = np.random.default_rng(seed)
    cust = rng.random((n, 2))
    if depot == "center":
        d = np.array([[0.5, 0.5]])
    elif depot == "corner":
        d = np.array([[0.05, 0.05]])
    else:
        d = rng.random((1, 2))
    coords = np.vstack([d, cust])  # (n+1, 2), index 0 = depot
    return {"coords": coords, "n": n, "seed": int(seed), "depot": depot, "kind": "uniform"}


def gen_clustered(n, seed, n_clusters=None, spread=0.06, depot="corner"):
    """Clustered demand (Solomon C-type style): customers drawn around random cluster
    centres (Gaussian, std=spread), clipped to the unit square. Depot at a corner by
    default (realistic for a depot outside the service area)."""
    rng = np.random.default_rng(seed)
    if n_clusters is None:
        n_clusters = max(3, n // 7)
    centres = rng.random((n_clusters, 2))
    assign = rng.integers(0, n_clusters, size=n)
    cust = np.clip(centres[assign] + rng.normal(0, spread, size=(n, 2)), 0.0, 1.0)
    if depot == "center":
        d = np.array([[0.5, 0.5]])
    elif depot == "corner":
        d = np.array([[0.05, 0.05]])
    else:
        d = rng.random((1, 2))
    coords = np.vstack([d, cust])
    return {"coords": coords, "n": n, "seed": int(seed), "depot": depot,
            "kind": "clustered", "n_clusters": int(n_clusters)}


def dist_matrix(coords, metric="euclidean"):
    """Pairwise distances. metric='euclidean' (drone, straight-line) or 'manhattan'
    (truck on a grid road network, L1)."""
    diff = coords[:, None, :] - coords[None, :, :]
    if metric == "manhattan":
        return np.abs(diff).sum(-1)
    return np.sqrt((diff ** 2).sum(-1))


# ----------------------------- DP split (single drone) -----------------------------
def tspd_split(order, Dtr, alpha, endurance=np.inf, max_span=None, Ddr=None):
    """
    Optimal min-makespan TSP-D decoding for a FIXED order (single drone).
    order: node indices beginning and ending at depot 0, e.g. [0, c1, ..., cn, 0].
    Dtr: truck distance matrix; Ddr: drone distance matrix (defaults to Dtr for the
    Euclidean case). Truck time = Dtr; drone time = Ddr/alpha.
    Returns (makespan, ops) with ops in {('truck',i,k), ('sortie',i,j,k)} (positions in order).
    """
    L = len(order) - 1
    if max_span is None:
        max_span = L
    if Ddr is None:
        Ddr = Dtr
    seq = np.ascontiguousarray(order, dtype=np.int64)
    endu = 1e18 if not np.isfinite(endurance) else float(endurance)
    ms, argi, argj = _split_full(seq, Dtr, Ddr, float(alpha), endu, int(max_span))
    ops = []
    k = L
    while k > 0:
        i = int(argi[k]); j = int(argj[k])
        ops.append(("truck", i, k) if j < 0 else ("sortie", i, j, k))
        k = i
    ops.reverse()
    return ms, ops


def _eligibility_array(eligible, n_nodes):
    """Normalize a customer eligibility mask for the JIT friction kernels."""
    if eligible is None:
        a = np.ones(n_nodes, dtype=np.uint8)
    else:
        a = np.ascontiguousarray(eligible, dtype=np.uint8)
        if a.shape != (n_nodes,):
            raise ValueError(f"eligible must have shape ({n_nodes},), got {a.shape}")
    a = a.copy()
    a[0] = 1  # the depot is never a drone customer; keep the sentinel admissible
    return a


def tspd_split_friction(order, Dtr, alpha, endurance=np.inf, max_span=None, Ddr=None,
                        t_service=0.0, t_launch=0.0, t_recover=0.0, eligible=None):
    """Fixed-order single-drone split with operational-friction parameters.

    ``t_service`` is paid once by the vehicle serving each customer. Launch and
    recovery durations are paid per sortie around the synchronized parallel interval.
    Endurance is still measured as flight distance, not elapsed sortie time.
    """
    seq = np.ascontiguousarray(order, dtype=np.int64)
    L = seq.shape[0] - 1
    if max_span is None:
        max_span = L
    if Ddr is None:
        Ddr = Dtr
    endu = 1e18 if not np.isfinite(endurance) else float(endurance)
    elig = _eligibility_array(eligible, Dtr.shape[0])
    ms, argi, argj = _split_full_friction(
        seq, Dtr, Ddr, float(alpha), endu, int(max_span), float(t_service),
        float(t_launch), float(t_recover), elig)
    ops = []
    k = L
    while k > 0:
        i = int(argi[k])
        if i < 0:
            raise RuntimeError("friction split failed to reconstruct a feasible path")
        j = int(argj[k])
        ops.append(("truck", i, k) if j < 0 else ("sortie", i, j, k))
        k = i
    ops.reverse()
    return ms, ops


def tspd_cost(order, Dtr, alpha, endurance=np.inf, max_span=None, Ddr=None):
    """Makespan-only (fast path; no ops reconstruction). Dtr=truck, Ddr=drone (default Dtr)."""
    L = len(order) - 1
    if max_span is None:
        max_span = L
    if Ddr is None:
        Ddr = Dtr
    seq = np.ascontiguousarray(order, dtype=np.int64)
    endu = 1e18 if not np.isfinite(endurance) else float(endurance)
    return _split_cost(seq, Dtr, Ddr, float(alpha), endu, int(max_span))


def tspd_cost_friction(order, Dtr, alpha, endurance=np.inf, max_span=None, Ddr=None,
                       t_service=0.0, t_launch=0.0, t_recover=0.0, eligible=None):
    """Makespan-only fast path for :func:`tspd_split_friction`."""
    seq = np.ascontiguousarray(order, dtype=np.int64)
    L = seq.shape[0] - 1
    if max_span is None:
        max_span = L
    if Ddr is None:
        Ddr = Dtr
    endu = 1e18 if not np.isfinite(endurance) else float(endurance)
    elig = _eligibility_array(eligible, Dtr.shape[0])
    return _split_cost_friction(
        seq, Dtr, Ddr, float(alpha), endu, int(max_span), float(t_service),
        float(t_launch), float(t_recover), elig)


@njit(cache=True, fastmath=True)
def _greedy_split(seq, Dtr, Ddr, alpha, endurance, max_span, argk, argj):
    """Myopic split used only as a deliberately weaker decoder baseline."""
    L = seq.shape[0] - 1
    i = 0
    total = 0.0
    t = 0
    while i < L:
        best_rate = Dtr[seq[i], seq[i + 1]]
        best_c = best_rate
        best_k = i + 1
        best_j = -1
        hi = min(i + max_span, L)
        legsum = Dtr[seq[i], seq[i + 1]]
        for k in range(i + 2, hi + 1):
            legsum += Dtr[seq[k - 1], seq[k]]
            si = seq[i]
            sk = seq[k]
            for j in range(i + 1, k):
                sj = seq[j]
                flight = Ddr[si, sj] + Ddr[sj, sk]
                if flight > endurance:
                    continue
                drone = flight / alpha
                truck = (legsum - Dtr[seq[j - 1], seq[j]]
                         - Dtr[seq[j], seq[j + 1]]
                         + Dtr[seq[j - 1], seq[j + 1]])
                cost = truck if truck > drone else drone
                rate = cost / (k - i)
                if rate < best_rate:
                    best_rate = rate
                    best_c = cost
                    best_k = k
                    best_j = j
        total += best_c
        argk[t] = best_k
        argj[t] = best_j
        t += 1
        i = best_k
    argk[t] = -1
    return total


def greedy_split_cost(order, Dtr, alpha, endurance=np.inf, max_span=None, Ddr=None):
    """Makespan-only path for the myopic decoder baseline."""
    seq = np.ascontiguousarray(order, dtype=np.int64)
    if Ddr is None:
        Ddr = Dtr
    if max_span is None:
        max_span = len(order) - 1
    argk = np.full(len(order), -1, dtype=np.int64)
    argj = np.full(len(order), -1, dtype=np.int64)
    endu = 1e18 if not np.isfinite(endurance) else float(endurance)
    return _greedy_split(seq, Dtr, Ddr, float(alpha), endu, int(max_span), argk, argj)


def greedy_split_ops(order, Dtr, alpha, endurance=np.inf, max_span=None, Ddr=None):
    """Myopic baseline decode with explicit operations."""
    seq = np.ascontiguousarray(order, dtype=np.int64)
    if Ddr is None:
        Ddr = Dtr
    if max_span is None:
        max_span = len(order) - 1
    argk = np.full(len(order), -1, dtype=np.int64)
    argj = np.full(len(order), -1, dtype=np.int64)
    endu = 1e18 if not np.isfinite(endurance) else float(endurance)
    makespan = _greedy_split(
        seq, Dtr, Ddr, float(alpha), endu, int(max_span), argk, argj)
    ops = []
    i = 0
    t = 0
    while argk[t] >= 0:
        k = int(argk[t])
        j = int(argj[t])
        ops.append(("truck", i, k) if j < 0 else ("sortie", i, j, k))
        i = k
        t += 1
    return makespan, ops


def tspd_cost_multi(order, Dtr, alpha, m, endurance=np.inf, max_span=6, Ddr=None):
    """Makespan-only with up to m parallel drones (small max_span recommended)."""
    if Ddr is None:
        Ddr = Dtr
    seq = np.ascontiguousarray(order, dtype=np.int64)
    endu = 1e18 if not np.isfinite(endurance) else float(endurance)
    return _split_cost_multi(seq, Dtr, Ddr, float(alpha), endu, int(max_span), int(m))


def tspd_cost_multi3(order, Dtr, alpha, m, endurance=np.inf, max_span=10, Ddr=None):
    """Makespan-only with up to m<=3 parallel drones; direct subset enumeration, so the
    same span cap as the single-drone search is affordable (symmetric decoding)."""
    if m > 3:
        raise ValueError("tspd_cost_multi3 supports m <= 3")
    if Ddr is None:
        Ddr = Dtr
    seq = np.ascontiguousarray(order, dtype=np.int64)
    endu = 1e18 if not np.isfinite(endurance) else float(endurance)
    return _split_cost_multi3(seq, Dtr, Ddr, float(alpha), endu, int(max_span), int(m))


def tspd_split_multi3(order, Dtr, alpha, m, endurance=np.inf, max_span=10, Ddr=None):
    """Full decode with up to m<=3 parallel drones. Returns (makespan, ops) with ops in
    {('truck', i, k), ('msortie', i, (j1[, j2[, j3]]), k)} (positions in order)."""
    if m > 3:
        raise ValueError("tspd_split_multi3 supports m <= 3")
    if Ddr is None:
        Ddr = Dtr
    L = len(order) - 1
    seq = np.ascontiguousarray(order, dtype=np.int64)
    endu = 1e18 if not np.isfinite(endurance) else float(endurance)
    ms, argi, argjs = _split_full_multi3(seq, Dtr, Ddr, float(alpha), endu,
                                         int(max_span), int(m))
    ops = []
    k = L
    while k > 0:
        i = int(argi[k])
        js = tuple(int(j) for j in argjs[k] if j >= 0)
        ops.append(("truck", i, k) if not js else ("msortie", i, js, k))
        k = i
    ops.reverse()
    return ms, ops


def obj_cost_lambda(order, Dtr, alpha, lam, endurance=np.inf, max_span=12, cl=1.0, cb=1.0, te=0.3, Ddr=None):
    """Scalarized makespan + lam*energy of the lambda-optimal decoding (search fast path)."""
    if Ddr is None:
        Ddr = Dtr
    seq = np.ascontiguousarray(order, dtype=np.int64)
    endu = 1e18 if not np.isfinite(endurance) else float(endurance)
    return _split_obj(seq, Dtr, Ddr, float(alpha), endu, int(max_span), float(lam),
                      float(cl), float(cb), float(te), 0)[0]


def decode_lambda(order, Dtr, alpha, lam, endurance=np.inf, max_span=12, cl=1.0, cb=1.0, te=0.3, Ddr=None):
    """Return (scalar_obj, makespan, drone_energy, ops) of the makespan+lam*energy-optimal decoding."""
    L = len(order) - 1
    if Ddr is None:
        Ddr = Dtr
    seq = np.ascontiguousarray(order, dtype=np.int64)
    endu = 1e18 if not np.isfinite(endurance) else float(endurance)
    obj, argi, argj = _split_obj(seq, Dtr, Ddr, float(alpha), endu, int(max_span), float(lam),
                                 float(cl), float(cb), float(te), 1)
    ms = 0.0; en = 0.0; ops = []; k = L
    while k > 0:
        i = int(argi[k]); j = int(argj[k])
        if j < 0:
            ms += Dtr[seq[i], seq[k]]; ops.append(("truck", i, k))
        else:
            t_out = Ddr[seq[i], seq[j]] / alpha; t_back = Ddr[seq[j], seq[k]] / alpha
            kept = [p for p in range(i, k + 1) if p != j]
            tr = sum(Dtr[seq[kept[a]], seq[kept[a + 1]]] for a in range(len(kept) - 1))
            ms += max(tr, t_out + t_back); en += cl * t_out + cb * (t_out + t_back)
            ops.append(("sortie", i, j, k))
        k = i
    ops.reverse()
    return obj, ms, en, ops


# ----------------------------- DP split (m drones) -----------------------------
def tspd_split_multi(order, Dt, alpha, m=1, endurance=np.inf, max_span=6):
    """
    Heuristic multi-drone extension: an operation seq[i]->seq[k] may launch up to m
    drones in parallel at seq[i], each serving one intermediate node, all rendezvous at
    seq[k]; the truck serves the remaining intermediate nodes in order. Operation cost =
    max(truck_time over kept nodes, max drone sortie time). Exact split for m=1.
    """
    if m == 1:
        return tspd_split(order, Dt, alpha, endurance, max_span)
    from itertools import combinations
    seq = order
    L = len(seq) - 1
    INF = float("inf")
    leg = [Dt[seq[a], seq[a + 1]] for a in range(L)]

    def truck_skip_set(i, k, S):
        """truck time over seq[i..k] skipping the set of positions S (in order)."""
        kept = [p for p in range(i, k + 1) if p == i or p == k or p not in S]
        t = 0.0
        for a in range(len(kept) - 1):
            t += Dt[seq[kept[a]], seq[kept[a + 1]]]
        return t

    def opcost(i, k):
        if k == i + 1:
            return leg[i], None
        inter = list(range(i + 1, k))
        best = INF
        bestS = None
        maxd = min(m, len(inter))
        # try assigning 1..maxd intermediate nodes to drones (subsets), rest to truck
        for nd in range(1, maxd + 1):
            for S in combinations(inter, nd):
                drt = []
                ok = True
                for j in S:
                    dr = (Dt[seq[i], seq[j]] + Dt[seq[j], seq[k]]) / alpha
                    if dr > endurance:
                        ok = False
                        break
                    drt.append(dr)
                if not ok:
                    continue
                tr = truck_skip_set(i, k, set(S))
                c = max(tr, max(drt))
                if c < best:
                    best, bestS = c, S
        return best, bestS

    best = [INF] * (L + 1)
    arg = [None] * (L + 1)
    best[0] = 0.0
    for k in range(1, L + 1):
        lo = max(0, k - max_span)
        for i in range(lo, k):
            if best[i] == INF:
                continue
            c, S = opcost(i, k)
            if c == INF:
                continue
            v = best[i] + c
            if v < best[k] - 1e-12:
                best[k] = v
                arg[k] = (i, S)
    ops = []
    k = L
    while k > 0:
        i, S = arg[k]
        ops.append(("truck", i, k) if S is None else ("msortie", i, tuple(S), k))
        k = i
    ops.reverse()
    return best[L], ops


# ----------------------------- helpers -----------------------------
def truck_only_length(order, Dt):
    return sum(Dt[order[a], order[a + 1]] for a in range(len(order) - 1))


def drone_customers(ops):
    """positions/nodes served by drone in a single-drone decoding."""
    out = []
    for op in ops:
        if op[0] == "sortie":
            out.append(op)  # (i, j, k)
    return out
