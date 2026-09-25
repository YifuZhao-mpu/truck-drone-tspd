"""
Adaptive Large Neighborhood Search for the min-makespan TSP-D.

Search space = customer visiting ORDER; each candidate order is decoded to its optimal
makespan TSP-D structure by the DP split (problem.tspd_split). Adaptive destroy/repair
(Pisinger & Ropke template) + simulated-annealing acceptance.

The H1 contribution is the set of FIRST-CLASS SORTIE-AWARE operators (sortie_aware=True):
drone-relocate, drone/truck swap, and sortie-window reversal — order perturbations informed
by the decoded sortie structure. The ablation (sortie_aware=False) replaces them with a
generic, mode-agnostic 2-opt/Or-opt local move of equal call frequency (fair compute).
"""
import math
import numpy as np
from problem import (tspd_split, tspd_cost, tspd_cost_multi3, tspd_split_multi3,
                     tspd_split_friction, tspd_cost_friction,
                     greedy_split_cost, greedy_split_ops,
                     obj_cost_lambda, decode_lambda)


# ----------------------------- destroy -----------------------------
def random_removal(order, q, rng, Dt=None):
    idx = rng.choice(len(order), size=q, replace=False)
    removed = [order[i] for i in idx]
    keep = [c for i, c in enumerate(order) if i not in set(idx)]
    return keep, removed


def worst_removal(order, q, rng, Dt):
    full = [0] + order + [0]
    det = []
    for p in range(1, len(full) - 1):
        a, c, b = full[p - 1], full[p], full[p + 1]
        det.append((Dt[a, c] + Dt[c, b] - Dt[a, b], order[p - 1]))
    det.sort(reverse=True)
    # randomized worst (Shaw-style determinism control)
    chosen = []
    pool = [c for _, c in det]
    while len(chosen) < q and pool:
        y = rng.random() ** 3
        k = int(y * len(pool))
        chosen.append(pool.pop(k))
    rem = set(chosen)
    keep = [c for c in order if c not in rem]
    return keep, chosen


def shaw_removal(order, q, rng, Dt):
    seed = order[rng.integers(len(order))]
    rel = sorted(order, key=lambda c: Dt[seed, c])
    removed = rel[:q]
    rem = set(removed)
    keep = [c for c in order if c not in rem]
    return keep, removed


# ----------------------------- repair -----------------------------
def _best_pos(order, c, Dt):
    full = [0] + order + [0]
    best, bp = float("inf"), 0
    for p in range(len(full) - 1):
        a, b = full[p], full[p + 1]
        cost = Dt[a, c] + Dt[c, b] - Dt[a, b]
        if cost < best:
            best, bp = cost, p
    return bp, best


def greedy_insertion(order, removed, rng, Dt):
    order = order[:]
    rng.shuffle(removed)
    for c in removed:
        bp, _ = _best_pos(order, c, Dt)
        order.insert(bp, c)
    return order


def regret2_insertion(order, removed, rng, Dt):
    order = order[:]
    removed = removed[:]
    while removed:
        best_c, best_regret, best_p = None, -1.0, 0
        for c in removed:
            full = [0] + order + [0]
            costs = []
            for p in range(len(full) - 1):
                a, b = full[p], full[p + 1]
                costs.append((Dt[a, c] + Dt[c, b] - Dt[a, b], p))
            costs.sort()
            c1 = costs[0][0]
            c2 = costs[1][0] if len(costs) > 1 else costs[0][0]
            c3 = costs[2][0] if len(costs) > 2 else c2
            regret = (c2 - c1) + (c3 - c1)
            if regret > best_regret:
                best_regret, best_c, best_p = regret, c, costs[0][1]
        order.insert(best_p, best_c)
        removed.remove(best_c)
    return order


# ----------------------------- sortie-aware local moves (H1 contribution) -----------------------------
# Moves receive the search's own `cost(order) -> scalar` and `split(order) -> (scalar, ops)`
# closures, so every move optimizes the SAME objective the acceptance criterion uses
# (makespan, or makespan + lam*total energy) with the SAME decoder (single- or multi-drone,
# one span cap). This closes the v1 inconsistencies flagged in review: multi-drone decoding
# is no longer weaker inside moves, and lambda-search moves no longer optimize makespan only.

def sortie_aware_move(order, cost, split, Dt, rng):
    """Try drone-relocate, drone/truck swap, and sortie-window reversal; keep best improving."""
    base, ops = split(order)
    sorties = [op for op in ops if op[0] in ("sortie", "msortie")]
    if not sorties:
        return order, base
    best_order, best_val = order, base
    full = [0] + order + [0]

    # (a) drone-relocate: move a drone customer next to a different launch candidate
    op = sorties[rng.integers(len(sorties))]
    if op[0] == "sortie":
        jpos = op[2]
    else:
        js = op[2]
        jpos = js[int(rng.integers(len(js)))]
    jnode = full[jpos]
    cand = [c for c in order if c != jnode]
    rng.shuffle(cand)
    for anchor in cand[:6]:
        tmp = [c for c in order if c != jnode]
        ai = tmp.index(anchor)
        for pos in (ai, ai + 1):
            cand_order = tmp[:pos] + [jnode] + tmp[pos:]
            v = cost(cand_order)
            if v < best_val - 1e-12:
                best_val, best_order = v, cand_order

    # (b) drone/truck swap: swap a drone customer with a nearby (spatial) truck customer
    truck_nodes = [full[p] for op2 in ops if op2[0] == "truck"
                   for p in (op2[2],) if full[p] != 0]
    if truck_nodes:
        tn = min(truck_nodes, key=lambda t: Dt[jnode, t])
        o = order[:]
        ia, ib = o.index(jnode), o.index(tn)
        o[ia], o[ib] = o[ib], o[ia]
        v = cost(o)
        if v < best_val - 1e-12:
            best_val, best_order = v, o

    # (c) sortie-window reversal: reverse truck nodes strictly between launch and rendezvous
    i, k = op[1], op[-1]
    if k - i >= 3:
        a, b = i, k - 2  # positions in `order` are full-pos-1
        if 0 <= a < b < len(order):
            o = order[:]
            o[a:b + 1] = o[a:b + 1][::-1]
            v = cost(o)
            if v < best_val - 1e-12:
                best_val, best_order = v, o
    return best_order, best_val


def generic_move(order, cost, split, Dt, rng):
    """Ablation: mode-agnostic local move (random 2-opt / Or-opt), equal call frequency."""
    base = cost(order)
    best_order, best_val = order, base
    L = len(order)
    if L < 4:
        return order, base
    for _ in range(4):
        if rng.random() < 0.5:  # 2-opt segment reversal
            a, b = sorted(rng.choice(L, size=2, replace=False))
            if b - a >= 1:
                o = order[:]
                o[a:b + 1] = o[a:b + 1][::-1]
        else:                    # Or-opt move of a single customer
            a = rng.integers(L)
            c = order[a]
            o = order[:a] + order[a + 1:]
            p = rng.integers(len(o) + 1)
            o = o[:p] + [c] + o[p:]
        v = cost(o)
        if v < best_val - 1e-12:
            best_val, best_order = v, o
    return best_order, best_val


# ----------------------------- ALNS driver -----------------------------
def alns(inst, Dt, alpha, endurance=np.inf, max_span=10, iters=8000,
         sortie_aware=True, seed=0, init_order=None, T0_frac=0.05, cool=0.994,
         m=1, Ddr=None, lam=0.0, cl=1.0, cb=1.0, te=0.3, final_span=None,
         t_service=0.0, t_launch=0.0, t_recover=0.0, eligible=None,
         decoder="dp"):
    """
    One ALNS for every experiment (v2 consistency fix). The search objective is
    makespan (lam=0) or makespan + lam * total energy (drone Dorling cl/cb + truck te per
    distance) via the lambda-aware exact split; multi-drone (m<=3) uses the direct-subset
    split at the SAME span cap and the SAME local moves as m=1 (symmetric decoding).
    Search decodes at max_span; the returned solution is re-decoded at final_span
    (default: unrestricted for m=1, min(L,14) for m>1 -- disclosed in Methods).
    Returns dict with makespan, order, ops, eval (full time/distance/energy audit),
    objective (scalar search objective of the returned solution).
    Dt=truck distance matrix; Ddr=drone distance matrix (defaults to Dt, Euclidean case).
    """
    if decoder not in ("dp", "greedy"):
        raise ValueError("decoder must be 'dp' or 'greedy'")
    if decoder == "greedy" and (m != 1 or lam > 0.0):
        raise NotImplementedError("greedy baseline decoding supports only m=1, lam=0")
    if lam > 0.0 and m > 1:
        raise NotImplementedError("lambda-objective is implemented for m=1 only")
    friction = (t_service != 0.0 or t_launch != 0.0 or t_recover != 0.0
                or eligible is not None)
    if friction and (m != 1 or lam > 0.0):
        raise NotImplementedError(
            "operational-friction decoding is implemented for m=1, lam=0 only")
    rng = np.random.default_rng(seed)
    n = inst["n"]
    if init_order is None:
        from baselines import truck_only_tsp
        to, _ = truck_only_tsp(inst, Dt, restarts=2, seed=seed)
        cur = [c for c in to if c != 0]
    else:
        cur = [c for c in init_order if c != 0]

    if m == 1:
        if lam > 0.0:
            def cost(o):
                return obj_cost_lambda([0] + o + [0], Dt, alpha, lam, endurance, max_span,
                                       cl, cb, te, Ddr=Ddr)

            def split(o):
                obj, _ms, _en, ops = decode_lambda([0] + o + [0], Dt, alpha, lam, endurance,
                                                   max_span, cl, cb, te, Ddr=Ddr)
                return obj, ops
        else:
            if decoder == "greedy":
                if friction:
                    raise NotImplementedError(
                        "greedy baseline is not defined for operational friction")

                def cost(o):
                    return greedy_split_cost(
                        [0] + o + [0], Dt, alpha, endurance, max_span, Ddr=Ddr)

                def split(o):
                    return greedy_split_ops(
                        [0] + o + [0], Dt, alpha, endurance, max_span, Ddr=Ddr)
            elif friction:
                def cost(o):
                    return tspd_cost_friction(
                        [0] + o + [0], Dt, alpha, endurance, max_span, Ddr=Ddr,
                        t_service=t_service, t_launch=t_launch, t_recover=t_recover,
                        eligible=eligible)

                def split(o):
                    return tspd_split_friction(
                        [0] + o + [0], Dt, alpha, endurance, max_span, Ddr=Ddr,
                        t_service=t_service, t_launch=t_launch, t_recover=t_recover,
                        eligible=eligible)
            else:
                def cost(o):
                    return tspd_cost([0] + o + [0], Dt, alpha, endurance, max_span, Ddr=Ddr)

                def split(o):
                    return tspd_split([0] + o + [0], Dt, alpha, endurance, max_span, Ddr=Ddr)
    else:
        def cost(o):
            return tspd_cost_multi3([0] + o + [0], Dt, alpha, m, endurance, max_span, Ddr=Ddr)

        def split(o):
            return tspd_split_multi3([0] + o + [0], Dt, alpha, m, endurance, max_span, Ddr=Ddr)

    cur_obj = cost(cur)
    best, best_obj = cur[:], cur_obj
    T = T0 = max(cur_obj * T0_frac, 1e-6)
    no_improve = 0
    restart_gap = max(400, n * 20)

    def double_bridge(o, rng):
        L = len(o)
        if L < 8:
            return o[:]
        p = sorted(rng.choice(range(1, L), size=3, replace=False))
        a, b, c = p
        return o[:a] + o[b:c] + o[a:b] + o[c:]

    destroys = [random_removal, worst_removal, shaw_removal]
    repairs = [greedy_insertion, regret2_insertion]
    dw = np.ones(len(destroys)); rw = np.ones(len(repairs))
    ds = np.zeros(len(destroys)); rs = np.zeros(len(repairs))
    dn = np.zeros(len(destroys)); rn = np.zeros(len(repairs))
    local = sortie_aware_move if sortie_aware else generic_move

    for it in range(iters):
        di = rng.choice(len(destroys), p=dw / dw.sum())
        ri = rng.choice(len(repairs), p=rw / rw.sum())
        q = int(rng.integers(2, max(3, n // 4)))
        keep, removed = destroys[di](cur, min(q, len(cur) - 1), rng, Dt)
        cand = repairs[ri](keep, removed, rng, Dt)
        cand, cm = local(cand, cost, split, Dt, rng)
        for _ in range(3):                      # iterate to local optimum (both arms, all m)
            cand, _ = generic_move(cand, cost, split, Dt, rng)
            cand, g = local(cand, cost, split, Dt, rng)
            if g >= cm - 1e-9:
                break
            cm = g
        cand_obj = cost(cand)

        reward = 0.0
        if cand_obj < best_obj - 1e-12:
            best, best_obj = cand[:], cand_obj
            reward = 3.0
            no_improve = 0
        else:
            no_improve += 1
        if cand_obj < cur_obj - 1e-12:
            cur, cur_obj = cand, cand_obj
            reward = max(reward, 2.0)
        elif rng.random() < math.exp(-(cand_obj - cur_obj) / max(T, 1e-9)):
            cur, cur_obj = cand, cand_obj
            reward = max(reward, 1.0)
        ds[di] += reward; dn[di] += 1; rs[ri] += reward; rn[ri] += 1
        T *= cool
        if no_improve >= restart_gap:                # stagnation: perturb best + reheat
            cur = double_bridge(best, rng)
            cur_obj = cost(cur)
            T = T0
            no_improve = 0
        if (it + 1) % 200 == 0:                      # adaptive weight update
            for arr_w, arr_s, arr_n in ((dw, ds, dn), (rw, rs, rn)):
                for a in range(len(arr_w)):
                    if arr_n[a] > 0:
                        arr_w[a] = 0.8 * arr_w[a] + 0.2 * (arr_s[a] / arr_n[a])
                        arr_w[a] = max(arr_w[a], 0.05)
            ds[:] = rs[:] = dn[:] = rn[:] = 0

    # final re-decode of best at a generous span (exact for m=1; near-exact, disclosed, for m>1)
    full_best = [0] + best + [0]
    if m == 1:
        fs = (len(best) + 1) if final_span is None else final_span
        if lam > 0.0:
            fin_obj, _ms, _en, fin_ops = decode_lambda(full_best, Dt, alpha, lam, endurance,
                                                       fs, cl, cb, te, Ddr=Ddr)
        elif decoder == "greedy":
            _ms, fin_ops = greedy_split_ops(
                full_best, Dt, alpha, endurance, fs, Ddr=Ddr)
        elif friction:
            _ms, fin_ops = tspd_split_friction(
                full_best, Dt, alpha, endurance, fs, Ddr=Ddr,
                t_service=t_service, t_launch=t_launch, t_recover=t_recover,
                eligible=eligible)
        else:
            _ms, fin_ops = tspd_split(full_best, Dt, alpha, endurance, fs, Ddr=Ddr)
    else:
        fs = min(len(best) + 1, 14) if final_span is None else final_span
        _ms, fin_ops = tspd_split_multi3(full_best, Dt, alpha, m, endurance, fs, Ddr=Ddr)
    from energy import evaluate_solution
    ev = evaluate_solution(full_best, fin_ops, Dt, Dt if Ddr is None else Ddr, alpha,
                           cl=cl, cb=cb, te=te)
    if decoder == "greedy":
        ev["operation_reconstruction_makespan"] = ev["makespan"]
        ev["makespan"] = float(_ms)
    if friction:
        ev["movement_makespan"] = ev["makespan"]
        ev["makespan"] = float(_ms)
        ev["operational_friction"] = {
            "t_service": float(t_service), "t_launch": float(t_launch),
            "t_recover": float(t_recover)}
    fin_ms = ev["makespan"]
    return {"makespan": fin_ms, "order": full_best, "ops": fin_ops, "eval": ev,
            "objective": fin_ms + lam * ev["e_total"], "search_objective": best_obj}


def alns_energy(inst, Dt, alpha, lam, endurance=np.inf, max_span=12, iters=6000, seed=0,
                cl=1.0, cb=1.0, te=0.3, Ddr=None):
    """DEPRECATED (v1 H3 solver, kept only so the superseded run_h3.py remains runnable
    for provenance). Review 2026-07-16 found it inconsistent: it searches on
    makespan + lam*(drone + truck) energy but returns drone-only energy, and it is a
    weaker non-adaptive search without the sortie moves. Use alns(lam=...) instead."""
    rng = np.random.default_rng(seed)
    n = inst["n"]
    from baselines import truck_only_tsp
    to, _ = truck_only_tsp(inst, Dt, restarts=2, seed=seed)
    cur = [c for c in to if c != 0]

    def scost(o):
        return obj_cost_lambda([0] + o + [0], Dt, alpha, lam, endurance, max_span, cl, cb, te, Ddr=Ddr)

    cur_obj = scost(cur)
    best, best_obj = cur[:], cur_obj
    T = T0 = max(cur_obj * 0.05, 1e-6)
    no_improve = 0
    restart_gap = max(400, n * 20)
    destroys = [random_removal, worst_removal, shaw_removal]
    repairs = [greedy_insertion, regret2_insertion]
    for it in range(iters):
        di = rng.integers(len(destroys)); ri = rng.integers(len(repairs))
        q = int(rng.integers(2, max(3, n // 4)))
        keep, removed = destroys[di](cur, min(q, len(cur) - 1), rng, Dt)
        cand = repairs[ri](keep, removed, rng, Dt)
        # cheap inline local move (mode-agnostic)
        if len(cand) >= 4:
            a, b = sorted(rng.choice(len(cand), size=2, replace=False))
            o2 = cand[:]; o2[a:b + 1] = o2[a:b + 1][::-1]
            if scost(o2) < scost(cand):
                cand = o2
        co = scost(cand)
        if co < best_obj - 1e-12:
            best, best_obj = cand[:], co; no_improve = 0
        else:
            no_improve += 1
        if co < cur_obj - 1e-12 or rng.random() < math.exp(-(co - cur_obj) / max(T, 1e-9)):
            cur, cur_obj = cand, co
        T *= 0.9995
        if no_improve >= restart_gap:
            L = len(best)
            if L >= 8:
                p = sorted(rng.choice(range(1, L), size=3, replace=False))
                cur = best[:p[0]] + best[p[1]:p[2]] + best[p[0]:p[1]] + best[p[2]:]
            else:
                cur = best[:]
            cur_obj = scost(cur); T = T0; no_improve = 0
    _, ms, en, _ = decode_lambda([0] + best + [0], Dt, alpha, lam, endurance, max_span, cl, cb, te, Ddr=Ddr)
    return ms, en
