"""
Decoder x operator 2x2 factorial (review fix item 7): the v1 claim "the decoder, not the
neighbourhood, is the dominant lever" was never tested causally -- both v1 ablation arms
used the exact DP split. Here we cross {exact DP split, greedy myopic split} x
{sortie-aware, generic} local moves at equal iteration budget (and equal split-call
structure), n=50, 30 instances, best-of-3, and report main effects and the interaction.
Wall-clock per arm is recorded (the greedy decoder is cheaper per call; equal-iteration
is the controlled comparison, disclosed).

Outputs experiments/P0-robustness/results/ops_factorial_v2.json
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from numba import njit
from scipy import stats

from problem import gen_instance, dist_matrix
from alns import (alns, random_removal, worst_removal, shaw_removal,
                  greedy_insertion, regret2_insertion, sortie_aware_move, generic_move)
import math

OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "P0-robustness",
                   "results", "ops_factorial_v2.json")
N = 50
SEEDS = range(30)
NSEED = 3
ITERS = 16000
SPAN = 12
ALPHA = 2.0


@njit(cache=True, fastmath=True)
def _greedy_split(seq, Dtr, Ddr, alpha, endurance, max_span, argk, argj):
    """Myopic decoder: from position i, take the single operation (truck leg, or sortie
    (i,j,k), j strictly between) with the lowest cost per customer served; repeat.
    Fills argk/argj (op end / drone position, -1 = truck leg) and returns total time."""
    L = seq.shape[0] - 1
    i = 0
    total = 0.0
    t = 0
    while i < L:
        best_rate = Dtr[seq[i], seq[i + 1]]  # truck leg serves 1 customer
        best_c = best_rate
        best_k = i + 1
        best_j = -1
        hi = i + max_span
        if hi > L:
            hi = L
        legsum = Dtr[seq[i], seq[i + 1]]
        for k in range(i + 2, hi + 1):
            legsum += Dtr[seq[k - 1], seq[k]]
            si = seq[i]
            sk = seq[k]
            for j in range(i + 1, k):
                sj = seq[j]
                fl = Ddr[si, sj] + Ddr[sj, sk]
                if fl > endurance:
                    continue
                dr = fl / alpha
                tr = legsum - Dtr[seq[j - 1], seq[j]] - Dtr[seq[j], seq[j + 1]] \
                    + Dtr[seq[j - 1], seq[j + 1]]
                cc = tr if tr > dr else dr
                rate = cc / (k - i)              # customers served = k - i
                if rate < best_rate:
                    best_rate = rate
                    best_c = cc
                    best_k = k
                    best_j = j
        total += best_c
        argk[t] = best_k
        argj[t] = best_j
        t += 1
        i = best_k
    argk[t] = -1
    return total


def greedy_cost(order, Dtr, Ddr, alpha, endurance, max_span):
    seq = np.ascontiguousarray(order, dtype=np.int64)
    argk = np.full(len(order), -1, dtype=np.int64)
    argj = np.full(len(order), -1, dtype=np.int64)
    endu = 1e18 if not np.isfinite(endurance) else float(endurance)
    return _greedy_split(seq, Dtr, Ddr, float(alpha), endu, int(max_span), argk, argj)


def greedy_split_ops(order, Dtr, Ddr, alpha, endurance, max_span):
    seq = np.ascontiguousarray(order, dtype=np.int64)
    argk = np.full(len(order), -1, dtype=np.int64)
    argj = np.full(len(order), -1, dtype=np.int64)
    endu = 1e18 if not np.isfinite(endurance) else float(endurance)
    ms = _greedy_split(seq, Dtr, Ddr, float(alpha), endu, int(max_span), argk, argj)
    ops = []
    i = 0
    t = 0
    while argk[t] >= 0:
        k = int(argk[t])
        j = int(argj[t])
        ops.append(("truck", i, k) if j < 0 else ("sortie", i, j, k))
        i = k
        t += 1
    return ms, ops


def alns_greedy_decoder(inst, Dt, alpha, iters, seed, sortie_aware):
    """Same ALNS loop as alns() but with the greedy myopic decoder as cost/split
    (the 'weak decoder' arms of the factorial). Final solution re-decoded greedily."""
    rng = np.random.default_rng(seed)
    n = inst["n"]
    from baselines import truck_only_tsp
    to, _ = truck_only_tsp(inst, Dt, restarts=2, seed=seed)
    cur = [c for c in to if c != 0]

    def cost(o):
        return greedy_cost([0] + o + [0], Dt, Dt, alpha, np.inf, SPAN)

    def split(o):
        return greedy_split_ops([0] + o + [0], Dt, Dt, alpha, np.inf, SPAN)

    cur_obj = cost(cur)
    best, best_obj = cur[:], cur_obj
    T = T0 = max(cur_obj * 0.05, 1e-6)
    no_improve = 0
    restart_gap = max(400, n * 20)
    destroys = [random_removal, worst_removal, shaw_removal]
    repairs = [greedy_insertion, regret2_insertion]
    dw = np.ones(3); rw = np.ones(2)
    ds = np.zeros(3); rs = np.zeros(2)
    dn = np.zeros(3); rn = np.zeros(2)
    local = sortie_aware_move if sortie_aware else generic_move

    def double_bridge(o, rng):
        L = len(o)
        if L < 8:
            return o[:]
        p = sorted(rng.choice(range(1, L), size=3, replace=False))
        a, b, c = p
        return o[:a] + o[b:c] + o[a:b] + o[c:]

    for it in range(iters):
        di = rng.choice(3, p=dw / dw.sum())
        ri = rng.choice(2, p=rw / rw.sum())
        q = int(rng.integers(2, max(3, n // 4)))
        keep, removed = destroys[di](cur, min(q, len(cur) - 1), rng, Dt)
        cand = repairs[ri](keep, removed, rng, Dt)
        cand, cm = local(cand, cost, split, Dt, rng)
        for _ in range(3):
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
        T *= 0.994
        if no_improve >= restart_gap:
            cur = double_bridge(best, rng)
            cur_obj = cost(cur)
            T = T0
            no_improve = 0
        if (it + 1) % 200 == 0:
            for arr_w, arr_s, arr_n in ((dw, ds, dn), (rw, rs, rn)):
                for a in range(len(arr_w)):
                    if arr_n[a] > 0:
                        arr_w[a] = 0.8 * arr_w[a] + 0.2 * (arr_s[a] / arr_n[a])
                        arr_w[a] = max(arr_w[a], 0.05)
            ds[:] = rs[:] = dn[:] = rn[:] = 0
    return best_obj


def job(args):
    seed, decoder, sortie = args
    inst = gen_instance(N, seed=seed)
    D = dist_matrix(inst["coords"])
    t0 = time.time()
    if decoder == "exact":
        ms = min(alns(inst, D, ALPHA, endurance=np.inf, max_span=SPAN, iters=ITERS,
                      sortie_aware=sortie, seed=s)["makespan"] for s in range(NSEED))
    else:
        ms = min(alns_greedy_decoder(inst, D, ALPHA, ITERS, s, sortie)
                 for s in range(NSEED))
    return {"seed": seed, "decoder": decoder, "sortie": sortie, "makespan": float(ms),
            "wall_s": time.time() - t0}


def main():
    t0 = time.time()
    jobs = [(s, d, so) for s in SEEDS for d in ("exact", "greedy") for so in (True, False)]
    print(f"{len(jobs)} factorial runs")
    with Pool(36) as pool:
        rows = pool.map(job, jobs)

    def arm(d, so):
        return {r["seed"]: r for r in rows if r["decoder"] == d and r["sortie"] == so}

    ee, eg = arm("exact", True), arm("exact", False)
    ge, gg = arm("greedy", True), arm("greedy", False)
    seeds = sorted(ee)
    out = {"config": {"n": N, "iters": ITERS, "nseed": NSEED, "span": SPAN,
                      "instances": len(seeds)},
           "arm_means": {"exact_sortie": float(np.mean([ee[s]["makespan"] for s in seeds])),
                         "exact_generic": float(np.mean([eg[s]["makespan"] for s in seeds])),
                         "greedy_sortie": float(np.mean([ge[s]["makespan"] for s in seeds])),
                         "greedy_generic": float(np.mean([gg[s]["makespan"] for s in seeds]))},
           "arm_wall_s_mean": {"exact_sortie": float(np.mean([ee[s]["wall_s"] for s in seeds])),
                               "exact_generic": float(np.mean([eg[s]["wall_s"] for s in seeds])),
                               "greedy_sortie": float(np.mean([ge[s]["wall_s"] for s in seeds])),
                               "greedy_generic": float(np.mean([gg[s]["wall_s"] for s in seeds]))}}
    dec_eff = np.array([(ge[s]["makespan"] + gg[s]["makespan"]) / 2
                        - (ee[s]["makespan"] + eg[s]["makespan"]) / 2 for s in seeds])
    op_eff = np.array([(eg[s]["makespan"] + gg[s]["makespan"]) / 2
                       - (ee[s]["makespan"] + ge[s]["makespan"]) / 2 for s in seeds])
    inter = np.array([(gg[s]["makespan"] - ge[s]["makespan"])
                      - (eg[s]["makespan"] - ee[s]["makespan"]) for s in seeds])
    for name, eff in (("decoder_effect", dec_eff), ("operator_effect", op_eff),
                      ("interaction", inter)):
        out[name] = {"mean": float(eff.mean()),
                     "rel_pct_of_exact_sortie": float(
                         eff.mean() / out["arm_means"]["exact_sortie"] * 100),
                     "wilcoxon_p": float(stats.wilcoxon(eff).pvalue)
                     if np.abs(eff).max() > 1e-12 else None}
    out["raw"] = rows
    out["config"]["runtime_s"] = time.time() - t0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: out[k] for k in ("arm_means", "decoder_effect", "operator_effect",
                                          "interaction")}, indent=1))
    print(f"ops factorial done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
