"""
Frontier completeness upgrade (round-4/5 fix: weighted sums sample only SUPPORTED
nondominated points, and the knee of a sampled curve is grid-sensitive).

Three steps per instance (n in {20,50}, seeds 0..29, nominal coefficients):
 1. ADAPTIVE REFINEMENT (one data-driven round): from the stored 13-lambda front, insert
    midpoint lambdas into the 6 intervals with the largest chord gap and run them
    (best-of-3, replicates archived).
 2. EXACT DECODING-LEVEL PARETO POST-PASS: for every archived winning order collected
    across all lambdas, compute the EXACT nondominated set of (makespan, E_total)
    decodings of that order by a Pareto-set dynamic program (composes per-operation
    (time, energy) pairs and prunes dominated states). This recovers nondominated
    decodings that the sampled weighted-sum searches missed (a sparse grid and heuristic
    search cannot prove unsupportedness), over the archived winner-order pool.
 3. GAP BOUND: for the final combined front, report the maximum possible improvement any
    missed point could still offer, bounded by the largest perpendicular chord gap between
    adjacent front points (normalized units).

Outputs experiments/H3-time-energy/results/h3_frontier.json
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from problem import gen_instance, dist_matrix
from alns import alns

HERE = os.path.dirname(__file__)
H3 = os.path.join(HERE, "..", "experiments", "H3-time-energy", "results", "h3_v2.json")
OUT = os.path.join(HERE, "..", "experiments", "H3-time-energy", "results", "h3_frontier.json")
ALPHA = 2.0
ITERS = {20: 9000, 50: 16000}
NSEED = 3
SPAN = 12
N_REFINE = 6


def pareto_decode(order, Dtr, alpha, cl=1.0, cb=1.0, te=0.3):
    """Exact nondominated (makespan, E_total) decodings of a FIXED order (single drone,
    unrestricted span). Pareto-set DP over positions; prunes dominated (time, energy)."""
    L = len(order) - 1
    seq = order
    states = [[] for _ in range(L + 1)]   # list of (time, energy) nondominated
    states[0] = [(0.0, 0.0)]

    def add(lst, t, e):
        lst.append((t, e))

    for k in range(1, L + 1):
        cand = []
        for i in range(k):
            if not states[i]:
                continue
            if k == i + 1:
                d = Dtr[seq[i], seq[k]]
                ops = [(d, te * d)]
            else:
                ops = []
                full = sum(Dtr[seq[a], seq[a + 1]] for a in range(i, k))
                for j in range(i + 1, k):
                    out_d = Dtr[seq[i], seq[j]]
                    back_d = Dtr[seq[j], seq[k]]
                    t_out = out_d / alpha
                    t_back = back_d / alpha
                    tr = full - Dtr[seq[j - 1], seq[j]] - Dtr[seq[j], seq[j + 1]] \
                        + Dtr[seq[j - 1], seq[j + 1]]
                    tm = max(tr, t_out + t_back)
                    en = cl * t_out + cb * (t_out + t_back) + te * tr
                    ops.append((tm, en))
            for (t0, e0) in states[i]:
                for (dt, de) in ops:
                    add(cand, t0 + dt, e0 + de)
        # prune to nondominated
        cand.sort()
        nd = []
        best_e = float("inf")
        for t, e in cand:
            if e < best_e - 1e-12:
                nd.append((t, e))
                best_e = e
        states[k] = nd
    return states[L]


def front_of(points):
    pts = sorted(points)
    out, be = [], float("inf")
    for t, e in pts:
        if e < be - 1e-12:
            out.append((t, e))
            be = e
    return out


def chord_gaps(front):
    """Max perpendicular slack between adjacent front points, normalized by the first
    (time-optimal) point; an unsupported point between i and i+1 can improve at most
    ~ half the gap rectangle. Returns the max normalized rectangle diagonal."""
    if len(front) < 2:
        return 0.0
    t0, e0 = front[0]
    g = 0.0
    for (t1, e1), (t2, e2) in zip(front, front[1:]):
        g = max(g, np.hypot((t2 - t1) / t0, (e1 - e2) / e0))
    return float(g)


def job_refine(args):
    n, seed, lam = args
    inst = gen_instance(n, seed=seed)
    D = dist_matrix(inst["coords"])
    reps = []
    best = None
    for s in range(NSEED):
        r = alns(inst, D, ALPHA, endurance=np.inf, max_span=SPAN, iters=ITERS[n],
                 sortie_aware=True, seed=s, lam=lam, cl=1.0, cb=1.0, te=0.3)
        ev = r["eval"]
        reps.append({"seed": s, "makespan": ev["makespan"], "e_total": ev["e_total"],
                     "objective": r["objective"]})
        if best is None or r["objective"] < best["objective"]:
            best = r
    ev = best["eval"]
    return {"n": n, "seed": seed, "lam": lam, "makespan": ev["makespan"],
            "e_total": ev["e_total"], "e_drone": ev["e_drone"],
            "replicates": reps, "order": [int(v) for v in best["order"]]}


def job_decode(args):
    n, seed, orders = args
    inst = gen_instance(n, seed=seed)
    D = dist_matrix(inst["coords"])
    pts = []
    for o in orders:
        pts.extend(pareto_decode(o, D, ALPHA))
    return {"n": n, "seed": seed, "decode_front": front_of(pts)}


def knee_of(front):
    if len(front) < 3:
        return None
    t0, e0 = front[0]
    x = np.array([t / t0 for t, e in front])
    y = np.array([e / e0 for t, e in front])
    chord = np.array([x[-1] - x[0], y[-1] - y[0]])
    cl_ = np.linalg.norm(chord) + 1e-12
    dxy = np.stack([x, y], 1) - np.array([x[0], y[0]])
    d = np.abs(chord[0] * dxy[:, 1] - chord[1] * dxy[:, 0]) / cl_
    ki = int(np.argmax(d))
    return (float((x[ki] - 1) * 100), float((1 - y[ki]) * 100))


def main():
    t0 = time.time()
    d = json.load(open(H3))
    main_rows = [r for r in d["raw"] if r["tag"] == "main"]

    # step 1: pick refinement lambdas per instance from the stored fronts
    refine_jobs = []
    stored = {}
    for n in (20, 50):
        for sd in range(30):
            pts = sorted([r for r in main_rows if r["n"] == n and r["seed"] == sd],
                         key=lambda p: p["lam"])
            stored[(n, sd)] = pts
            fr = front_of([(p["makespan"], p["e_total"]) for p in pts])
            lam_of = {}
            for p in pts:
                lam_of[(round(p["makespan"], 12), round(p["e_total"], 12))] = p["lam"]
            gaps = []
            t0p, e0p = fr[0]
            for (a, b) in zip(fr, fr[1:]):
                g = np.hypot((b[0] - a[0]) / t0p, (a[1] - b[1]) / e0p)
                la = lam_of.get((round(a[0], 12), round(a[1], 12)))
                lb = lam_of.get((round(b[0], 12), round(b[1], 12)))
                if la is not None and lb is not None and abs(lb - la) > 1e-9:
                    gaps.append((g, (la + lb) / 2))
            gaps.sort(reverse=True)
            for g, lam in gaps[:N_REFINE]:
                refine_jobs.append((n, sd, round(lam, 6)))
    print(f"{len(refine_jobs)} refinement runs")
    with Pool(36) as pool:
        new_rows = pool.map(job_refine, refine_jobs)

        # step 2: decode-level Pareto post-pass over all winning orders
        decode_jobs = []
        for n in (20, 50):
            for sd in range(30):
                orders = [d["routes"][f"main-n{n}-s{sd}-l{r['lam']}-cl1.0-te0.3"]
                          for r in stored[(n, sd)]]
                orders += [r["order"] for r in new_rows if r["n"] == n and r["seed"] == sd]
                # dedupe
                seen, uniq = set(), []
                for o in orders:
                    k = tuple(o)
                    if k not in seen:
                        seen.add(k)
                        uniq.append(list(o))
                decode_jobs.append((n, sd, uniq))
        dec = pool.map(job_decode, decode_jobs)

    out = {"config": {"n_refine_per_instance": N_REFINE, "nseed": NSEED,
                      "iters": {str(k): v for k, v in ITERS.items()}, "span": SPAN,
                      "runtime_s": None},
           "summary": {}, "refined_rows": [{k: r[k] for k in r if k != "order"}
                                           for r in new_rows]}
    decmap = {(r["n"], r["seed"]): r["decode_front"] for r in dec}
    for n in (20, 50):
        knees, gaps_before, gaps_after, unsupported_gain = [], [], [], []
        for sd in range(30):
            base_pts = [(p["makespan"], p["e_total"]) for p in stored[(n, sd)]]
            ref_pts = [(r["makespan"], r["e_total"]) for r in new_rows
                       if r["n"] == n and r["seed"] == sd]
            f_before = front_of(base_pts)
            f_mid = front_of(base_pts + ref_pts)
            f_after = front_of(f_mid + decmap[(n, sd)])
            gaps_before.append(chord_gaps(f_before))
            gaps_after.append(chord_gaps(f_after))
            k = knee_of(f_after)
            if k:
                knees.append(k)
            # how much did decode-level (unsupported-capable) points improve the front?
            area_gain = 0.0
            for (t, e) in f_after:
                dom = any(bt <= t + 1e-12 and be <= e + 1e-12 for bt, be in f_mid)
                if not dom:
                    dmin = min(max((t - bt) / f_mid[0][0], (e - be) / f_mid[0][1])
                               for bt, be in f_mid)
                    area_gain = max(area_gain, -dmin)
            unsupported_gain.append(float(area_gain))
        dms = np.array([k[0] for k in knees])
        de = np.array([k[1] for k in knees])
        out["summary"][str(n)] = {
            "knee_dms_median": float(np.median(dms)),
            "knee_dms_iqr": [float(np.percentile(dms, 25)), float(np.percentile(dms, 75))],
            "knee_de_median": float(np.median(de)),
            "knee_de_iqr": [float(np.percentile(de, 25)), float(np.percentile(de, 75))],
            "max_chord_gap_before_mean": float(np.mean(gaps_before)),
            "max_chord_gap_after_mean": float(np.mean(gaps_after)),
            "unsupported_front_improvement_max": float(np.max(unsupported_gain)),
            "n_knees": len(knees)}
    # archive the COMPLETED per-instance fronts and the completeness audit (round-10 fix:
    # these fields were produced by the original session but not written by this driver;
    # they are now first-class outputs so the artifact regenerates end-to-end), and keep
    # the refinement winner ORDERS so the decode pass is reproducible from the archive.
    out["refined_orders"] = {f"n{r['n']}-s{r['seed']}-l{r['lam']}": r["order"] for r in new_rows}
    out["combined_fronts"] = {}
    out["completeness_audit"] = {}
    for n in (20, 50):
        fronts = {}
        eps_list, extra_total = [], 0
        g_base, g_mid, g_comb = [], [], []
        for sd in range(30):
            base_pts = [(p["makespan"], p["e_total"]) for p in stored[(n, sd)]]
            ref_pts = [(r["makespan"], r["e_total"]) for r in new_rows
                       if r["n"] == n and r["seed"] == sd]
            f_mid = front_of(base_pts + ref_pts)
            f_after = front_of(f_mid + decmap[(n, sd)])
            fronts[str(sd)] = [[t, e] for t, e in f_after]
            g_base.append(chord_gaps(front_of(base_pts)))
            g_mid.append(chord_gaps(f_mid))
            g_comb.append(chord_gaps(f_after))
            extra_total += sum(1 for (t, e) in f_after
                               if not any(bt <= t + 1e-12 and be <= e + 1e-12 for bt, be in f_mid))
            # additive-epsilon indicator of the weighted-sum front vs the completed front:
            # how far (relative to the time-optimal endpoint scales) f_mid must shift to
            # weakly dominate every completed-front point
            t0n, e0n = f_after[0][0], f_after[0][1]
            eps = 0.0
            for (t, e) in f_after:
                d = min(max((bt - t) / t0n, (be - e) / e0n) for bt, be in f_mid)
                eps = max(eps, d)
            eps_list.append(100.0 * max(0.0, eps))
        out["combined_fronts"][str(n)] = fronts
        out["completeness_audit"][str(n)] = {
            "eps_indicator_ws_vs_complete_pct": {"mean": float(np.mean(eps_list)),
                                                 "max": float(np.max(eps_list))},
            "chord_gap_base_mean": float(np.mean(g_base)),
            "chord_gap_after_refinement_only_mean": float(np.mean(g_mid)),
            "chord_gap_combined_mean": float(np.mean(g_comb)),
            "extra_front_points_total": extra_total}
    out["config"]["runtime_s"] = time.time() - t0
    json.dump(out, open(OUT, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(out["summary"], indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
