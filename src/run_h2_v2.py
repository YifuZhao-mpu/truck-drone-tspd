"""
H2 v2 -- TRUE alpha x endurance x m factorial (review fix: v1 ran alpha x E at m=1 only
plus an m-sweep at a single (alpha, E, n=20) slice, with asymmetric multi-drone decoding).

Full crossed design: alpha in {1,1.5,2,2.5,3} x E in {0.5,1,2,inf} x m in {1,2,3},
n in {20,50}, 30 instances (seeds 0..29), best-of-3 solver seeds. SYMMETRIC solver: same
ALNS, same sortie-aware moves, same span cap (12) for every m. Savings are measured
against the certified LKH truck-only reference (tsp_ref), not the internal heuristic.
Per-instance raws (incl. routes) are stored.

Outputs experiments/H2-design-space/results/h2_v2.json
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from problem import gen_instance, dist_matrix
from alns import alns
from tsp_ref import _load_cache, key_of

OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "H2-design-space",
                   "results", "h2_v2.json")
ALPHAS = [1.0, 1.5, 2.0, 2.5, 3.0]
ENDUR = [0.5, 1.0, 2.0, np.inf]
ENDUR_KEY = ["0.5", "1.0", "2.0", "inf"]
MS = [1, 2, 3]
NS = [20, 50]
SEEDS = range(30)
NSEED = 3
ITERS = {20: 9000, 50: 16000}
SPAN = 12

REF = _load_cache()


def job(args):
    n, seed, ai, ei, m = args
    inst = gen_instance(n, seed=seed)
    D = dist_matrix(inst["coords"])
    ref = REF[key_of("uniform", n, seed, "center", "euclidean")]["length"]
    best = None
    for s in range(NSEED):
        r = alns(inst, D, ALPHAS[ai], endurance=ENDUR[ei], max_span=SPAN,
                 iters=ITERS[n], sortie_aware=True, seed=s, m=m)
        if best is None or r["makespan"] < best["makespan"]:
            best = r
    ev = best["eval"]
    return {"n": n, "seed": seed, "ai": ai, "ei": ei, "m": m,
            "makespan": ev["makespan"], "truck_ref": ref,
            "saving_pct": (ref - ev["makespan"]) / ref * 100.0,
            "e_drone": ev["e_drone"], "e_truck": ev["e_truck"],
            "n_sorties": ev["n_sorties"], "order": [int(v) for v in best["order"]]}


def boot_ci(vals, nres=20000, seed=0):
    a = np.asarray(vals, dtype=float)
    bs = np.random.default_rng(seed).choice(a, size=(nres, len(a))).mean(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    t0 = time.time()
    jobs = [(n, s, ai, ei, m) for n in NS for s in SEEDS
            for ai in range(len(ALPHAS)) for ei in range(len(ENDUR)) for m in MS]
    print(f"{len(jobs)} factorial runs (grid {len(ALPHAS)}x{len(ENDUR)}x{len(MS)}, "
          f"{len(list(SEEDS))} instances, best-of-{NSEED})")
    with Pool(36) as pool:
        rows = pool.map(job, jobs)

    out = {"config": {"alphas": ALPHAS, "endurance": ENDUR_KEY, "ms": MS,
                      "n_instances": len(list(SEEDS)), "nseed": NSEED,
                      "iters": {str(k): v for k, v in ITERS.items()}, "span": SPAN,
                      "denominator": "best-found LKH truck-only reference (no optimality certificate)"},
           "cells": {}, "marginals": {}, "interactions": {}}
    for n in NS:
        cells = {}
        for ai, a in enumerate(ALPHAS):
            for ei, ek in enumerate(ENDUR_KEY):
                for m in MS:
                    sub = [r["saving_pct"] for r in rows
                           if r["n"] == n and r["ai"] == ai and r["ei"] == ei and r["m"] == m]
                    cells[f"a{a}_E{ek}_m{m}"] = {
                        "mean_saving": float(np.mean(sub)),
                        "ci95": boot_ci(sub, seed=ai * 100 + ei * 10 + m),
                        "n": len(sub),
                        "ceiling": (1 - 1 / (a * m + 1)) * 100}
        out["cells"][str(n)] = cells
        # marginal gains along each axis + two-way interaction tables
        marg = {}
        for m in MS:
            surf = np.array([[cells[f"a{a}_E{ek}_m{m}"]["mean_saving"]
                              for ek in ENDUR_KEY] for a in ALPHAS])
            marg[f"m{m}_surface"] = surf.tolist()
        drone_gain = {}
        for ai, a in enumerate(ALPHAS):
            for ei, ek in enumerate(ENDUR_KEY):
                g12 = cells[f"a{a}_E{ek}_m2"]["mean_saving"] - cells[f"a{a}_E{ek}_m1"]["mean_saving"]
                g23 = cells[f"a{a}_E{ek}_m3"]["mean_saving"] - cells[f"a{a}_E{ek}_m2"]["mean_saving"]
                drone_gain[f"a{a}_E{ek}"] = {"m1_to_2": g12, "m2_to_3": g23}
        out["marginals"][str(n)] = marg
        out["interactions"][str(n)] = {"drone_marginal_gain": drone_gain}
    out["raw"] = [{k: r[k] for k in r if k != "order"} for r in rows]
    out["routes"] = {f"n{r['n']}-s{r['seed']}-a{r['ai']}-e{r['ei']}-m{r['m']}": r["order"]
                     for r in rows}
    out["config"]["runtime_s"] = time.time() - t0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for n in NS:
        c = out["cells"][str(n)]
        print(f"n={n}: a2_E1: m1 {c['a2.0_E1.0_m1']['mean_saving']:.1f}% "
              f"m2 {c['a2.0_E1.0_m2']['mean_saving']:.1f}% m3 {c['a2.0_E1.0_m3']['mean_saving']:.1f}%")
    print(f"H2 v2 done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
