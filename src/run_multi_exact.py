"""
Exact multi-drone validation (round-4/5 fix): the multi-drone arm previously had no
independent optimality reference. Within the batch model (up to m drones sharing a launch
and rendezvous per operation), the exact optimum over ALL customer orders is computable at
small n by exhaustive enumeration with the full-span multi3 decode. We compare the unified
ALNS (same configuration as every primary experiment) against this exact reference for
m in {1,2,3} at n=8 and n=9, at two endurance levels, 10 instances each.

All replicate outcomes and seeds are archived (round-4 archiving fix applies to new runs).

Outputs experiments/VALIDATION/results/multi_exact_v2.json
"""
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from problem import gen_instance, dist_matrix, tspd_cost_multi3
from alns import alns

OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "VALIDATION", "results",
                   "multi_exact_v2.json")
ALPHA = 2.0
NSEED = 3


def brute_force_multi(inst, D, m, endurance):
    """Exact min-makespan over all orders under the batch multi-drone decode (full span)."""
    n = inst["n"]
    best = float("inf")
    for perm in itertools.permutations(range(1, n + 1)):
        order = [0] + list(perm) + [0]
        ms = tspd_cost_multi3(order, D, ALPHA, m, endurance=endurance, max_span=n + 1)
        if ms < best:
            best = ms
    return best


def job(args):
    n, seed, m, E = args
    inst = gen_instance(n, seed=seed)
    D = dist_matrix(inst["coords"])
    opt = brute_force_multi(inst, D, m, E)
    reps = []
    for s in range(NSEED):
        r = alns(inst, D, ALPHA, endurance=E, max_span=12, iters=5000, sortie_aware=True,
                 seed=s, m=m)
        reps.append({"seed": s, "makespan": r["makespan"]})
    best = min(rp["makespan"] for rp in reps)
    return {"n": n, "seed": seed, "m": m, "E": (None if np.isinf(E) else E),
            "exact": float(opt), "alns_best": float(best),
            "gap_pct": float((best - opt) / opt * 100), "replicates": reps}


def main():
    t0 = time.time()
    jobs = [(n, sd, m, E) for n in (8, 9) for sd in range(10) for m in (1, 2, 3)
            for E in (np.inf, 0.8)]
    print(f"{len(jobs)} exact-reference comparisons")
    with Pool(36) as pool:
        rows = pool.map(job, jobs)
    gaps = np.array([r["gap_pct"] for r in rows])
    out = {"config": {"alpha": ALPHA, "nseed": NSEED, "iters": 5000, "span": 12,
                      "note": "exact = exhaustive over all orders, full-span batch decode",
                      "runtime_s": time.time() - t0},
           "summary": {"count": len(rows),
                       "exact_matches": int((gaps < 1e-9).sum()),
                       "max_gap_pct": float(gaps.max()),
                       "by_m": {str(m): {"count": int(sum(1 for r in rows if r["m"] == m)),
                                         "exact": int(sum(1 for r in rows
                                                          if r["m"] == m and r["gap_pct"] < 1e-9)),
                                         "max_gap_pct": float(max(r["gap_pct"] for r in rows
                                                                  if r["m"] == m))}
                                for m in (1, 2, 3)}},
           "raw": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(out["summary"], indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
