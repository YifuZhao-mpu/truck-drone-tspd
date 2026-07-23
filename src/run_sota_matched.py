"""
Matched-budget, pinned-version comparison vs TSPDrone.jl (round-8 optional item).

Protocol: both solvers receive the SAME wall-clock budget per instance (T = our standard
best-of-3 time for that size, measured on this machine), run single-threaded, launch
complete runs (fresh seeds) until the budget is exhausted, and keep the best. Every
replicate (cost + time) is archived on both sides. TSPDrone.jl pinned at commit
f42d27c0369dd0e8ed6fc719a1176a8131b1c3cf; both sides JIT-warmed before the clock starts.

This file runs OUR side and writes ours_matched.json; run_sota_matched.jl runs theirs.
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from problem import dist_matrix
from alns import alns

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "SOTA", "results")
BUDGET = {"bench": 30.0, 20: 35.0, 50: 140.0, 100: 430.0}
ITERS = {"bench": 9000, 20: 9000, 50: 16000, 100: 16000}
SPAN = {"bench": 20, 20: 12, 50: 12, 100: 12}


def job(rec):
    coords = np.stack([np.array(rec["x"]), np.array(rec["y"])], 1)
    n = len(rec["x"]) - 1
    inst = {"coords": coords, "n": n, "seed": 0, "depot": "x", "kind": "sota"}
    D = dist_matrix(coords)
    alpha = rec["truck_cost_factor"] / rec["drone_cost_factor"]
    key = "bench" if rec["kind"] == "benchmark" else rec["n"]
    T, iters, span = BUDGET[key], ITERS[key], SPAN[key]
    # JIT warmup outside the clock
    alns(inst, D, alpha, endurance=np.inf, max_span=span, iters=50, seed=999)
    reps = []
    best = float("inf")
    t0 = time.perf_counter()
    s = 0
    while time.perf_counter() - t0 < T:
        r = alns(inst, D, alpha, endurance=np.inf, max_span=span, iters=iters, seed=s)
        reps.append({"seed": s, "makespan": r["makespan"],
                     "t": time.perf_counter() - t0})
        best = min(best, r["makespan"])
        s += 1
    return {"id": rec["id"], "kind": rec["kind"], "n": rec["n"], "budget_s": T,
            "elapsed_s": time.perf_counter() - t0, "runs": s, "best": best,
            "replicates": reps}


def main():
    t0 = time.time()
    insts = json.load(open(os.path.join(OUTDIR, "instances.json")))
    with Pool(36) as pool:
        rows = pool.map(job, insts)
    json.dump({"config": {"budgets": {str(k): v for k, v in BUDGET.items()},
                          "iters": {str(k): v for k, v in ITERS.items()},
                          "protocol": "complete runs until budget exhausted, best kept, "
                                      "single-threaded, JIT-warmed", "runtime_s": time.time() - t0},
               "rows": rows},
              open(os.path.join(OUTDIR, "ours_matched.json"), "w"), indent=1)
    print(f"our side done in {time.time()-t0:.0f}s; "
          f"mean runs/instance = {np.mean([r['runs'] for r in rows]):.1f}")


if __name__ == "__main__":
    main()
