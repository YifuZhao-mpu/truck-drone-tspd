"""
Precompute certified truck-only TSP references (LKH-3) for every instance set used by
the v2 experiments, and validate them against the internal heuristic denominator.

Covers:
- uniform / central depot, n in {10, 20, 50, 100}, seeds 0..29, Euclidean (+ Manhattan n>=50)
- depot x demand deconfound grid at n in {50, 100}: {uniform, clustered} x {center, corner}
  x {euclidean, manhattan}
Output: cache experiments/TSPREF/results/tspref_cache.json
        summary experiments/TSPREF/results/tspref_summary.json
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from problem import gen_instance, gen_clustered, dist_matrix
from baselines import truck_only_tsp
from tsp_ref import reference_tsp, key_of, _coords_hash, _load_cache, _save_cache, CACHE_PATH

SEEDS = range(30)
OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "TSPREF", "results",
                   "tspref_summary.json")


def make_inst(kind, n, seed, depot):
    return (gen_instance(n, seed=seed, depot=depot) if kind == "uniform"
            else gen_clustered(n, seed=seed, depot=depot))


def jobs():
    out = []
    for n in [10, 20, 50, 100]:
        for s in SEEDS:
            out.append(("uniform", n, s, "center", "euclidean"))
    for n in [50, 100]:
        for s in SEEDS:
            for kind in ("uniform", "clustered"):
                for depot in ("center", "corner"):
                    for metric in ("euclidean", "manhattan"):
                        if kind == "uniform" and depot == "center" and metric == "euclidean":
                            continue  # already added above
                        out.append((kind, n, s, depot, metric))
    return out


def run_one(args):
    kind, n, seed, depot, metric = args
    inst = make_inst(kind, n, seed, depot)
    D = dist_matrix(inst["coords"], metric=metric)
    ref_len, ref_tour = reference_tsp(inst["coords"], metric=metric, runs=8, seed=1)
    _, heur_len = truck_only_tsp(inst, D, restarts=4, seed=0)
    return (key_of(kind, n, seed, depot, metric), _coords_hash(inst["coords"]),
            ref_len, ref_tour, float(heur_len))


def main():
    t0 = time.time()
    jl = jobs()
    print(f"{len(jl)} TSP references to compute")
    with Pool(36) as pool:
        res = pool.map(run_one, jl)
    cache = _load_cache()
    worse = 0
    gaps = []
    for key, ch, ref_len, tour, heur_len in res:
        cache[key] = {"coords_hash": ch, "length": ref_len, "tour": tour,
                      "internal_heuristic": heur_len}
        gaps.append((heur_len - ref_len) / ref_len * 100)
        if ref_len > heur_len + 1e-9:
            worse += 1
            print(f"WARNING: LKH worse than internal heuristic on {key}")
    _save_cache(cache)
    summary = {"count": len(res), "lkh_worse_than_internal": worse,
               "internal_gap_vs_lkh_pct": {"mean": float(np.mean(gaps)),
                                           "median": float(np.median(gaps)),
                                           "max": float(np.max(gaps)),
                                           "min": float(np.min(gaps))},
               "runtime_s": time.time() - t0}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"cache -> {CACHE_PATH}")


if __name__ == "__main__":
    main()
