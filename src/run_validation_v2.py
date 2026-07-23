"""
V2 solver validation with STORED artifacts (review fix: the 1e-15 / brute-force / 20us
claims previously had no stored evidence).

 (a) Benchmark: gap to published exact optima, Agatz/Bouman uniform n=11..17 (identical
     config to v1: span 20, iters 9000, best-of-3) -- regression target: mean gap 0.13%.
 (b) DP == independent event-simulator on random decodes (single, multi3, lambda-aware).
 (c) Objective algebra: decode_lambda scalar == makespan + lam * e_total(evaluate_solution),
     and lambda=0 kernel == makespan kernel.
 (d) Span-cap sensitivity: final ALNS orders re-decoded at spans 6/8/10/12/14/full.
 (e) Split timing (us/decode) for n=50/100, m=1/2/3.
 (f) Brute-force exact check at n=8 (ALNS == exhaustive optimum over orders).
Outputs experiments/VALIDATION/results/validation_v2.json
"""
import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from problem import (gen_instance, dist_matrix, tspd_cost, tspd_split, tspd_split_multi3,
                     tspd_cost_multi3, obj_cost_lambda, decode_lambda)
from alns import alns
from baselines import truck_only_tsp
from benchmark import load_instance, load_solution
from energy import evaluate_solution
from exact import simulate_makespan, brute_force_optimal

BASE = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "external", "uniform")
OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "VALIDATION", "results",
                   "validation_v2.json")


def job_bench(args):
    n, idx = args
    ip = f"{BASE}/uniform-{idx}-n{n}.txt"
    sp = f"{BASE}/solutions/uniform-{idx}-n{n}-DP.txt"
    if not (os.path.exists(ip) and os.path.exists(sp)):
        return None
    inst = load_instance(ip)
    opt = load_solution(sp)["total"]
    Dt = dist_matrix(inst["coords"])
    ms = min(alns(inst, Dt, inst["alpha"], endurance=np.inf, max_span=20, iters=9000,
                  sortie_aware=True, seed=s)["makespan"] for s in range(3))
    return {"n": n, "idx": idx, "opt": opt, "alns": ms, "gap": (ms - opt) / opt * 100}


def sim_check(_):
    rng = np.random.default_rng()
    worst = 0.0
    for _ in range(20):
        n = int(rng.integers(10, 60))
        inst = gen_instance(n, seed=int(rng.integers(10000)))
        D = dist_matrix(inst["coords"])
        order = [0] + list(rng.permutation(range(1, n + 1))) + [0]
        E = [np.inf, 1.0, 0.5][int(rng.integers(3))]
        alpha = [1.5, 2.0, 3.0][int(rng.integers(3))]
        for m in (1, 2, 3):
            ms, ops = tspd_split_multi3(order, D, alpha, m, endurance=E, max_span=12)
            sim, feas, served = simulate_makespan(order, ops, D, alpha, endurance=E)
            ev = evaluate_solution(order, ops, D, D, alpha)
            assert feas and len(served) == n
            worst = max(worst, abs(ms - sim), abs(sim - ev["makespan"]))
        lam = float(rng.random() * 2)
        obj, msl, enl, opsl = decode_lambda(order, D, alpha, lam, E, 12)
        ev = evaluate_solution(order, opsl, D, D, alpha)
        worst = max(worst, abs(obj - (ev["makespan"] + lam * ev["e_total"])),
                    abs(msl - ev["makespan"]), abs(enl - ev["e_drone"]),
                    abs(obj_cost_lambda(order, D, alpha, 0.0, E, 12)
                        - tspd_cost(order, D, alpha, E, 12)))
    return worst


def span_check(args):
    n, seed, m = args
    inst = gen_instance(n, seed=seed)
    D = dist_matrix(inst["coords"])
    r = alns(inst, D, 2.0, endurance=np.inf, max_span=12, iters=6000, seed=0, m=m)
    order = r["order"]
    spans = [6, 8, 10, 12, 14]
    vals = {}
    for sp in spans:
        if m == 1:
            vals[sp] = tspd_cost(order, D, 2.0, np.inf, sp)
        else:
            vals[sp] = tspd_cost_multi3(order, D, 2.0, m, endurance=np.inf, max_span=sp)
    if m == 1:
        vals["full"] = tspd_cost(order, D, 2.0, np.inf, len(order) - 1)
    ref = min(vals.values())
    return {"n": n, "seed": seed, "m": m,
            "rel_excess_pct": {str(k): (v - ref) / ref * 100 for k, v in vals.items()}}


def bf_check(seed):
    inst = gen_instance(8, seed=seed)
    D = dist_matrix(inst["coords"])
    opt, _ = brute_force_optimal(inst, D, 2.0, endurance=np.inf)
    ms = min(alns(inst, D, 2.0, endurance=np.inf, max_span=9, iters=3000,
                  seed=s)["makespan"] for s in range(2))
    return {"seed": seed, "brute": float(opt), "alns": float(ms),
            "gap": float((ms - opt) / opt * 100)}


def main():
    t0 = time.time()
    out = {}
    with Pool(36) as pool:
        bench = [r for r in pool.map(job_bench,
                                     [(n, i) for n in range(11, 18) for i in range(1, 11)])
                 if r]
        gaps = np.array([r["gap"] for r in bench])
        boot = np.random.default_rng(0).choice(gaps, size=(20000, len(gaps))).mean(1)
        out["benchmark"] = {
            "count": len(bench), "mean_gap_pct": float(gaps.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "exact_pct": float((gaps < 1e-9).mean() * 100),
            "within1_pct": float((gaps < 1.0).mean() * 100),
            "per_size_exact_pct": {str(n): float(np.mean([r["gap"] < 1e-9 for r in bench
                                                          if r["n"] == n]) * 100)
                                   for n in range(11, 18)},
            "raw": bench}

        worst = max(pool.map(sim_check, range(12)))
        out["dp_vs_simulator_and_algebra_maxdiff"] = float(worst)

        spans = pool.map(span_check, [(n, s, m) for n in (50, 100) for s in range(5)
                                      for m in (1, 2, 3)])
        agg = {}
        for r in spans:
            for k, v in r["rel_excess_pct"].items():
                agg.setdefault(f"m{r['m']}_span{k}", []).append(v)
        out["span_sensitivity"] = {"max_rel_excess_pct":
                                   {k: float(np.max(v)) for k, v in sorted(agg.items())},
                                   "raw": spans}

        out["bruteforce_n8"] = pool.map(bf_check, range(10))

    # timing (single process, warm JIT)
    tim = {}
    for n in (50, 100):
        inst = gen_instance(n, seed=0)
        D = dist_matrix(inst["coords"])
        o = [0] + list(np.random.default_rng(0).permutation(range(1, n + 1))) + [0]
        tspd_cost(o, D, 2.0, np.inf, 12)
        tspd_cost_multi3(o, D, 2.0, 3, endurance=np.inf, max_span=12)
        for m in (1, 2, 3):
            t1 = time.perf_counter()
            N = 300
            for _ in range(N):
                if m == 1:
                    tspd_cost(o, D, 2.0, np.inf, 12)
                else:
                    tspd_cost_multi3(o, D, 2.0, m, endurance=np.inf, max_span=12)
            tim[f"n{n}_m{m}_span12_us"] = (time.perf_counter() - t1) / N * 1e6
    out["split_timing_us"] = tim
    out["config"] = {"runtime_s": time.time() - t0}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"benchmark mean gap {out['benchmark']['mean_gap_pct']:.3f}% "
          f"CI {out['benchmark']['ci95']}, exact {out['benchmark']['exact_pct']:.0f}%")
    print(f"dp==sim maxdiff {out['dp_vs_simulator_and_algebra_maxdiff']:.2e}")
    print(f"span max excess: {out['span_sensitivity']['max_rel_excess_pct']}")
    print(f"bruteforce gaps: {[round(r['gap'], 4) for r in out['bruteforce_n8']]}")
    print(f"timing: { {k: round(v, 1) for k, v in tim.items()} }")
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
