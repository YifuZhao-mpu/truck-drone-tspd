"""
Export the exact instances (same fixed seeds) for the TSPDrone.jl (HGA-TAC) SOTA comparison,
and run our ALNS on them (parallelized). Writes instances.json (coords + our ALNS makespan).
run_sota.jl reads the text export and appends HGA-TAC total_cost.
"""
import os, json
import numpy as np
from multiprocessing import Pool
from problem import gen_instance, dist_matrix
from benchmark import load_instance, load_solution
from alns import alns

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "SOTA", "results")
BENCH = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "external", "uniform")
ALPHA = 2.0
ITERS = {20: 9000, 50: 14000, 100: 9000}
NSYN = {20: 10, 50: 10, 100: 5}


def job_syn(args):
    n, seed = args
    inst = gen_instance(n, seed=seed); Dt = dist_matrix(inst["coords"])
    ms = min(alns(inst, Dt, ALPHA, max_span=min(n + 1, 12), iters=ITERS[n],
                  sortie_aware=True, seed=s)["makespan"] for s in range(3))
    return {"id": f"uniform-n{n}-s{seed}", "kind": "synthetic", "n": n,
            "x": inst["coords"][:, 0].tolist(), "y": inst["coords"][:, 1].tolist(),
            "truck_cost_factor": 1.0, "drone_cost_factor": 1.0 / ALPHA, "our_alns": ms}


def job_bench(args):
    n, idx = args
    ip = f"{BENCH}/uniform-{idx}-n{n}.txt"; sp = f"{BENCH}/solutions/uniform-{idx}-n{n}-DP.txt"
    if not (os.path.exists(ip) and os.path.exists(sp)):
        return None
    bi = load_instance(ip); opt = load_solution(sp)["total"]; Dt = dist_matrix(bi["coords"])
    ms = min(alns(bi, Dt, bi["alpha"], max_span=20, iters=9000, sortie_aware=True, seed=s)["makespan"]
             for s in range(3))
    return {"id": f"bench-{idx}-n{n}", "kind": "benchmark", "n": bi["n"],
            "x": bi["coords"][:, 0].tolist(), "y": bi["coords"][:, 1].tolist(),
            "truck_cost_factor": bi["truck_f"], "drone_cost_factor": bi["drone_f"],
            "opt": opt, "our_alns": ms}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    syn_jobs = [(n, s) for n in [20, 50, 100] for s in range(NSYN[n])]
    bench_jobs = [(n, idx) for n in [11, 13, 15, 17] for idx in range(1, 6)]
    with Pool(36) as pool:
        syn = pool.map(job_syn, syn_jobs)
        bench = [r for r in pool.map(job_bench, bench_jobs) if r]
    items = syn + bench
    json.dump(items, open(os.path.join(OUTDIR, "instances.json"), "w"))
    print(f"exported {len(items)} instances ({len(syn)} synthetic + {len(bench)} benchmark)")


if __name__ == "__main__":
    main()
