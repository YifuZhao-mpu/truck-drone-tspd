"""
H1 experiments:
 (A) external validation — gap to PUBLISHED EXACT OPTIMA on Agatz/Bouman uniform n=11..17.
 (B) synthetic ablation — truck-only / greedy / generic-ALNS / sortie-ALNS on n=10,20,50,
     with savings vs truck-only and the paired sortie-vs-generic operator test.
Outputs experiments/H1-alns-backbone/results/h1.json
"""
import os, json, glob, time
import numpy as np
from multiprocessing import Pool
from problem import dist_matrix
from baselines import truck_only_tsp, greedy_tspd
from alns import alns
from benchmark import load_instance, load_solution
import problem as P

BASE = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "external", "uniform")
OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "H1-alns-backbone", "results", "h1.json")
ITERS = {10: 7000, 20: 9000, 50: 16000}
NSEED = 3


def best_alns(inst, Dt, alpha, endurance, max_span, iters, sortie, nseed):
    best = None
    for s in range(nseed):
        r = alns(inst, Dt, alpha, endurance=endurance, max_span=max_span,
                 iters=iters, sortie_aware=sortie, seed=s)
        if best is None or r["makespan"] < best:
            best = r["makespan"]
    return best


def job_bench(args):
    n, idx = args
    ip = f"{BASE}/uniform-{idx}-n{n}.txt"; sp = f"{BASE}/solutions/uniform-{idx}-n{n}-DP.txt"
    if not (os.path.exists(ip) and os.path.exists(sp)):
        return None
    inst = load_instance(ip); opt = load_solution(sp)["total"]
    Dt = dist_matrix(inst["coords"])
    ms = best_alns(inst, Dt, inst["alpha"], np.inf, 20, 9000, True, NSEED)
    return {"n": n, "idx": idx, "opt": opt, "alns": ms, "gap": (ms - opt) / opt * 100}


def job_synth(args):
    n, seed = args
    from problem import gen_instance
    inst = gen_instance(n, seed=seed)
    Dt = dist_matrix(inst["coords"])
    alpha, endu, span = 2.0, np.inf, min(n + 1, 12)
    it = ITERS[n]
    _, truck = truck_only_tsp(inst, Dt, restarts=4, seed=seed)
    greedy, _, _ = greedy_tspd(inst, Dt, alpha)
    sortie = best_alns(inst, Dt, alpha, endu, span, it, True, NSEED)
    generic = best_alns(inst, Dt, alpha, endu, span, it, False, NSEED)
    return {"n": n, "seed": seed, "truck": truck, "greedy": greedy,
            "sortie": sortie, "generic": generic,
            "save_sortie": (truck - sortie) / truck * 100,
            "save_generic": (truck - generic) / truck * 100}


def main():
    t0 = time.time()
    bench_jobs = [(n, idx) for n in [11, 12, 13, 14, 15, 16, 17] for idx in range(1, 11)]
    synth_jobs = [(n, s) for n in [10, 20, 50] for s in range(30)]
    with Pool(38) as pool:
        bench = [r for r in pool.map(job_bench, bench_jobs) if r]
        synth = pool.map(job_synth, synth_jobs)

    # aggregate benchmark gaps by n
    bench_by_n = {}
    for n in sorted(set(r["n"] for r in bench)):
        g = np.array([r["gap"] for r in bench if r["n"] == n])
        bench_by_n[n] = {"mean_gap": float(g.mean()), "max_gap": float(g.max()),
                         "median_gap": float(np.median(g)),
                         "pct_optimal": float((g < 1e-6).mean() * 100), "count": int(len(g))}
    # aggregate synthetic by n
    synth_by_n = {}
    for n in [10, 20, 50]:
        rows = [r for r in synth if r["n"] == n]
        ss = np.array([r["sortie"] for r in rows]); gg = np.array([r["generic"] for r in rows])
        sv = np.array([r["save_sortie"] for r in rows]); gv = np.array([r["save_generic"] for r in rows])
        tr = np.array([r["truck"] for r in rows]); gr = np.array([r["greedy"] for r in rows])
        paired = ss - gg
        synth_by_n[n] = {
            "truck_mean": float(tr.mean()), "greedy_mean": float(gr.mean()),
            "sortie_mean": float(ss.mean()), "generic_mean": float(gg.mean()),
            "saving_sortie_mean": float(sv.mean()), "saving_sortie_std": float(sv.std()),
            "saving_generic_mean": float(gv.mean()),
            "greedy_gap_vs_alns": float(((gr - ss) / ss * 100).mean()),
            "paired_sortie_minus_generic_mean": float(paired.mean()),
            "sortie_wins": int((paired < -1e-9).sum()), "generic_wins": int((paired > 1e-9).sum()),
            "ties": int((np.abs(paired) <= 1e-9).sum()), "count": len(rows)}

    res = {"benchmark_gap_to_optimal": bench_by_n, "synthetic": synth_by_n,
           "config": {"nseed": NSEED, "iters": ITERS, "alpha": 2.0, "runtime_s": time.time() - t0},
           "raw_bench": bench, "raw_synth": synth}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"H1 done in {time.time()-t0:.0f}s -> {OUT}")
    print("BENCH gap-to-optimal:", json.dumps(bench_by_n, indent=2))
    print("SYNTH:", json.dumps(synth_by_n, indent=2))


if __name__ == "__main__":
    main()
