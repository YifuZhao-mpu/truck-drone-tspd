"""
H2 design-space "when do drones help" map (one consistent solver = our ALNS).
 Part A: makespan SAVING vs truck-only over the speed-ratio (alpha) x endurance (E) grid, m=1.
 Part B: SAVING vs number of drones m at alpha=2 (diminishing-returns axis).
Saving compared to the worst-case ceiling 1 - 1/(alpha*m + 1) and the ~30% (alpha=2, m=1)
asymptotic reference (Lee et al. 2026). Outputs experiments/H2-design-space/results/h2.json
"""
import os, json, time
import numpy as np
from multiprocessing import Pool
from problem import gen_instance, dist_matrix
from baselines import truck_only_tsp
from alns import alns

OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "H2-design-space", "results", "h2.json")
ALPHAS = [1.0, 1.5, 2.0, 2.5, 3.0]
ENDUR = [0.5, 1.0, 2.0, 1e18]                 # flight-distance budgets (unit square); 1e18=inf
ENDUR_LABELS = ["0.5", "1.0", "2.0", "inf"]
N20_INST, N50_INST = 20, 12
NSEED = 2


def _truck(inst, Dt, seed):
    _, t = truck_only_tsp(inst, Dt, restarts=4, seed=seed)
    return t


def job_grid(args):
    n, seed, ai, ei = args
    inst = gen_instance(n, seed=seed); Dt = dist_matrix(inst["coords"])
    truck = _truck(inst, Dt, seed)
    best = min(alns(inst, Dt, ALPHAS[ai], endurance=ENDUR[ei], max_span=10,
                    iters=9000, sortie_aware=True, seed=s)["makespan"] for s in range(NSEED))
    return ("grid", n, seed, ai, ei, truck, best, (truck - best) / truck * 100)


def job_drones(args):
    n, seed, m, ei = args
    inst = gen_instance(n, seed=seed); Dt = dist_matrix(inst["coords"])
    truck = _truck(inst, Dt, seed)
    best = min(alns(inst, Dt, 2.0, endurance=ENDUR[ei], max_span=6,
                    iters=8000, seed=s, m=m)["makespan"] for s in range(NSEED))
    return ("drone", n, seed, m, ei, truck, best, (truck - best) / truck * 100)


def main():
    t0 = time.time()
    grid_jobs = ([(20, s, ai, ei) for s in range(N20_INST) for ai in range(len(ALPHAS)) for ei in range(len(ENDUR))]
                 + [(50, s, ai, ei) for s in range(N50_INST) for ai in range(len(ALPHAS)) for ei in range(len(ENDUR))])
    drone_jobs = [(20, s, m, ei) for s in range(N20_INST) for m in [1, 2, 3] for ei in [1, 3]]  # E=1.0, inf
    with Pool(38) as pool:
        grid = pool.map(job_grid, grid_jobs)
        drone = pool.map(job_drones, drone_jobs)

    # aggregate grid: mean saving per (n, alpha, E)
    surface = {}
    for n in [20, 50]:
        surface[n] = {}
        for ai, a in enumerate(ALPHAS):
            row = {}
            for ei, el in enumerate(ENDUR_LABELS):
                sv = [r[7] for r in grid if r[1] == n and r[3] == ai and r[4] == ei]
                row[el] = {"mean_saving": float(np.mean(sv)), "std": float(np.std(sv)),
                           "ceiling": (1 - 1 / (a * 1 + 1)) * 100}
            surface[n][f"alpha_{a}"] = row
    # aggregate drones: mean saving per (m, E) at alpha=2, n=20
    drones = {}
    for ei in [1, 3]:
        el = ENDUR_LABELS[ei]
        drones[el] = {}
        for m in [1, 2, 3]:
            sv = [r[7] for r in drone if r[3] == m and r[4] == ei]
            drones[el][f"m_{m}"] = {"mean_saving": float(np.mean(sv)), "std": float(np.std(sv)),
                                    "ceiling": (1 - 1 / (2 * m + 1)) * 100}

    res = {"surface_saving_vs_truckonly": surface, "drones_axis_alpha2": drones,
           "alphas": ALPHAS, "endurance": ENDUR_LABELS,
           "asymptotic_ref_alpha2_m1_pct": 30.0,
           "config": {"n20_inst": N20_INST, "n50_inst": N50_INST, "nseed": NSEED,
                      "runtime_s": time.time() - t0},
           "raw_grid": grid, "raw_drone": drone}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"H2 done in {time.time()-t0:.0f}s -> {OUT}")
    print("SURFACE n=20:", json.dumps(surface[20], indent=2))
    print("SURFACE n=50:", json.dumps(surface[50], indent=2))
    print("DRONES axis:", json.dumps(drones, indent=2))


if __name__ == "__main__":
    main()
