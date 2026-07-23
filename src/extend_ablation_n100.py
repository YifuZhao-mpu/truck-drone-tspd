"""Extend the P0 n=100 sortie-vs-generic ablation from 10 to 30 paired instances
(same pipeline config as run_p0.job_ablation: uniform demand, alpha=2, E=inf,
iters=7000, best-of-2 seeds per arm). Updates ablation_n100 in p0.json in place
and stores the per-instance raw values so P1 can run an exact sign test."""
import os, json, time
import numpy as np
from multiprocessing import Pool
from problem import gen_instance, dist_matrix
from baselines import truck_only_tsp
from alns import alns

OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "P0-robustness", "results", "p0.json")
N_INST = 30


def job(seed):
    n = 100
    inst = gen_instance(n, seed=seed)
    D = dist_matrix(inst["coords"])
    _, truck = truck_only_tsp(inst, D, restarts=4, seed=seed)
    sortie = min(alns(inst, D, 2.0, endurance=np.inf, max_span=12, iters=7000,
                      sortie_aware=True, seed=s)["makespan"] for s in range(2))
    generic = min(alns(inst, D, 2.0, endurance=np.inf, max_span=12, iters=7000,
                       sortie_aware=False, seed=s)["makespan"] for s in range(2))
    return (seed, truck, sortie, generic)


def main():
    t0 = time.time()
    with Pool(30) as pool:
        rows = pool.map(job, list(range(N_INST)))
    tr = np.array([r[1] for r in rows])
    ss = np.array([r[2] for r in rows])
    gg = np.array([r[3] for r in rows])
    paired = ss - gg
    res = json.load(open(OUT))
    res["ablation_n100"] = {
        "saving_sortie_mean": float(((tr - ss) / tr * 100).mean()),
        "sortie_mean": float(ss.mean()), "generic_mean": float(gg.mean()),
        "paired_sortie_minus_generic": float(paired.mean()),
        "sortie_wins": int((paired < -1e-9).sum()), "generic_wins": int((paired > 1e-9).sum()),
        "ties": int((np.abs(paired) <= 1e-9).sum()), "count": len(rows),
        "raw": [{"seed": int(r[0]), "truck": float(r[1]), "sortie": float(r[2]),
                 "generic": float(r[3])} for r in rows],
        "note": "30 paired instances, same config as run_p0.job_ablation (iters=7000, best-of-2)"}
    json.dump(res, open(OUT, "w"), indent=2)
    a = res["ablation_n100"]
    print(f"done in {time.time()-t0:.0f}s; wins {a['sortie_wins']}/{a['generic_wins']}/{a['ties']}"
          f" meanD {paired.mean():+.4f}")


if __name__ == "__main__":
    main()
