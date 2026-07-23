"""Re-run the n=100 sortie-vs-generic ablation with the (tuned, fair) solver for Table 2."""
import json, os
import numpy as np
from multiprocessing import Pool
from scipy import stats
from problem import gen_instance, dist_matrix
from alns import alns

OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "P0-robustness", "results", "ablation_n100_v2.json")
ITERS = 11000


def job(seed):
    inst = gen_instance(100, seed=seed); D = dist_matrix(inst["coords"])
    s = min(alns(inst, D, 2.0, endurance=np.inf, max_span=12, iters=ITERS, sortie_aware=True, seed=k)["makespan"] for k in (0, 1))
    g = min(alns(inst, D, 2.0, endurance=np.inf, max_span=12, iters=ITERS, sortie_aware=False, seed=k)["makespan"] for k in (0, 1))
    return s, g


def main():
    with Pool(20) as pool:
        res = pool.map(job, range(10))
    s = np.array([r[0] for r in res]); g = np.array([r[1] for r in res]); d = s - g
    k = int((d < -1e-9).sum()); N = int((d < -1e-9).sum() + (d > 1e-9).sum())
    p = float(stats.binomtest(k, N, 0.5, alternative="greater").pvalue) if N > 0 else None
    out = {"n": 100, "sortie_wins": k, "generic_wins": int((d > 1e-9).sum()),
           "ties": int((np.abs(d) <= 1e-9).sum()), "mean_diff": float(d.mean()),
           "sign_test_p": p, "count": len(res), "iters": ITERS}
    json.dump(out, open(OUT, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
