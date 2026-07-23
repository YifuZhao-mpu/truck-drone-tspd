"""
P0 v2 -- robustness with ONE solver config and the depot/demand deconfound
(review fixes: v1 ran robustness with sortie moves off, 1 seed, and confounded
clustered demand with a corner depot; CIs were normal-approx while the paper said
bootstrap; the n=100 ablation artifact was stale).

 (A) Deconfound grid: {uniform, clustered} demand x {center, corner} depot x
     {euclidean, manhattan} truck metric, at alpha=2, E=1, m=1; n in {50, 100};
     30 instances; best-of-3; LKH denominators; bootstrap CIs.
 (B) alpha x E surface at n=100 (uniform, center, euclidean), 30 instances, best-of-3.
 (C) Operator ablation (sortie vs generic) at n in {20, 50, 100}, 30 paired instances,
     best-of-3 per arm, one config -- replaces the stale ablation_n100_v2.json.
 (D) Budget-sensitivity diagnostic (NOT an optimality bound): standard vs 2.5x budget,
     best-of-3 both, 15 instances at n in {50, 100}.

Outputs experiments/P0-robustness/results/p0_v2.json
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy import stats

from problem import gen_instance, gen_clustered, dist_matrix
from alns import alns
from tsp_ref import _load_cache, key_of

OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "P0-robustness",
                   "results", "p0_v2.json")
ALPHAS = [1.0, 1.5, 2.0, 2.5, 3.0]
ENDUR = [0.5, 1.0, 2.0, np.inf]
ENDUR_KEY = ["0.5", "1.0", "2.0", "inf"]
SEEDS = range(30)
NSEED = 3
ITERS = {20: 9000, 50: 16000, 100: 16000}
LONG_ITERS = {50: 40000, 100: 40000}
SPAN = 12
REF = _load_cache()


def make_inst(kind, n, seed, depot):
    return (gen_instance(n, seed=seed, depot=depot) if kind == "uniform"
            else gen_clustered(n, seed=seed, depot=depot))


def matrices(inst, metric):
    Dtr = dist_matrix(inst["coords"], metric=metric)
    Ddr = dist_matrix(inst["coords"], metric="euclidean")
    return Dtr, Ddr


def best_ms(inst, Dtr, Ddr, alpha, E, iters, nseed, sortie=True):
    return min(alns(inst, Dtr, alpha, endurance=E, max_span=SPAN, iters=iters,
                    sortie_aware=sortie, seed=s, Ddr=Ddr)["makespan"]
               for s in range(nseed))


def job_cell(args):
    kind, depot, metric, n, seed = args
    inst = make_inst(kind, n, seed, depot)
    Dtr, Ddr = matrices(inst, metric)
    ref = REF[key_of(kind, n, seed, depot, metric)]["length"]
    ms = best_ms(inst, Dtr, Ddr, 2.0, 1.0, ITERS[n], NSEED)
    return ("cell", kind, depot, metric, n, seed, ms, ref,
            (ref - ms) / ref * 100.0)


def job_surface(args):
    n, seed, ai, ei = args
    inst = gen_instance(n, seed=seed)
    D = dist_matrix(inst["coords"])
    ref = REF[key_of("uniform", n, seed, "center", "euclidean")]["length"]
    ms = best_ms(inst, D, D, ALPHAS[ai], ENDUR[ei], ITERS[n], NSEED)
    return ("surf", n, seed, ai, ei, ms, ref, (ref - ms) / ref * 100.0)


def job_ablation(args):
    n, seed = args
    inst = gen_instance(n, seed=seed)
    D = dist_matrix(inst["coords"])
    s_ms = best_ms(inst, D, D, 2.0, np.inf, ITERS[n], NSEED, sortie=True)
    g_ms = best_ms(inst, D, D, 2.0, np.inf, ITERS[n], NSEED, sortie=False)
    return ("abl", n, seed, s_ms, g_ms)


def job_budget(args):
    n, seed = args
    inst = gen_instance(n, seed=seed)
    D = dist_matrix(inst["coords"])
    std = best_ms(inst, D, D, 2.0, np.inf, ITERS[n], NSEED)
    lng = best_ms(inst, D, D, 2.0, np.inf, LONG_ITERS[n], NSEED)
    return ("bud", n, seed, std, lng)


def boot_ci(vals, nres=20000, seed=0):
    a = np.asarray(vals, dtype=float)
    bs = np.random.default_rng(seed).choice(a, size=(nres, len(a))).mean(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    t0 = time.time()
    cell_jobs = [(kind, depot, metric, n, s)
                 for kind in ("uniform", "clustered") for depot in ("center", "corner")
                 for metric in ("euclidean", "manhattan") for n in (50, 100) for s in SEEDS]
    surf_jobs = [(100, s, ai, ei) for s in SEEDS
                 for ai in range(len(ALPHAS)) for ei in range(len(ENDUR))]
    abl_jobs = [(n, s) for n in (20, 50, 100) for s in SEEDS]
    bud_jobs = [(n, s) for n in (50, 100) for s in range(15)]
    print(f"{len(cell_jobs)} cells + {len(surf_jobs)} surface + {len(abl_jobs)} ablation "
          f"+ {len(bud_jobs)} budget runs")
    with Pool(36) as pool:
        res = pool.map(job_cell, cell_jobs)
        res += pool.map(job_surface, surf_jobs)
        res += pool.map(job_ablation, abl_jobs)
        res += pool.map(job_budget, bud_jobs)

    out = {"config": {"nseed": NSEED, "iters": {str(k): v for k, v in ITERS.items()},
                      "long_iters": {str(k): v for k, v in LONG_ITERS.items()},
                      "span": SPAN, "sortie_aware": True,
                      "n_instances": len(list(SEEDS)),
                      "denominator": "best-found LKH truck-only reference (no optimality certificate)",
                      "ci_method": "bootstrap 20000 resamples"}}

    cells = {}
    for kind in ("uniform", "clustered"):
        for depot in ("center", "corner"):
            for metric in ("euclidean", "manhattan"):
                for n in (50, 100):
                    sub = [r[8] for r in res if r[0] == "cell" and r[1] == kind
                           and r[2] == depot and r[3] == metric and r[4] == n]
                    cells[f"{kind}_{depot}_{metric}_n{n}"] = {
                        "mean_saving": float(np.mean(sub)), "ci95": boot_ci(sub),
                        "n": len(sub)}
    out["deconfound_cells"] = cells

    surf = {}
    for ai, a in enumerate(ALPHAS):
        for ei, ek in enumerate(ENDUR_KEY):
            sub = [r[7] for r in res if r[0] == "surf" and r[3] == ai and r[4] == ei]
            surf[f"a{a}_E{ek}"] = {"mean_saving": float(np.mean(sub)),
                                   "ci95": boot_ci(sub), "n": len(sub)}
    out["surface_n100"] = surf

    abl = {}
    for n in (20, 50, 100):
        pairs = [(r[3], r[4]) for r in res if r[0] == "abl" and r[1] == n]
        d = np.array([s - g for s, g in pairs])
        wins = int((d < -1e-9).sum())
        losses = int((d > 1e-9).sum())
        ties = len(d) - wins - losses
        nz = d[np.abs(d) > 1e-9]
        sign_p = float(stats.binomtest(min(wins, losses), wins + losses).pvalue) \
            if wins + losses else 1.0
        wil_p = float(stats.wilcoxon(nz, alternative="less").pvalue) if len(nz) >= 5 else None
        abl[str(n)] = {"sortie_wins": wins, "generic_wins": losses, "ties": ties,
                       "mean_diff": float(d.mean()),
                       "mean_rel_diff_pct": float(np.mean(
                           [(s - g) / g * 100 for s, g in pairs])),
                       "sign_test_p": sign_p, "wilcoxon_p": wil_p,
                       "raw": [{"seed": r[2], "sortie": r[3], "generic": r[4]}
                               for r in res if r[0] == "abl" and r[1] == n]}
    out["ablation"] = abl

    bud = {}
    for n in (50, 100):
        pairs = [(r[3], r[4]) for r in res if r[0] == "bud" and r[1] == n]
        gaps = [(s - l) / l * 100 for s, l in pairs]
        bud[str(n)] = {"mean_gap_to_2p5x_budget_pct": float(np.mean(gaps)),
                       "max_gap_pct": float(np.max(gaps)),
                       "note": "budget-sensitivity diagnostic, NOT an optimality bound",
                       "raw": [{"seed": r[2], "std": r[3], "long": r[4]}
                               for r in res if r[0] == "bud" and r[1] == n]}
    out["budget_sensitivity"] = bud

    out["raw_cells"] = [{"kind": r[1], "depot": r[2], "metric": r[3], "n": r[4],
                         "seed": r[5], "makespan": r[6], "truck_ref": r[7],
                         "saving_pct": r[8]} for r in res if r[0] == "cell"]
    out["raw_surface"] = [{"n": r[1], "seed": r[2], "ai": r[3], "ei": r[4],
                           "makespan": r[5], "truck_ref": r[6], "saving_pct": r[7]}
                          for r in res if r[0] == "surf"]
    out["config"]["runtime_s"] = time.time() - t0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: v for k, v in cells.items() if "_n50" in k}, indent=1)[:800])
    for n in ("20", "50", "100"):
        a = abl[n]
        print(f"ablation n={n}: {a['sortie_wins']}W/{a['generic_wins']}L/{a['ties']}T "
              f"sign p={a['sign_test_p']:.4f}")
    print(f"P0 v2 done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
