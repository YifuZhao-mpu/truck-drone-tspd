"""
H3 v2 -- time-energy trade-off, rebuilt per review 2026-07-16.

ONE consistent objective everywhere: the search, the seed selection, and the reported
frontier all use makespan + lambda * E_total, where E_total = drone energy (Dorling
affine, loaded out / empty return) + truck energy (te per unit distance). The solver is
the SAME validated adaptive sortie-aware ALNS as every other experiment (alns(lam=...)),
not the v1 simplified search. Per instance we store makespan, truck distance, drone
energy, truck energy, total energy and the route for EVERY (lambda, seed) point, filter
to the non-dominated set per instance, and compute per-instance knees with uncertainty.
Sensitivity: payload cl in {0.5, 1, 2} and truck coefficient te in {0.15, 0.3, 0.6}
on a 15-instance subset (disclosed).

Outputs experiments/H3-time-energy/results/h3_v2.json
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from problem import gen_instance, dist_matrix
from alns import alns

OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "H3-time-energy",
                   "results", "h3_v2.json")
LAMBDAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.45, 0.65, 1.0, 1.5, 3.0, 8.0]  # contains LAMBDAS_SENS
LAMBDAS_SENS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.65, 1.5, 8.0]
ALPHA = 2.0
ITERS = {20: 9000, 50: 16000}
NSEED = 3
SPAN = 12
SEEDS_MAIN = range(30)
SEEDS_SENS = range(15)


def job(args):
    tag, n, seed, lam, cl, te = args
    inst = gen_instance(n, seed=seed)
    D = dist_matrix(inst["coords"])
    best = None
    for s in range(NSEED):
        r = alns(inst, D, ALPHA, endurance=np.inf, max_span=SPAN, iters=ITERS[n],
                 sortie_aware=True, seed=s, lam=lam, cl=cl, cb=1.0, te=te)
        if best is None or r["objective"] < best["objective"]:
            best = r
    ev = best["eval"]
    return {"tag": tag, "n": n, "seed": seed, "lam": lam, "cl": cl, "te": te,
            "makespan": ev["makespan"], "truck_dist": ev["truck_dist"],
            "e_drone": ev["e_drone"], "e_truck": ev["e_truck"], "e_total": ev["e_total"],
            "n_sorties": ev["n_sorties"], "objective": best["objective"],
            "order": [int(v) for v in best["order"]]}


def pareto_filter(points):
    """Non-dominated subset of (makespan, e_total) point dicts (min both)."""
    pts = sorted(points, key=lambda p: (p["makespan"], p["e_total"]))
    front = []
    best_e = float("inf")
    for p in pts:
        if p["e_total"] < best_e - 1e-12:
            front.append(p)
            best_e = p["e_total"]
    return front


def knee_of(front):
    """Max-chord knee of one instance's non-dominated front, normalized by its
    time-optimal (min-makespan) endpoint. Returns None for degenerate fronts."""
    if len(front) < 3:
        return None
    ms0 = front[0]["makespan"]
    e0 = front[0]["e_total"]
    x = np.array([p["makespan"] / ms0 for p in front])
    y = np.array([p["e_total"] / e0 for p in front])
    p0 = np.array([x[0], y[0]])
    p1 = np.array([x[-1], y[-1]])
    chord = p1 - p0
    cl_ = np.linalg.norm(chord) + 1e-12
    dxy = np.stack([x, y], 1) - p0
    d = np.abs(chord[0] * dxy[:, 1] - chord[1] * dxy[:, 0]) / cl_
    ki = int(np.argmax(d))
    # alternate knee definition: last point with marginal energy/time rate >= 1
    ki_rate = 0
    for a in range(1, len(x)):
        dx = x[a] - x[a - 1]
        dy = y[a - 1] - y[a]
        if dx > 1e-12 and dy / dx >= 1.0:
            ki_rate = a
    return {"knee_lambda": front[ki]["lam"],
            "dms_pct": (x[ki] - 1) * 100, "de_pct": (1 - y[ki]) * 100,
            "dms_pct_rate": (x[ki_rate] - 1) * 100, "de_pct_rate": (1 - y[ki_rate]) * 100,
            "n_front": len(front)}


def summarize(rows, lambdas):
    """Per-n summary: per-instance fronts, knees (median/IQR), mean curve for display."""
    out = {}
    for n in sorted(set(r["n"] for r in rows)):
        rs = [r for r in rows if r["n"] == n]
        seeds = sorted(set(r["seed"] for r in rs))
        fronts = {}
        knees = []
        strictly_dominated = 0
        displaced_lam0 = 0
        for sd in seeds:
            pts = [r for r in rs if r["seed"] == sd]
            pts.sort(key=lambda p: p["lam"])
            front = pareto_filter(pts)
            fronts[sd] = front
            lam0 = [p for p in pts if p["lam"] == 0.0][0]
            # strict domination = search noise; displacement incl. same-time ties = expected
            # tie-breaking under a time-only objective (round-2/3 review fix)
            if any(p["makespan"] < lam0["makespan"] - 1e-12
                   and p["e_total"] < lam0["e_total"] - 1e-12 for p in pts):
                strictly_dominated += 1
            if not any(f["lam"] == 0.0 for f in front):
                displaced_lam0 += 1
            k = knee_of(front)
            if k:
                knees.append(k)
        # mean display curve over lambda (normalized per instance by its lam=0 point)
        curve = []
        for lam in lambdas:
            nm, ne, nd = [], [], []
            for sd in seeds:
                pts = {p["lam"]: p for p in rs if p["seed"] == sd}
                if lam not in pts:
                    continue
                base = pts[0.0]
                nm.append(pts[lam]["makespan"] / base["makespan"])
                ne.append(pts[lam]["e_total"] / base["e_total"])
                nd.append(pts[lam]["e_drone"] / base["e_drone"]
                          if base["e_drone"] > 1e-9 else 0.0)
            curve.append({"lambda": lam, "norm_makespan": float(np.mean(nm)),
                          "norm_e_total": float(np.mean(ne)),
                          "norm_e_drone": float(np.mean(nd))})
        def kstats(vals):
            if not len(vals):
                return None
            v = np.asarray(vals, dtype=float)
            return {"median": float(np.median(v)),
                    "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
                    "mean": float(v.mean())}
        out[str(n)] = {
            "n_instances": len(seeds),
            "n_knees": len(knees),
            "knee_dms_pct": kstats([k["dms_pct"] for k in knees]),
            "knee_de_pct": kstats([k["de_pct"] for k in knees]),
            "knee_rate_dms_pct_median": (float(np.median([k["dms_pct_rate"] for k in knees]))
                                         if knees else None),
            "knee_rate_de_pct_median": (float(np.median([k["de_pct_rate"] for k in knees]))
                                        if knees else None),
            "lam0_strictly_dominated": strictly_dominated,
            "lam0_displaced_from_front": displaced_lam0,
            "front_sizes": [len(fronts[sd]) for sd in seeds],
            "mean_curve": curve,
            "e_total_span_pct_mean": float(np.mean(
                [(1 - fronts[sd][-1]["e_total"] / fronts[sd][0]["e_total"]) * 100
                 for sd in seeds if fronts[sd][0]["e_total"] > 1e-9]) if seeds else 0.0),
            "ms_span_pct_mean": float(np.mean(
                [(fronts[sd][-1]["makespan"] / fronts[sd][0]["makespan"] - 1) * 100
                 for sd in seeds]) if seeds else 0.0)}
    return out


def main():
    t0 = time.time()
    jobs = [("main", n, sd, lam, 1.0, 0.3)
            for n in (20, 50) for sd in SEEDS_MAIN for lam in LAMBDAS]
    jobs += [("payload", 50, sd, lam, cl, 0.3)
             for sd in SEEDS_SENS for lam in LAMBDAS_SENS for cl in (0.5, 2.0)]
    jobs += [("truckcoef", 50, sd, lam, 1.0, te)
             for sd in SEEDS_SENS for lam in LAMBDAS_SENS for te in (0.15, 0.6)]
    print(f"{len(jobs)} H3 v2 runs")
    with Pool(36) as pool:
        rows = pool.map(job, jobs)

    main_rows = [r for r in rows if r["tag"] == "main"]
    out = {"config": {"lambdas": LAMBDAS, "lambdas_sens": LAMBDAS_SENS, "alpha": ALPHA,
                      "iters": {str(k): v for k, v in ITERS.items()}, "nseed": NSEED,
                      "span": SPAN, "objective": "makespan + lambda * (e_drone + te*truck_dist)",
                      "cl": 1.0, "cb": 1.0, "te": 0.3,
                      "n_instances_main": len(list(SEEDS_MAIN)),
                      "n_instances_sens": len(list(SEEDS_SENS))},
           "main": summarize(main_rows, LAMBDAS),
           "sensitivity": {}}
    for tag, param, vals in (("payload", "cl", (0.5, 2.0)), ("truckcoef", "te", (0.15, 0.6))):
        for v in vals:
            sub = [r for r in rows if r["tag"] == tag and abs(r[param] - v) < 1e-12]
            if sub:
                out["sensitivity"][f"{tag}_{v}"] = summarize(sub, LAMBDAS_SENS)
    out["raw"] = [{k: r[k] for k in r if k != "order"} for r in rows]
    out["routes"] = {f"{r['tag']}-n{r['n']}-s{r['seed']}-l{r['lam']}-cl{r['cl']}-te{r['te']}":
                     r["order"] for r in rows}
    out["config"]["runtime_s"] = time.time() - t0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for n in ("20", "50"):
        s = out["main"][n]
        if s["knee_dms_pct"]:
            print(f"n={n}: knee median +{s['knee_dms_pct']['median']:.1f}% time / "
                  f"-{s['knee_de_pct']['median']:.1f}% TOTAL energy "
                  f"(IQR {s['knee_dms_pct']['iqr']}, {s['knee_de_pct']['iqr']}); "
                  f"lam0 strictly dominated on {s['lam0_strictly_dominated']}/{s['n_instances']}, "
                  f"displaced on {s['lam0_displaced_from_front']}/{s['n_instances']}")
        else:
            print(f"n={n}: NO KNEES FOUND ({s['n_knees']}/{s['n_instances']})")
    print(f"H3 v2 done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
