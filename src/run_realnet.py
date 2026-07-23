"""
Real road-network case study (round-7 roadmap): two OSM districts with contrasting
street patterns; truck = road shortest-path distances, drone = straight-line; savings vs
LKH best-found references on the explicit road matrix; m in {1,2,3}; best-of-3 with all
replicates archived. Compares against the synthetic unit-square predictions (uniform,
central depot: Euclidean-metric and L1-proxy cells).

Outputs experiments/REALNET/results/realnet.json
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from realnet import get_district, build_instance, circuity
from tsp_ref import reference_tsp_matrix
from alns import alns

OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "REALNET", "results",
                   "realnet.json")
DISTRICT_NAMES = ["manhattan", "paris"]
N = 50
SEEDS = range(30)
MS = [1, 2, 3]
ALPHA = 2.0
E = 1.0
ITERS = 16000
NSEED = 3
SPAN = 12


def job(args):
    dname, seed, m = args
    G = get_district(dname)               # cached graphml; cheap per worker
    inst, Dtr, Ddr, scale = build_instance(G, N, seed)
    ref, _ = reference_tsp_matrix(Dtr)
    reps, best = [], None
    for s in range(NSEED):
        r = alns(inst, Dtr, ALPHA, endurance=E, max_span=SPAN, iters=ITERS,
                 sortie_aware=True, seed=s, m=m, Ddr=Ddr)
        reps.append({"seed": s, "makespan": r["makespan"]})
        if best is None or r["makespan"] < best["makespan"]:
            best = r
    return {"district": dname, "seed": seed, "m": m, "makespan": best["makespan"],
            "truck_ref": float(ref), "saving_pct": (ref - best["makespan"]) / ref * 100,
            "circuity": circuity(Dtr, Ddr), "scale_m": scale, "replicates": reps,
            "order": [int(v) for v in best["order"]]}


def boot_ci(vals, seed=0):
    a = np.asarray(vals, dtype=float)
    bs = np.random.default_rng(seed).choice(a, size=(20000, len(a))).mean(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    t0 = time.time()
    for dn in DISTRICT_NAMES:            # warm the graph cache before forking
        get_district(dn)
    jobs = [(dn, sd, m) for dn in DISTRICT_NAMES for sd in SEEDS for m in MS]
    print(f"{len(jobs)} real-network runs")
    with Pool(36) as pool:
        rows = pool.map(job, jobs)
    h2 = json.load(open(os.path.join(os.path.dirname(__file__), "..", "experiments",
                                     "H2-design-space", "results", "h2_v2.json")))
    p0 = json.load(open(os.path.join(os.path.dirname(__file__), "..", "experiments",
                                     "P0-robustness", "results", "p0_v2.json")))
    out = {"config": {"districts": DISTRICT_NAMES, "n": N, "alpha": ALPHA, "endurance": E,
                      "iters": ITERS, "nseed": NSEED, "span": SPAN,
                      "n_instances": len(list(SEEDS)),
                      "note": "truck = undirected OSM shortest paths (one-ways ignored); "
                              "drone = Euclidean; coords normalized by largest bbox side; "
                              "customers/depot sampled from road nodes (depot = centroid-"
                              "nearest node); LKH explicit-matrix references"},
           "by_district": {}, "synthetic_reference_cells": {
               "uniform_center_euclid_m1": h2["cells"]["50"]["a2.0_E1.0_m1"]["mean_saving"],
               "uniform_center_euclid_m2": h2["cells"]["50"]["a2.0_E1.0_m2"]["mean_saving"],
               "uniform_center_euclid_m3": h2["cells"]["50"]["a2.0_E1.0_m3"]["mean_saving"],
               "uniform_center_L1_m1": p0["deconfound_cells"]["uniform_center_manhattan_n50"]["mean_saving"]}}
    for dn in DISTRICT_NAMES:
        sub = {}
        for m in MS:
            v = [r["saving_pct"] for r in rows if r["district"] == dn and r["m"] == m]
            sub[f"m{m}"] = {"mean_saving": float(np.mean(v)), "ci95": boot_ci(v), "n": len(v)}
        circ = [r["circuity"] for r in rows if r["district"] == dn and r["m"] == 1]
        sub["circuity_mean"] = float(np.mean(circ))
        out["by_district"][dn] = sub
    out["raw"] = [{k: r[k] for k in r if k != "order"} for r in rows]
    out["routes"] = {f"{r['district']}-s{r['seed']}-m{r['m']}": r["order"] for r in rows}
    out["config"]["runtime_s"] = time.time() - t0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("synthetic refs:", {k: round(v, 1) for k, v in out["synthetic_reference_cells"].items()})
    for dn in DISTRICT_NAMES:
        s = out["by_district"][dn]
        print(f"{dn}: circuity {s['circuity_mean']:.3f}; savings m1/2/3 = "
              f"{s['m1']['mean_saving']:.1f}/{s['m2']['mean_saving']:.1f}/{s['m3']['mean_saving']:.1f}%")
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
