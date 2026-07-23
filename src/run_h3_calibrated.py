"""
Physically calibrated time-energy analysis (round-4/5 fix: "model units are not
operator-facing").

Anchoring (literature values, converted to model units where drone empty power = 1):
 - small delivery multirotor: empty cruise power ~0.35-0.5 kW; loaded (+2 kg) roughly
   +50-100% (Dorling et al. 2017 affine power-in-payload) -> cl = cb = 1 retained.
 - urban truck effective speed ~25-30 km/h; drone ~50-60 km/h -> alpha = 2 retained.
 - truck energy per km: battery-electric van ~0.3-0.5 kWh/km; diesel van ~1 kWh/km fuel.
   Truck POWER = (kWh/km) x (km/h): EV ~ 9-15 kW, diesel ~ 25-30 kW.
   In model units (truck speed 1, drone empty power 1):
       te = truck power / drone empty power  ~  20-60   (vs the 0.3 "model unit" default)
   We run te in {1, 6, 30, 60}: the drone-cost-parity threshold (~1), a light electric
   platform (6), a battery-electric van (30), and a diesel van (60).

The lambda grid is scaled by (0.3 / te) so the effective trade intensities match the
nominal design; instances and grid otherwise match the sensitivity design (n=50, seeds
0..14, 8 scaled lambdas, best-of-3). ALL replicate outcomes and seeds are archived.

Outputs experiments/H3-time-energy/results/h3_calibrated.json
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from problem import gen_instance, dist_matrix
from alns import alns
from tsp_ref import _load_cache, key_of

OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "H3-time-energy",
                   "results", "h3_calibrated.json")
LAMBDAS_BASE = [0.0, 0.05, 0.1, 0.2, 0.35, 0.65, 1.5, 8.0]
TES = [1.0, 6.0, 30.0, 60.0]
ALPHA = 2.0
ITERS = 16000
NSEED = 3
SPAN = 12
SEEDS = range(15)
REF = _load_cache()


def job(args):
    seed, te, lam = args
    inst = gen_instance(50, seed=seed)
    D = dist_matrix(inst["coords"])
    reps = []
    best = None
    for s in range(NSEED):
        r = alns(inst, D, ALPHA, endurance=np.inf, max_span=SPAN, iters=ITERS,
                 sortie_aware=True, seed=s, lam=lam, cl=1.0, cb=1.0, te=te)
        ev = r["eval"]
        reps.append({"seed": s, "makespan": ev["makespan"], "e_total": ev["e_total"],
                     "objective": r["objective"]})
        if best is None or r["objective"] < best["objective"]:
            best = r
    ev = best["eval"]
    ref = REF[key_of("uniform", 50, seed, "center", "euclidean")]["length"]
    return {"seed": seed, "te": te, "lam": lam,
            "makespan": ev["makespan"], "truck_dist": ev["truck_dist"],
            "e_drone": ev["e_drone"], "e_truck": ev["e_truck"], "e_total": ev["e_total"],
            "n_sorties": ev["n_sorties"],
            "truck_only_ref": ref, "truck_only_e_total": te * ref,
            "replicates": reps, "order": [int(v) for v in best["order"]]}


def main():
    t0 = time.time()
    jobs = [(sd, te, round(lb * 0.3 / te, 10)) for te in TES for sd in SEEDS
            for lb in LAMBDAS_BASE]
    print(f"{len(jobs)} calibrated H3 runs")
    with Pool(36) as pool:
        rows = pool.map(job, jobs)

    out = {"config": {"tes": TES, "lambda_base": LAMBDAS_BASE,
                      "lambda_scaling": "lambda = base * 0.3 / te",
                      "alpha": ALPHA, "iters": ITERS, "nseed": NSEED, "span": SPAN,
                      "n_instances": len(list(SEEDS)),
                      "anchoring": "te = truck power / drone empty power; EV van ~30, "
                                   "diesel ~60, light electric ~6, parity ~1"},
           "by_te": {}}
    for te in TES:
        per = {}
        for r in rows:
            if r["te"] == te:
                per.setdefault(r["seed"], []).append(r)
        # per-instance: is truck-only the energy max or min? does time-opt dominate?
        summ = {"time_opt_vs_truck_only_energy": [], "energy_min_vs_truck_only_energy": [],
                "time_opt_saving_pct": [], "energy_min_extra_time_pct": [],
                "energy_min_saving_vs_truckonly_energy_pct": []}
        for sd, pts in per.items():
            pts.sort(key=lambda p: p["lam"])
            t0p = pts[0]                     # lambda=0 (time-optimal)
            emin = min(pts, key=lambda p: p["e_total"])
            ref_e = t0p["truck_only_e_total"]
            summ["time_opt_vs_truck_only_energy"].append(t0p["e_total"] / ref_e)
            summ["energy_min_vs_truck_only_energy"].append(emin["e_total"] / ref_e)
            summ["time_opt_saving_pct"].append(
                (t0p["truck_only_ref"] - t0p["makespan"]) / t0p["truck_only_ref"] * 100)
            summ["energy_min_extra_time_pct"].append(
                (emin["makespan"] / t0p["makespan"] - 1) * 100)
            summ["energy_min_saving_vs_truckonly_energy_pct"].append(
                (1 - emin["e_total"] / ref_e) * 100)
        out["by_te"][str(te)] = {k: {"mean": float(np.mean(v)),
                                     "median": float(np.median(v))}
                                 for k, v in summ.items()}
    out["raw"] = [{k: r[k] for k in r if k not in ("order",)} for r in rows]
    out["routes"] = {f"te{r['te']}-s{r['seed']}-l{r['lam']}": r["order"] for r in rows}
    out["config"]["runtime_s"] = time.time() - t0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for te in TES:
        s = out["by_te"][str(te)]
        print(f"te={te}: time-opt E/truck-only E = {s['time_opt_vs_truck_only_energy']['mean']:.3f}; "
              f"energy-min E/truck-only E = {s['energy_min_vs_truck_only_energy']['mean']:.3f} "
              f"(+{s['energy_min_extra_time_pct']['mean']:.1f}% time)")
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
