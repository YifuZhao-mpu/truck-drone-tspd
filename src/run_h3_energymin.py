"""Round-5 fix: the calibrated 'energy-minimal' comparator was just the lowest of eight
finite-lambda samples and omitted the truck-only plan. Add (a) a dedicated pure-energy
arm (effective lambda 100x the largest sampled trade intensity) and (b) the truck-only
plan as an explicit candidate; recompute the by_te summaries with the true minimum over
{all sampled points, energy arm, truck-only}. Replicates archived."""
import json, os
from multiprocessing import Pool
import numpy as np
from problem import gen_instance, dist_matrix
from alns import alns
from tsp_ref import _load_cache, key_of

OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "H3-time-energy",
                   "results", "h3_calibrated.json")
REF = _load_cache()
TES = [1.0, 6.0, 30.0, 60.0]

def job(args):
    seed, te = args
    lam = 100.0 * 0.3 / te
    inst = gen_instance(50, seed=seed)
    D = dist_matrix(inst["coords"])
    reps, best = [], None
    for s in range(3):
        r = alns(inst, D, 2.0, endurance=np.inf, max_span=12, iters=16000,
                 sortie_aware=True, seed=s, lam=lam, cl=1.0, cb=1.0, te=te)
        ev = r["eval"]
        reps.append({"seed": s, "makespan": ev["makespan"], "e_total": ev["e_total"]})
        if best is None or r["objective"] < best["objective"]:
            best = r
    ev = best["eval"]
    return {"seed": seed, "te": te, "lam": lam, "makespan": ev["makespan"],
            "e_total": ev["e_total"], "e_drone": ev["e_drone"], "tag": "energymin",
            "replicates": reps, "order": [int(v) for v in best["order"]]}

def main():
    with Pool(36) as pool:
        rows = pool.map(job, [(sd, te) for te in TES for sd in range(15)])
    d = json.load(open(OUT))
    d["energymin_rows"] = [{k: r[k] for k in r if k != "order"} for r in rows]
    for r in rows:
        d["routes"][f"emin-te{r['te']}-s{r['seed']}"] = r["order"]
    # recompute by_te with true minimum incl. truck-only candidate
    for te in TES:
        per = {}
        for r in d["raw"]:
            if r["te"] == te:
                per.setdefault(r["seed"], []).append(r)
        summ = {"time_opt_vs_truck_only_energy": [], "energy_min_vs_truck_only_energy": [],
                "time_opt_saving_pct": [], "energy_min_extra_time_pct": [],
                "energy_min_is_truck_only_count": 0}
        for sd, pts in per.items():
            pts = sorted(pts, key=lambda p: p["lam"])
            ref = REF[key_of("uniform", 50, sd, "center", "euclidean")]["length"]
            ref_e = te * ref
            t0p = pts[0]
            cands = [(p["makespan"], p["e_total"]) for p in pts]
            cands += [(r["makespan"], r["e_total"]) for r in rows
                      if r["te"] == te and r["seed"] == sd]
            cands.append((ref, ref_e))          # truck-only plan itself
            emin = min(cands, key=lambda c: c[1])
            if abs(emin[1] - ref_e) < 1e-9:
                summ["energy_min_is_truck_only_count"] += 1
            summ["time_opt_vs_truck_only_energy"].append(t0p["e_total"] / ref_e)
            summ["energy_min_vs_truck_only_energy"].append(emin[1] / ref_e)
            summ["time_opt_saving_pct"].append((ref - t0p["makespan"]) / ref * 100)
            summ["energy_min_extra_time_pct"].append((emin[0] / t0p["makespan"] - 1) * 100)
        d["by_te"][str(te)] = {k: ({"mean": float(np.mean(v)), "median": float(np.median(v))}
                                   if isinstance(v, list) else v) for k, v in summ.items()}
    d["config"]["energymin_arm"] = "lambda = 100*0.3/te, best-of-3; truck-only plan included as candidate"
    json.dump(d, open(OUT, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for te in TES:
        s = d["by_te"][str(te)]
        print(f"te={te}: time-opt {s['time_opt_vs_truck_only_energy']['mean']:.3f} / "
              f"TRUE energy-min {s['energy_min_vs_truck_only_energy']['mean']:.3f} "
              f"(+{s['energy_min_extra_time_pct']['mean']:.1f}% time; "
              f"truck-only is emin on {s['energy_min_is_truck_only_count']}/15)")

if __name__ == "__main__":
    main()
