"""Committed producer for h3_calibrated.json's pooled by_te endpoints (round-11 fix).

Reproduces, from archived data + deterministic re-decoding only (no search), the pooled
per-instance endpoints consumed by Fig. 6a and the Results regime paragraph: for each
(te, seed), the candidate pool is
  (a) every winner row at that (te, seed) and all its replicates,
  (b) the dedicated energy-minimization arm row and its replicates,
  (c) the truck-only plan (T = L0, E = te*L0),
  (d) the exact Pareto-set re-decoding (run_h3_frontier.pareto_decode) of every archived
      order for that seed, evaluated at that te   [variant A: same-te orders only;
      variant B: all archived orders across te arms],
with makespan ties consolidated at relative 1e-12. time-opt endpoint = min-T (tie: min E);
energy-min endpoint = min E. Ratios are relative to truck-only energy te*L0.
Output: experiments/H3-time-energy/results/calibrated_endpoints_check.json
"""
import json
import os
from multiprocessing import Pool

import numpy as np

from problem import gen_instance, dist_matrix
from run_h3_frontier import pareto_decode

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "experiments", "H3-time-energy", "results")
ALPHA = 2.0
TIME_RTOL = 1e-12
TES = (1.0, 6.0, 30.0, 60.0)
SEEDS = range(15)


def decode_job(args):
    sd, te, orders = args
    inst = gen_instance(50, seed=sd)
    D = dist_matrix(inst["coords"])
    pts = []
    for o in orders:
        pts.extend(pareto_decode(o, D, ALPHA, cl=1.0, cb=1.0, te=te))
    return (sd, te, pts)


def endpoints(pool_pts, ref_e):
    pts = sorted(set(pool_pts))
    t0 = pts[0][0]
    e_timeopt = min(e for t, e in pts if t <= t0 * (1 + TIME_RTOL))
    e_min = min(e for _, e in pts)
    return e_timeopt / ref_e, e_min / ref_e


def main():
    cal = json.load(open(os.path.join(RES, "h3_calibrated.json")))
    rows = cal["raw"]
    emin_rows = cal["energymin_rows"]
    routes = cal["routes"]

    orders_by_seed_te, orders_by_seed = {}, {}
    for k, o in routes.items():
        parts = k.split("-")
        if parts[0] == "emin":          # 'emin-te{X}-s{sd}': energy-minimization arm
            te = float(parts[1][2:])
            sd = int(parts[2][1:])
        else:                            # 'te{X}-s{sd}-l{lam}'
            te = float(parts[0][2:])
            sd = int(parts[1][1:])
        orders_by_seed_te.setdefault((sd, te), []).append(o)
        orders_by_seed.setdefault(sd, []).append(o)

    def dedupe(os_):
        seen, out = set(), []
        for o in os_:
            t = tuple(o)
            if t not in seen:
                seen.add(t)
                out.append(o)
        return out

    jobs_a = [(sd, te, dedupe(orders_by_seed_te.get((sd, te), []))) for sd in SEEDS for te in TES]
    jobs_b = [(sd, te, dedupe(orders_by_seed[sd])) for sd in SEEDS for te in TES]
    with Pool(36) as p:
        dec_a = {(sd, te): pts for sd, te, pts in p.map(decode_job, jobs_a)}
        dec_b = {(sd, te): pts for sd, te, pts in p.map(decode_job, jobs_b)}

    out = {"pool": "winners+replicates+energymin(+reps)+truck-only+exact re-decodings",
           "variants": {}}
    for name, dec in (("A_same_te_orders", dec_a), ("B_all_te_orders", dec_b)):
        by_te = {}
        for te in TES:
            to_r, em_r, emin_extra_t, truck_only_is_min = [], [], [], 0
            for sd in SEEDS:
                rws = [r for r in rows if r["te"] == te and r["seed"] == sd]
                L0 = rws[0]["truck_only_ref"]
                ref_e = te * L0
                pool_pts = [(r["makespan"], r["e_total"]) for r in rws]
                pool_pts += [(rep["makespan"], rep["e_total"]) for r in rws
                             for rep in r.get("replicates", [])]
                for r in emin_rows:
                    if r["te"] == te and r["seed"] == sd:
                        pool_pts.append((r["makespan"], r["e_total"]))
                        pool_pts += [(rep["makespan"], rep["e_total"])
                                     for rep in r.get("replicates", [])]
                pool_pts.append((L0, ref_e))
                pool_pts += dec[(sd, te)]
                to, em = endpoints(pool_pts, ref_e)
                to_r.append(to)
                em_r.append(em)
                if abs(em * ref_e - ref_e) <= 1e-9:
                    truck_only_is_min += 1
            by_te[str(te)] = {"time_opt_vs_truck_only_energy_mean": float(np.mean(to_r)),
                              "energy_min_vs_truck_only_energy_mean": float(np.mean(em_r)),
                              "energy_min_is_truck_only_count": truck_only_is_min,
                              "archived_time_opt_mean": cal["by_te"][str(te)]["time_opt_vs_truck_only_energy"]["mean"],
                              "archived_energy_min_mean": cal["by_te"][str(te)]["energy_min_vs_truck_only_energy"]["mean"]}
            print(f"{name} te={te}: time-opt {np.mean(to_r):.6f} (archived "
                  f"{by_te[str(te)]['archived_time_opt_mean']:.6f}) | energy-min {np.mean(em_r):.6f} "
                  f"(archived {by_te[str(te)]['archived_energy_min_mean']:.6f})")
        out["variants"][name] = by_te
    json.dump(out, open(os.path.join(RES, "calibrated_endpoints_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
