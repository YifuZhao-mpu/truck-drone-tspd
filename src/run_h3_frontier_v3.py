"""Archive-conditional time-energy sets from EVERY retained H3-v3 order (protocol §A/§I8).

The submitted analysis (run_h3_frontier.py / analyze_service_level.py) enumerated the
exact fixed-order Pareto decodings of the archived *winning* orders only. The locked
revision protocol requires the completed archive-conditional fronts to be built on "the
union of all retained orders". This driver therefore, for every instance of the nominal
sweep (n in {20, 50}, alpha=2, unlimited endurance, cl=1, te=0.3):

  1. collects every retained solver-replicate order over the 13-lambda grid from
     h3_v3.json (three replicates x 13 lambdas; exact duplicates and reversals are
     decoded once - the model is symmetric, so a reversed order has the same decodings);
  2. computes the exact non-dominated (makespan, total-energy) decodings of each order
     with the unchanged Pareto-set dynamic program of run_h3_frontier.py;
  3. adds the archived adaptive-refinement points of h3_frontier.json (their winner
     orders were not archived, so they contribute weighted-sum points, not decodings);
  4. takes the non-dominated union, and reports the max-chord knee and the best energy
     reduction under +5/+10/+20% delivery-time caps relative to the union's time-optimal
     baseline (time ties consolidated at relative tolerance 1e-12, as in v2);
  5. reports, per instance, how much the non-winner orders add relative to the
     winner-only pool under the same baseline (completion gain, non-negative by
     construction).

Everything is post-processing of archived orders: no solver runs, no re-optimization.
The sets remain conditional on the heuristic order pool and are never global fronts.
Output: experiments/H3-time-energy/results/h3_frontier_v3.json
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from problem import dist_matrix, gen_instance
from revision_utils import git_commit
from run_h3_frontier import front_of, knee_of, pareto_decode

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "experiments", "H3-time-energy", "results")
H3 = os.path.join(RES, "h3_v3.json")
LEGACY_FRONTIER = os.path.join(RES, "h3_frontier.json")
OUT = os.path.join(RES, "h3_frontier_v3.json")
ALPHA = 2.0
DELTAS = (0.05, 0.10, 0.20)
TIME_RTOL = 1e-12


def canonical(order):
    fwd = tuple(int(v) for v in order)
    rev = tuple(reversed(fwd))
    return min(fwd, rev)


def baseline(front):
    t0 = front[0][0]
    e0 = min(e for t, e in front if t <= t0 * (1 + TIME_RTOL))
    return t0, e0


def reduction_at_caps(front, t0, e0):
    out = {}
    for d in DELTAS:
        cap = (1 + d) * t0
        feasible = [e for t, e in front if t <= cap * (1 + TIME_RTOL)]
        out[d] = 100.0 * (1.0 - min(feasible) / e0)
    return out


def job(args):
    n, seed, all_orders, winner_orders, extra_points = args
    inst = gen_instance(n, seed=seed)
    D = dist_matrix(inst["coords"])
    decoded = {}
    for order in all_orders:
        decoded[order] = pareto_decode(list(order), D, ALPHA)
    union_pts = [p for order in all_orders for p in decoded[order]] + extra_points
    winner_pts = [p for order in winner_orders for p in decoded[order]] + extra_points
    front_all = front_of(union_pts)
    front_win = front_of(winner_pts)
    t0, e0 = baseline(front_all)          # common baseline: union front is a superset
    red_all = reduction_at_caps(front_all, t0, e0)
    red_win = reduction_at_caps(front_win, t0, e0)
    knee = knee_of(front_all)
    return {"n": n, "seed": seed, "orders_decoded": len(all_orders),
            "winner_orders": len(winner_orders), "front": front_all,
            "front_points": len(front_all), "winner_front_points": len(front_win),
            "knee_dms_pct": None if knee is None else knee[0],
            "knee_de_pct": None if knee is None else knee[1],
            "reduction_pct": {str(int(d * 100)): red_all[d] for d in DELTAS},
            "winner_only_reduction_pct": {str(int(d * 100)): red_win[d] for d in DELTAS},
            "completion_gain_points": {str(int(d * 100)): red_all[d] - red_win[d]
                                       for d in DELTAS}}


def q(values, p):
    return float(np.percentile(np.asarray(values, dtype=float), p))


def main():
    started = time.time()
    h3 = json.load(open(H3))
    legacy = json.load(open(LEGACY_FRONTIER))
    pools, winners = {}, {}
    for row in h3["rows"]:
        if row["tag"] != "main" or row["cl"] != 1.0 or row["te"] != 0.3:
            continue
        key = (row["n"], row["instance_seed"])
        for rep in row["replicates"]:
            pools.setdefault(key, set()).add(canonical(rep["order"]))
            if rep["solver_seed"] == row["winner_solver_seed"]:
                winners.setdefault(key, set()).add(canonical(rep["order"]))
    extra = {}
    for r in legacy["refined_rows"]:
        key = (r["n"], r["seed"])
        extra.setdefault(key, []).append((r["makespan"], r["e_total"]))
        for rep in r.get("replicates", []):
            extra[key].append((rep["makespan"], rep["e_total"]))
    jobs = [(n, seed, sorted(pools[(n, seed)]), sorted(winners[(n, seed)]),
             extra.get((n, seed), [])) for n in (20, 50) for seed in range(30)]
    print(f"{len(jobs)} instances; orders to decode: "
          f"{sum(len(j[2]) for j in jobs)} (winners {sum(len(j[3]) for j in jobs)})")
    with Pool(36) as pool:
        rows = pool.map(job, jobs)

    summary = {}
    for n in (20, 50):
        rs = [r for r in rows if r["n"] == n]
        knees_t = [r["knee_dms_pct"] for r in rs if r["knee_dms_pct"] is not None]
        knees_e = [r["knee_de_pct"] for r in rs if r["knee_de_pct"] is not None]
        summary[str(n)] = {
            "n_instances": len(rs), "n_knees": len(knees_t),
            "orders_decoded_total": sum(r["orders_decoded"] for r in rs),
            "winner_orders_total": sum(r["winner_orders"] for r in rs),
            "knee_dms_median": q(knees_t, 50), "knee_dms_iqr": [q(knees_t, 25), q(knees_t, 75)],
            "knee_de_median": q(knees_e, 50), "knee_de_iqr": [q(knees_e, 25), q(knees_e, 75)],
            "reduction_pct": {lvl: {"median": q([r["reduction_pct"][lvl] for r in rs], 50),
                                    "iqr": [q([r["reduction_pct"][lvl] for r in rs], 25),
                                            q([r["reduction_pct"][lvl] for r in rs], 75)]}
                              for lvl in ("5", "10", "20")},
            "completion_gain_points": {lvl: {"median": q([r["completion_gain_points"][lvl] for r in rs], 50),
                                             "max": float(max(r["completion_gain_points"][lvl] for r in rs)),
                                             "frac_positive": float(np.mean([r["completion_gain_points"][lvl] > 1e-9 for r in rs]))}
                                       for lvl in ("5", "10", "20")},
            "front_points_median": q([r["front_points"] for r in rs], 50),
        }
    out = {"config": {"source": "h3_v3.json all retained replicate orders (nominal sweep) + "
                                "archived refinement points of h3_frontier.json",
                      "alpha": ALPHA, "endurance": "inf", "cl": 1.0, "te": 0.3,
                      "decoder": "exact Pareto-set DP over a fixed order, unrestricted span "
                                 "(run_h3_frontier.pareto_decode, unchanged)",
                      "refinement_orders": "not archived in h3_frontier.json; their "
                                           "weighted-sum points enter the pool as points",
                      "deltas_pct": [int(d * 100) for d in DELTAS], "time_rtol": TIME_RTOL,
                      "scope": "archive-conditional; never a global front",
                      "git_commit": git_commit(os.path.join(HERE, "..")),
                      "runtime_s": time.time() - started},
           "summary": summary,
           "combined_fronts": {str(n): {str(r["seed"]): r["front"] for r in rows if r["n"] == n}
                               for n in (20, 50)},
           "per_instance": [{k: v for k, v in r.items() if k != "front"} for r in rows]}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"done in {time.time() - started:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
