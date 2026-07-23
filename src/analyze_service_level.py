"""Energy attainable under hard service-time caps, from the COMPLETED per-instance fronts.

v2 definition (post round-10 review): (1) machine-equal makespans are consolidated with a
relative time tolerance (1e-12) so the time-optimal baseline is the LOWEST energy among
time-tied points; (2) the completed pool is the UNION of the archived completed front and
the full pre-completion pool (so it is a superset by construction); (3) both pools are
evaluated against the COMMON time-optimal baseline (T0, E0) of the union front, so
completion gains are non-negative and like-for-like.
Pre-completion pool = 13-lambda weighted-sum winners (h3_v2.json, tag=main) + all
adaptive-refinement rows and replicates (h3_frontier.json). Archived artifacts only.
Output: experiments/H3-time-energy/results/service_level_v2.json
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "experiments", "H3-time-energy", "results")
DELTAS = (0.05, 0.10, 0.20)
TIME_RTOL = 1e-12


def pareto(points):
    """points: iterable of (T, E) -> non-dominated set sorted by T (strict E decrease)."""
    pts = sorted(set(points))
    front = []
    best_e = float("inf")
    for t, e in pts:
        if e < best_e - 1e-15:
            front.append((t, e))
            best_e = e
    return front


def baseline(front):
    """(T0, E0): minimum time and the lowest energy among time-tied minimum points."""
    t0 = front[0][0]
    e0 = min(e for t, e in front if t <= t0 * (1 + TIME_RTOL))
    return t0, e0


def reduction_at_caps(front, t0, e0):
    out = {}
    for d in DELTAS:
        cap = (1 + d) * t0
        feas = [e for t, e in front if t <= cap * (1 + TIME_RTOL)]
        out[d] = 100.0 * (1.0 - min(feas) / e0)
    return out


def q(x, p):
    return float(np.percentile(np.asarray(x, dtype=float), p))


def main():
    h3f = json.load(open(os.path.join(RES, "h3_frontier.json")))
    h3 = json.load(open(os.path.join(RES, "h3_v2.json")))

    pre_pool = {}
    for r in h3["raw"]:
        if r["tag"] == "main" and r["te"] == 0.3 and r["cl"] == 1.0:
            pre_pool.setdefault((r["n"], r["seed"]), []).append((r["makespan"], r["e_total"]))
    for r in h3f["refined_rows"]:
        pre_pool.setdefault((r["n"], r["seed"]), []).append((r["makespan"], r["e_total"]))
        for rep in r.get("replicates", []):
            pre_pool[(r["n"], r["seed"])].append((rep["makespan"], rep["e_total"]))

    out = {"deltas_pct": [int(d * 100) for d in DELTAS], "time_rtol": TIME_RTOL, "per_n": {}}
    for n in (20, 50):
        cf = h3f["combined_fronts"][str(n)]
        red_c = {d: [] for d in DELTAS}
        gain = {d: [] for d in DELTAS}
        per_inst = []
        for sid, pts in cf.items():
            pre_pts = pre_pool[(n, int(sid))]
            front_c = pareto([tuple(p) for p in pts] + pre_pts)  # union: superset by construction
            front_p = pareto(pre_pts)
            t0, e0 = baseline(front_c)  # common baseline for both pools
            rc = reduction_at_caps(front_c, t0, e0)
            rp = reduction_at_caps(front_p, t0, e0)
            per_inst.append({"seed": int(sid),
                             **{f"c{int(d*100)}": rc[d] for d in DELTAS},
                             **{f"p{int(d*100)}": rp[d] for d in DELTAS}})
            for d in DELTAS:
                red_c[d].append(rc[d])
                gain[d].append(rc[d] - rp[d])
        out["per_n"][str(n)] = {
            "completed_reduction_pct": {str(int(d * 100)): {"median": q(red_c[d], 50),
                                                            "iqr": [q(red_c[d], 25), q(red_c[d], 75)]}
                                        for d in DELTAS},
            "completion_gain_pct_points": {str(int(d * 100)): {"median": q(gain[d], 50),
                                                               "max": float(max(gain[d])),
                                                               "min": float(min(gain[d])),
                                                               "frac_positive": float(np.mean([g > 1e-9 for g in gain[d]]))}
                                           for d in DELTAS},
            "per_instance": per_inst}
        print(f"n={n}")
        for d in DELTAS:
            print(f"  +{int(d*100)}%: completed median {q(red_c[d],50):.1f} "
                  f"[{q(red_c[d],25):.1f},{q(red_c[d],75):.1f}] | gain: median "
                  f"{q(gain[d],50):.2f}pt max {max(gain[d]):.1f}pt min {min(gain[d]):.2f}")
    out["note"] = ("Reductions relative to the COMMON time-optimal baseline (T0, E0) of the "
                   "union front (archived completed front UNION pre-completion pool -> "
                   "Pareto), with time ties consolidated at relative tolerance 1e-12; the "
                   "union makes the completed pool a superset of the pre-pool, so gains are "
                   "non-negative. Fronts are exact over the archived winner-order pool only.")
    json.dump(out, open(os.path.join(RES, "service_level_v2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
