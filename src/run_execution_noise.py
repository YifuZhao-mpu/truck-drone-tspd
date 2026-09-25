"""Post-hoc execution-noise stress test on all central-cell H2-v3 routes.

Routes are fixed (no stochastic re-optimization).  Mean-one lognormal multipliers are
drawn per physical leg.  A shared truck edge receives the same draw in the hybrid and
truck-only route within a realization, providing common random numbers where routes
overlap.
"""
import json
import math
import os

import numpy as np

from problem import dist_matrix, gen_instance, tspd_split
from revision_utils import bootstrap_ci, git_commit
from tsp_ref import _load_cache, key_of


HERE = os.path.dirname(os.path.abspath(__file__))
H2 = os.path.join(HERE, "..", "experiments", "H2-design-space", "results",
                  "h2_v3.json")
OUT = os.path.join(HERE, "..", "experiments", "P0-robustness", "results",
                   "execution_noise_v3.json")
ALPHA = 2.0
ENDURANCE = 1.0
N_REALIZATIONS = 10000
CVS = (0.10, 0.20)
RECONSTRUCTION_TOL = 1e-12  # absolute; see experiments/protocol-deviations.md row 19
REF = _load_cache()


def op_legs(order, op):
    if op[0] == "truck":
        return [("truck", order[op[1]], order[op[2]], 1.0)], []
    i, j, k = op[1], op[2], op[3]
    kept = [p for p in range(i, k + 1) if p != j]
    truck = [("truck", order[kept[a]], order[kept[a + 1]], 1.0)
             for a in range(len(kept) - 1)]
    drone = [("drone", order[i], order[j], 1.0 / ALPHA),
             ("drone", order[j], order[k], 1.0 / ALPHA)]
    return truck, drone


def simulate_route(seed, order, ops, D, lkh_tour, cv):
    sigma = math.sqrt(math.log1p(cv * cv))
    mu = -0.5 * sigma * sigma
    factors = {}

    def factor(vehicle, u, v):
        # Directed-edge key; the seed depends only on the full physical-leg identity.
        key = (vehicle, int(u), int(v))
        if key not in factors:
            vehicle_code = 1 if vehicle == "truck" else 2
            edge_rng = np.random.default_rng(
                np.random.SeedSequence([20260901, seed, int(round(cv * 1000)),
                                        vehicle_code, int(u), int(v)]))
            factors[key] = edge_rng.lognormal(mu, sigma, size=N_REALIZATIONS)
        return factors[key]

    hybrid = np.zeros(N_REALIZATIONS)
    for op in ops:
        truck_legs, drone_legs = op_legs(order, op)
        truck = np.zeros(N_REALIZATIONS)
        drone = np.zeros(N_REALIZATIONS)
        for vehicle, u, v, speed_factor in truck_legs:
            truck += D[u, v] * speed_factor * factor(vehicle, u, v)
        for vehicle, u, v, speed_factor in drone_legs:
            drone += D[u, v] * speed_factor * factor(vehicle, u, v)
        hybrid += np.maximum(truck, drone)

    truck_only = np.zeros(N_REALIZATIONS)
    for a in range(len(lkh_tour) - 1):
        u, v = lkh_tour[a], lkh_tour[a + 1]
        truck_only += D[u, v] * factor("truck", u, v)
    return (truck_only - hybrid) / truck_only * 100.0, len(factors)


def main():
    h2 = json.load(open(H2))
    rows = []
    nominal = []
    max_reconstruction_delta = 0.0
    central = {r["instance_seed"]: r for r in h2["rows"]
               if r["n"] == 50 and r["ai"] == 2 and r["ei"] == 1
               and r["fleet"] == 1}
    for seed in range(30):
        inst = gen_instance(50, seed=seed)
        D = dist_matrix(inst["coords"])
        h2row = central[seed]
        ref = REF[key_of("uniform", 50, seed, "center", "euclidean")]
        nominal.append(float(np.mean([p["saving_pct"] for p in h2row["replicates"]])))
        for cv in CVS:
            replicate_draws = []
            edge_counts = []
            for rep in h2row["replicates"]:
                order = rep["order"]
                makespan, ops = tspd_split(order, D, ALPHA, ENDURANCE, max_span=None)
                # The archived makespan is the driver's full-audit resummation of the
                # same plan; it can differ from the split DP value by float association
                # only (deviation row 19). Structure must match exactly at both the
                # unrestricted and the archived span-12 decode.
                makespan_12, ops_12 = tspd_split(order, D, ALPHA, ENDURANCE, max_span=12)
                delta = abs(makespan - rep["makespan"])
                if delta > RECONSTRUCTION_TOL or ops != ops_12:
                    raise RuntimeError(f"route reconstruction failed for instance {seed}, "
                                       f"solver seed {rep['solver_seed']} (|delta|={delta})")
                max_reconstruction_delta = max(max_reconstruction_delta, delta)
                draws, n_edges = simulate_route(seed, order, ops, D, ref["tour"], cv)
                replicate_draws.append(draws)
                edge_counts.append(n_edges)
            # Protocol estimand: solver-replicate mean first, then summarize within instance.
            realized = np.mean(np.asarray(replicate_draws), axis=0)
            rows.append({
                "instance_seed": seed, "cv": cv,
                "nominal_all_seed_saving_pct": nominal[-1],
                "median_realized_saving_pct": float(np.median(realized)),
                "q05_realized_saving_pct": float(np.percentile(realized, 5)),
                "q95_realized_saving_pct": float(np.percentile(realized, 95)),
                "probability_realized_saving_positive": float(np.mean(realized > 0.0)),
                "mean_realized_saving_pct": float(np.mean(realized)),
                "unique_directed_edges_by_replicate": edge_counts,
            })

    summary = {}
    nominal_mean = float(np.mean(nominal))
    for cv in CVS:
        group = [r for r in rows if r["cv"] == cv]
        medians = [r["median_realized_saving_pct"] for r in group]
        q05 = [r["q05_realized_saving_pct"] for r in group]
        q95 = [r["q95_realized_saving_pct"] for r in group]
        probs = [r["probability_realized_saving_positive"] for r in group]
        summary[str(cv)] = {
            "nominal_mean_saving_pct": nominal_mean,
            "mean_instance_median_realized_saving_pct": float(np.mean(medians)),
            "mean_instance_median_ci95": bootstrap_ci(medians, 1000 + int(cv * 100)),
            "mean_instance_q05_realized_saving_pct": float(np.mean(q05)),
            "mean_instance_q05_ci95": bootstrap_ci(q05, 2000 + int(cv * 100)),
            "mean_instance_q95_realized_saving_pct": float(np.mean(q95)),
            "mean_instance_q95_ci95": bootstrap_ci(q95, 3000 + int(cv * 100)),
            "mean_probability_realized_saving_positive": float(np.mean(probs)),
            "probability_positive_ci95": bootstrap_ci(probs, 4000 + int(cv * 100)),
            "median_change_from_nominal_points": float(np.mean(medians) - nominal_mean),
        }
    commit = git_commit(os.path.join(HERE, ".."))
    out = {
        "config": {"n": 50, "alpha": ALPHA, "endurance": ENDURANCE,
                   "cvs": CVS, "realizations_per_route": N_REALIZATIONS,
                   "distribution": "mean-one lognormal independent by vehicle/directed edge; "
                                   "common random numbers on shared truck edges",
                   "scope": "post-hoc fixed-route stress test; no stochastic re-optimization",
                   "aggregation": "mean over 3 solver routes within each instance/realization; "
                                  "distribution statistics within instance; mean/bootstrap over instances",
                   "bootstrap_resamples": 20000, "bootstrap_unit": "instance",
                   "rng_seed_rule": "SeedSequence([20260901, instance, round(1000*CV), "
                                    "vehicle_code, directed_from, directed_to])",
                   "route_reconstruction": {"tolerance_abs": RECONSTRUCTION_TOL,
                                            "max_abs_delta_observed": max_reconstruction_delta,
                                            "ops_identical_unrestricted_vs_span12": True},
                   "git_commit": commit},
        "summary": summary,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
