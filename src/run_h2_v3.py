"""H2 v3: exact v2 rerun with every algorithm seed and route retained.

The experimental design and solver settings are unchanged. A hard reproduction gate
requires every archived winner field and route to reproduce exactly. New summaries
use the within-instance mean across all three algorithm seeds.
"""
import argparse
import hashlib
import json
import os
import time
import numpy as np

from alns import alns
from problem import dist_matrix, gen_instance
from revision_utils import bootstrap_ci, git_commit, run_checkpointed
from tsp_ref import _load_cache, key_of


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "experiments", "H2-design-space", "results",
                   "h2_v3.json")
OLD = os.path.join(HERE, "..", "experiments", "H2-design-space", "results",
                   "h2_v2.json")
CHECKPOINT = os.path.join(HERE, "..", "experiments", "H2-design-space", "results",
                          "h2_v3.jsonl")
ALPHAS = [1.0, 1.5, 2.0, 2.5, 3.0]
ENDURANCE = [0.5, 1.0, 2.0, np.inf]
ENDURANCE_KEYS = ["0.5", "1.0", "2.0", "inf"]
FLEETS = [1, 2, 3]
SIZES = [20, 50]
INSTANCE_SEEDS = range(30)
SOLVER_SEEDS = range(3)
ITERS = {20: 9000, 50: 16000}
SPAN = 12
REF = _load_cache()


def source_digest():
    h = hashlib.sha256()
    for name in ("problem.py", "alns.py", "run_h2_v3.py"):
        with open(os.path.join(HERE, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def job(args):
    n, instance_seed, ai, ei, fleet = args
    inst = gen_instance(n, seed=instance_seed)
    D = dist_matrix(inst["coords"])
    ref = REF[key_of("uniform", n, instance_seed, "center", "euclidean")]["length"]
    reps = []
    for solver_seed in SOLVER_SEEDS:
        started = time.perf_counter()
        result = alns(inst, D, ALPHAS[ai], endurance=ENDURANCE[ei], max_span=SPAN,
                      iters=ITERS[n], sortie_aware=True, seed=solver_seed, m=fleet)
        ev = result["eval"]
        reps.append({
            "solver_seed": solver_seed, "makespan": float(ev["makespan"]),
            "saving_pct": float((ref - ev["makespan"]) / ref * 100),
            "e_drone": float(ev["e_drone"]), "e_truck": float(ev["e_truck"]),
            "n_sorties": int(ev["n_sorties"]),
            "wall_s": float(time.perf_counter() - started),
            "order": [int(v) for v in result["order"]],
        })
    winner = min(reps, key=lambda r: (r["makespan"], r["solver_seed"]))
    return {"n": n, "instance_seed": instance_seed, "ai": ai, "ei": ei,
            "fleet": fleet, "truck_ref": float(ref),
            "winner_solver_seed": winner["solver_seed"], "replicates": reps}


def row_key(row):
    return [row["n"], row["instance_seed"], row["ai"], row["ei"], row["fleet"]]


def reproduction_gate(rows, expected_count=3600):
    old = json.load(open(OLD))
    expected = {(r["n"], r["seed"], r["ai"], r["ei"], r["m"]): r
                for r in old["raw"]}
    float_fields = ("makespan", "saving_pct", "e_drone", "e_truck")
    field_checks = 0
    route_checks = 0
    for row in rows:
        winner = min(row["replicates"], key=lambda r: (r["makespan"], r["solver_seed"]))
        key = (row["n"], row["instance_seed"], row["ai"], row["ei"], row["fleet"])
        oldrow = expected[key]
        observed = {**winner, "truck_ref": row["truck_ref"]}
        for field in float_fields + ("truck_ref",):
            field_checks += 1
            if observed[field] != oldrow[field]:
                raise RuntimeError(f"H2 field gate failed at {key}/{field}: "
                                   f"{observed[field]} != {oldrow[field]}")
        field_checks += 1
        if observed["n_sorties"] != oldrow["n_sorties"]:
            raise RuntimeError(f"H2 field gate failed at {key}/n_sorties")
        route_key = (f"n{row['n']}-s{row['instance_seed']}-a{row['ai']}-"
                     f"e{row['ei']}-m{row['fleet']}")
        route_checks += 1
        if winner["order"] != old["routes"][route_key]:
            raise RuntimeError(f"H2 route gate failed at {key}")
    if len(rows) != expected_count:
        raise RuntimeError(f"H2 gate row count {len(rows)} != {expected_count}")
    return {"status": "PASS", "rows": len(rows),
            "archived_fields_checked": field_checks, "routes_checked": route_checks,
            "comparison": "exact Python equality"}


def summaries(rows):
    cells = {}
    per_instance = {}
    pooled_advantage = {}
    for n in SIZES:
        cells[str(n)] = {}
        for ai, alpha in enumerate(ALPHAS):
            for ei, ek in enumerate(ENDURANCE_KEYS):
                for fleet in FLEETS:
                    group = sorted((r for r in rows if r["n"] == n and r["ai"] == ai
                                    and r["ei"] == ei and r["fleet"] == fleet),
                                   key=lambda r: r["instance_seed"])
                    means = [float(np.mean([p["saving_pct"] for p in r["replicates"]]))
                             for r in group]
                    best = [max(p["saving_pct"] for p in r["replicates"]) for r in group]
                    worst = [min(p["saving_pct"] for p in r["replicates"]) for r in group]
                    advantages = np.asarray(best) - np.asarray(means)
                    spreads = [(max(p["makespan"] for p in r["replicates"])
                                - min(p["makespan"] for p in r["replicates"]))
                               / min(p["makespan"] for p in r["replicates"]) * 100
                               for r in group]
                    key = f"a{alpha}_E{ek}_m{fleet}"
                    per_instance[(n, ai, ei, fleet)] = means
                    for r, value in zip(group, advantages):
                        pooled_advantage.setdefault((n, r["instance_seed"]), []).append(
                            float(value))
                    cells[str(n)][key] = {
                        "all_seed_mean_saving": float(np.mean(means)),
                        "all_seed_ci95": bootstrap_ci(means, n * 1000 + ai * 100 + ei * 10 + fleet),
                        "best_of_3_mean_saving": float(np.mean(best)),
                        "worst_seed_mean_saving": float(np.mean(worst)),
                        "selection_gain_points": float(np.mean(advantages)),
                        "selection_gain_ci95": bootstrap_ci(
                            advantages, 600000 + n * 1000 + ai * 100 + ei * 10 + fleet),
                        "selection_gain_distribution": {
                            "median": float(np.median(advantages)),
                            "iqr": [float(np.percentile(advantages, 25)),
                                    float(np.percentile(advantages, 75))],
                            "min": float(np.min(advantages)),
                            "max": float(np.max(advantages))},
                        "median_replicate_spread_pct": float(np.median(spreads)),
                    }
    interactions = {}
    complementarity = {}
    for n in SIZES:
        interactions[str(n)] = {}
        for ai, alpha in enumerate(ALPHAS):
            for ei, ek in enumerate(ENDURANCE_KEYS):
                marginal_12 = (np.asarray(per_instance[(n, ai, ei, 2)])
                               - np.asarray(per_instance[(n, ai, ei, 1)]))
                marginal_23 = (np.asarray(per_instance[(n, ai, ei, 3)])
                               - np.asarray(per_instance[(n, ai, ei, 2)]))
                interactions[str(n)][f"a{alpha}_E{ek}"] = {
                    "m1_to_2_points": float(np.mean(marginal_12)),
                    "m1_to_2_ci95": bootstrap_ci(
                        marginal_12, 700000 + n * 100 + ai * 10 + ei),
                    "m2_to_3_points": float(np.mean(marginal_23)),
                    "m2_to_3_ci95": bootstrap_ci(
                        marginal_23, 800000 + n * 100 + ai * 10 + ei),
                }
        complementarity[str(n)] = {}
        for ei, ek in enumerate(ENDURANCE_KEYS):
            for lower, upper in ((1, 2), (2, 3)):
                low = (np.asarray(per_instance[(n, 0, ei, upper)])
                       - np.asarray(per_instance[(n, 0, ei, lower)]))
                high = (np.asarray(per_instance[(n, 4, ei, upper)])
                        - np.asarray(per_instance[(n, 4, ei, lower)]))
                delta = high - low
                ckey = f"E{ek}_m{lower}_to_{upper}_alpha3_minus_alpha1"
                complementarity[str(n)][ckey] = {
                    "mean_points": float(np.mean(delta)),
                    "ci95": bootstrap_ci(delta, 900000 + n * 100 + ei * 10 + lower),
                }
    selection = [v["selection_gain_points"] for byn in cells.values() for v in byn.values()]
    pooled_instance = [float(np.mean(pooled_advantage[key]))
                       for key in sorted(pooled_advantage)]
    return (cells, interactions, complementarity,
            {"median_cell_mean_points": float(np.median(selection)),
             "max_points": float(np.max(selection)),
             "min_points": float(np.min(selection)),
             "pooled_instance_mean_points": float(np.mean(pooled_instance)),
             "pooled_instance_ci95": bootstrap_ci(pooled_instance, 999001),
             "pooled_instance_distribution": {
                 "median": float(np.median(pooled_instance)),
                 "iqr": [float(np.percentile(pooled_instance, 25)),
                         float(np.percentile(pooled_instance, 75))]}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--workers", type=int, default=36)
    args = parser.parse_args()
    started = time.time()
    digest = source_digest()
    jobs = [(n, seed, ai, ei, fleet) for n in SIZES for seed in INSTANCE_SEEDS
            for ai in range(len(ALPHAS)) for ei in range(len(ENDURANCE))
            for fleet in FLEETS]
    if args.canary:
        picks = (0, 1, 59, 60, 1200, 1800, 2400, 3599)
        jobs = [jobs[i] for i in picks]
    print(f"{len(jobs)} H2 v3 jobs, 3 retained replicates each")
    commit = git_commit(os.path.join(HERE, ".."))
    checkpoint = CHECKPOINT.replace(".jsonl", ".canary.jsonl") if args.canary else CHECKPOINT
    metadata = {"driver": "run_h2_v3.py", "git_commit": commit,
                "source_sha256": digest, "canary": args.canary,
                "solver_seeds": list(SOLVER_SEEDS), "iters": ITERS, "span": SPAN}
    rows = run_checkpointed(jobs, job, lambda x: list(x), row_key, checkpoint,
                            metadata, processes=args.workers)
    gate = reproduction_gate(rows, len(jobs))
    if args.canary:
        print(json.dumps({"gate": gate, "checkpoint": checkpoint}, indent=2))
        return
    cells, interactions, complementarity, selection = summaries(rows)
    raw = []
    for row in rows:
        winner = min(row["replicates"], key=lambda r: (r["makespan"], r["solver_seed"]))
        raw.append({"n": row["n"], "seed": row["instance_seed"], "ai": row["ai"],
                    "ei": row["ei"], "m": row["fleet"],
                    "makespan": winner["makespan"], "truck_ref": row["truck_ref"],
                    "saving_pct": winner["saving_pct"], "e_drone": winner["e_drone"],
                    "e_truck": winner["e_truck"], "n_sorties": winner["n_sorties"],
                    "winner_solver_seed": winner["solver_seed"]})
    out = {
        "config": {"alphas": ALPHAS, "endurance": ENDURANCE_KEYS, "ms": FLEETS,
                   "n_instances": len(list(INSTANCE_SEEDS)),
                   "solver_seeds": list(SOLVER_SEEDS), "iters": ITERS, "span": SPAN,
                   "aggregation": "replicate mean within instance; bootstrap over instances",
                   "bootstrap_resamples": 20000, "bootstrap_unit": "instance",
                   "git_commit": commit, "source_sha256": digest,
                   "checkpoint": os.path.relpath(checkpoint, os.path.join(HERE, "..")),
                   "runtime_s": time.time() - started},
        "reproduction_gate": gate, "cells": cells, "interactions": interactions,
        "complementarity_alpha3_minus_alpha1": complementarity,
        "selection_gain_summary": selection, "raw": raw, "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    for n in SIZES:
        c = cells[str(n)]
        print(f"n={n} central all-seed savings: "
              f"{c['a2.0_E1.0_m1']['all_seed_mean_saving']:.2f}/"
              f"{c['a2.0_E1.0_m2']['all_seed_mean_saving']:.2f}/"
              f"{c['a2.0_E1.0_m3']['all_seed_mean_saving']:.2f}%")
    print(f"gate={gate}; selection={selection}; runtime={time.time()-started:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
