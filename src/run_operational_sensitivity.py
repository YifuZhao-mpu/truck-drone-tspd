"""Revision experiment: service, handling, eligibility and reserve sensitivity.

The design is locked in ``drafts/revision/revision_protocol.md``.  All three solver
replicates and routes are retained.  The base arm must reproduce the corresponding H2
v2 winner exactly before the artifact is accepted.
"""
import argparse
import json
import os
import time
import numpy as np

from alns import alns
from problem import (dist_matrix, gen_instance, tspd_cost_friction,
                     tspd_split_friction)
from revision_utils import bootstrap_ci, git_commit, run_checkpointed
from tsp_ref import _load_cache, key_of
from validate_friction import evaluate_operations


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "experiments", "P0-robustness", "results",
                   "operational_sensitivity_v3.json")
H2 = os.path.join(HERE, "..", "experiments", "H2-design-space", "results",
                  "h2_v3.json")
CHECKPOINT = os.path.join(HERE, "..", "experiments", "P0-robustness", "results",
                          "operational_sensitivity_v3.jsonl")
N = 50
SEEDS = range(30)
SOLVER_SEEDS = range(3)
ALPHA = 2.0
E = 1.0
ITERS = 16000
SPAN = 12
REF = _load_cache()


def median_interstop_time():
    legs = []
    for seed in SEEDS:
        inst = gen_instance(N, seed=seed)
        D = dist_matrix(inst["coords"])
        ent = REF[key_of("uniform", N, seed, "center", "euclidean")]
        tour = ent["tour"]
        legs.extend(float(D[tour[a], tour[a + 1]]) for a in range(len(tour) - 1))
    return float(np.median(legs))


TAU = median_interstop_time()
CELLS = {
    "base": {"t_service": 0.0, "t_launch": 0.0, "t_recover": 0.0,
             "eligibility": 1.0, "endurance": E},
    "service_0.25tau": {"t_service": 0.25 * TAU, "t_launch": 0.0,
                        "t_recover": 0.0, "eligibility": 1.0, "endurance": E},
    "service_0.50tau": {"t_service": 0.50 * TAU, "t_launch": 0.0,
                        "t_recover": 0.0, "eligibility": 1.0, "endurance": E},
    "handling_0.25tau_each": {"t_service": 0.0, "t_launch": 0.25 * TAU,
                              "t_recover": 0.25 * TAU, "eligibility": 1.0,
                              "endurance": E},
    "handling_0.50tau_each": {"t_service": 0.0, "t_launch": 0.50 * TAU,
                              "t_recover": 0.50 * TAU, "eligibility": 1.0,
                              "endurance": E},
    "joint_0.25tau": {"t_service": 0.25 * TAU, "t_launch": 0.25 * TAU,
                      "t_recover": 0.25 * TAU, "eligibility": 1.0,
                      "endurance": E},
    "eligible_75pct": {"t_service": 0.0, "t_launch": 0.0, "t_recover": 0.0,
                       "eligibility": 0.75, "endurance": E},
    "eligible_50pct": {"t_service": 0.0, "t_launch": 0.0, "t_recover": 0.0,
                       "eligibility": 0.50, "endurance": E},
    "reserve_20pct": {"t_service": 0.0, "t_launch": 0.0, "t_recover": 0.0,
                      "eligibility": 1.0, "endurance": 0.8},
}


def eligibility_mask(instance_seed, fraction):
    if fraction >= 1.0:
        return None
    rng = np.random.default_rng((instance_seed, 0xE119))
    priority = rng.permutation(np.arange(1, N + 1))
    mask = np.zeros(N + 1, dtype=bool)
    mask[0] = True
    mask[priority[:int(np.ceil(fraction * N))]] = True
    return mask


def full_path_error(result, D, cfg, mask):
    """Audit every friction-bearing result at the search and final spans."""
    if not (cfg["t_service"] or cfg["t_launch"] or cfg["t_recover"]
            or mask is not None):
        return 0.0
    kwargs = {"t_service": cfg["t_service"], "t_launch": cfg["t_launch"],
              "t_recover": cfg["t_recover"], "eligible": mask}
    errors = []
    for span, stored in ((SPAN, result["search_objective"]),
                         (N + 1, result["makespan"])):
        fast = tspd_cost_friction(result["order"], D, ALPHA, cfg["endurance"],
                                  span, **kwargs)
        full, ops = tspd_split_friction(result["order"], D, ALPHA,
                                        cfg["endurance"], span, **kwargs)
        event = evaluate_operations(result["order"], ops, D, D, ALPHA,
                                    cfg["t_service"], cfg["t_launch"],
                                    cfg["t_recover"])
        errors.extend((abs(fast - full), abs(fast - event), abs(fast - stored)))
    maximum = max(errors)
    if maximum >= 1e-9:
        raise RuntimeError(f"operational-friction full-path audit failed: {maximum}")
    return float(maximum)


def job(args):
    instance_seed, cell_name = args
    cfg = CELLS[cell_name]
    inst = gen_instance(N, seed=instance_seed)
    D = dist_matrix(inst["coords"])
    mask = eligibility_mask(instance_seed, cfg["eligibility"])
    ref = REF[key_of("uniform", N, instance_seed, "center", "euclidean")]["length"]
    adjusted_ref = ref + N * cfg["t_service"]
    reps = []
    for solver_seed in SOLVER_SEEDS:
        started = time.perf_counter()
        kwargs = {}
        if (cfg["t_service"] or cfg["t_launch"] or cfg["t_recover"]
                or mask is not None):
            kwargs = {"t_service": cfg["t_service"], "t_launch": cfg["t_launch"],
                      "t_recover": cfg["t_recover"], "eligible": mask}
        result = alns(inst, D, ALPHA, endurance=cfg["endurance"], max_span=SPAN,
                      iters=ITERS, sortie_aware=True, seed=solver_seed, **kwargs)
        ev = result["eval"]
        path_error = full_path_error(result, D, cfg, mask)
        reps.append({
            "solver_seed": solver_seed,
            "makespan": float(result["makespan"]),
            "saving_pct": float((adjusted_ref - result["makespan"]) / adjusted_ref * 100),
            "movement_makespan": float(ev.get("movement_makespan", ev["makespan"])),
            "n_sorties": int(ev["n_sorties"]),
            "e_drone": float(ev["e_drone"]), "e_truck": float(ev["e_truck"]),
            "full_path_max_abs_error": path_error,
            "wall_s": float(time.perf_counter() - started),
            "order": [int(v) for v in result["order"]],
        })
    winner = min(reps, key=lambda r: (r["makespan"], r["solver_seed"]))
    return {"cell": cell_name, "instance_seed": instance_seed,
            "truck_ref": float(ref), "adjusted_truck_ref": float(adjusted_ref),
            "eligible_nodes": (None if mask is None else
                               [int(v) for v in np.flatnonzero(mask) if v != 0]),
            "replicates": reps, "winner_solver_seed": winner["solver_seed"]}


def summarize(rows):
    out = {}
    base = {r["instance_seed"]: r for r in rows if r["cell"] == "base"}
    for ci, name in enumerate(CELLS):
        group = sorted((r for r in rows if r["cell"] == name),
                       key=lambda r: r["instance_seed"])
        best = [max(rp["saving_pct"] for rp in r["replicates"]) for r in group]
        all_seed = [float(np.mean([rp["saving_pct"] for rp in r["replicates"]]))
                    for r in group]
        worst = [min(rp["saving_pct"] for rp in r["replicates"]) for r in group]
        spreads = [(max(rp["makespan"] for rp in r["replicates"])
                    - min(rp["makespan"] for rp in r["replicates"]))
                   / min(rp["makespan"] for rp in r["replicates"]) * 100 for r in group]
        absolute_changes = []
        relative_changes = []
        for row in group:
            base_reps = {p["solver_seed"]: p for p in base[row["instance_seed"]]["replicates"]}
            changes = [p["makespan"] - base_reps[p["solver_seed"]]["makespan"]
                       for p in row["replicates"]]
            relative = [100.0 * (p["makespan"] / base_reps[p["solver_seed"]]["makespan"] - 1.0)
                        for p in row["replicates"]]
            absolute_changes.append(float(np.mean(changes)))
            relative_changes.append(float(np.mean(relative)))
        out[name] = {
            "best_of_3_mean_saving_pct": float(np.mean(best)),
            "best_of_3_ci95": bootstrap_ci(best, 100 + ci),
            "all_seed_mean_saving_pct": float(np.mean(all_seed)),
            "all_seed_ci95": bootstrap_ci(all_seed, 200 + ci),
            "worst_seed_mean_saving_pct": float(np.mean(worst)),
            "selection_gain_points": float(np.mean(best) - np.mean(all_seed)),
            "median_replicate_spread_pct": float(np.median(spreads)),
            "hybrid_makespan_change_vs_base": {
                "absolute_mean": float(np.mean(absolute_changes)),
                "absolute_ci95": bootstrap_ci(absolute_changes, 300 + ci),
                "relative_mean_pct": float(np.mean(relative_changes)),
                "relative_ci95": bootstrap_ci(relative_changes, 400 + ci),
            },
        }
    return out


def check_base(rows):
    old = json.load(open(H2))
    expected = {r["instance_seed"]: r for r in old["rows"]
                if r["n"] == N and r["ai"] == 2 and r["ei"] == 1
                and r["fleet"] == 1}
    checks = 0
    for row in rows:
        if row["cell"] != "base":
            continue
        expected_reps = {p["solver_seed"]: p for p in expected[row["instance_seed"]]["replicates"]}
        for observed in row["replicates"]:
            archived = expected_reps[observed["solver_seed"]]
            for field in ("makespan", "saving_pct", "e_drone", "e_truck",
                          "n_sorties", "order"):
                checks += 1
                if observed[field] != archived[field]:
                    raise RuntimeError(f"base all-seed gate failed: instance "
                                       f"{row['instance_seed']}, solver seed "
                                       f"{observed['solver_seed']}, field {field}")
    nbase = sum(r["cell"] == "base" for r in rows)
    if nbase != 30:
        raise RuntimeError(f"base reproduction gate expected 30 rows, got {nbase}")
    return {"status": "PASS", "instance_rows": nbase,
            "replicate_field_checks": checks, "comparison": "exact Python equality"}


def row_key(row):
    return [row["instance_seed"], row["cell"]]


def reused_base_rows(h2):
    rows = []
    for source in h2["rows"]:
        if not (source["n"] == N and source["ai"] == 2 and source["ei"] == 1
                and source["fleet"] == 1):
            continue
        reps = []
        for rep in source["replicates"]:
            reps.append({**rep, "movement_makespan": rep["makespan"],
                         "full_path_max_abs_error": 0.0})
        rows.append({"cell": "base", "instance_seed": source["instance_seed"],
                     "truck_ref": source["truck_ref"],
                     "adjusted_truck_ref": source["truck_ref"],
                     "eligible_nodes": None, "replicates": reps,
                     "winner_solver_seed": source["winner_solver_seed"],
                     "provenance": "reused H2-v3 central-cell rows"})
    if len(rows) != 30:
        raise RuntimeError(f"expected 30 reused H2-v3 base rows, got {len(rows)}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=36)
    args = parser.parse_args()
    started = time.time()
    h2 = json.load(open(H2))
    base_rows = reused_base_rows(h2)
    jobs = [(seed, cell) for cell in CELLS if cell != "base" for seed in SEEDS]
    print(f"{len(jobs)} operational-sensitivity jobs, 3 retained replicates each")
    commit = git_commit(os.path.join(HERE, ".."))
    metadata = {"driver": "run_operational_sensitivity.py", "git_commit": commit,
                "cells": CELLS, "tau": TAU, "solver_seeds": list(SOLVER_SEEDS),
                "iters": ITERS, "span": SPAN}
    new_rows = run_checkpointed(jobs, job, lambda x: list(x), row_key, CHECKPOINT,
                                metadata, processes=args.workers)
    rows = base_rows + new_rows
    gate = check_base(rows)
    out = {
        "config": {"n": N, "alpha": ALPHA, "nominal_endurance": E,
                   "iters": ITERS, "span": SPAN, "solver_seeds": list(SOLVER_SEEDS),
                   "instances": len(list(SEEDS)), "median_interstop_time": TAU,
                   "cells": CELLS,
                   "eligibility_seed_rule": "default_rng((instance_seed, 0xE119)); nested ceil masks",
                   "service_semantics": "charged once to serving vehicle; rendezvous service "
                                        "may overlap drone flight; depot service zero",
                   "truck_reference": "LKH length + n*t_service",
                   "base_source": "reused H2-v3 central-cell rows; 8 new cells",
                   "bootstrap_resamples": 20000, "bootstrap_unit": "instance",
                   "git_commit": commit,
                   "checkpoint": os.path.relpath(CHECKPOINT, os.path.join(HERE, "..")),
                   "runtime_s": time.time() - started},
        "reproduction_gate": gate,
        "summary": summarize(rows),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["summary"], indent=1))
    print(f"done in {time.time() - started:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
