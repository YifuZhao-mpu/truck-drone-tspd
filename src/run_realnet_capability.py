"""Two-district road-geometry stress test across five capability points.

This extends, but does not turn, the two cached OSM districts into deployment
validation.  All solver seeds and routes are retained.  The original (alpha=2,E=1)
arm is rerun as a bit-level reproduction gate.
"""
import argparse
import json
import os
import time
import numpy as np

from alns import alns
from realnet import build_instance, circuity, get_district
from revision_utils import bootstrap_ci, git_commit, run_checkpointed
from tsp_ref import reference_tsp_matrix


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "experiments", "REALNET", "results",
                   "realnet_capability_v3.json")
OLD = os.path.join(HERE, "..", "experiments", "REALNET", "results", "realnet.json")
CHECKPOINT = os.path.join(HERE, "..", "experiments", "REALNET", "results",
                          "realnet_capability_v3.jsonl")
DISTRICTS = ("manhattan", "paris")
SEEDS = range(30)
SOLVER_SEEDS = range(3)
CELLS = {
    "a1_E1": (1.0, 1.0),
    "a2_E0.5": (2.0, 0.5),
    "a2_E1": (2.0, 1.0),
    "a2_Einf": (2.0, np.inf),
    "a3_E1": (3.0, 1.0),
}
N = 50
ITERS = 16000
SPAN = 12


def job(args):
    district, instance_seed = args
    graph = get_district(district)
    inst, Dtr, Ddr, scale = build_instance(graph, N, instance_seed)
    ref, ref_tour = reference_tsp_matrix(Dtr)
    rows = []
    for cell, (alpha, endurance) in CELLS.items():
        reps = []
        for solver_seed in SOLVER_SEEDS:
            started = time.perf_counter()
            result = alns(inst, Dtr, alpha, endurance=endurance, max_span=SPAN,
                          iters=ITERS, sortie_aware=True, seed=solver_seed,
                          m=1, Ddr=Ddr)
            reps.append({
                "solver_seed": solver_seed,
                "makespan": float(result["makespan"]),
                "saving_pct": float((ref - result["makespan"]) / ref * 100),
                "n_sorties": int(result["eval"]["n_sorties"]),
                "wall_s": float(time.perf_counter() - started),
                "order": [int(v) for v in result["order"]],
            })
        winner = min(reps, key=lambda r: (r["makespan"], r["solver_seed"]))
        rows.append({"district": district, "instance_seed": instance_seed,
                     "cell": cell, "alpha": alpha,
                     "endurance": "inf" if not np.isfinite(endurance) else endurance,
                     "truck_ref": float(ref), "scale_m": float(scale),
                     "circuity": circuity(Dtr, Ddr),
                     "reference_tour": [int(v) for v in ref_tour],
                     "winner_solver_seed": winner["solver_seed"],
                     "replicates": reps})
    return rows


def summarize(rows):
    arms = {}
    contrasts = {}
    for di, district in enumerate(DISTRICTS):
        by_cell = {}
        for ci, cell in enumerate(CELLS):
            group = sorted((r for r in rows if r["district"] == district
                            and r["cell"] == cell), key=lambda r: r["instance_seed"])
            all_seed = [float(np.mean([p["saving_pct"] for p in r["replicates"]]))
                        for r in group]
            best = [max(p["saving_pct"] for p in r["replicates"]) for r in group]
            by_cell[cell] = all_seed
            arms[f"{district}_{cell}"] = {
                "all_seed_mean_saving_pct": float(np.mean(all_seed)),
                "all_seed_ci95": bootstrap_ci(all_seed, 100 * di + ci),
                "best_of_3_mean_saving_pct": float(np.mean(best)),
                "selection_gain_points": float(np.mean(best) - np.mean(all_seed)),
                "circuity_mean": float(np.mean([r["circuity"] for r in group])),
            }
        base = np.asarray(by_cell["a2_E1"])
        for ci, cell in enumerate(CELLS):
            if cell == "a2_E1":
                continue
            paired = np.asarray(by_cell[cell]) - base
            contrasts[f"{district}_{cell}_minus_a2_E1"] = {
                "mean_paired_saving_difference_points": float(np.mean(paired)),
                "ci95": bootstrap_ci(paired, 5000 + 100 * di + ci),
            }
    return arms, contrasts


def row_key(group):
    first = group[0]
    return [first["district"], first["instance_seed"]]


def reproduction_gate(rows):
    old = json.load(open(OLD))
    expected = {(r["district"], r["seed"]): r for r in old["raw"]
                if r["m"] == 1}
    checks = 0
    for row in rows:
        if row["cell"] != "a2_E1":
            continue
        key = (row["district"], row["instance_seed"])
        archived = expected[key]
        expected_reps = {p["seed"]: p["makespan"] for p in archived["replicates"]}
        for rep in row["replicates"]:
            checks += 1
            if rep["makespan"] != expected_reps[rep["solver_seed"]]:
                raise RuntimeError(f"real-network replicate gate failed at {key}/"
                                   f"{rep['solver_seed']}")
        winner = min(row["replicates"], key=lambda p: (p["makespan"], p["solver_seed"]))
        values = {"makespan": winner["makespan"], "truck_ref": row["truck_ref"],
                  "saving_pct": winner["saving_pct"], "circuity": row["circuity"],
                  "scale_m": row["scale_m"]}
        for field, value in values.items():
            checks += 1
            if value != archived[field]:
                raise RuntimeError(f"real-network field gate failed at {key}/{field}")
        route_key = f"{row['district']}-s{row['instance_seed']}-m1"
        checks += 1
        if winner["order"] != old["routes"][route_key]:
            raise RuntimeError(f"real-network route gate failed at {key}")
    nbase = sum(r["cell"] == "a2_E1" for r in rows)
    if nbase != 60:
        raise RuntimeError(f"real-network gate expected 60 base rows, got {nbase}")
    return {"status": "PASS", "instance_rows": nbase,
            "archived_fields_checked": checks, "comparison": "exact Python equality"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=36)
    args = parser.parse_args()
    started = time.time()
    for district in DISTRICTS:
        get_district(district)
    jobs = [(district, seed) for district in DISTRICTS for seed in SEEDS]
    print(f"{len(jobs)} district-instance jobs x {len(CELLS)} cells x 3 replicates")
    commit = git_commit(os.path.join(HERE, ".."))
    metadata = {"driver": "run_realnet_capability.py", "git_commit": commit,
                "districts": list(DISTRICTS), "cells": CELLS,
                "solver_seeds": list(SOLVER_SEEDS), "iters": ITERS, "span": SPAN}
    nested = run_checkpointed(jobs, job, lambda x: list(x), row_key, CHECKPOINT,
                              metadata, processes=args.workers)
    rows = [row for group in nested for row in group]
    gate = reproduction_gate(rows)
    arms, contrasts = summarize(rows)
    out = {
        "config": {"districts": DISTRICTS, "n": N, "cells": CELLS,
                   "instances": len(list(SEEDS)), "solver_seeds": list(SOLVER_SEEDS),
                   "iters": ITERS, "span": SPAN,
                   "scope": "two-district road-geometry stress test; not deployment validation",
                   "network_model": "cached undirected OSM shortest paths; one-way and congestion omitted",
                   "bootstrap_resamples": 20000, "bootstrap_unit": "district draw",
                   "git_commit": commit,
                   "checkpoint": os.path.relpath(CHECKPOINT, os.path.join(HERE, "..")),
                   "runtime_s": time.time() - started},
        "reproduction_gate": gate,
        "summary": arms,
        "paired_capability_contrasts": contrasts,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["summary"], indent=1))
    print(f"done in {time.time() - started:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
