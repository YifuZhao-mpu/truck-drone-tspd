"""Fixed-customer-density control for the endurance-by-size interpretation.

The n=20 unit square is the anchor.  Coordinates (equivalently, both distance
matrices) are scaled by sqrt(n/20), so area grows as n/20 while customer density and
physical endurance remain fixed.  Every solver replicate and route is retained.
"""
import argparse
import json
import math
import os
import time
import numpy as np

from alns import alns
from problem import dist_matrix, gen_instance
from revision_utils import bootstrap_ci, git_commit, run_checkpointed
from tsp_ref import _load_cache, key_of


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "experiments", "P0-robustness", "results",
                   "density_control_v3.json")
H2 = os.path.join(HERE, "..", "experiments", "H2-design-space", "results",
                  "h2_v3.json")
P0 = os.path.join(HERE, "..", "experiments", "P0-robustness", "results",
                  "p0_v3.json")
CHECKPOINT = os.path.join(HERE, "..", "experiments", "P0-robustness", "results",
                          "density_control_v3.jsonl")
NS = (20, 50, 100)
SEEDS = range(30)
SOLVER_SEEDS = range(3)
ENDURANCE = (0.5, 1.0, np.inf)
ENDURANCE_KEYS = ("0.5", "1.0", "inf")
FLEETS = (1, 2, 3)
ITERS = {20: 9000, 50: 16000, 100: 16000}
SPAN = 12
ALPHA = 2.0
REF = _load_cache()


def job(args):
    n, instance_seed, endurance_index, fleet = args
    scale = math.sqrt(n / 20.0)
    inst = gen_instance(n, seed=instance_seed)
    D = dist_matrix(inst["coords"]) * scale
    ref0 = REF[key_of("uniform", n, instance_seed, "center", "euclidean")]["length"]
    ref = ref0 * scale
    reps = []
    for solver_seed in SOLVER_SEEDS:
        started = time.perf_counter()
        result = alns(inst, D, ALPHA, endurance=ENDURANCE[endurance_index],
                      max_span=SPAN, iters=ITERS[n], sortie_aware=True,
                      seed=solver_seed, m=fleet)
        reps.append({
            "solver_seed": solver_seed,
            "makespan": float(result["makespan"]),
            "saving_pct": float((ref - result["makespan"]) / ref * 100),
            "n_sorties": int(result["eval"]["n_sorties"]),
            "wall_s": float(time.perf_counter() - started),
            "order": [int(v) for v in result["order"]],
        })
    winner = min(reps, key=lambda r: (r["makespan"], r["solver_seed"]))
    return {"n": n, "instance_seed": instance_seed, "endurance_index": endurance_index,
            "endurance": ENDURANCE_KEYS[endurance_index], "fleet": fleet,
            "distance_scale": scale, "truck_ref": float(ref),
            "winner_solver_seed": winner["solver_seed"], "replicates": reps}


def row_key(row):
    return [row["n"], row["instance_seed"], row["endurance_index"], row["fleet"]]


def h2_anchor_rows(h2):
    """Map the n=20 H2-v3 central cell into the fixed-density artifact."""
    old_to_new_ei = {0: 0, 1: 1, 3: 2}
    rows = []
    for source in h2["rows"]:
        if source["n"] != 20 or source["ai"] != 2 or source["ei"] not in old_to_new_ei:
            continue
        ei = old_to_new_ei[source["ei"]]
        rows.append({"n": 20, "instance_seed": source["instance_seed"],
                     "endurance_index": ei, "endurance": ENDURANCE_KEYS[ei],
                     "fleet": source["fleet"], "distance_scale": 1.0,
                     "truck_ref": source["truck_ref"],
                     "winner_solver_seed": source["winner_solver_seed"],
                     "replicates": source["replicates"],
                     "provenance": "reused H2-v3 n=20 all-seed anchor"})
    if len(rows) != 270:
        raise RuntimeError(f"expected 270 H2-v3 anchor rows, got {len(rows)}")
    return rows


def fixed_area_penalties(h2, p0):
    """Return paired fixed-area endurance penalties from archived all-seed reruns."""
    out = {}
    for n in (20, 50):
        for fleet in FLEETS:
            selected = [r for r in h2["rows"] if r["n"] == n and r["ai"] == 2
                        and r["fleet"] == fleet and r["ei"] in (0, 1, 3)]
            values = {(r["instance_seed"], r["ei"]): np.mean(
                [p["saving_pct"] for p in r["replicates"]]) for r in selected}
            for old_ei, ek in ((0, "0.5"), (1, "1.0")):
                out[(n, fleet, ek)] = np.asarray(
                    [values[(seed, 3)] - values[(seed, old_ei)] for seed in SEEDS])
    selected = [r for r in p0["rows"] if r["type"] == "surface" and r["ai"] == 2
                and r["ei"] in (0, 1, 3)]
    values = {(r["instance_seed"], r["ei"]): np.mean(
        [p["saving_pct"] for p in r["replicates"]]) for r in selected}
    for old_ei, ek in ((0, "0.5"), (1, "1.0")):
        out[(100, 1, ek)] = np.asarray(
            [values[(seed, 3)] - values[(seed, old_ei)] for seed in SEEDS])
    return out


def summarize(rows, h2, p0):
    cells = {}
    per_instance = {}
    for n in NS:
        for ei, ek in enumerate(ENDURANCE_KEYS):
            for fleet in FLEETS:
                group = sorted((r for r in rows if r["n"] == n
                                and r["endurance_index"] == ei and r["fleet"] == fleet),
                               key=lambda r: r["instance_seed"])
                means = [float(np.mean([p["saving_pct"] for p in r["replicates"]]))
                         for r in group]
                best = [max(p["saving_pct"] for p in r["replicates"]) for r in group]
                key = f"n{n}_E{ek}_m{fleet}"
                cells[key] = {
                    "all_seed_mean_saving_pct": float(np.mean(means)),
                    "all_seed_ci95": bootstrap_ci(means, n * 100 + ei * 10 + fleet),
                    "best_of_3_mean_saving_pct": float(np.mean(best)),
                    "selection_gain_points": float(np.mean(best) - np.mean(means)),
                }
                per_instance[key] = means
    contrasts = {}
    penalties = {}
    for n in NS:
        for fleet in FLEETS:
            inf = np.asarray(per_instance[f"n{n}_Einf_m{fleet}"])
            for ek in ("0.5", "1.0"):
                penalty = inf - np.asarray(per_instance[f"n{n}_E{ek}_m{fleet}"])
                penalties[(n, fleet, ek)] = penalty
                contrasts[f"n{n}_E{ek}_m{fleet}"] = {
                    "mean_endurance_penalty_points": float(np.mean(penalty)),
                    "ci95": bootstrap_ci(penalty, 5000 + n + fleet),
                }
    fades = {"fixed_density_n20_minus_n100": {},
             "fixed_area_n20_minus_n50": {},
             "fixed_area_n20_minus_n100_m1": {}}
    for fleet in FLEETS:
        for ek in ("0.5", "1.0"):
            fade = penalties[(20, fleet, ek)] - penalties[(100, fleet, ek)]
            key = f"E{ek}_m{fleet}"
            fades["fixed_density_n20_minus_n100"][key] = {
                "mean_fade_points": float(np.mean(fade)),
                "ci95": bootstrap_ci(fade, 7000 + fleet * 10 + int(float(ek) * 2)),
            }
    fixed = fixed_area_penalties(h2, p0)
    for fleet in FLEETS:
        for ek in ("0.5", "1.0"):
            fade = fixed[(20, fleet, ek)] - fixed[(50, fleet, ek)]
            key = f"E{ek}_m{fleet}"
            fades["fixed_area_n20_minus_n50"][key] = {
                "mean_fade_points": float(np.mean(fade)),
                "ci95": bootstrap_ci(fade, 8000 + fleet * 10 + int(float(ek) * 2)),
            }
    for ek in ("0.5", "1.0"):
        fade = fixed[(20, 1, ek)] - fixed[(100, 1, ek)]
        fades["fixed_area_n20_minus_n100_m1"][f"E{ek}_m1"] = {
            "mean_fade_points": float(np.mean(fade)),
            "ci95": bootstrap_ci(fade, 9000 + int(float(ek) * 2)),
        }
    return cells, contrasts, fades


def anchor_gate(canary, anchors):
    expected = next(r for r in anchors if row_key(r) == row_key(canary))
    checks = 0
    if canary["truck_ref"] != expected["truck_ref"]:
        raise RuntimeError("density s=1 canary truck reference mismatch")
    checks += 1
    expected_reps = {p["solver_seed"]: p for p in expected["replicates"]}
    for observed in canary["replicates"]:
        archived = expected_reps[observed["solver_seed"]]
        for field in ("makespan", "saving_pct", "n_sorties", "order"):
            checks += 1
            if observed[field] != archived[field]:
                raise RuntimeError(f"density s=1 canary failed at seed "
                                   f"{observed['solver_seed']}/{field}")
    return {"status": "PASS", "job": row_key(canary),
            "archived_fields_checked": checks, "comparison": "exact Python equality"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=36)
    args = parser.parse_args()
    started = time.time()
    h2 = json.load(open(H2))
    p0 = json.load(open(P0))
    anchors = h2_anchor_rows(h2)
    canary = job((20, 0, 0, 1))
    gate = anchor_gate(canary, anchors)
    jobs = [(n, seed, ei, fleet) for n in (50, 100) for seed in SEEDS
            for ei in range(len(ENDURANCE)) for fleet in FLEETS]
    print(f"{len(jobs)} new fixed-density jobs, 3 retained replicates each")
    commit = git_commit(os.path.join(HERE, ".."))
    metadata = {"driver": "run_density_control.py", "git_commit": commit,
                "sizes": [50, 100], "solver_seeds": list(SOLVER_SEEDS),
                "endurance": list(ENDURANCE_KEYS), "fleets": list(FLEETS),
                "iters": ITERS, "span": SPAN}
    new_rows = run_checkpointed(jobs, job, lambda x: list(x), row_key, CHECKPOINT,
                                metadata, processes=args.workers)
    rows = anchors + new_rows
    cells, contrasts, fades = summarize(rows, h2, p0)
    out = {
        "config": {"sizes": NS, "instances": len(list(SEEDS)), "alpha": ALPHA,
                   "endurance": ENDURANCE_KEYS, "fleets": FLEETS,
                   "solver_seeds": list(SOLVER_SEEDS), "iters": ITERS, "span": SPAN,
                   "distance_scale": "sqrt(n/20); area=n/20; density=20 per unit^2",
                   "truck_reference": "unit-square LKH route length multiplied by scale",
                   "n20_source": "H2-v3 central-cell all-seed rows (not recomputed)",
                   "bootstrap_resamples": 20000, "bootstrap_unit": "instance",
                   "git_commit": commit,
                   "checkpoint": os.path.relpath(CHECKPOINT, os.path.join(HERE, "..")),
                   "runtime_s": time.time() - started},
        "anchor_reproduction_gate": gate,
        "cells": cells,
        "endurance_penalties": contrasts,
        "endurance_fade_contrasts": fades,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(contrasts, indent=1))
    print(f"done in {time.time() - started:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
