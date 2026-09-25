"""P0 v3: reproduce every P0-v2 arm while retaining all seeds and routes."""
import argparse
import hashlib
import json
import os
import time
import numpy as np
from scipy import stats

from alns import alns
from problem import dist_matrix, gen_clustered, gen_instance
from revision_utils import bootstrap_ci, git_commit, run_checkpointed
from tsp_ref import _load_cache, key_of


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "experiments", "P0-robustness", "results", "p0_v3.json")
OLD = os.path.join(HERE, "..", "experiments", "P0-robustness", "results", "p0_v2.json")
CHECKPOINT = os.path.join(HERE, "..", "experiments", "P0-robustness", "results",
                          "p0_v3.jsonl")
ALPHAS = [1.0, 1.5, 2.0, 2.5, 3.0]
ENDURANCE = [0.5, 1.0, 2.0, np.inf]
ENDURANCE_KEYS = ["0.5", "1.0", "2.0", "inf"]
INSTANCE_SEEDS = range(30)
SOLVER_SEEDS = range(3)
ITERS = {20: 9000, 50: 16000, 100: 16000}
LONG_ITERS = {50: 40000, 100: 40000}
SPAN = 12
REF = _load_cache()


def source_digest():
    h = hashlib.sha256()
    for name in ("problem.py", "alns.py", "run_p0_v3.py"):
        with open(os.path.join(HERE, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def make_inst(kind, n, seed, depot):
    return (gen_instance(n, seed=seed, depot=depot) if kind == "uniform"
            else gen_clustered(n, seed=seed, depot=depot))


def matrices(inst, metric):
    return (dist_matrix(inst["coords"], metric=metric),
            dist_matrix(inst["coords"], metric="euclidean"))


def solve_reps(inst, Dtr, Ddr, alpha, endurance, iters, sortie=True):
    reps = []
    for solver_seed in SOLVER_SEEDS:
        started = time.perf_counter()
        result = alns(inst, Dtr, alpha, endurance=endurance, max_span=SPAN,
                      iters=iters, sortie_aware=sortie, seed=solver_seed, Ddr=Ddr)
        reps.append({"solver_seed": solver_seed, "makespan": float(result["makespan"]),
                     "wall_s": float(time.perf_counter() - started),
                     "order": [int(v) for v in result["order"]]})
    return reps


def job_cell(args):
    kind, depot, metric, n, seed = args
    inst = make_inst(kind, n, seed, depot)
    Dtr, Ddr = matrices(inst, metric)
    ref = REF[key_of(kind, n, seed, depot, metric)]["length"]
    reps = solve_reps(inst, Dtr, Ddr, 2.0, 1.0, ITERS[n])
    for p in reps:
        p["saving_pct"] = (ref - p["makespan"]) / ref * 100
    return {"type": "cell", "kind": kind, "depot": depot, "metric": metric,
            "n": n, "instance_seed": seed, "truck_ref": float(ref), "replicates": reps}


def job_surface(args):
    seed, ai, ei = args
    inst = gen_instance(100, seed=seed)
    D = dist_matrix(inst["coords"])
    ref = REF[key_of("uniform", 100, seed, "center", "euclidean")]["length"]
    reps = solve_reps(inst, D, D, ALPHAS[ai], ENDURANCE[ei], ITERS[100])
    for p in reps:
        p["saving_pct"] = (ref - p["makespan"]) / ref * 100
    return {"type": "surface", "n": 100, "instance_seed": seed, "ai": ai, "ei": ei,
            "truck_ref": float(ref), "replicates": reps}


def job_ablation(args):
    n, seed = args
    inst = gen_instance(n, seed=seed)
    D = dist_matrix(inst["coords"])
    return {"type": "ablation", "n": n, "instance_seed": seed,
            "sortie_replicates": solve_reps(inst, D, D, 2.0, np.inf, ITERS[n], True),
            "generic_replicates": solve_reps(inst, D, D, 2.0, np.inf, ITERS[n], False)}


def job_budget(args):
    n, seed = args
    inst = gen_instance(n, seed=seed)
    D = dist_matrix(inst["coords"])
    return {"type": "budget", "n": n, "instance_seed": seed,
            "standard_replicates": solve_reps(inst, D, D, 2.0, np.inf, ITERS[n]),
            "long_replicates": solve_reps(inst, D, D, 2.0, np.inf, LONG_ITERS[n])}


def best(reps):
    return min(reps, key=lambda r: (r["makespan"], r["solver_seed"]))["makespan"]


def run_job(spec):
    task, payload = spec[0], spec[1:]
    return {"cell": job_cell, "surface": job_surface, "ablation": job_ablation,
            "budget": job_budget}[task](payload)


def spec_key(spec):
    return list(spec)


def row_key(row):
    if row["type"] == "cell":
        return ["cell", row["kind"], row["depot"], row["metric"], row["n"],
                row["instance_seed"]]
    if row["type"] == "surface":
        return ["surface", row["instance_seed"], row["ai"], row["ei"]]
    return [row["type"], row["n"], row["instance_seed"]]


def reproduction_gate(rows, expected_jobs=None):
    old = json.load(open(OLD))
    checks = 0
    old_cells = {(r["kind"], r["depot"], r["metric"], r["n"], r["seed"]): r
                 for r in old["raw_cells"]}
    old_surface = {(r["seed"], r["ai"], r["ei"]): r
                   for r in old["raw_surface"]}
    for r in rows:
        if r["type"] == "cell":
            key = (r["kind"], r["depot"], r["metric"], r["n"], r["instance_seed"])
            winner = min(r["replicates"], key=lambda p: (p["makespan"], p["solver_seed"]))
            expected = old_cells[key]
            observed = {"makespan": winner["makespan"], "truck_ref": r["truck_ref"],
                        "saving_pct": winner["saving_pct"]}
            for field, old_field in (("makespan", "makespan"),
                                     ("truck_ref", "truck_ref"),
                                     ("saving_pct", "saving_pct")):
                checks += 1
                if observed[field] != expected[old_field]:
                    raise RuntimeError(f"P0 cell gate failed at {key}/{field}")
        elif r["type"] == "surface":
            key = (r["instance_seed"], r["ai"], r["ei"])
            winner = min(r["replicates"], key=lambda p: (p["makespan"], p["solver_seed"]))
            expected = old_surface[key]
            for field, observed in (("makespan", winner["makespan"]),
                                    ("truck_ref", r["truck_ref"]),
                                    ("saving_pct", winner["saving_pct"])):
                checks += 1
                if observed != expected[field]:
                    raise RuntimeError(f"P0 surface gate failed at {key}/{field}")
        elif r["type"] == "ablation":
            oldrow = next(x for x in old["ablation"][str(r["n"])]["raw"]
                          if x["seed"] == r["instance_seed"])
            for field, observed in (("sortie", best(r["sortie_replicates"])),
                                    ("generic", best(r["generic_replicates"]))):
                checks += 1
                if observed != oldrow[field]:
                    raise RuntimeError(f"P0 ablation gate failed at "
                                       f"{r['n']}/{r['instance_seed']}/{field}")
        elif r["type"] == "budget":
            oldrow = next(x for x in old["budget_sensitivity"][str(r["n"])]["raw"]
                          if x["seed"] == r["instance_seed"])
            for field, observed in (("std", best(r["standard_replicates"])),
                                    ("long", best(r["long_replicates"]))):
                checks += 1
                if observed != oldrow[field]:
                    raise RuntimeError(f"P0 budget gate failed at "
                                       f"{r['n']}/{r['instance_seed']}/{field}")
    if expected_jobs is not None and len(rows) != expected_jobs:
        raise RuntimeError(f"P0 gate job count {len(rows)} != {expected_jobs}")
    return {"status": "PASS", "jobs": len(rows),
            "archived_fields_checked": checks, "comparison": "exact Python equality",
            "route_note": "P0-v2 did not archive routes; v3 routes retained prospectively"}


def summarize(rows):
    cells = {}
    for ci, (kind, depot, metric, n) in enumerate(
            (k, d, m, n) for k in ("uniform", "clustered")
            for d in ("center", "corner") for m in ("euclidean", "manhattan")
            for n in (50, 100)):
        group = [r for r in rows if r["type"] == "cell" and r["kind"] == kind
                 and r["depot"] == depot and r["metric"] == metric and r["n"] == n]
        means = [float(np.mean([p["saving_pct"] for p in r["replicates"]])) for r in group]
        wins = [max(p["saving_pct"] for p in r["replicates"]) for r in group]
        advantage = np.asarray(wins) - np.asarray(means)
        cells[f"{kind}_{depot}_{metric}_n{n}"] = {
            "all_seed_mean_saving": float(np.mean(means)),
            "all_seed_ci95": bootstrap_ci(means, 100 + ci),
            "best_of_3_mean_saving": float(np.mean(wins)),
            "selection_gain_points": float(np.mean(advantage)),
            "selection_gain_ci95": bootstrap_ci(advantage, 700 + ci),
            "selection_gain_distribution": {
                "median": float(np.median(advantage)),
                "iqr": [float(np.percentile(advantage, 25)),
                        float(np.percentile(advantage, 75))],
                "min": float(np.min(advantage)), "max": float(np.max(advantage))}}
    surface = {}
    for ai, alpha in enumerate(ALPHAS):
        for ei, ek in enumerate(ENDURANCE_KEYS):
            group = [r for r in rows if r["type"] == "surface" and r["ai"] == ai
                     and r["ei"] == ei]
            means = [float(np.mean([p["saving_pct"] for p in r["replicates"]])) for r in group]
            wins = [max(p["saving_pct"] for p in r["replicates"]) for r in group]
            advantage = np.asarray(wins) - np.asarray(means)
            surface[f"a{alpha}_E{ek}"] = {
                "all_seed_mean_saving": float(np.mean(means)),
                "all_seed_ci95": bootstrap_ci(means, 500 + ai * 10 + ei),
                "best_of_3_mean_saving": float(np.mean(wins)),
                "selection_gain_points": float(np.mean(advantage)),
                "selection_gain_ci95": bootstrap_ci(
                    advantage, 900 + ai * 10 + ei),
                "selection_gain_distribution": {
                    "median": float(np.median(advantage)),
                    "iqr": [float(np.percentile(advantage, 25)),
                            float(np.percentile(advantage, 75))],
                    "min": float(np.min(advantage)), "max": float(np.max(advantage))}}
    ablation = {}
    for n in (20, 50, 100):
        group = [r for r in rows if r["type"] == "ablation" and r["n"] == n]
        s = np.array([np.mean([p["makespan"] for p in r["sortie_replicates"]]) for r in group])
        g = np.array([np.mean([p["makespan"] for p in r["generic_replicates"]]) for r in group])
        d = s - g
        relative = d / g * 100
        nz = d[np.abs(d) > 1e-12]
        ablation[str(n)] = {"all_seed_mean_relative_diff_pct": float(np.mean(relative)),
                            "all_seed_mean_relative_diff_ci95": bootstrap_ci(
                                relative, 2000 + n),
                            "sortie_wins": int(np.sum(d < -1e-9)),
                            "generic_wins": int(np.sum(d > 1e-9)),
                            "ties": int(np.sum(np.abs(d) <= 1e-9)),
                            "wilcoxon_p_one_sided": (float(stats.wilcoxon(nz, alternative="less").pvalue)
                                                       if len(nz) >= 5 else None)}
    budget = {}
    for n in (50, 100):
        group = [r for r in rows if r["type"] == "budget" and r["n"] == n]
        all_gaps = [(np.mean([p["makespan"] for p in r["standard_replicates"]])
                     - np.mean([p["makespan"] for p in r["long_replicates"]]))
                    / np.mean([p["makespan"] for p in r["long_replicates"]]) * 100
                    for r in group]
        best_gaps = [(best(r["standard_replicates"]) - best(r["long_replicates"]))
                     / best(r["long_replicates"]) * 100 for r in group]
        budget[str(n)] = {"all_seed_mean_gap_pct": float(np.mean(all_gaps)),
                          "all_seed_mean_gap_ci95": bootstrap_ci(all_gaps, 3000 + n),
                          "best_of_3_mean_gap_pct": float(np.mean(best_gaps)),
                          "note": "search-stability diagnostic, not optimality bound"}
    return cells, surface, ablation, budget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--workers", type=int, default=36)
    args = parser.parse_args()
    started = time.time()
    jobs_cell = [(k, d, m, n, s) for k in ("uniform", "clustered")
                 for d in ("center", "corner") for m in ("euclidean", "manhattan")
                 for n in (50, 100) for s in INSTANCE_SEEDS]
    jobs_surface = [(s, ai, ei) for s in INSTANCE_SEEDS for ai in range(len(ALPHAS))
                    for ei in range(len(ENDURANCE))]
    jobs_ablation = [(n, s) for n in (20, 50, 100) for s in INSTANCE_SEEDS]
    jobs_budget = [(n, s) for n in (50, 100) for s in range(15)]
    specs = ([('cell', *x) for x in jobs_cell]
             + [('surface', *x) for x in jobs_surface]
             + [('ablation', *x) for x in jobs_ablation]
             + [('budget', *x) for x in jobs_budget])
    if args.canary:
        specs = [specs[i] for i in (0, 239, 479, 480, 1079, 1080, 1169, 1170)]
    print(f"P0 v3: {len(specs)} jobs (canary={args.canary})")
    commit = git_commit(os.path.join(HERE, ".."))
    digest = source_digest()
    checkpoint = CHECKPOINT.replace(".jsonl", ".canary.jsonl") if args.canary else CHECKPOINT
    metadata = {"driver": "run_p0_v3.py", "git_commit": commit,
                "source_sha256": digest, "canary": args.canary,
                "solver_seeds": list(SOLVER_SEEDS), "iters": ITERS,
                "long_iters": LONG_ITERS, "span": SPAN}
    rows = run_checkpointed(specs, run_job, spec_key, row_key, checkpoint, metadata,
                            processes=args.workers)
    gate = reproduction_gate(rows, len(specs))
    if args.canary:
        print(json.dumps({"gate": gate, "checkpoint": checkpoint}, indent=2))
        return
    cells, surface, ablation, budget = summarize(rows)
    out = {"config": {"solver_seeds": list(SOLVER_SEEDS), "iters": ITERS,
                      "long_iters": LONG_ITERS, "span": SPAN,
                      "aggregation": "replicate mean within instance; bootstrap over instances",
                      "bootstrap_resamples": 20000, "bootstrap_unit": "instance",
                      "git_commit": commit, "source_sha256": digest,
                      "checkpoint": os.path.relpath(checkpoint, os.path.join(HERE, "..")),
                      "runtime_s": time.time() - started},
           "reproduction_gate": gate, "deconfound_cells": cells,
           "surface_n100": surface, "ablation": ablation,
           "budget_sensitivity": budget, "rows": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(json.dumps({"gate": gate, "ablation": ablation, "budget": budget}, indent=2))
    print(f"done in {time.time()-started:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
