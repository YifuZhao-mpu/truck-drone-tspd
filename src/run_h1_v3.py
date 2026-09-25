"""H1 all-seed rerun for the small-instance validation and synthetic baselines."""
import argparse
import json
import os
import time
import numpy as np

from alns import alns
from baselines import greedy_tspd, truck_only_tsp
from benchmark import load_instance, load_solution
from problem import dist_matrix, gen_instance
from revision_utils import bootstrap_ci, git_commit, run_checkpointed


HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "benchmarks", "external", "uniform")
OUT = os.path.join(HERE, "..", "experiments", "H1-alns-backbone", "results", "h1_v3.json")
OLD = os.path.join(HERE, "..", "experiments", "H1-alns-backbone", "results", "h1.json")
CHECKPOINT = os.path.join(HERE, "..", "experiments", "H1-alns-backbone", "results",
                          "h1_v3.jsonl")
ITERS = {10: 7000, 20: 9000, 50: 16000}
SOLVER_SEEDS = range(3)
ARCHIVE_ABS_TOL = 1e-12


def solve_reps(inst, D, alpha, endurance, span, iters, sortie):
    reps = []
    for solver_seed in SOLVER_SEEDS:
        started = time.perf_counter()
        result = alns(inst, D, alpha, endurance=endurance, max_span=span,
                      iters=iters, sortie_aware=sortie, seed=solver_seed)
        reps.append({"solver_seed": solver_seed, "makespan": float(result["makespan"]),
                     "wall_s": float(time.perf_counter() - started),
                     "order": [int(v) for v in result["order"]]})
    return reps


def job_benchmark(args):
    n, idx = args
    instance_path = f"{BASE}/uniform-{idx}-n{n}.txt"
    solution_path = f"{BASE}/solutions/uniform-{idx}-n{n}-DP.txt"
    if not (os.path.exists(instance_path) and os.path.exists(solution_path)):
        return None
    inst = load_instance(instance_path)
    opt = load_solution(solution_path)["total"]
    D = dist_matrix(inst["coords"])
    reps = solve_reps(inst, D, inst["alpha"], np.inf, 20, 9000, True)
    for p in reps:
        p["gap_pct"] = (p["makespan"] - opt) / opt * 100
    return {"type": "benchmark", "n": n, "index": idx, "optimum": float(opt),
            "replicates": reps}


def job_synthetic(args):
    n, instance_seed = args
    inst = gen_instance(n, seed=instance_seed)
    D = dist_matrix(inst["coords"])
    span = min(n + 1, 12)
    _, truck = truck_only_tsp(inst, D, restarts=4, seed=instance_seed)
    greedy, greedy_order, _ = greedy_tspd(inst, D, 2.0)
    return {"type": "synthetic", "n": n, "instance_seed": instance_seed,
            "truck": float(truck), "greedy": float(greedy),
            "greedy_order": [int(v) for v in greedy_order],
            "sortie_replicates": solve_reps(inst, D, 2.0, np.inf, span, ITERS[n], True),
            "generic_replicates": solve_reps(inst, D, 2.0, np.inf, span, ITERS[n], False)}


def best(reps):
    return min(reps, key=lambda p: (p["makespan"], p["solver_seed"]))["makespan"]


def run_job(spec):
    return job_benchmark(spec[1:]) if spec[0] == "benchmark" else job_synthetic(spec[1:])


def spec_key(spec):
    return list(spec)


def row_key(row):
    if row["type"] == "benchmark":
        return ["benchmark", row["n"], row["index"]]
    return ["synthetic", row["n"], row["instance_seed"]]


def reproduction_gate(bench, synth, expected_jobs=None):
    old = json.load(open(OLD))
    old_b = {(r["n"], r["idx"]): r for r in old["raw_bench"]}
    old_s = {(r["n"], r["seed"]): r for r in old["raw_synth"]}
    checks = 0
    max_abs_delta = 0.0

    def check_archived(value, expected, label):
        nonlocal checks, max_abs_delta
        checks += 1
        delta = abs(value - expected)
        max_abs_delta = max(max_abs_delta, delta)
        if delta > ARCHIVE_ABS_TOL:
            raise RuntimeError(
                f"H1 archive gate failed at {label}: {value} != {expected} "
                f"(absolute delta {delta} > {ARCHIVE_ABS_TOL})")

    for r in bench:
        expected = old_b[(r["n"], r["index"])]
        observed = best(r["replicates"])
        values = {"opt": r["optimum"], "alns": observed,
                  "gap": (observed - r["optimum"]) / r["optimum"] * 100}
        for field, value in values.items():
            check_archived(value, expected[field],
                           f"benchmark/{r['n']}/{r['index']}/{field}")
    for r in synth:
        expected = old_s[(r["n"], r["instance_seed"])]
        sortie, generic = best(r["sortie_replicates"]), best(r["generic_replicates"])
        values = {"truck": r["truck"], "greedy": r["greedy"],
                  "sortie": sortie, "generic": generic,
                  "save_sortie": (r["truck"] - sortie) / r["truck"] * 100,
                  "save_generic": (r["truck"] - generic) / r["truck"] * 100}
        for field, value in values.items():
            check_archived(value, expected[field],
                           f"synthetic/{r['n']}/{r['instance_seed']}/{field}")
    if expected_jobs is not None and len(bench) + len(synth) != expected_jobs:
        raise RuntimeError("H1 gate job count mismatch")
    return {"status": "PASS", "jobs": len(bench) + len(synth),
            "archived_fields_checked": checks,
            "comparison": f"absolute tolerance {ARCHIVE_ABS_TOL:g}",
            "max_abs_delta": max_abs_delta,
            "route_note": "H1-v1 did not archive ALNS routes; v3 routes retained prospectively"}


def summarize(bench, synth):
    b = {}
    for n in sorted({r["n"] for r in bench}):
        group = [r for r in bench if r["n"] == n]
        per_instance = [np.mean([p["gap_pct"] for p in r["replicates"]]) for r in group]
        advantage = [np.mean([p["gap_pct"] for p in r["replicates"]])
                     - min(p["gap_pct"] for p in r["replicates"]) for r in group]
        b[str(n)] = {"all_seed_mean_gap_pct": float(np.mean(per_instance)),
                     "all_seed_mean_gap_ci95": bootstrap_ci(per_instance, 1000 + n),
                     "best_of_3_mean_gap_pct": float(np.mean(
                         [min(p["gap_pct"] for p in r["replicates"]) for r in group])),
                     "best_of_3_selection_advantage_gap_points": float(np.mean(advantage)),
                     "selection_advantage_ci95": bootstrap_ci(advantage, 1500 + n),
                     "all_seed_percent_exact": float(np.mean(
                         [p["gap_pct"] < 1e-9 for r in group for p in r["replicates"]]) * 100)}
    s = {}
    for n in (10, 20, 50):
        group = [r for r in synth if r["n"] == n]
        sortie = np.array([np.mean([p["makespan"] for p in r["sortie_replicates"]])
                            for r in group])
        generic = np.array([np.mean([p["makespan"] for p in r["generic_replicates"]])
                            for r in group])
        greedy = np.array([r["greedy"] for r in group])
        truck = np.array([r["truck"] for r in group])
        savings = (truck - sortie) / truck * 100
        operator_effect = (generic - sortie) / sortie * 100
        selection = np.asarray([
            (np.mean([p["makespan"] for p in r["sortie_replicates"]])
             - min(p["makespan"] for p in r["sortie_replicates"]))
            / np.mean([p["makespan"] for p in r["sortie_replicates"]]) * 100
            for r in group])
        s[str(n)] = {"all_seed_sortie_mean": float(np.mean(sortie)),
                     "all_seed_generic_mean": float(np.mean(generic)),
                     "greedy_gap_vs_all_seed_sortie_pct": float(np.mean(
                         (greedy - sortie) / sortie * 100)),
                     "all_seed_mean_saving_pct": float(np.mean(savings)),
                     "all_seed_mean_saving_ci95": bootstrap_ci(savings, 2000 + n),
                     "sortie_selection_advantage_pct": float(np.mean(selection)),
                     "sortie_selection_advantage_ci95": bootstrap_ci(selection, 2500 + n),
                     "generic_minus_sortie_pct": float(np.mean(operator_effect)),
                     "generic_minus_sortie_ci95": bootstrap_ci(operator_effect, 3000 + n)}
    return b, s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--workers", type=int, default=36)
    args = parser.parse_args()
    started = time.time()
    bench_jobs = [(n, idx) for n in range(11, 18) for idx in range(1, 11)]
    synth_jobs = [(n, seed) for n in (10, 20, 50) for seed in range(30)]
    specs = ([('benchmark', *x) for x in bench_jobs]
             + [('synthetic', *x) for x in synth_jobs])
    if args.canary:
        specs = [specs[i] for i in (0, 9, 60, 69, 70, 99, 100, 159)]
    commit = git_commit(os.path.join(HERE, ".."))
    checkpoint = CHECKPOINT.replace(".jsonl", ".canary.jsonl") if args.canary else CHECKPOINT
    metadata = {"driver": "run_h1_v3.py", "git_commit": commit,
                "canary": args.canary, "solver_seeds": list(SOLVER_SEEDS),
                "iters": ITERS}
    rows = run_checkpointed(specs, run_job, spec_key, row_key, checkpoint, metadata,
                            processes=args.workers)
    bench = [r for r in rows if r is not None and r["type"] == "benchmark"]
    synth = [r for r in rows if r["type"] == "synthetic"]
    gate = reproduction_gate(bench, synth, len(specs))
    if args.canary:
        print(json.dumps({"gate": gate, "checkpoint": checkpoint}, indent=2))
        return
    bench_summary, synth_summary = summarize(bench, synth)
    out = {"config": {"solver_seeds": list(SOLVER_SEEDS), "iters": ITERS,
                      "git_commit": commit,
                      "checkpoint": os.path.relpath(checkpoint, os.path.join(HERE, "..")),
                      "runtime_s": time.time() - started},
           "reproduction_gate": gate, "benchmark": bench_summary,
           "synthetic": synth_summary, "benchmark_rows": bench,
           "synthetic_rows": synth}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(json.dumps({"gate": gate, "benchmark": bench_summary,
                      "synthetic": synth_summary}, indent=2))
    print(f"done in {time.time()-started:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
