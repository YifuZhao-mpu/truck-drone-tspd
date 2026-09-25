"""Stronger decoder-baseline curve requested in major revision (Reviewer 5.9).

Crosses greedy decoding and restricted dynamic-programming spans 4, 6 and 12 with
sortie-aware/generic operators. Search and final decoding use the same span.  All
replicates and routes are retained; the design is equal-iteration and reports runtime.
"""
import argparse
import json
import os
import time
import numpy as np
from scipy import stats

from alns import alns
from problem import dist_matrix, gen_instance
from revision_utils import bootstrap_ci, git_commit, run_checkpointed
from tsp_ref import _load_cache, key_of


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "experiments", "P0-robustness", "results",
                   "decoder_strength_v3.json")
OLD = os.path.join(HERE, "..", "experiments", "P0-robustness", "results",
                   "ops_factorial_v2.json")
CHECKPOINT = os.path.join(HERE, "..", "experiments", "P0-robustness", "results",
                          "decoder_strength_v3.jsonl")
N = 50
SEEDS = range(30)
SOLVER_SEEDS = range(3)
DECODERS = ("greedy", "dp4", "dp6", "dp12")
OPERATORS = ("sortie", "generic")
ITERS = 16000
ALPHA = 2.0
REF = _load_cache()


def decoder_span(name):
    return 12 if name == "greedy" else int(name[2:])


def job(args):
    instance_seed, decoder, operator = args
    inst = gen_instance(N, seed=instance_seed)
    D = dist_matrix(inst["coords"])
    ref = REF[key_of("uniform", N, instance_seed, "center", "euclidean")]["length"]
    reps = []
    for solver_seed in SOLVER_SEEDS:
        started = time.perf_counter()
        span = decoder_span(decoder)
        result = alns(inst, D, ALPHA, endurance=np.inf, max_span=span,
                      final_span=span, iters=ITERS,
                      sortie_aware=(operator == "sortie"), seed=solver_seed,
                      decoder="greedy" if decoder == "greedy" else "dp")
        reps.append({
            "solver_seed": solver_seed,
            "makespan": float(result["makespan"]),
            "saving_pct": float((ref - result["makespan"]) / ref * 100),
            "wall_s": float(time.perf_counter() - started),
            "order": [int(v) for v in result["order"]],
        })
    winner = min(reps, key=lambda r: (r["makespan"], r["solver_seed"]))
    return {"instance_seed": instance_seed, "decoder": decoder,
            "operator": operator, "truck_ref": float(ref),
            "winner_solver_seed": winner["solver_seed"], "replicates": reps}


def row_key(row):
    return [row["instance_seed"], row["decoder"], row["operator"]]


def greedy_reproduction_gate(rows, expected_instances):
    old = json.load(open(OLD))
    expected = {(r["seed"], "sortie" if r["sortie"] else "generic"): r["makespan"]
                for r in old["raw"] if r["decoder"] == "greedy"}
    checks = 0
    for row in rows:
        if row["decoder"] != "greedy":
            continue
        observed = min(p["makespan"] for p in row["replicates"])
        key = (row["instance_seed"], row["operator"])
        checks += 1
        if observed != expected[key]:
            raise RuntimeError(f"shared-loop greedy reproduction failed at {key}: "
                               f"{observed} != {expected[key]}")
    if checks != 2 * expected_instances:
        raise RuntimeError(f"expected {2 * expected_instances} greedy checks, got {checks}")
    return {"status": "PASS", "archived_greedy_arms_checked": checks,
            "comparison": "best-of-three exact Python equality",
            "dp12_note": "not compared: v2 silently used unrestricted final re-decoding"}


def summarize(rows):
    arms = {}
    instance_values = {}
    for decoder in DECODERS:
        for operator in OPERATORS:
            group = sorted((r for r in rows if r["decoder"] == decoder
                            and r["operator"] == operator),
                           key=lambda r: r["instance_seed"])
            all_ms = [float(np.mean([p["makespan"] for p in r["replicates"]]))
                      for r in group]
            all_save = [float(np.mean([p["saving_pct"] for p in r["replicates"]]))
                        for r in group]
            best_ms = [min(p["makespan"] for p in r["replicates"]) for r in group]
            key = f"{decoder}_{operator}"
            instance_values[key] = all_ms
            arms[key] = {
                "all_seed_mean_makespan": float(np.mean(all_ms)),
                "all_seed_mean_saving_pct": float(np.mean(all_save)),
                "best_of_3_mean_makespan": float(np.mean(best_ms)),
                "selection_gain_makespan_pct": float(
                    (np.mean(all_ms) - np.mean(best_ms)) / np.mean(all_ms) * 100),
                "mean_wall_s_per_replicate": float(np.mean(
                    [p["wall_s"] for r in group for p in r["replicates"]])),
                "all_seed_mean_saving_ci95": bootstrap_ci(
                    all_save, 10000 + DECODERS.index(decoder) * 10
                    + OPERATORS.index(operator)),
            }

    effects = {}
    ref = np.asarray(instance_values["dp12_sortie"])
    for decoder in DECODERS:
        vals = (np.asarray(instance_values[f"{decoder}_sortie"])
                + np.asarray(instance_values[f"{decoder}_generic"])) / 2
        dp12 = (np.asarray(instance_values["dp12_sortie"])
                + np.asarray(instance_values["dp12_generic"])) / 2
        effects[f"{decoder}_vs_dp12"] = {
            "mean_makespan_penalty_pct": float(np.mean((vals - dp12) / dp12 * 100)),
            "ci95": bootstrap_ci((vals - dp12) / dp12 * 100,
                                  20000 + DECODERS.index(decoder)),
        }
    for decoder in DECODERS:
        s = np.asarray(instance_values[f"{decoder}_sortie"])
        g = np.asarray(instance_values[f"{decoder}_generic"])
        contrast = (g - s) / s * 100
        effects[f"operator_effect_at_{decoder}"] = {
            "generic_minus_sortie_pct": float(np.mean(contrast)),
            "ci95": bootstrap_ci(contrast, 30000 + DECODERS.index(decoder)),
            "wilcoxon_p_two_sided": (float(stats.wilcoxon(contrast).pvalue)
                                      if np.max(np.abs(contrast)) > 1e-12 else None),
        }
    return arms, effects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--workers", type=int, default=36)
    args = parser.parse_args()
    started = time.time()
    jobs = [(seed, decoder, operator) for seed in SEEDS for decoder in DECODERS
            for operator in OPERATORS]
    if args.canary:
        jobs = [(0, "greedy", "sortie"), (0, "greedy", "generic")]
    print(f"{len(jobs)} decoder-strength jobs, 3 retained replicates each")
    commit = git_commit(os.path.join(HERE, ".."))
    checkpoint = CHECKPOINT.replace(".jsonl", ".canary.jsonl") if args.canary else CHECKPOINT
    metadata = {"driver": "run_decoder_strength.py", "git_commit": commit,
                "canary": args.canary, "decoders": list(DECODERS),
                "operators": list(OPERATORS), "solver_seeds": list(SOLVER_SEEDS),
                "iters": ITERS}
    rows = run_checkpointed(jobs, job, lambda x: list(x), row_key, checkpoint,
                            metadata, processes=args.workers)
    gate = greedy_reproduction_gate(rows, 1 if args.canary else 30)
    if args.canary:
        print(json.dumps({"gate": gate, "checkpoint": checkpoint}, indent=2))
        return
    arms, effects = summarize(rows)
    out = {
        "config": {"n": N, "instances": len(list(SEEDS)), "alpha": ALPHA,
                   "endurance": "inf", "decoders": DECODERS, "operators": OPERATORS,
                   "solver_seeds": list(SOLVER_SEEDS), "iters": ITERS,
                   "comparison": "equal iteration; search span equals final span; runtimes reported",
                   "search_framework": "one alns() loop; decoder is a registered option",
                   "bootstrap_resamples": 20000, "bootstrap_unit": "instance",
                   "git_commit": commit,
                   "checkpoint": os.path.relpath(checkpoint, os.path.join(HERE, "..")),
                   "runtime_s": time.time() - started},
        "greedy_reproduction_gate": gate,
        "arms": arms,
        "effects": effects,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(effects, indent=2))
    print(f"done in {time.time() - started:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
