"""H3 v3: reproduce the full v2 energy sweep and retain every algorithm seed/route."""
import argparse
import hashlib
import json
import os
import time
import numpy as np

from alns import alns
from problem import dist_matrix, gen_instance
from revision_utils import bootstrap_ci, git_commit, run_checkpointed
from run_h3_v2 import summarize


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "experiments", "H3-time-energy", "results", "h3_v3.json")
OLD = os.path.join(HERE, "..", "experiments", "H3-time-energy", "results", "h3_v2.json")
CHECKPOINT = os.path.join(HERE, "..", "experiments", "H3-time-energy", "results",
                          "h3_v3.jsonl")
LAMBDAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.45, 0.65, 1.0, 1.5, 3.0, 8.0]
LAMBDAS_SENS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.65, 1.5, 8.0]
ALPHA = 2.0
ITERS = {20: 9000, 50: 16000}
SOLVER_SEEDS = range(3)
SPAN = 12
INSTANCE_SEEDS_MAIN = range(30)
INSTANCE_SEEDS_SENS = range(15)


def source_digest():
    h = hashlib.sha256()
    for name in ("problem.py", "alns.py", "run_h3_v3.py"):
        with open(os.path.join(HERE, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def job(args):
    tag, n, instance_seed, lam, cl, te = args
    inst = gen_instance(n, seed=instance_seed)
    D = dist_matrix(inst["coords"])
    reps = []
    for solver_seed in SOLVER_SEEDS:
        started = time.perf_counter()
        result = alns(inst, D, ALPHA, endurance=np.inf, max_span=SPAN,
                      iters=ITERS[n], sortie_aware=True, seed=solver_seed,
                      lam=lam, cl=cl, cb=1.0, te=te)
        ev = result["eval"]
        reps.append({"solver_seed": solver_seed,
                     "makespan": float(ev["makespan"]),
                     "truck_dist": float(ev["truck_dist"]),
                     "e_drone": float(ev["e_drone"]),
                     "e_truck": float(ev["e_truck"]),
                     "e_total": float(ev["e_total"]),
                     "n_sorties": int(ev["n_sorties"]),
                     "objective": float(result["objective"]),
                     "wall_s": float(time.perf_counter() - started),
                     "order": [int(v) for v in result["order"]]})
    winner = min(reps, key=lambda r: (r["objective"], r["solver_seed"]))
    return {"tag": tag, "n": n, "instance_seed": instance_seed, "lam": lam,
            "cl": cl, "te": te, "winner_solver_seed": winner["solver_seed"],
            "replicates": reps}


def winner_row(row):
    p = min(row["replicates"], key=lambda x: (x["objective"], x["solver_seed"]))
    return {"tag": row["tag"], "n": row["n"], "seed": row["instance_seed"],
            "lam": row["lam"], "cl": row["cl"], "te": row["te"],
            **{k: p[k] for k in ("makespan", "truck_dist", "e_drone", "e_truck",
                                  "e_total", "n_sorties", "objective")},
            "winner_solver_seed": p["solver_seed"], "order": p["order"]}


def replicate_row(row, solver_seed):
    p = next(x for x in row["replicates"] if x["solver_seed"] == solver_seed)
    return {"tag": row["tag"], "n": row["n"], "seed": row["instance_seed"],
            "lam": row["lam"], "cl": row["cl"], "te": row["te"],
            **{k: p[k] for k in ("makespan", "truck_dist", "e_drone", "e_truck",
                                  "e_total", "n_sorties", "objective")}}


def row_key(row):
    return [row["tag"], row["n"], row["instance_seed"], row["lam"], row["cl"],
            row["te"]]


def reproduction_gate(winners, expected_count=1260):
    old = json.load(open(OLD))
    expected = {(r["tag"], r["n"], r["seed"], r["lam"], r["cl"], r["te"]): r
                for r in old["raw"]}
    fields = ("makespan", "truck_dist", "e_drone", "e_truck", "e_total",
              "n_sorties", "objective")
    checks = 0
    route_checks = 0
    for r in winners:
        oldrow = expected[(r["tag"], r["n"], r["seed"], r["lam"], r["cl"], r["te"])]
        for field in fields:
            checks += 1
            if r[field] != oldrow[field]:
                raise RuntimeError(f"H3 field gate failed at "
                                   f"{row_key({'tag': r['tag'], 'n': r['n'], 'instance_seed': r['seed'], 'lam': r['lam'], 'cl': r['cl'], 'te': r['te']})}/{field}")
        route_key = (f"{r['tag']}-n{r['n']}-s{r['seed']}-l{r['lam']}-"
                     f"cl{r['cl']}-te{r['te']}")
        route_checks += 1
        if r["order"] != old["routes"][route_key]:
            raise RuntimeError(f"H3 route gate failed at {route_key}")
    if len(winners) != expected_count:
        raise RuntimeError(f"H3 gate row count {len(winners)} != {expected_count}")
    return {"status": "PASS", "rows": len(winners),
            "archived_fields_checked": checks, "routes_checked": route_checks,
            "comparison": "exact Python equality"}


def selection_summary(rows):
    gains = []
    spreads = []
    by_instance = {}
    for row in rows:
        vals = np.array([p["objective"] for p in row["replicates"]])
        mean = float(np.mean(vals))
        gain = (mean - float(np.min(vals))) / mean * 100 if mean else 0.0
        gains.append(gain)
        by_instance.setdefault((row["tag"], row["n"], row["instance_seed"]), []).append(gain)
        spreads.append((float(np.max(vals)) - float(np.min(vals))) / float(np.min(vals)) * 100)
    pooled = [float(np.mean(by_instance[key])) for key in sorted(by_instance)]
    return {"median_objective_selection_gain_pct": float(np.median(gains)),
            "objective_selection_gain_iqr_pct": [float(np.percentile(gains, 25)),
                                                  float(np.percentile(gains, 75))],
            "max_objective_selection_gain_pct": float(np.max(gains)),
            "pooled_instance_mean_selection_gain_pct": float(np.mean(pooled)),
            "pooled_instance_selection_gain_ci95": bootstrap_ci(pooled, 555001),
            "median_objective_replicate_spread_pct": float(np.median(spreads)),
            "max_objective_replicate_spread_pct": float(np.max(spreads))}


def between_seed_front_summary(fronts):
    out = {}
    for n in (20, 50):
        metrics = {}
        paths = {
            "knee_time_increase_median_pct": ("knee_dms_pct", "median"),
            "knee_energy_reduction_median_pct": ("knee_de_pct", "median"),
            "endpoint_time_span_mean_pct": ("ms_span_pct_mean",),
            "endpoint_energy_span_mean_pct": ("e_total_span_pct_mean",),
        }
        for name, path in paths.items():
            vals = []
            for solver_seed in SOLVER_SEEDS:
                value = fronts[str(solver_seed)][str(n)]
                for key in path:
                    value = value[key]
                vals.append(float(value))
            metrics[name] = {"by_solver_seed": vals,
                             "min": float(min(vals)), "max": float(max(vals)),
                             "range": float(max(vals) - min(vals))}
        out[str(n)] = metrics
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--workers", type=int, default=36)
    args = parser.parse_args()
    started = time.time()
    jobs = [("main", n, sd, lam, 1.0, 0.3)
            for n in (20, 50) for sd in INSTANCE_SEEDS_MAIN for lam in LAMBDAS]
    jobs += [("payload", 50, sd, lam, cl, 0.3)
             for sd in INSTANCE_SEEDS_SENS for lam in LAMBDAS_SENS for cl in (0.5, 2.0)]
    jobs += [("truckcoef", 50, sd, lam, 1.0, te)
             for sd in INSTANCE_SEEDS_SENS for lam in LAMBDAS_SENS for te in (0.15, 0.6)]
    if args.canary:
        jobs = [jobs[i] for i in (0, 12, 389, 390, 779, 780, 1019, 1259)]
    print(f"{len(jobs)} H3 v3 jobs, 3 retained replicates each")
    commit = git_commit(os.path.join(HERE, ".."))
    digest = source_digest()
    checkpoint = CHECKPOINT.replace(".jsonl", ".canary.jsonl") if args.canary else CHECKPOINT
    metadata = {"driver": "run_h3_v3.py", "git_commit": commit,
                "source_sha256": digest, "canary": args.canary,
                "solver_seeds": list(SOLVER_SEEDS), "iters": ITERS, "span": SPAN}
    rows = run_checkpointed(jobs, job, lambda x: list(x), row_key, checkpoint,
                            metadata, processes=args.workers)
    winners = [winner_row(r) for r in rows]
    gate = reproduction_gate(winners, len(jobs))
    if args.canary:
        print(json.dumps({"gate": gate, "checkpoint": checkpoint}, indent=2))
        return
    winner_main = [r for r in winners if r["tag"] == "main"]
    fixed_seed_fronts = {}
    for solver_seed in SOLVER_SEEDS:
        reps = [replicate_row(r, solver_seed) for r in rows if r["tag"] == "main"]
        fixed_seed_fronts[str(solver_seed)] = summarize(reps, LAMBDAS)
    out = {
        "config": {"lambdas": LAMBDAS, "lambdas_sens": LAMBDAS_SENS,
                   "alpha": ALPHA, "iters": ITERS, "solver_seeds": list(SOLVER_SEEDS),
                   "span": SPAN, "objective": "makespan + lambda*(e_drone+te*truck_dist)",
                   "aggregation": "all replicates retained; fixed-solver-seed fronts reported",
                   "git_commit": commit, "source_sha256": digest,
                   "checkpoint": os.path.relpath(checkpoint, os.path.join(HERE, "..")),
                   "runtime_s": time.time() - started},
        "reproduction_gate": gate,
        "winner_summary": summarize(winner_main, LAMBDAS),
        "fixed_solver_seed_fronts": fixed_seed_fronts,
        "between_solver_seed_front_spread": between_seed_front_summary(fixed_seed_fronts),
        "selection_summary": selection_summary(rows),
        "raw": winners,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(json.dumps({"gate": gate, "selection": out["selection_summary"],
                      "fixed_seed_fronts": fixed_seed_fronts}, indent=1)[:10000])
    print(f"done in {time.time()-started:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
