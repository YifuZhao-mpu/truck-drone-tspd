"""§F held-out long-budget comparator cross-check (major revision, R5.7).

Design locked in drafts/revision/revision_protocol.md §F and operationalized here:

* Instances: never-used uniform/center/euclidean seeds 30-39 at n in {20, 50} and
  30-34 at n=100, at the study capability point alpha=2 (truck factor 1, drone
  factor 0.5) and unlimited endurance, exactly as the archived matched comparison.
* Budgets: 4x the v2 matched per-size wall budgets (35/140/430 s -> 140/560/1720 s),
  used exactly as in the archived v2 comparison: complete solver runs are launched back
  to back until the cap, and every inner run that finishes after the cap is archived but
  excluded (the v2 hard cutoff, applied to every side).  Comparator processes are killed
  15 minutes after their cap (the warm-up is off the clock); their partial output is
  kept because every inner run is written as soon as it finishes.
* Replicates and seeds: three such restart loops per (solver, instance) with seeds
  {1, 2, 3} passed explicitly to every solver's RNG (Julia: Random.seed!(seed) once
  before the loop; ours: restart k of loop s uses ALNS seed 1000*s + k); a loop's value
  is the best cost among its inner runs inside the cap, the per-solver per-instance
  value is the best of its own completed loops.  Every inner run's cost and cumulative
  time, each loop's status and best route/solution structure are archived.
* Comparators run at package defaults from the committed pinned environment
  (experiments/SOTA/julia); ours runs the study configuration (span 12,
  9k/16k/16k iterations).  Tuning effort is therefore not matched (R5.8) and the
  manuscript says so.
* Report: per instance, each solver's relative gap to the pooled best-found value
  over all completed replicates of all solvers (the pooled best includes each solver
  itself and is not a bound); pairwise win/tie/loss with the v2 tie tolerance 1e-6
  relative; per-solver median/IQR/max gap; per-replicate spread; exclusion and failure
  counts.  Descriptive only -- no ranking, no optimality inference.

Every process is single-threaded (JULIA_NUM_THREADS=1, BLAS/Numba threads 1); jobs are
spread over independent processes.  Run on an otherwise quiet machine: the budgets are
wall-clock.  Usage: python run_heldout_comparators.py [--smoke] [--workers N]
"""
import argparse
import hashlib
import json
import os
import subprocess
import time

import numpy as np

from alns import alns
from problem import dist_matrix, gen_instance
from revision_utils import git_commit, run_checkpointed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUTDIR = os.path.join(ROOT, "experiments", "SOTA", "results")
OUT = os.path.join(OUTDIR, "heldout_comparators_v3.json")
CHECKPOINT = os.path.join(OUTDIR, "heldout_comparators_v3.jsonl")
INPUT_TXT = os.path.join(OUTDIR, "heldout_input.txt")
INSTANCES_JSON = os.path.join(OUTDIR, "heldout_instances.json")
JULIA_PROJECT = os.path.join(ROOT, "experiments", "SOTA", "julia")
JULIA_BIN = os.environ.get("JULIA_BIN", os.path.expanduser("~/.local/opt/julia-1.10.4/bin/julia"))
JULIA_DRIVER = os.path.join(HERE, "run_heldout_comparator.jl")

HELDOUT = {20: list(range(30, 40)), 50: list(range(30, 40)), 100: list(range(30, 35))}
V2_BUDGET = {20: 35.0, 50: 140.0, 100: 430.0}
BUDGET_FACTOR = 4.0
CAP = {n: BUDGET_FACTOR * b for n, b in V2_BUDGET.items()}
KILL_GRACE_S = 900.0
ITERS = {20: 9000, 50: 16000, 100: 16000}
SPAN = 12
ALPHA = 2.0
TRUCK_FACTOR, DRONE_FACTOR = 1.0, 1.0 / ALPHA
REP_SEEDS = (1, 2, 3)
SOLVERS = ("ours", "dps", "hgatac")
TIE_TOL = 1e-6
PACKAGES = {
    "dps": "chkwon/TSPDrone.jl @ f42d27c (DPS / TSP-ep-all of Bogyrbayeva et al.)",
    "hgatac": "Sasanm88/TSPDroneHGATAC.jl @ 8f1f345 (HGA-TAC of Mahmoudinazlou & Kwon)",
}
SINGLE_THREAD_ENV = {"JULIA_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                     "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                     "NUMBA_NUM_THREADS": "1"}


def instance_records(smoke):
    if smoke:
        spec = {20: [30, 31]}
    else:
        spec = HELDOUT
    recs = []
    for n, seeds in spec.items():
        for seed in seeds:
            inst = gen_instance(n, seed=seed)
            recs.append({"id": f"uniform-n{n}-s{seed}", "n": n, "instance_seed": seed,
                         "x": inst["coords"][:, 0].tolist(),
                         "y": inst["coords"][:, 1].tolist(),
                         "truck_cost_factor": TRUCK_FACTOR,
                         "drone_cost_factor": DRONE_FACTOR})
    return recs


def write_inputs(recs, txt_path, json_path):
    with open(txt_path, "w") as f:
        for it in recs:
            f.write(f"ID {it['id']} {it['n']} {it['truck_cost_factor']} {it['drone_cost_factor']}\n")
            f.write("X " + " ".join(repr(v) for v in it["x"]) + "\n")
            f.write("Y " + " ".join(repr(v) for v in it["y"]) + "\n")
    with open(json_path, "w") as f:
        json.dump(recs, f)


def source_digest():
    h = hashlib.sha256()
    for name in ("problem.py", "alns.py", "run_heldout_comparators.py",
                 "run_heldout_comparator.jl"):
        with open(os.path.join(HERE, name), "rb") as f:
            h.update(f.read())
    for name in ("Project.toml", "Manifest.toml"):
        with open(os.path.join(JULIA_PROJECT, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def run_ours(rec, seed):
    os.environ.update(SINGLE_THREAD_ENV)
    coords = np.stack([np.array(rec["x"]), np.array(rec["y"])], 1)
    n = rec["n"]
    cap = CAP[n]
    inst = {"coords": coords, "n": n, "seed": rec["instance_seed"], "depot": "center",
            "kind": "uniform"}
    D = dist_matrix(coords)
    alns(inst, D, ALPHA, endurance=np.inf, max_span=SPAN, iters=50, seed=999)  # JIT warm-up
    runs = []
    best, best_order = float("inf"), None
    t0 = time.perf_counter()
    k = 0
    while time.perf_counter() - t0 < cap:
        alns_seed = 1000 * seed + k
        r = alns(inst, D, ALPHA, endurance=np.inf, max_span=SPAN, iters=ITERS[n],
                 seed=alns_seed)
        el = time.perf_counter() - t0
        k += 1
        runs.append({"k": k, "alns_seed": alns_seed, "cost": float(r["makespan"]),
                     "t_s": el})
        if r["makespan"] < best and el <= cap:
            best, best_order = float(r["makespan"]), [int(v) for v in r["order"]]
    return {"runs": runs, "elapsed_s": time.perf_counter() - t0,
            "best_order": best_order, "stderr_tail": ""}


def run_comparator(rec, solver, seed, block, input_txt):
    n = rec["n"]
    # One working directory per process: Concorde (used by TSPDrone.jl) writes temporary
    # .tsp/.sol files into the cwd with names drawn from the global RNG, so two
    # processes seeded identically would otherwise collide on the same file names.
    job_dir = os.path.join(OUTDIR, "heldout_tmp", f"{rec['id']}.{solver}.s{seed}")
    os.makedirs(job_dir, exist_ok=True)
    out_path = os.path.join(job_dir, "replicate.txt")
    if os.path.exists(out_path):
        os.remove(out_path)
    cmd = [JULIA_BIN, f"--project={JULIA_PROJECT}", JULIA_DRIVER, os.path.abspath(input_txt),
           str(block), solver, str(seed), repr(CAP[n]), out_path]
    env = dict(os.environ, **SINGLE_THREAD_ENV)
    t0 = time.perf_counter()
    killed, stderr_tail, returncode = False, "", 0
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=job_dir,
                              timeout=CAP[n] + KILL_GRACE_S)
        stderr_tail, returncode = proc.stderr[-500:], proc.returncode
    except subprocess.TimeoutExpired as exc:
        killed = True
        stderr_tail = exc.stderr[-500:] if isinstance(exc.stderr, str) else ""
    elapsed = time.perf_counter() - t0
    runs, best_truck, best_drone = [], None, None
    if os.path.exists(out_path):
        for line in open(out_path):
            f = line.split()
            if not f:
                continue
            if f[0] == "RUN":
                runs.append({"k": int(f[1]), "cost": float(f[2]), "t_s": float(f[3])})
            elif f[0] == "BEST":
                best_truck = [int(v) for v in f[2].split(",")]
                best_drone = [int(v) for v in f[3].split(",")]
    for leftover in os.listdir(job_dir):          # Concorde temp files, if any
        if leftover != "replicate.txt":
            os.remove(os.path.join(job_dir, leftover))
    res = {"runs": runs, "elapsed_s": elapsed, "best_truck_route": best_truck,
           "best_drone_route": best_drone, "stderr_tail": stderr_tail,
           "process_killed": killed, "returncode": returncode}
    if returncode != 0 and not runs:
        res["status"] = "crashed"
    return res


def job(spec):
    solver, n, instance_seed, seed, block, input_txt, rec = spec
    if solver == "ours":
        res = run_ours(rec, seed)
    else:
        res = run_comparator(rec, solver, seed, block, input_txt)
    cap = CAP[n]
    inside = [r["cost"] for r in res["runs"] if r["t_s"] <= cap + 1e-9]
    status = res.pop("status", None)
    if status is None:
        status = "completed" if inside else "no_run_inside_cap"
    return {"solver": solver, "id": rec["id"], "n": n, "instance_seed": instance_seed,
            "replicate_seed": seed, "cap_s": cap, "status": status,
            "cost": min(inside) if inside else None,
            "runs_inside_cap": len(inside), "runs_total": len(res["runs"]), **res}


def spec_key(spec):
    return [spec[0], spec[1], spec[2], spec[3]]


def row_key(row):
    return [row["solver"], row["n"], row["instance_seed"], row["replicate_seed"]]


def summarize(rows, recs):
    by_inst = {}
    for r in rows:
        by_inst.setdefault(r["id"], {}).setdefault(r["solver"], []).append(r)
    per_instance = []
    for rec in recs:
        iid = rec["id"]
        solvers = by_inst.get(iid, {})
        completed = {s: [r for r in solvers.get(s, []) if r["status"] == "completed"]
                     for s in SOLVERS}
        pooled = [r["cost"] for s in SOLVERS for r in completed[s]]
        pooled_best = min(pooled) if pooled else None
        entry = {"id": iid, "n": rec["n"], "pooled_best": pooled_best}
        for s in SOLVERS:
            reps = completed[s]
            if reps:
                best = min(r["cost"] for r in reps)
                worst = max(r["cost"] for r in reps)
                entry[s] = {"best": best, "completed": len(reps),
                            "excluded": len(solvers.get(s, [])) - len(reps),
                            "gap_pct": (best - pooled_best) / pooled_best * 100.0,
                            "replicate_spread_pct": (worst - best) / best * 100.0,
                            "runs_inside_cap": [r["runs_inside_cap"] for r in reps]}
            else:
                entry[s] = {"best": None, "completed": 0,
                            "excluded": len(solvers.get(s, [])), "gap_pct": None,
                            "replicate_spread_pct": None, "runs_inside_cap": []}
        per_instance.append(entry)

    def block(entries):
        out = {"instances": len(entries)}
        for s in SOLVERS:
            gaps = [e[s]["gap_pct"] for e in entries if e[s]["gap_pct"] is not None]
            spreads = [e[s]["replicate_spread_pct"] for e in entries
                       if e[s]["replicate_spread_pct"] is not None]
            out[s] = {
                "instances_with_result": len(gaps),
                "instances_excluded_no_completed_replicate": len(entries) - len(gaps),
                "median_gap_pct": float(np.median(gaps)) if gaps else None,
                "iqr_gap_pct": ([float(np.percentile(gaps, 25)), float(np.percentile(gaps, 75))]
                                if gaps else None),
                "max_gap_pct": float(max(gaps)) if gaps else None,
                "mean_gap_pct": float(np.mean(gaps)) if gaps else None,
                "matches_pooled_best": int(sum(g <= TIE_TOL * 100 for g in gaps)),
                "median_replicate_spread_pct": float(np.median(spreads)) if spreads else None,
                "max_replicate_spread_pct": float(max(spreads)) if spreads else None,
            }
        pairwise = {}
        for a in SOLVERS:
            for b in SOLVERS:
                if a >= b:
                    continue
                w = l = t = 0
                usable = 0
                for e in entries:
                    if e[a]["best"] is None or e[b]["best"] is None:
                        continue
                    usable += 1
                    va, vb = e[a]["best"], e[b]["best"]
                    if abs(va - vb) <= TIE_TOL * max(va, vb):
                        t += 1
                    elif va < vb:
                        w += 1
                    else:
                        l += 1
                pairwise[f"{a}_vs_{b}"] = {"first_lower": w, "second_lower": l, "ties": t,
                                          "instances_compared": usable}
        out["pairwise"] = pairwise
        return out

    summary = {"pooled": block(per_instance)}
    for n in sorted({e["n"] for e in per_instance}):
        summary[f"n{n}"] = block([e for e in per_instance if e["n"] == n])
    statuses = {}
    for r in rows:
        statuses.setdefault(r["solver"], {}).setdefault(r["status"], 0)
        statuses[r["solver"]][r["status"]] += 1
    summary["replicate_statuses"] = statuses
    return per_instance, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="pre-F gate: two n=20 instances, one replicate, all solvers")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    started = time.time()
    if not os.path.exists(JULIA_BIN):
        raise SystemExit(f"julia not found at {JULIA_BIN}; set JULIA_BIN")
    recs = instance_records(args.smoke)
    suffix = ".smoke" if args.smoke else ""
    input_txt = INPUT_TXT.replace(".txt", f"{suffix}.txt")
    instances_json = INSTANCES_JSON.replace(".json", f"{suffix}.json")
    write_inputs(recs, input_txt, instances_json)
    seeds = REP_SEEDS[:1] if args.smoke else REP_SEEDS
    specs = [(solver, rec["n"], rec["instance_seed"], seed, block, input_txt, rec)
             for block, rec in enumerate(recs, start=1)
             for solver in SOLVERS for seed in seeds]
    print(f"held-out comparator cross-check: {len(recs)} instances, {len(specs)} jobs "
          f"(smoke={args.smoke})")
    julia_version = subprocess.run([JULIA_BIN, "--version"], capture_output=True,
                                   text=True).stdout.strip()
    commit = git_commit(ROOT)
    metadata = {"driver": "run_heldout_comparators.py", "git_commit": commit,
                "source_sha256": source_digest(), "smoke": args.smoke,
                "caps_s": {str(k): v for k, v in CAP.items()},
                "replicate_seeds": list(seeds), "iters": {str(k): v for k, v in ITERS.items()},
                "span": SPAN, "julia": julia_version}
    checkpoint = CHECKPOINT.replace(".jsonl", f"{suffix}.jsonl")
    rows = run_checkpointed(specs, job, spec_key, row_key, checkpoint, metadata,
                            processes=args.workers)
    per_instance, summary = summarize(rows, recs)
    out = {"config": {**metadata, "packages": PACKAGES, "julia_project": "experiments/SOTA/julia",
                      "v2_budgets_s": {str(k): v for k, v in V2_BUDGET.items()},
                      "budget_factor": BUDGET_FACTOR, "kill_grace_s": KILL_GRACE_S,
                      "alpha": ALPHA, "endurance": "inf", "tie_tolerance_relative": TIE_TOL,
                      "heldout_seeds": {str(k): v for k, v in HELDOUT.items()},
                      "protocol": "each replicate is a seeded restart loop until the 4x "
                                  "cap (v2 shape); inner runs finishing after the cap are "
                                  "excluded on every side; comparators at package defaults, ours at "
                                  "the study configuration (tuning effort not matched); "
                                  "descriptive cross-check against the pooled best-found "
                                  "value, which includes each solver and is not a bound",
                      "checkpoint": os.path.relpath(checkpoint, ROOT),
                      "runtime_s": time.time() - started},
           "summary": summary, "per_instance": per_instance, "rows": rows}
    out_path = OUT.replace(".json", f"{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"done in {time.time() - started:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
