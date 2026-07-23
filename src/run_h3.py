"""
H3 time-vs-drone-energy Pareto frontier (single drone, alpha=2, endurance=inf).
For each instance we sweep the energy weight lambda in the ALNS objective
makespan + lambda * drone_energy, recording (makespan, drone_energy) of the best solution.
lambda=0 is time-optimal (max energy); large lambda -> avoid drones (-> truck-only, ~0 energy).
We test whether the front is non-degenerate (time-opt != energy-opt) and find its knee.
Outputs experiments/H3-time-energy/results/h3.json
"""
import os, json, time
import numpy as np
from multiprocessing import Pool
from problem import gen_instance, dist_matrix
from alns import alns_energy

OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "H3-time-energy", "results", "h3.json")
LAMBDAS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 2.0, 8.0]
ALPHA = 2.0
NSEED = 2


def job(args):
    n, seed, li = args
    lam = LAMBDAS[li]
    inst = gen_instance(n, seed=seed); Dt = dist_matrix(inst["coords"])
    best_ms, best_e, best_obj = None, None, None
    for s in range(NSEED):
        ms, e = alns_energy(inst, Dt, ALPHA, lam, endurance=np.inf, max_span=12, iters=8000, seed=s)
        obj = ms + lam * e
        if best_obj is None or obj < best_obj:
            best_obj, best_ms, best_e = obj, ms, e
    return (n, seed, li, best_ms, best_e)


def main():
    t0 = time.time()
    jobs = ([(20, s, li) for s in range(20) for li in range(len(LAMBDAS))]
            + [(50, s, li) for s in range(12) for li in range(len(LAMBDAS))])
    with Pool(38) as pool:
        raw = pool.map(job, jobs)

    out = {"lambdas": LAMBDAS, "by_n": {}, "raw": raw,
           "config": {"alpha": ALPHA, "nseed": NSEED, "runtime_s": time.time() - t0}}
    for n in [20, 50]:
        rows = [r for r in raw if r[0] == n]
        seeds = sorted(set(r[1] for r in rows))
        # per-instance normalize by lambda=0 (time-opt makespan, max energy)
        norm_ms = {li: [] for li in range(len(LAMBDAS))}
        norm_e = {li: [] for li in range(len(LAMBDAS))}
        abs_curve = {li: {"ms": [], "e": []} for li in range(len(LAMBDAS))}
        for sd in seeds:
            base = [r for r in rows if r[1] == sd]
            ms0 = [r[3] for r in base if r[2] == 0][0]
            e0 = [r[4] for r in base if r[2] == 0][0]
            for r in base:
                li = r[2]
                norm_ms[li].append(r[3] / ms0)
                norm_e[li].append(r[4] / e0 if e0 > 1e-9 else 0.0)
                abs_curve[li]["ms"].append(r[3]); abs_curve[li]["e"].append(r[4])
        curve = []
        for li, lam in enumerate(LAMBDAS):
            curve.append({"lambda": lam,
                          "norm_makespan": float(np.mean(norm_ms[li])),
                          "norm_energy": float(np.mean(norm_e[li])),
                          "abs_makespan": float(np.mean(abs_curve[li]["ms"])),
                          "abs_energy": float(np.mean(abs_curve[li]["e"]))})
        # degeneracy + knee on the averaged normalized front
        x = np.array([c["norm_makespan"] for c in curve])   # increases with lambda
        y = np.array([c["norm_energy"] for c in curve])      # decreases with lambda
        # knee = max perpendicular distance to the chord between endpoints
        p0 = np.array([x[0], y[0]]); p1 = np.array([x[-1], y[-1]])
        chord = p1 - p0; cl = np.linalg.norm(chord) + 1e-12
        dists = [float(abs(np.cross(chord, np.array([x[i], y[i]]) - p0)) / cl) for i in range(len(x))]
        knee_i = int(np.argmax(dists))
        time_span = float(x.max() - x.min()); energy_span = float(y.max() - y.min())
        out["by_n"][n] = {
            "pareto": curve,
            "time_opt": {"makespan": curve[0]["abs_makespan"], "energy": curve[0]["abs_energy"]},
            "energy_opt": {"makespan": curve[-1]["abs_makespan"], "energy": curve[-1]["abs_energy"]},
            "makespan_span_frac": time_span, "energy_span_frac": energy_span,
            "non_degenerate": bool(time_span > 0.05 and energy_span > 0.05),
            "knee": {"lambda": curve[knee_i]["lambda"],
                     "norm_makespan": curve[knee_i]["norm_makespan"],
                     "norm_energy": curve[knee_i]["norm_energy"],
                     "makespan_increase_pct": (curve[knee_i]["norm_makespan"] - 1) * 100,
                     "energy_reduction_pct": (1 - curve[knee_i]["norm_energy"]) * 100}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"H3 done in {time.time()-t0:.0f}s -> {OUT}")
    for n in [20, 50]:
        b = out["by_n"][n]
        print(f"n={n}: non_degenerate={b['non_degenerate']} "
              f"makespan_span={b['makespan_span_frac']:.3f} energy_span={b['energy_span_frac']:.3f} "
              f"knee: +{b['knee']['makespan_increase_pct']:.1f}% time buys -{b['knee']['energy_reduction_pct']:.1f}% energy")


if __name__ == "__main__":
    main()
