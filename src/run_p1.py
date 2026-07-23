"""
P1 statistical rigor + strong-reference convergence (reviewer-facing).
 A. Formal tests on existing results: bootstrap 95% CIs (benchmark gap, savings) and paired
    Wilcoxon signed-rank / exact sign tests (sortie vs generic; ALNS vs greedy).
 B. Convergence reference: standard-budget ALNS vs a high-budget ALNS-long at n=20/50/100,
    bounding the optimality gap where no exact solver scales (no Julia/SOTA here).
Outputs experiments/P1-stats/results/p1.json
"""
import os, json, time
import numpy as np
from scipy import stats
from multiprocessing import Pool
from problem import gen_instance, dist_matrix
from baselines import truck_only_tsp
from alns import alns

OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "P1-stats", "results", "p1.json")
RES = os.path.join(os.path.dirname(__file__), "..", "experiments")


def boot_ci(x, B=20000, seed=12345):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------- Part B workers ----------
ITERS_STD = {20: 9000, 50: 16000, 100: 12000}
ITERS_LONG = {20: 24000, 50: 40000, 100: 30000}
N_CONV = {20: 10, 50: 10, 100: 6}


def job_conv(args):
    n, seed = args
    inst = gen_instance(n, seed=seed); D = dist_matrix(inst["coords"])
    span = min(n + 1, 12)
    std = min(alns(inst, D, 2.0, max_span=span, iters=ITERS_STD[n], sortie_aware=True, seed=s)["makespan"]
              for s in range(2))
    lng = min(alns(inst, D, 2.0, max_span=span, iters=ITERS_LONG[n], sortie_aware=True, seed=s)["makespan"]
              for s in range(3))
    return (n, seed, std, lng)


def main():
    t0 = time.time()
    h1 = json.load(open(f"{RES}/H1-alns-backbone/results/h1.json"))
    h2 = json.load(open(f"{RES}/H2-design-space/results/h2.json"))
    p0 = json.load(open(f"{RES}/P0-robustness/results/p0.json"))
    out = {}

    # A1 — benchmark optimality gap: bootstrap CI over 70 instances
    gaps = [r["gap"] for r in h1["raw_bench"]]
    m, lo, hi = boot_ci(gaps)
    out["benchmark_gap"] = {"mean": m, "ci95": [lo, hi], "n": len(gaps),
                            "pct_within_1pct": float(np.mean(np.array(gaps) < 1.0) * 100)}

    # A2 — sortie vs generic: paired Wilcoxon (n=20,50) + exact sign test (n=100)
    out["sortie_vs_generic"] = {}
    for n in [20, 50]:
        rows = [r for r in h1["raw_synth"] if r["n"] == n]
        s = np.array([r["sortie"] for r in rows]); g = np.array([r["generic"] for r in rows])
        d = s - g
        nz = d[np.abs(d) > 1e-12]
        try:
            w = stats.wilcoxon(s[np.abs(d) > 1e-12], g[np.abs(d) > 1e-12], alternative="less")
            p = float(w.pvalue)
        except Exception:
            p = None
        out["sortie_vs_generic"][f"n{n}"] = {
            "mean_diff": float(d.mean()), "wins": int((d < -1e-12).sum()),
            "losses": int((d > 1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum()),
            "wilcoxon_p_one_sided": p, "n_nonzero": int(len(nz))}
    ab = p0["ablation_n100"]
    k, N = ab["sortie_wins"], ab["sortie_wins"] + ab["generic_wins"]
    sign_p = float(stats.binomtest(k, N, 0.5, alternative="greater").pvalue) if N > 0 else None
    out["sortie_vs_generic"]["n100"] = {"wins": ab["sortie_wins"], "losses": ab["generic_wins"],
                                        "ties": ab["ties"], "mean_diff": ab["paired_sortie_minus_generic"],
                                        "sign_test_p_one_sided": sign_p}

    # A3 — ALNS vs greedy: paired Wilcoxon (n=10/20/50)
    out["alns_vs_greedy"] = {}
    for n in [10, 20, 50]:
        rows = [r for r in h1["raw_synth"] if r["n"] == n]
        a = np.array([r["sortie"] for r in rows]); gr = np.array([r["greedy"] for r in rows])
        w = stats.wilcoxon(a, gr, alternative="less")
        out["alns_vs_greedy"][f"n{n}"] = {"median_improve_pct": float(np.median((gr - a) / gr * 100)),
                                          "wilcoxon_p_one_sided": float(w.pvalue)}

    # A4 — savings bootstrap CI for headline cells (uniform alpha=2, E=1.0; ai=2, ei=1)
    out["saving_ci"] = {}
    for n in [20, 50]:
        sv = [r[7] for r in h2["raw_grid"] if r[1] == n and r[3] == 2 and r[4] == 1]
        m, lo, hi = boot_ci(sv)
        out["saving_ci"][f"uniform_a2_E1_n{n}"] = {"mean": m, "ci95": [lo, hi], "n": len(sv)}
    # clustered+manhattan cells from p0 (have mean/std/count -> normal-approx CI)
    cs = p0["cross_setting_cell_a2_e1"]
    for key in ["clustered_manhattan_n50", "clustered_manhattan_n100", "uniform_manhattan_n50"]:
        v = cs[key]; se = v["std"] / np.sqrt(v["count"])
        out["saving_ci"][key] = {"mean": v["mean_saving"], "ci95": [v["mean_saving"] - 1.96 * se, v["mean_saving"] + 1.96 * se],
                                 "n": v["count"], "method": "normal_approx"}

    # B — convergence vs high-budget ALNS-long
    jobs = [(n, s) for n in [20, 50, 100] for s in range(N_CONV[n])]
    with Pool(36) as pool:
        conv = pool.map(job_conv, jobs)
    out["convergence"] = {}
    for n in [20, 50, 100]:
        rows = [r for r in conv if r[0] == n]
        std = np.array([r[2] for r in rows]); lng = np.array([r[3] for r in rows])
        gap = (std - lng) / lng * 100
        m, lo, hi = boot_ci(gap)
        out["convergence"][f"n{n}"] = {"mean_gap_to_long_pct": m, "ci95": [lo, hi],
                                       "max_gap_pct": float(gap.max()), "n": len(rows)}
    out["config"] = {"iters_std": ITERS_STD, "iters_long": ITERS_LONG, "runtime_s": time.time() - t0}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"P1 done in {time.time()-t0:.0f}s -> {OUT}")
    print(json.dumps({k: out[k] for k in ["benchmark_gap", "sortie_vs_generic", "alns_vs_greedy", "convergence"]}, indent=2))
    print("SAVING CIs:", json.dumps(out["saving_ci"], indent=2))


if __name__ == "__main__":
    main()
