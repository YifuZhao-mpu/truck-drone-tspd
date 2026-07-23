"""Street-network transfer as a prediction-error test (archived artifacts only).

Predictions frozen from the synthetic factorial (h2_v2.json alpha=2, E=1, n=50 cell and
p0_v2.json uniform/center/L1 cell); measurements from realnet.json. Reported as
prediction errors (district minus synthetic) with independent two-sample bootstrap CIs
over instances (20k resamples). Output: experiments/REALNET/results/transfer_test_v2.json
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H2 = os.path.join(HERE, "..", "experiments", "H2-design-space", "results", "h2_v2.json")
P0 = os.path.join(HERE, "..", "experiments", "P0-robustness", "results", "p0_v2.json")
RN = os.path.join(HERE, "..", "experiments", "REALNET", "results", "realnet.json")
OUT = os.path.join(HERE, "..", "experiments", "REALNET", "results", "transfer_test_v2.json")
RNG = np.random.default_rng(20260719)
B = 20000


def boot_diff(a, b):
    """mean(a) - mean(b) with independent bootstrap CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = [RNG.choice(a, a.size).mean() - RNG.choice(b, b.size).mean() for _ in range(B)]
    return {"diff": float(a.mean() - b.mean()),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "n_a": int(a.size), "n_b": int(b.size)}


def main():
    h2 = json.load(open(H2))
    p0 = json.load(open(P0))
    rn = json.load(open(RN))

    # synthetic per-instance savings, alpha=2 (ai=2), E=1 (ei=1), n=50, by m
    syn = {m: [r["saving_pct"] for r in h2["raw"]
               if r["n"] == 50 and r["ai"] == 2 and r["ei"] == 1 and r["m"] == m]
           for m in (1, 2, 3)}
    # synthetic per-instance increments (paired within seed)
    syn_by_seed = {}
    for r in h2["raw"]:
        if r["n"] == 50 and r["ai"] == 2 and r["ei"] == 1:
            syn_by_seed.setdefault(r["seed"], {})[r["m"]] = r["saving_pct"]
    syn_inc2 = [v[2] - v[1] for v in syn_by_seed.values()]
    syn_inc3 = [v[3] - v[2] for v in syn_by_seed.values()]
    # L1-proxy m=1 prediction cell (uniform, center, rectilinear truck metric, n=50)
    l1 = [r["saving_pct"] for r in p0["raw_cells"]
          if r["kind"] == "uniform" and r["depot"] == "center"
          and r["metric"] == "manhattan" and r["n"] == 50]

    # district measurements (paired increments within draw)
    dist_rows = {}
    for r in rn["raw"]:
        dist_rows.setdefault(r["district"], {}).setdefault(r["seed"], {})[r["m"]] = r["saving_pct"]

    out = {"bootstrap": B, "predictions": {
        "synthetic_euclid_m1_mean": float(np.mean(syn[1])),
        "synthetic_L1proxy_m1_mean": float(np.mean(l1)),
        "synthetic_inc2_mean": float(np.mean(syn_inc2)),
        "synthetic_inc3_mean": float(np.mean(syn_inc3))}, "districts": {}}

    for d, by_seed in dist_rows.items():
        m1 = [v[1] for v in by_seed.values()]
        inc2 = [v[2] - v[1] for v in by_seed.values()]
        inc3 = [v[3] - v[2] for v in by_seed.values()]
        out["districts"][d] = {
            "m1_saving_mean": float(np.mean(m1)),
            "inc2_mean": float(np.mean(inc2)), "inc3_mean": float(np.mean(inc3)),
            "err_inc2_vs_synthetic": boot_diff(inc2, syn_inc2),
            "err_inc3_vs_synthetic": boot_diff(inc3, syn_inc3),
            "err_m1_vs_L1proxy": boot_diff(m1, l1),
            "err_m1_vs_euclid": boot_diff(m1, syn[1])}
        print(f"{d}: m1 {np.mean(m1):.1f}%; inc2 err {out['districts'][d]['err_inc2_vs_synthetic']['diff']:+.2f} "
              f"{out['districts'][d]['err_inc2_vs_synthetic']['ci95']}; "
              f"inc3 err {out['districts'][d]['err_inc3_vs_synthetic']['diff']:+.2f} "
              f"{out['districts'][d]['err_inc3_vs_synthetic']['ci95']}; "
              f"m1 vs L1proxy {out['districts'][d]['err_m1_vs_L1proxy']['diff']:+.2f} "
              f"{out['districts'][d]['err_m1_vs_L1proxy']['ci95']}")

    out["note"] = ("Prediction errors, not equivalence tests (no preregistered margin). "
                   "Independent two-sample bootstrap: district draws and synthetic instances "
                   "are different instance sets.")
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
