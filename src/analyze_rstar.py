"""Exact green-and-fast boundary r* = E_D / (L0 - D_T) for every archived route.

Under the affine energy model, a hybrid plan p beats truck-only in TOTAL energy at
truck-to-drone power ratio r iff  E_D,p + r*D_T,p < r*L0  <=>  r > r*_p = E_D/(L0-D_T)
(whenever L0 > D_T). Inputs are ARCHIVED artifacts only (no solver runs):
  - h2_v2.json: 3600 factorial winners; e_truck stored at TRUCK_E=0.3 -> D_T = e_truck/0.3
  - h3_calibrated.json: independent lambda=0 arms (te=1.0 rows), truck_dist stored directly
Output: experiments/H2-design-space/results/rstar_v2.json
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H2 = os.path.join(HERE, "..", "experiments", "H2-design-space", "results", "h2_v2.json")
H3C = os.path.join(HERE, "..", "experiments", "H3-time-energy", "results", "h3_calibrated.json")
OUT = os.path.join(HERE, "..", "experiments", "H2-design-space", "results", "rstar_v2.json")
TE_H2 = 0.3  # energy.TRUCK_E used by run_h2_v2.py's evaluate_solution accounting


def q(x, p):
    return float(np.percentile(np.asarray(x, dtype=float), p))


def main():
    h2 = json.load(open(H2))
    alphas = h2["config"]["alphas"]
    ms = h2["config"]["ms"]

    rows = []
    nonpos = 0
    for r in h2["raw"]:
        d_t = r["e_truck"] / TE_H2
        denom = r["truck_ref"] - d_t
        if denom <= 0:
            nonpos += 1
            rstar = float("inf")
        else:
            rstar = r["e_drone"] / denom
        rows.append({"n": r["n"], "seed": r["seed"], "ai": r["ai"], "ei": r["ei"],
                     "m": r["m"], "rstar": rstar, "L0_minus_DT": denom,
                     "e_drone": r["e_drone"], "saving_pct": r["saving_pct"]})

    allr = [r["rstar"] for r in rows if np.isfinite(r["rstar"])]
    out = {"model": "rstar = e_drone / (truck_ref - e_truck/0.3); exact identity under the "
                    "affine model for BEST-FOUND plans (no optimality certificate); "
                    "conditional on the stated energy model",
           "n_records": len(rows), "n_nonpositive_denominator": nonpos,
           "overall": {"median": q(allr, 50), "iqr": [q(allr, 25), q(allr, 75)],
                       "min": float(min(allr)), "max": float(max(allr))},
           "frac_green_at_r": {str(rv): float(np.mean([r["rstar"] < rv for r in rows]))
                               for rv in (1, 2, 3, 6, 30, 60)}}

    # central cell (alpha=2 -> ai=2, E=1 -> ei=1) per (n, m)
    cc = {}
    for n in (20, 50):
        for m in ms:
            v = [r["rstar"] for r in rows if r["n"] == n and r["ai"] == 2 and r["ei"] == 1 and r["m"] == m]
            cc[f"n{n}-m{m}"] = {"median": q(v, 50), "iqr": [q(v, 25), q(v, 75)], "count": len(v)}
    out["central_cell_alpha2_E1"] = cc

    # capability structure: median r* by alpha (m=1, E=1, pooled n) and by m (alpha=2, E=1)
    out["by_alpha_m1_E1"] = {str(alphas[ai]): q([r["rstar"] for r in rows
                                                 if r["ai"] == ai and r["ei"] == 1 and r["m"] == 1], 50)
                             for ai in range(len(alphas))}
    out["by_m_alpha2_E1"] = {str(m): q([r["rstar"] for r in rows
                                        if r["ai"] == 2 and r["ei"] == 1 and r["m"] == m], 50)
                             for m in ms}
    # full per-cell medians for the phase-chart figure
    med_cell = {}
    for n in (20, 50):
        for ai in range(len(alphas)):
            for ei in range(4):
                for m in ms:
                    v = [r["rstar"] for r in rows if r["n"] == n and r["ai"] == ai
                         and r["ei"] == ei and r["m"] == m]
                    med_cell[f"n{n}-a{ai}-e{ei}-m{m}"] = q(v, 50)
    out["cell_medians"] = med_cell

    # independent check: h3_calibrated lambda=0 arm (te=1.0 rows store truck_dist directly)
    h3c = json.load(open(H3C))
    v = [r["e_drone"] / (r["truck_only_ref"] - r["truck_dist"])
         for r in h3c["raw"] if r["lam"] == 0.0 and r["te"] == 1.0
         and r["truck_only_ref"] > r["truck_dist"]]
    out["h3_calibrated_lambda0_check"] = {"median": q(v, 50), "iqr": [q(v, 25), q(v, 75)],
                                          "count": len(v)}

    out["per_record"] = rows
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"records={len(rows)} nonpos_denom={nonpos}")
    print(f"overall median={out['overall']['median']:.3f} range=[{out['overall']['min']:.3f},"
          f"{out['overall']['max']:.3f}]")
    print("central cell:", {k: round(v['median'], 2) for k, v in cc.items()})
    print("frac green:", {k: round(v, 3) for k, v in out['frac_green_at_r'].items()})
    print("by alpha (m=1,E=1):", {k: round(v, 2) for k, v in out['by_alpha_m1_E1'].items()})
    print("by m (a=2,E=1):", {k: round(v, 2) for k, v in out['by_m_alpha2_E1'].items()})
    print("h3_calibrated check:", {k: (round(v, 3) if isinstance(v, float) else v)
                                   for k, v in out['h3_calibrated_lambda0_check'].items()})


if __name__ == "__main__":
    main()
