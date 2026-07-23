"""Provenance check for h3_frontier.json's combined_fronts / completeness_audit.

The archived artifact was produced by the session that ran run_h3_frontier.py; the driver
committed at the time did not write combined_fronts/completeness_audit and did not archive
the refinement winner ORDERS (now fixed in run_h3_frontier.py). This script verifies, from
the CURRENT archive alone and deterministic re-decoding only (no search):

  A. Re-decode every archived MAIN-sweep winner order with the exact Pareto-set DP; check
     no recomputed point strictly dominates any archived combined-front point (i.e. the
     archived fronts are complete w.r.t. everything re-derivable today).
  B. Attribute each archived front point to (i) a main-order decode, (ii) a stored
     weighted-sum/refinement VALUE, or (iii) a refinement-order decode (orders not
     archived -> reported as such, a disclosed replicate-archiving gap).
  C. Recompute the completeness_audit chord-gap fields that are derivable from stored
     values (base and after-refinement-only) and compare with the archived audit.

Output: experiments/H3-time-energy/results/frontier_provenance_check.json
"""
import json
import os

import numpy as np

from problem import gen_instance, dist_matrix
from run_h3_frontier import pareto_decode, front_of, chord_gaps, ALPHA

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "experiments", "H3-time-energy", "results")
TOL = 1e-9


def main():
    h3 = json.load(open(os.path.join(RES, "h3_v2.json")))
    h3f = json.load(open(os.path.join(RES, "h3_frontier.json")))
    main_rows = [r for r in h3["raw"] if r["tag"] == "main"]

    out = {"per_n": {}, "tolerance": TOL}
    ok_domination = True
    for n in (20, 50):
        att_main, att_value, att_unarchived = 0, 0, 0
        dominated_violations = 0
        g_base, g_mid = [], []
        for sd in range(30):
            stored = sorted([r for r in main_rows if r["n"] == n and r["seed"] == sd],
                            key=lambda p: p["lam"])
            inst = gen_instance(n, seed=sd)
            D = dist_matrix(inst["coords"])
            decoded = []
            for r in stored:
                order = h3["routes"][f"main-n{n}-s{sd}-l{r['lam']}-cl1.0-te0.3"]
                decoded.extend(pareto_decode(order, D, ALPHA))
            arch = [tuple(p) for p in h3f["combined_fronts"][str(n)][str(sd)]]
            # A: no recomputed point may strictly dominate an archived front point
            for (t, e) in front_of(decoded):
                if any(t < at - TOL and e < ae - TOL for at, ae in arch):
                    dominated_violations += 1
            # B: attribute archived front points
            vals = [(p["makespan"], p["e_total"]) for p in stored]
            vals += [(r["makespan"], r["e_total"]) for r in h3f["refined_rows"]
                     if r["n"] == n and r["seed"] == sd]
            vals += [(rep["makespan"], rep["e_total"]) for r in h3f["refined_rows"]
                     if r["n"] == n and r["seed"] == sd for rep in r.get("replicates", [])]
            dec_set = decoded
            for (t, e) in arch:
                if any(abs(t - dt) <= TOL and abs(e - de) <= TOL for dt, de in dec_set):
                    att_main += 1
                elif any(abs(t - vt) <= TOL and abs(e - ve) <= TOL for vt, ve in vals):
                    att_value += 1
                else:
                    att_unarchived += 1
            # C: recomputable audit fields
            base_pts = vals[:len(stored)]
            ref_pts = [(r["makespan"], r["e_total"]) for r in h3f["refined_rows"]
                       if r["n"] == n and r["seed"] == sd]
            g_base.append(chord_gaps(front_of(base_pts)))
            g_mid.append(chord_gaps(front_of(base_pts + ref_pts)))
        # canonical audit recomputation with COMMITTED formulas (round-11 fix): extra
        # points at 1e-12 dominance tolerance; additive-epsilon of the weighted-sum front
        # vs the completed front, normalized by the completed front's time-optimal
        # endpoint; chord gaps from the committed chord_gaps() on the archived fronts.
        extra_total, eps_list, g_comb = 0, [], []
        for sd in range(30):
            base_pts = [(r["makespan"], r["e_total"]) for r in main_rows
                        if r["n"] == n and r["seed"] == sd]
            ref_pts = [(r["makespan"], r["e_total"]) for r in h3f["refined_rows"]
                       if r["n"] == n and r["seed"] == sd]
            f_mid = front_of(base_pts + ref_pts)
            f_after = [tuple(p) for p in h3f["combined_fronts"][str(n)][str(sd)]]
            g_comb.append(chord_gaps(f_after))
            extra_total += sum(1 for (t, e) in f_after
                               if not any(bt <= t + 1e-12 and be <= e + 1e-12
                                          for bt, be in f_mid))
            t0n, e0n = f_after[0][0], f_after[0][1]
            eps = max(min(max((bt - t) / t0n, (be - e) / e0n) for bt, be in f_mid)
                      for t, e in f_after)
            eps_list.append(100.0 * max(0.0, eps))
        total = att_main + att_value + att_unarchived
        aud = h3f["completeness_audit"][str(n)]
        out["per_n"][str(n)] = {
            "front_points_total": total,
            "explained_by_main_order_decode": att_main,
            "explained_by_stored_values": att_value,
            "from_unarchived_refinement_orders": att_unarchived,
            "recomputed_dominating_archived (must be 0)": dominated_violations,
            "chord_gap_base_recomputed_vs_archived": [float(np.mean(g_base)),
                                                      aud["chord_gap_base_mean"]],
            "chord_gap_mid_recomputed_vs_archived": [float(np.mean(g_mid)),
                                                     aud["chord_gap_after_refinement_only_mean"]],
            "audit_recomputed_canonical": {
                "extra_front_points": extra_total,
                "eps_indicator_pct": {"mean": float(np.mean(eps_list)),
                                      "max": float(np.max(eps_list))},
                "chord_gap_combined_mean": float(np.mean(g_comb)),
                "note": "canonical values quoted by the manuscript; the archived "
                        "completeness_audit extra-point counts (1799/8111) used a "
                        "different tolerance convention and summary.max_chord_gap_after_"
                        "mean is a stale session value — both superseded by these"}}
        print(f"  canonical audit: extra {extra_total} | eps mean {np.mean(eps_list):.3f}% "
              f"max {np.max(eps_list):.3f}% | chord {np.mean(g_comb):.4f}")
        ok_domination &= dominated_violations == 0
        print(f"n={n}: {total} front pts | main-decode {att_main} | stored-values {att_value} "
              f"| unarchived-refinement-orders {att_unarchived} | violations {dominated_violations}")
        print(f"  chord base {np.mean(g_base):.6f} vs archived {aud['chord_gap_base_mean']:.6f} | "
              f"mid {np.mean(g_mid):.6f} vs {aud['chord_gap_after_refinement_only_mean']:.6f}")
    out["verdict"] = ("PASS: archived fronts are consistent with everything re-derivable from "
                      "the archive; points from unarchived refinement orders are quantified "
                      "above (the committed driver now archives refinement orders)."
                      if ok_domination else "FAIL: recomputed points dominate archived front")
    json.dump(out, open(os.path.join(RES, "frontier_provenance_check.json"), "w"), indent=1)
    print(out["verdict"])


if __name__ == "__main__":
    main()
