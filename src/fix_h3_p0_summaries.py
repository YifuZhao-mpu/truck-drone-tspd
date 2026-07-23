"""
Round-2 review fixes recomputed from STORED raw results (no solver reruns):

H3 (h3_v2.json):
 - Correct the lam=0 survival accounting (round-2 finding: the v2 counter marked lam=0 as
   surviving whenever any front point shared its makespan). Report BOTH:
     * strictly_dominated: some sweep point beats lam=0 in makespan AND energy (search noise)
     * weakly_dominated: a same-makespan lower-energy point displaced it (expected
       tie-breaking under a time-only objective)
 - Matched-control sensitivity: recompute the NOMINAL knee on exactly the sensitivity
   design (seeds 0..14, the 8-lambda grid) so coefficient comparisons are like-for-like,
   and report the nominal knee's grid sensitivity (12-lambda vs 8-lambda on same seeds).

P0 (p0_v2.json):
 - Full paired factor contrasts from raw_cells: demand, depot, metric effects for every
   combination of the other two factors and size, plus the demand x metric interaction.

Writes: experiments/H3-time-energy/results/h3_v2_fixes.json
        experiments/P0-robustness/results/p0_v2_contrasts.json
and patches the corrected lam0 accounting into h3_v2.json (summary fields only).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
H3 = os.path.join(HERE, "..", "experiments", "H3-time-energy", "results", "h3_v2.json")
P0 = os.path.join(HERE, "..", "experiments", "P0-robustness", "results", "p0_v2.json")
OUT_H3 = os.path.join(HERE, "..", "experiments", "H3-time-energy", "results", "h3_v2_fixes.json")
OUT_P0 = os.path.join(HERE, "..", "experiments", "P0-robustness", "results", "p0_v2_contrasts.json")

LAMBDAS_SENS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.65, 1.5, 8.0]


def front_of(pts):
    pts = sorted(pts, key=lambda p: (p["makespan"], p["e_total"]))
    out, be = [], float("inf")
    for p in pts:
        if p["e_total"] < be - 1e-12:
            out.append(p)
            be = p["e_total"]
    return out


def knee_of(front):
    if len(front) < 3:
        return None
    x = np.array([p["makespan"] / front[0]["makespan"] for p in front])
    y = np.array([p["e_total"] / front[0]["e_total"] for p in front])
    chord = np.array([x[-1] - x[0], y[-1] - y[0]])
    cl = np.linalg.norm(chord) + 1e-12
    dxy = np.stack([x, y], 1) - np.array([x[0], y[0]])
    d = np.abs(chord[0] * dxy[:, 1] - chord[1] * dxy[:, 0]) / cl
    ki = int(np.argmax(d))
    return (x[ki] - 1) * 100, (1 - y[ki]) * 100


def main():
    h3 = json.load(open(H3))
    fixes = {"lam0_accounting": {}, "matched_sensitivity": {}}
    for n in (20, 50):
        strict = weak = 0
        for sd in range(30):
            pts = [r for r in h3["raw"] if r["tag"] == "main" and r["n"] == n
                   and r["seed"] == sd]
            lam0 = [p for p in pts if p["lam"] == 0.0][0]
            if any(p["makespan"] < lam0["makespan"] - 1e-12
                   and p["e_total"] < lam0["e_total"] - 1e-12 for p in pts):
                strict += 1
            fr = front_of(pts)
            if not any(f["lam"] == 0.0 for f in fr):
                weak += 1
        fixes["lam0_accounting"][str(n)] = {
            "strictly_dominated": strict,
            "displaced_from_front_incl_ties": weak,
            "note": "strict = a sweep point beats lam=0 in BOTH makespan and energy "
                    "(search noise); displaced = includes same-makespan lower-energy "
                    "points (expected tie-breaking under a time-only objective)"}
        # patch the summary in h3_v2.json
        h3["main"][str(n)]["lam0_strictly_dominated"] = strict
        h3["main"][str(n)]["lam0_displaced_from_front"] = weak
        h3["main"][str(n)]["lam0_dominated_count_v2_BUGGY"] = \
            h3["main"][str(n)].pop("lam0_dominated_count")

    # matched-control sensitivity: nominal knee on seeds 0..14 under both grids
    for n in (50,):
        for grid, label in ((None, "nominal_12lam_seeds15"),
                            (LAMBDAS_SENS, "nominal_8lam_seeds15")):
            knees = []
            for sd in range(15):
                pts = [r for r in h3["raw"] if r["tag"] == "main" and r["n"] == n
                       and r["seed"] == sd and (grid is None or r["lam"] in grid)]
                k = knee_of(front_of(pts))
                if k:
                    knees.append(k)
            dms = np.array([k[0] for k in knees])
            de = np.array([k[1] for k in knees])
            fixes["matched_sensitivity"][label] = {
                "knee_dms_median": float(np.median(dms)),
                "knee_de_median": float(np.median(de)),
                "n_knees": len(knees)}
    json.dump(h3, open(H3, "w"),
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    json.dump(fixes, open(OUT_H3, "w"), indent=2)
    print("H3 fixes:", json.dumps(fixes, indent=1))

    # ---- P0 contrasts ----
    p0 = json.load(open(P0))
    rows = p0["raw_cells"]

    def cellmap(kind, depot, metric, n):
        return {r["seed"]: r["saving_pct"] for r in rows if r["kind"] == kind
                and r["depot"] == depot and r["metric"] == metric and r["n"] == n}

    contrasts = {"demand": {}, "depot": {}, "metric": {}, "demand_x_metric": {}}
    for n in (50, 100):
        for depot in ("center", "corner"):
            for metric in ("euclidean", "manhattan"):
                u, c = cellmap("uniform", depot, metric, n), cellmap("clustered", depot, metric, n)
                d = [u[s] - c[s] for s in u]
                contrasts["demand"][f"{depot}_{metric}_n{n}"] = {
                    "mean": float(np.mean(d)),
                    "ci95": [float(np.percentile(np.random.default_rng(0).choice(
                        d, size=(20000, len(d))).mean(1), q)) for q in (2.5, 97.5)]}
        for kind in ("uniform", "clustered"):
            for metric in ("euclidean", "manhattan"):
                ce, co = cellmap(kind, "center", metric, n), cellmap(kind, "corner", metric, n)
                d = [ce[s] - co[s] for s in ce]
                contrasts["depot"][f"{kind}_{metric}_n{n}"] = {"mean": float(np.mean(d))}
            for depot in ("center", "corner"):
                eu, ma = cellmap(kind, depot, "euclidean", n), cellmap(kind, depot, "manhattan", n)
                d = [ma[s] - eu[s] for s in eu]
                contrasts["metric"][f"{kind}_{depot}_n{n}"] = {"mean": float(np.mean(d))}
        # demand x metric interaction (center depot): (uniformM - uniformE) - (clusteredM - clusteredE)
        for depot in ("center", "corner"):
            ue, um = cellmap("uniform", depot, "euclidean", n), cellmap("uniform", depot, "manhattan", n)
            ce_, cm = cellmap("clustered", depot, "euclidean", n), cellmap("clustered", depot, "manhattan", n)
            d = [(um[s] - ue[s]) - (cm[s] - ce_[s]) for s in ue]
            contrasts["demand_x_metric"][f"{depot}_n{n}"] = {"mean": float(np.mean(d))}
    json.dump(contrasts, open(OUT_P0, "w"), indent=2)
    dm = [v["mean"] for v in contrasts["demand"].values()]
    dp = [v["mean"] for v in contrasts["depot"].values()]
    mt = [v["mean"] for v in contrasts["metric"].values()]
    print(f"demand effect range: {min(dm):.2f}..{max(dm):.2f} pt")
    print(f"depot effect range: {min(dp):.2f}..{max(dp):.2f} pt")
    print(f"metric effect range: {min(mt):.2f}..{max(mt):.2f} pt")
    print("demand x metric:", {k: round(v['mean'], 2) for k, v in contrasts['demand_x_metric'].items()})


if __name__ == "__main__":
    main()
