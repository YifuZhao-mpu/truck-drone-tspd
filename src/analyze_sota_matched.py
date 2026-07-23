"""Matched-budget analysis, v2 (targeted-audit fixes): three-way (ours vs DPS/TSP-ep-all
vs HGA-TAC), a HARD wall-clock cutoff applied symmetrically to every side (replicates
finishing after the nominal budget are discarded), conventional 'ours X% lower'
denominators, disclosed tie tolerance, and tie decomposition. Writes sota_matched.json."""
import json
import os
import re

import numpy as np

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "SOTA", "results")
BENCH = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "external", "uniform")
TIE_TOL = 1e-9


def budget_of(iid, n):
    if iid.startswith("bench"):
        return 30.0
    return {20: 35.0, 50: 140.0, 100: 430.0}[n]


def load_julia(path):
    rows = {}
    for line in open(path):
        p = line.split()
        reps = []
        if len(p) > 4:
            for tok in p[4].split(","):
                c, t = tok.split(":")
                reps.append((float(c), float(t)))
        rows[p[0]] = reps
    return rows


def hard_best(reps, T):
    """Best cost among replicates that FINISHED within the nominal budget."""
    vals = [c for c, t in reps if t <= T + 1e-9]
    return min(vals) if vals else float("inf")


def main():
    ours_raw = json.load(open(os.path.join(OUTDIR, "ours_matched.json")))["rows"]
    ours = {}
    for r in ours_raw:
        T = budget_of(r["id"], r["n"])
        reps = [(rp["makespan"], rp["t"]) for rp in r["replicates"]]
        ours[r["id"]] = {"reps": reps, "hard": hard_best(reps, T), "T": T}
    dps = load_julia(os.path.join(OUTDIR, "theirs_matched.txt"))
    hga = load_julia(os.path.join(OUTDIR, "hgatac_matched.txt"))

    from benchmark import load_solution
    out = {"config": {
        "dps_package": "chkwon/TSPDrone.jl (DPS / TSP-ep-all of Bogyrbayeva et al.) "
                       "@ f42d27c0369dd0e8ed6fc719a1176a8131b1c3cf",
        "hgatac_package": "Sasanm88/TSPDroneHGATAC.jl (HGA-TAC of Mahmoudinazlou & Kwon) "
                          "@ 8f1f345c93ede2d7bb9345b18b6fe01880e133f8",
        "julia": "1.10.4 (env status archived in julia_env_status.txt)",
        "protocol": "identical nominal per-instance budgets (30/35/140/430s); "
                    "single-threaded; JIT-warmed; complete default-settings runs launched "
                    "until the deadline; HARD cutoff applied symmetrically in analysis "
                    "(replicates finishing after the budget discarded on every side); "
                    f"tie tolerance {TIE_TOL}",
    }, "per_size": {}, "raw": []}

    groups = {}
    for iid in ours:
        m = re.match(r"bench-(\d+)-n(\d+)", iid)
        if m:
            key = f"bench{m.group(2)}"
            opt = load_solution(f"{BENCH}/solutions/uniform-{m.group(1)}-n{m.group(2)}-DP.txt")["total"]
        else:
            key = "n" + re.match(r"uniform-n(\d+)-", iid).group(1)
            opt = None
        groups.setdefault(key, []).append((iid, opt))

    for key in sorted(groups):
        rows = []
        for iid, opt in groups[key]:
            T = ours[iid]["T"]
            rows.append({"id": iid, "opt": opt,
                         "ours": ours[iid]["hard"],
                         "dps": hard_best(dps[iid], T),
                         "hgatac": hard_best(hga[iid], T),
                         "runs": {"ours": len(ours[iid]["reps"]),
                                  "dps": len(dps[iid]), "hgatac": len(hga[iid])}})
        out["raw"] += rows
        o = np.array([r["ours"] for r in rows])
        stats = {"count": len(rows), "ours_mean": float(o.mean())}
        for rival in ("dps", "hgatac"):
            t = np.array([r[rival] for r in rows])
            wins = int((o < t - TIE_TOL).sum())
            losses = int((t < o - TIE_TOL).sum())
            ties = len(rows) - wins - losses
            ties_at_opt = 0
            if groups[key][0][1] is not None:
                for r in rows:
                    if abs(r["ours"] - r[rival]) <= TIE_TOL and \
                       abs(r["ours"] - r["opt"]) / r["opt"] < 1e-9:
                        ties_at_opt += 1
            stats[rival] = {"mean": float(t.mean()),
                            "wins": wins, "losses": losses, "ties": ties,
                            "ties_at_known_optimum": ties_at_opt,
                            "ours_lower_pct": float(np.mean((t - o) / t * 100))}
            if groups[key][0][1] is not None:
                opts = np.array([r["opt"] for r in rows])
                stats[rival]["gap_pct"] = float(np.mean((t - opts) / opts * 100))
        if groups[key][0][1] is not None:
            opts = np.array([r["opt"] for r in rows])
            stats["ours_gap_pct"] = float(np.mean((o - opts) / opts * 100))
        out["per_size"][key] = stats

    json.dump(out, open(os.path.join(OUTDIR, "sota_matched.json"), "w"), indent=1)
    tw = {r: [0, 0, 0] for r in ("dps", "hgatac")}
    for k, s in out["per_size"].items():
        line = f"{k}: ours {s['ours_mean']:.4f}"
        if "ours_gap_pct" in s:
            line += f" (gap {s['ours_gap_pct']:.2f}%)"
        for rival in ("dps", "hgatac"):
            v = s[rival]
            tw[rival][0] += v["wins"]; tw[rival][1] += v["losses"]; tw[rival][2] += v["ties"]
            line += (f" | {rival} {v['mean']:.4f}"
                     + (f" (gap {v['gap_pct']:.2f}%)" if "gap_pct" in v else "")
                     + f" {v['wins']}W/{v['losses']}L/{v['ties']}T ours {v['ours_lower_pct']:.2f}% lower")
        print(line)
    for rival in ("dps", "hgatac"):
        print(f"TOTAL vs {rival}: {tw[rival][0]}W / {tw[rival][1]}L / {tw[rival][2]}T (hard cutoff)")


if __name__ == "__main__":
    main()
