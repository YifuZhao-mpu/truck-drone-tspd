"""
P0 robustness round (reviewer-facing external validity):
 A. Distribution x distance-metric robustness: clustered demand + road-network (Manhattan)
    truck distance (drone stays Euclidean). Full alpha x E saving surface for the most
    realistic setting (clustered + Manhattan) at n=50, plus a cross-setting comparison.
 B. Scale to n=100: H1 savings + sortie/generic ablation, and the alpha x E surface.
 C. Energy-model payload sensitivity: H3 time-energy knee at 3 payload levels.
Outputs experiments/P0-robustness/results/p0.json
"""
import os, json, time
import numpy as np
from multiprocessing import Pool
from problem import gen_instance, gen_clustered, dist_matrix
from baselines import truck_only_tsp
from alns import alns, alns_energy

OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "P0-robustness", "results", "p0.json")
ALPHAS = [1.0, 1.5, 2.0, 2.5, 3.0]
ENDUR = [0.5, 1.0, 2.0, 1e18]
ELAB = ["0.5", "1.0", "2.0", "inf"]
LAMBDAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 8.0]
PAYLOADS = [0.5, 1.0, 2.0]                 # cl = a*payload (a=1); cb (base) = 1.0


def make_inst(kind, n, seed):
    return gen_clustered(n, seed=seed) if kind == "clustered" else gen_instance(n, seed=seed)


def matrices(inst, truck_metric):
    Dtr = dist_matrix(inst["coords"], truck_metric)
    Ddr = dist_matrix(inst["coords"], "euclidean")
    return Dtr, Ddr


def best_saving(inst, Dtr, Ddr, alpha, E, iters, nseed):
    _, truck = truck_only_tsp(inst, Dtr, restarts=4, seed=0)
    best = min(alns(inst, Dtr, alpha, endurance=E, max_span=10, iters=iters,
                    sortie_aware=False, seed=s, Ddr=Ddr)["makespan"] for s in range(nseed))
    return (truck - best) / truck * 100, truck, best


# ---- workers ----
def job_grid(args):
    kind, tm, n, seed, ai, ei, iters = args
    inst = make_inst(kind, n, seed); Dtr, Ddr = matrices(inst, tm)
    sv, _, _ = best_saving(inst, Dtr, Ddr, ALPHAS[ai], ENDUR[ei], iters, 1)
    return ("grid", kind, tm, n, seed, ai, ei, sv)


def job_cell(args):
    kind, tm, n, seed, iters = args               # alpha=2, E=1.0 cross-setting cell
    inst = make_inst(kind, n, seed); Dtr, Ddr = matrices(inst, tm)
    sv, _, _ = best_saving(inst, Dtr, Ddr, 2.0, 1.0, iters, 1)
    return ("cell", kind, tm, n, seed, sv)


def job_ablation(args):
    n, seed, iters = args                          # uniform-euclid, n=100 sortie vs generic
    inst = gen_instance(n, seed=seed); D = dist_matrix(inst["coords"])
    _, truck = truck_only_tsp(inst, D, restarts=4, seed=seed)
    sortie = min(alns(inst, D, 2.0, endurance=np.inf, max_span=12, iters=iters,
                      sortie_aware=True, seed=s)["makespan"] for s in range(2))
    generic = min(alns(inst, D, 2.0, endurance=np.inf, max_span=12, iters=iters,
                       sortie_aware=False, seed=s)["makespan"] for s in range(2))
    return ("abl", n, seed, truck, sortie, generic)


def job_energy(args):
    n, seed, li, pi, iters = args
    inst = gen_instance(n, seed=seed); D = dist_matrix(inst["coords"])
    cl = PAYLOADS[pi]
    ms, e = alns_energy(inst, D, 2.0, LAMBDAS[li], endurance=np.inf, max_span=12,
                        iters=iters, seed=0, cl=cl, cb=1.0, te=0.3)
    return ("ener", n, seed, li, pi, ms, e)


def knee(curve):
    x = np.array([c[0] for c in curve]); y = np.array([c[1] for c in curve])
    p0 = np.array([x[0], y[0]]); p1 = np.array([x[-1], y[-1]]); ch = p1 - p0
    cl_ = np.linalg.norm(ch) + 1e-12
    d = [abs(np.cross(ch, np.array([x[i], y[i]]) - p0)) / cl_ for i in range(len(x))]
    i = int(np.argmax(d))
    return {"makespan_increase_pct": float((x[i] - 1) * 100),
            "energy_reduction_pct": float((1 - y[i]) * 100)}


def main():
    t0 = time.time()
    SETTINGS = [("uniform", "euclidean"), ("clustered", "euclidean"),
                ("uniform", "manhattan"), ("clustered", "manhattan")]
    # A1: full grid for clustered+manhattan at n=50 (12 inst)
    gridA = [("clustered", "manhattan", 50, s, ai, ei, 6000)
             for s in range(12) for ai in range(5) for ei in range(4)]
    # B2: full grid uniform+euclid at n=100 (6 inst)
    gridB = [("uniform", "euclidean", 100, s, ai, ei, 7000)
             for s in range(6) for ai in range(5) for ei in range(4)]
    # A2: cross-setting cell (alpha=2,E=1.0) at n=50 and n=100 (12 inst)
    cells = [(k, m, n, s, 6000 if n == 50 else 7000)
             for (k, m) in SETTINGS for n in (50, 100) for s in range(12)]
    # B1: n=100 ablation (10 inst)
    abls = [(100, s, 7000) for s in range(10)]
    # C: energy payload sweep n=50 (8 inst, best-of-1)
    eners = [(50, s, li, pi, 5000) for s in range(8) for li in range(len(LAMBDAS)) for pi in range(len(PAYLOADS))]

    with Pool(38) as pool:
        rGrid = pool.map(job_grid, gridA + gridB)
        rCell = pool.map(job_cell, cells)
        rAbl = pool.map(job_ablation, abls)
        rEner = pool.map(job_energy, eners)

    res = {"config": {"alphas": ALPHAS, "endurance": ELAB, "lambdas": LAMBDAS,
                      "payloads": PAYLOADS, "runtime_s": time.time() - t0}}

    # A1 + B2 surfaces
    res["surfaces"] = {}
    for (kind, tm, n) in [("clustered", "manhattan", 50), ("uniform", "euclidean", 100)]:
        surf = {}
        for ai, a in enumerate(ALPHAS):
            row = {}
            for ei, el in enumerate(ELAB):
                sv = [r[7] for r in rGrid if r[1] == kind and r[2] == tm and r[3] == n and r[5] == ai and r[6] == ei]
                row[el] = {"mean_saving": float(np.mean(sv)), "std": float(np.std(sv))}
            surf[f"alpha_{a}"] = row
        res["surfaces"][f"{kind}_{tm}_n{n}"] = surf

    # A2 cross-setting cell
    res["cross_setting_cell_a2_e1"] = {}
    for (k, m) in SETTINGS:
        for n in (50, 100):
            sv = [r[5] for r in rCell if r[1] == k and r[2] == m and r[3] == n]
            res["cross_setting_cell_a2_e1"][f"{k}_{m}_n{n}"] = {
                "mean_saving": float(np.mean(sv)), "std": float(np.std(sv)), "count": len(sv)}

    # B1 ablation n=100
    ss = np.array([r[4] for r in rAbl]); gg = np.array([r[5] for r in rAbl])
    tr = np.array([r[3] for r in rAbl]); paired = ss - gg
    res["ablation_n100"] = {
        "saving_sortie_mean": float(((tr - ss) / tr * 100).mean()),
        "sortie_mean": float(ss.mean()), "generic_mean": float(gg.mean()),
        "paired_sortie_minus_generic": float(paired.mean()),
        "sortie_wins": int((paired < -1e-9).sum()), "generic_wins": int((paired > 1e-9).sum()),
        "ties": int((np.abs(paired) <= 1e-9).sum()), "count": len(rAbl)}

    # C energy payload knees
    res["energy_payload"] = {}
    for pi, pl in enumerate(PAYLOADS):
        # average normalized curve across instances
        seeds = sorted(set(r[2] for r in rEner if r[4] == pi))
        norm = {li: {"ms": [], "e": []} for li in range(len(LAMBDAS))}
        for sd in seeds:
            rows = [r for r in rEner if r[4] == pi and r[2] == sd]
            ms0 = [r[5] for r in rows if r[3] == 0][0]; e0 = [r[6] for r in rows if r[3] == 0][0]
            for r in rows:
                norm[r[3]]["ms"].append(r[5] / ms0)
                norm[r[3]]["e"].append(r[6] / e0 if e0 > 1e-9 else 0.0)
        curve = [(float(np.mean(norm[li]["ms"])), float(np.mean(norm[li]["e"]))) for li in range(len(LAMBDAS))]
        res["energy_payload"][f"payload_{pl}"] = {"curve": curve, "knee": knee(curve)}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"P0 done in {time.time()-t0:.0f}s -> {OUT}")
    print("CROSS-SETTING (alpha=2,E=1.0):", json.dumps(res["cross_setting_cell_a2_e1"], indent=2))
    print("ABLATION n=100:", json.dumps(res["ablation_n100"], indent=2))
    print("ENERGY payload knees:", json.dumps({k: v["knee"] for k, v in res["energy_payload"].items()}, indent=2))
    print("clustered+manhattan n=50 surface:", json.dumps(res["surfaces"]["clustered_manhattan_n50"], indent=2))
    print("uniform n=100 surface:", json.dumps(res["surfaces"]["uniform_euclidean_n100"], indent=2))


if __name__ == "__main__":
    main()
