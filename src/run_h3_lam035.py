"""Round-3 fix: the nominal main grid lacked lambda=0.35 while every sensitivity arm had it,
so the 'matched 8-lambda control' had only 7 points. Run the missing nominal lambda=0.35
arm (both sizes, all 30 seeds, best-of-3) and merge into h3_v2.json raw."""
import json, os
from multiprocessing import Pool
from run_h3_v2 import job, OUT, LAMBDAS

def main():
    jobs = [("main", n, sd, 0.35, 1.0, 0.3) for n in (20, 50) for sd in range(30)]
    with Pool(36) as pool:
        rows = pool.map(job, jobs)
    d = json.load(open(OUT))
    have = {(r["tag"], r["n"], r["seed"], r["lam"], r["cl"], r["te"]) for r in d["raw"]}
    added = 0
    for r in rows:
        key = (r["tag"], r["n"], r["seed"], r["lam"], r["cl"], r["te"])
        if key not in have:
            d["raw"].append({k: r[k] for k in r if k != "order"})
            d["routes"][f"{r['tag']}-n{r['n']}-s{r['seed']}-l{r['lam']}-cl{r['cl']}-te{r['te']}"] = r["order"]
            added += 1
    if 0.35 not in d["config"]["lambdas"]:
        d["config"]["lambdas"] = sorted(d["config"]["lambdas"] + [0.35])
        d["config"]["lambda_0.35_added"] = "round-3 fix: nominal grid now contains the full sensitivity grid"
    json.dump(d, open(OUT, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"merged {added} new (n,seed) lambda=0.35 rows into h3_v2.json")

if __name__ == "__main__":
    main()
