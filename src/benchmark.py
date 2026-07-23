"""
Loader for the Agatz/Bouman public TSP-D instances (github.com/pcbouman-eur/TSP-D-Instances).
Validates our makespan model against their published exact optima (DP solutions, n<=17).

Instance grammar: truck cost factor, drone cost factor, num nodes (incl depot), then
`x y name` per node (depot first). Drone speed factor alpha = truck_factor / drone_factor.
Solution grammar: num operations, then `start end fly #internal [internal ids...]` per op.
"""
import os
import re
import numpy as np


def _nums(line):
    return re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)


def strip_comments(text):
    return re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)


def load_instance(path):
    raw = open(path).read()
    # capture #MAXFLY (endurance) before stripping
    maxfly = None
    mm = re.search(r"#MAXFLY\s+([\d.]+)", raw)
    if mm:
        maxfly = float(mm.group(1))
    txt = strip_comments(raw)
    # remove directive lines
    txt = "\n".join(l for l in txt.splitlines() if not l.strip().startswith("#"))
    toks = txt.split()
    truck_f = float(toks[0]); drone_f = float(toks[1]); nnodes = int(float(toks[2]))
    rest = toks[3:]
    coords = []
    i = 0
    while len(coords) < nnodes and i + 1 < len(rest):
        try:
            x = float(rest[i]); y = float(rest[i + 1])
        except ValueError:
            i += 1
            continue
        coords.append((x, y))
        i += 3  # x y name
    coords = np.array(coords[:nnodes])
    alpha = truck_f / drone_f
    return {"coords": coords, "n": nnodes - 1, "alpha": alpha,
            "endurance": (maxfly / drone_f * drone_f) if False else (maxfly if maxfly else np.inf),
            "truck_f": truck_f, "drone_f": drone_f, "path": path}


def load_solution(path):
    txt = strip_comments(open(path).read())
    total = None
    m = re.search(r"Total cost\s*:?\s*([\d.]+)", open(path).read())
    if m:
        total = float(m.group(1))
    lines = [l for l in txt.splitlines() if l.strip()]
    nops = int(_nums(lines[0])[0])
    ops = []
    for l in lines[1:]:
        nums = _nums(l)
        if len(nums) < 4:
            continue
        s, e, fly, ni = int(float(nums[0])), int(float(nums[1])), int(float(nums[2])), int(float(nums[3]))
        internals = [int(float(x)) for x in nums[4:4 + ni]]
        ops.append((s, e, fly, internals))
        if len(ops) >= nops:
            break
    return {"ops": ops, "total": total}


def eval_solution(inst, sol):
    """Recompute makespan of a benchmark solution with our model (truck_f, drone_f)."""
    coords = inst["coords"]; tf = inst["truck_f"]; df = inst["drone_f"]
    D = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    total = 0.0
    for (s, e, fly, internals) in sol["ops"]:
        path = [s] + internals + [e]
        tr = sum(D[path[a], path[a + 1]] for a in range(len(path) - 1)) * tf
        if fly is not None and fly > 0:
            dr = (D[s, fly] + D[fly, e]) * df
            total += max(tr, dr)
        else:
            total += tr
    return total
