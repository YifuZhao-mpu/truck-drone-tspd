"""
Certified-quality truck-only TSP reference via LKH-3 (Helsgaun).

The review of submission-v5 showed the internal NN+2-opt/Or-opt truck_only_tsp is
suboptimal (its routes were beaten by zero-drone ALNS routes on 12/12 n=50 instances),
inflating every reported saving. All savings denominators are therefore computed here:
LKH-3 with multiple runs on the exact instance coordinates, re-evaluated in the true
float metric, then polished by (and cross-checked against) the internal local search.

Coordinates in [0,1]^2 are scaled by 1e6 to integers for TSPLIB (EUC_2D / MAN_2D round
edge lengths to the nearest integer, a <=5e-7 relative perturbation used only for tour
*choice*; the returned length is always recomputed exactly in the float metric).

Results are cached in experiments/TSPREF/results/tspref_cache.json keyed by
(kind, n, seed, depot, metric) with a coordinate hash guard.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

SCALE = 1_000_000
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "experiments", "TSPREF",
                          "results", "tspref_cache.json")
LKH_BIN = os.environ.get("LKH_BIN", os.path.expanduser("~/.local/bin/LKH"))


def _coords_hash(coords):
    return hashlib.sha1(np.ascontiguousarray(np.round(coords * SCALE)).astype(np.int64)
                        .tobytes()).hexdigest()[:16]


def tour_length(tour, D):
    return float(sum(D[tour[a], tour[a + 1]] for a in range(len(tour) - 1)))


def _write_tsplib(path, coords, metric):
    n = coords.shape[0]
    ew = "EUC_2D" if metric == "euclidean" else "MAN_2D"
    with open(path, "w") as f:
        f.write(f"NAME : inst\nTYPE : TSP\nDIMENSION : {n}\nEDGE_WEIGHT_TYPE : {ew}\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords, start=1):
            f.write(f"{i} {int(round(x * SCALE))} {int(round(y * SCALE))}\n")
        f.write("EOF\n")


def lkh_tour(coords, metric="euclidean", runs=8, seed=1):
    """Run LKH-3, return (length_in_true_float_metric, closed tour starting/ending at 0)."""
    tmp = tempfile.mkdtemp(prefix="lkh_")
    try:
        prob = os.path.join(tmp, "p.tsp")
        parf = os.path.join(tmp, "p.par")
        tourf = os.path.join(tmp, "p.tour")
        _write_tsplib(prob, coords, metric)
        with open(parf, "w") as f:
            f.write(f"PROBLEM_FILE = {prob}\nTOUR_FILE = {tourf}\n"
                    f"RUNS = {runs}\nSEED = {seed}\nTRACE_LEVEL = 0\n")
        subprocess.run([LKH_BIN, parf], check=True, capture_output=True, timeout=600)
        tour = []
        with open(tourf) as f:
            in_sec = False
            for line in f:
                t = line.strip()
                if t == "TOUR_SECTION":
                    in_sec = True
                    continue
                if in_sec:
                    v = int(t.split()[0])
                    if v == -1:
                        break
                    tour.append(v - 1)  # TSPLIB is 1-indexed
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # rotate so tour starts at depot 0, close it
    z = tour.index(0)
    tour = tour[z:] + tour[:z] + [0]
    from problem import dist_matrix
    D = dist_matrix(coords, metric=metric)
    return tour_length(tour, D), tour


def reference_tsp(coords, metric="euclidean", runs=8, seed=1, polish=True):
    """LKH tour re-evaluated in the exact float metric, then polished with the internal
    2-opt/Or-opt under that metric (never worse than raw LKH). Returns (length, tour)."""
    from problem import dist_matrix
    from baselines import two_opt, or_opt
    length, tour = lkh_tour(coords, metric=metric, runs=runs, seed=seed)
    if polish:
        D = dist_matrix(coords, metric=metric)
        pol = or_opt(two_opt(tour, D), D)
        pl = tour_length(pol, D)
        if pl < length - 1e-12:
            length, tour = pl, pol
    return length, tour


def _load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH + ".tmp", "w") as f:
        json.dump(cache, f)
    os.replace(CACHE_PATH + ".tmp", CACHE_PATH)


def key_of(kind, n, seed, depot, metric):
    return f"{kind}-n{n}-seed{seed}-{depot}-{metric}"


def get_reference(inst, metric="euclidean", cache=None):
    """Cached certified truck-only reference length for a generated instance dict.
    Raises KeyError if absent and cache is read-only (dict passed); computes if cache is None."""
    key = key_of(inst["kind"], inst["n"], inst["seed"], inst["depot"], metric)
    ch = _coords_hash(inst["coords"])
    if cache is None:
        cache = _load_cache()
    if key in cache:
        ent = cache[key]
        if ent["coords_hash"] != ch:
            raise ValueError(f"coords hash mismatch for {key}")
        return ent["length"]
    raise KeyError(f"no TSP reference cached for {key}; run run_tspref.py first")


def reference_tsp_matrix(D, runs=8, seed=1, polish=True):
    """LKH best-found tour for an EXPLICIT symmetric distance matrix (e.g. road-network
    shortest-path distances), re-evaluated in the float matrix and polished by
    2-opt/Or-opt. Returns (length, closed tour starting/ending at 0)."""
    from baselines import two_opt, or_opt
    n = D.shape[0]
    tmp = tempfile.mkdtemp(prefix="lkhm_")
    try:
        prob = os.path.join(tmp, "p.tsp")
        parf = os.path.join(tmp, "p.par")
        tourf = os.path.join(tmp, "p.tour")
        with open(prob, "w") as f:
            f.write(f"NAME : m\nTYPE : TSP\nDIMENSION : {n}\n"
                    "EDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\n"
                    "EDGE_WEIGHT_SECTION\n")
            for i in range(n):
                f.write(" ".join(str(int(round(D[i, j] * SCALE))) for j in range(n)) + "\n")
            f.write("EOF\n")
        with open(parf, "w") as f:
            f.write(f"PROBLEM_FILE = {prob}\nTOUR_FILE = {tourf}\n"
                    f"RUNS = {runs}\nSEED = {seed}\nTRACE_LEVEL = 0\n")
        subprocess.run([LKH_BIN, parf], check=True, capture_output=True, timeout=600)
        tour = []
        with open(tourf) as f:
            in_sec = False
            for line in f:
                t = line.strip()
                if t == "TOUR_SECTION":
                    in_sec = True
                    continue
                if in_sec:
                    v = int(t.split()[0])
                    if v == -1:
                        break
                    tour.append(v - 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    z = tour.index(0)
    tour = tour[z:] + tour[:z] + [0]
    length = tour_length(tour, D)
    if polish:
        pol = or_opt(two_opt(tour, D), D)
        pl = tour_length(pol, D)
        if pl < length - 1e-12:
            length, tour = pl, pol
    return length, tour
