"""Energy-model extensions requested in major revision.

1. Re-derive waiting times from every archived H2 operation and evaluate the corrected
   idle/hover/takeoff-and-landing power-ratio boundary.
2. Re-account the same time-oriented routes with empirical energy intensities reported
   by Rodrigues et al. (2022) and Stolaroff et al. (2018).

The calculation validates route accounting; it does not calibrate a deployment fleet
or prove that the archived routes are globally optimal.
"""
import gzip
import json
import os

import numpy as np

from problem import (dist_matrix, gen_instance, tspd_split,
                     tspd_split_multi3)
from revision_utils import bootstrap_ci, git_commit


HERE = os.path.dirname(os.path.abspath(__file__))
H2 = os.path.join(HERE, "..", "experiments", "H2-design-space", "results",
                  "h2_v3.json")
OUT = os.path.join(HERE, "..", "experiments", "H2-design-space", "results",
                   "energy_extensions_v3.json.gz")  # ~70 MB uncompressed; stored gzipped
TE_NOMINAL = 0.3

# MJ/km.  These paired values are taken from the cited papers' common accounting
# tables; multiplying every geometric distance by the same physical scale cancels in
# each hybrid/truck-only ratio.
EMPIRICAL_SETS = {
    "rodrigues_quadcopter_electric_van": {
        "drone_MJ_per_km": 0.08, "truck_MJ_per_km": 1.65,
        "source": "Rodrigues et al. 2022 version of record, Table 3"},
    "rodrigues_quadcopter_diesel_van": {
        "drone_MJ_per_km": 0.08, "truck_MJ_per_km": 4.90,
        "source": "Rodrigues et al. 2022 version of record, Table 3"},
    "stolaroff_quadcopter_class4_ev": {
        "drone_MJ_per_km": 0.032, "truck_MJ_per_km": 2.44,
        "source": "Stolaroff et al. 2018 (32 J/m small quadcopter; Class-4 EV 2.44 MJ/km)"},
    "stolaroff_quadcopter_class4_diesel": {
        "drone_MJ_per_km": 0.032, "truck_MJ_per_km": 7.30,
        "source": "Stolaroff et al. 2018 (32 J/m small quadcopter; Class-4 diesel 7.3 MJ/km)"},
}


def cell_first(plan_values, keys):
    """Replicate-first, instance-first aggregation (protocol global contract).

    plan_values: iterable of (n, instance_seed, cell_key, value) with value possibly
    non-finite. Returns {(n, instance): [cell means]} where a cell mean is the mean over
    its solver replicates and a cell with any non-finite replicate is dropped (counted
    by the caller), plus the number of dropped cells.
    """
    cells = {}
    for n, inst, cell, value in plan_values:
        cells.setdefault((n, inst, cell), []).append(value)
    by_instance, dropped = {}, 0
    for (n, inst, _cell), vals in cells.items():
        if all(np.isfinite(v) for v in vals):
            by_instance.setdefault((n, inst), []).append(float(np.mean(vals)))
        else:
            dropped += 1
    return by_instance, dropped


def quantiles(values):
    a = np.asarray(values, dtype=float)
    return {"median": float(np.median(a)),
            "iqr": [float(np.percentile(a, 25)), float(np.percentile(a, 75))],
            "mean": float(np.mean(a)), "min": float(np.min(a)),
            "max": float(np.max(a))}


def bootstrap_median(values, seed):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(20000, dtype=float)
    for b in range(len(draws)):
        draws[b] = np.median(rng.choice(a, size=len(a), replace=True))
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def operation_audit(order, ops, D, alpha):
    truck_dist = 0.0
    drone_dist = 0.0
    truck_wait = 0.0
    drone_wait = 0.0
    sortie_energies = []
    for op in ops:
        if op[0] == "truck":
            truck_dist += D[order[op[1]], order[op[2]]]
            continue
        i, k = op[1], op[-1]
        js = (op[2],) if op[0] == "sortie" else tuple(op[2])
        kept = [p for p in range(i, k + 1) if p not in set(js)]
        truck = sum(D[order[kept[a]], order[kept[a + 1]]]
                    for a in range(len(kept) - 1))
        drone_times = []
        for j in js:
            outbound = D[order[i], order[j]]
            inbound = D[order[j], order[k]]
            drone_dist += outbound + inbound
            drone_times.append((outbound + inbound) / alpha)
            sortie_energies.append(2.0 * outbound / alpha + inbound / alpha)
        duration = max([truck] + drone_times)
        truck_dist += truck
        truck_wait += duration - truck
        drone_wait += sum(duration - t for t in drone_times)
    return {"truck_dist": truck_dist, "drone_dist": drone_dist,
            "truck_wait": truck_wait, "drone_wait": drone_wait,
            "sortie_energies": sortie_energies}


def main():
    h2 = json.load(open(H2))
    alphas = h2["config"]["alphas"]
    endurance_values = [0.5, 1.0, 2.0, np.inf]
    audits = []
    reserve_audits = []
    all_sortie_energies = []
    reconstruction_errors = []
    energy_errors = []
    truck_distance_errors = []
    for source in h2["rows"]:
        n, seed = source["n"], source["instance_seed"]
        alpha = alphas[source["ai"]]
        endurance = endurance_values[source["ei"]]
        D = dist_matrix(gen_instance(n, seed=seed)["coords"])
        for rep in source["replicates"]:
            order = rep["order"]
            if source["fleet"] == 1:
                ms, ops = tspd_split(order, D, alpha, endurance, max_span=None)
            else:
                ms, ops = tspd_split_multi3(
                    order, D, alpha, source["fleet"], endurance,
                    max_span=min(n + 1, 14))
            reconstruction_errors.append(abs(ms - rep["makespan"]))
            audit = operation_audit(order, ops, D, alpha)
            energy_errors.append(abs(sum(audit["sortie_energies"]) - rep["e_drone"]))
            truck_distance_errors.append(abs(audit["truck_dist"] - rep["e_truck"] / TE_NOMINAL))
            all_sortie_energies.extend(audit["sortie_energies"])
            audit.update({"n": n, "instance_seed": seed, "solver_seed": rep["solver_seed"],
                          "ai": source["ai"], "ei": source["ei"],
                          "fleet": source["fleet"], "alpha": alpha,
                          "truck_ref": source["truck_ref"], "e_drone": rep["e_drone"],
                          "n_sorties": rep["n_sorties"],
                          "time_saving_pct": rep["saving_pct"],
                          "historical_winner": rep["solver_seed"]
                                               == source["winner_solver_seed"]})
            audits.append(audit)
            if source["fleet"] == 1 and np.isfinite(endurance):
                reserve_ms, reserve_ops = tspd_split(
                    order, D, alpha, 0.8 * endurance, max_span=None)
                reserve = operation_audit(order, reserve_ops, D, alpha)
                nominal_denom = source["truck_ref"] - audit["truck_dist"]
                reserve_denom = source["truck_ref"] - reserve["truck_dist"]
                reserve.update({"n": n, "instance_seed": seed,
                                "solver_seed": rep["solver_seed"],
                                "ai": source["ai"], "ei": source["ei"],
                                "fleet": 1, "alpha": alpha,
                                "original_endurance": endurance,
                                "usable_endurance": 0.8 * endurance,
                                "truck_ref": source["truck_ref"],
                                "e_drone": sum(reserve["sortie_energies"]),
                                "n_sorties": len(reserve["sortie_energies"]),
                                "makespan": reserve_ms,
                                "nominal_makespan": rep["makespan"],
                                "nominal_rstar": (rep["e_drone"] / nominal_denom
                                                  if nominal_denom > 0.0 else float("inf")),
                                "rstar": (sum(reserve["sortie_energies"]) / reserve_denom
                                          if reserve_denom > 0.0 else float("inf")),
                                "historical_winner": rep["solver_seed"]
                                                     == source["winner_solver_seed"],
                                "fixed_order": True})
                reserve_audits.append(reserve)
    if max(reconstruction_errors + energy_errors + truck_distance_errors) > 1e-9:
        raise RuntimeError(
            "route accounting gate failed: "
            f"time={max(reconstruction_errors)}, energy={max(energy_errors)}, "
            f"truck_distance={max(truck_distance_errors)}")

    median_sortie_energy = float(np.median(all_sortie_energies))
    boundary_rows = []
    boundary_summary = {}
    for idle in (0.0, 0.25, 0.5):
        for hover in (0.0, 0.5, 1.0):
            for surcharge_fraction in (0.0, 0.1):
                surcharge = surcharge_fraction * median_sortie_energy
                values = []
                nonfinite = 0
                plan_values = []
                for a in audits:
                    denom = (a["truck_ref"] - a["truck_dist"]
                             - idle * a["truck_wait"])
                    numer = (a["e_drone"] + hover * a["drone_wait"]
                             + surcharge * a["n_sorties"])
                    rstar = numer / denom if denom > 0.0 else float("inf")
                    plan_values.append((a["n"], a["instance_seed"],
                                        (a["ai"], a["ei"], a["fleet"]), rstar))
                    if np.isfinite(rstar):
                        values.append(rstar)
                    else:
                        nonfinite += 1
                    boundary_rows.append({"n": a["n"], "instance_seed": a["instance_seed"],
                                          "ai": a["ai"], "ei": a["ei"],
                                          "fleet": a["fleet"], "idle_fraction": idle,
                                          "hover_fraction": hover,
                                          "surcharge_fraction_of_median_sortie": surcharge_fraction,
                                          "rstar": rstar})
                skey = f"idle{idle}_hover{hover}_tol{surcharge_fraction}"
                by_instance, dropped_cells = cell_first(plan_values, None)
                instance_medians = [float(np.median(v)) for v in by_instance.values()]
                instance_fractions = [float(np.mean(np.asarray(v) < 6.0))
                                      for v in by_instance.values()]
                boundary_summary[skey] = {
                    "aggregation": "replicate mean within (instance, capability cell); "
                                   "instance median / fraction over cells; distribution "
                                   "and bootstrap across instances",
                    "plan_level_descriptive": quantiles(values),
                    "cells_dropped_nonfinite": dropped_cells,
                    "instance_median_rstar": quantiles(instance_medians),
                    "instance_mean_fraction_below_r6": float(np.mean(instance_fractions)),
                    "instance_mean_fraction_below_r6_ci95": bootstrap_ci(
                        instance_fractions, 40000 + int(idle * 100)
                        + int(hover * 10) + int(surcharge_fraction * 10)),
                    "nonfinite": nonfinite}

    reserve_boundary_rows = []
    reserve_boundary_summary = {}
    for idle in (0.0, 0.25, 0.5):
        for hover in (0.0, 0.5, 1.0):
            for surcharge_fraction in (0.0, 0.1):
                surcharge = surcharge_fraction * median_sortie_energy
                values = []
                nonfinite = 0
                plan_values = []
                for a in reserve_audits:
                    denom = a["truck_ref"] - a["truck_dist"] - idle * a["truck_wait"]
                    numer = (a["e_drone"] + hover * a["drone_wait"]
                             + surcharge * a["n_sorties"])
                    rstar = numer / denom if denom > 0.0 else float("inf")
                    plan_values.append((a["n"], a["instance_seed"], (a["ai"], a["ei"]), rstar))
                    if np.isfinite(rstar):
                        values.append(rstar)
                    else:
                        nonfinite += 1
                    reserve_boundary_rows.append({
                        "n": a["n"], "instance_seed": a["instance_seed"],
                        "solver_seed": a["solver_seed"], "ai": a["ai"],
                        "ei": a["ei"], "idle_fraction": idle,
                        "hover_fraction": hover,
                        "surcharge_fraction_of_median_sortie": surcharge_fraction,
                        "rstar": rstar})
                skey = f"idle{idle}_hover{hover}_tol{surcharge_fraction}"
                by_instance, dropped_cells = cell_first(plan_values, None)
                instance_medians = [float(np.median(v)) for v in by_instance.values()]
                instance_fractions = [float(np.mean(np.asarray(v) < 6.0))
                                      for v in by_instance.values()]
                reserve_boundary_summary[skey] = {
                    "aggregation": "replicate mean within (instance, cell); instance median "
                                   "/ fraction over cells; distribution across instances",
                    "plan_level_descriptive": quantiles(values),
                    "cells_dropped_nonfinite": dropped_cells,
                    "instance_median_rstar": quantiles(instance_medians),
                    "instance_mean_fraction_below_r6": float(np.mean(instance_fractions)),
                    "instance_mean_fraction_below_r6_ci95": bootstrap_ci(
                        instance_fractions, 50000 + int(idle * 100)
                        + int(hover * 10) + int(surcharge_fraction * 10)),
                    "nonfinite": nonfinite}

    # Battery reserve (protocol G.1): every m=1, finite-E solver replicate was already
    # re-decoded above at usable endurance E*(1-0.2).  Keep the customer order fixed
    # and summarize the resulting sortie reassignment without accessing the legacy v2
    # winner-route layout.
    RESERVE = 0.2
    reserve_rows = [{
        "n": a["n"], "instance_seed": a["instance_seed"],
        "solver_seed": a["solver_seed"], "ai": a["ai"], "ei": a["ei"],
        "makespan_reserve": a["makespan"],
        "makespan_nominal": a["nominal_makespan"],
        "rstar_reserve": a["rstar"],
        "rstar_nominal": a["nominal_rstar"],
        "historical_winner": a["historical_winner"],
    } for a in reserve_audits]
    reserve_deltas = [a["rstar"] - a["nominal_rstar"] for a in reserve_audits
                      if np.isfinite(a["rstar"]) and np.isfinite(a["nominal_rstar"])]
    shift_by_instance, shift_cells_dropped = cell_first(
        [(a["n"], a["instance_seed"], (a["ai"], a["ei"]),
          (a["rstar"] - a["nominal_rstar"]) if np.isfinite(a["rstar"]) else float("inf"))
         for a in reserve_audits], None)
    instance_median_shift = [float(np.median(v)) for v in shift_by_instance.values()]
    instance_mean_shift = [float(np.mean(v)) for v in shift_by_instance.values()]
    reserve_summary = {
        "usable_endurance_factor": 1.0 - RESERVE,
        "scope": "fixed-order re-decode of all archived m=1 finite-E solver "
                 "replicates; archive-conditional (order held fixed, sorties re-assigned)",
        "n_plans": len(reserve_rows),
        "aggregation": "replicate mean within (instance, cell), cells with a non-finite "
                       "reserve boundary dropped; instance median and instance mean over "
                       "cells; distribution and bootstrap across instances",
        "cells_dropped_nonfinite": shift_cells_dropped,
        "instance_median_rstar_shift": quantiles(instance_median_shift),
        "instance_mean_rstar_shift": quantiles(instance_mean_shift),
        "instance_mean_rstar_shift_ci95": bootstrap_ci(instance_mean_shift, 70001),
        "plan_level_rstar_shift_descriptive": quantiles(reserve_deltas) if reserve_deltas else None,
        "nonfinite": int(sum(1 for r in reserve_rows
                             if not np.isfinite(r["rstar_reserve"]))),
    }

    empirical_summary = {}
    empirical_rows = []
    for name, cfg in EMPIRICAL_SETS.items():
        ratios = []
        savings = []
        properties = []
        for a in audits:
            hybrid = (cfg["drone_MJ_per_km"] * a["drone_dist"]
                      + cfg["truck_MJ_per_km"] * a["truck_dist"])
            truck_only = cfg["truck_MJ_per_km"] * a["truck_ref"]
            ratio = hybrid / truck_only
            ratios.append(ratio)
            savings.append((1.0 - ratio) * 100.0)
            property_value = float(a["time_saving_pct"] > 0.0 and ratio < 1.0)
            properties.append(property_value)
            empirical_rows.append({"coefficient_set": name, "n": a["n"],
                                   "instance_seed": a["instance_seed"], "ai": a["ai"],
                                   "ei": a["ei"], "fleet": a["fleet"],
                                   "solver_seed": a["solver_seed"],
                                   "energy_ratio_to_truck_only": ratio,
                                   "energy_saving_pct": (1.0 - ratio) * 100.0,
                                   "time_saving_pct": a["time_saving_pct"],
                                   "faster_and_lower_energy": bool(property_value)})
        cluster_values = []
        cluster_magnitudes = []
        prop_by_instance, _ = cell_first(
            [(a["n"], a["instance_seed"], (a["ai"], a["ei"], a["fleet"]), properties[i])
             for i, a in enumerate(audits)], None)
        save_by_cell = {}
        for i, a in enumerate(audits):
            save_by_cell.setdefault((a["n"], a["instance_seed"], (a["ai"], a["ei"], a["fleet"])),
                                    []).append((savings[i], properties[i]))
        for n in (20, 50):
            for instance_seed in range(30):
                cluster_values.append(float(np.mean(prop_by_instance[(n, instance_seed)])))
                green_cells = [float(np.mean([s for s, _p in vals]))
                               for (cn, ci, _cell), vals in save_by_cell.items()
                               if cn == n and ci == instance_seed
                               and np.mean([p for _s, p in vals]) > 0.5]
                cluster_magnitudes.append(float(np.median(green_cells))
                                          if green_cells else float("nan"))
        finite_magnitudes = [v for v in cluster_magnitudes if np.isfinite(v)]
        winner_idx = [i for i, a in enumerate(audits) if a["historical_winner"]]
        empirical_summary[name] = {
            "source": cfg["source"],
            "truck_to_drone_distance_intensity_ratio": (
                cfg["truck_MJ_per_km"] / cfg["drone_MJ_per_km"]),
            "energy_saving_pct": quantiles(savings),
            "fraction_faster_and_lower_energy": float(np.mean(cluster_values)),
            "fraction_faster_and_lower_energy_ci95": bootstrap_ci(
                cluster_values, 60000 + list(EMPIRICAL_SETS).index(name)),
            "median_instance_energy_saving_among_green_and_fast_pct": float(
                np.median(finite_magnitudes)),
            "median_instance_energy_saving_ci95": bootstrap_median(
                finite_magnitudes, 61000 + list(EMPIRICAL_SETS).index(name)),
            "instances_with_at_least_one_green_and_fast_plan": len(finite_magnitudes),
            "historical_v2_winner_fraction_faster_and_lower_energy": float(np.mean(
                [properties[i] for i in winner_idx])),
            "aggregation": "replicate mean within (instance, capability cell); instance "
                           "fraction = mean over cells, instance magnitude = median over "
                           "green-and-fast cells; then median / median-bootstrap across 60 instances",
        }

    commit = git_commit(os.path.join(HERE, ".."))
    out = {
        "config": {"source": "all H2-v3 solver-replicate routes; no re-optimization",
                   "generalized_boundary": "(E_D+h*W_D+c*N_s)/(L0-D_T-i*W_T)",
                   "idle_fractions": [0, 0.25, 0.5],
                   "hover_fractions": [0, 0.5, 1],
                   "tol_surcharge_fractions_of_median_sortie": [0, 0.1],
                   "median_nominal_sortie_energy": median_sortie_energy,
                   "empirical_coefficient_sets": EMPIRICAL_SETS,
                   "empirical_model_lock": "experiments/REVISION/energy_models.md",
                   "distance_mapping": "1 model distance unit = 1 km; energy ratio is "
                                       "invariant to common distance scaling",
                   "bootstrap_resamples": 20000,
                   "bootstrap_unit": "(n, instance_seed)",
                   "git_commit": commit,
                   "scope": "route-accounting check, not fleet-specific calibration or LCA"},
        "reconstruction_gate": {
            "status": "PASS", "n": len(audits),
            "max_makespan_abs_error": max(reconstruction_errors),
            "max_drone_energy_abs_error": max(energy_errors),
            "max_truck_distance_abs_error": max(truck_distance_errors)},
        "nominal_route_audits": [{k: v for k, v in a.items()
                                   if k != "sortie_energies"} for a in audits],
        "boundary_summary": boundary_summary,
        "boundary_rows": boundary_rows,
        "reserve_fixed_order_summary": reserve_boundary_summary,
        "reserve_fixed_order_rows": reserve_boundary_rows,
        "reserve_fixed_order_audits": [{k: v for k, v in a.items()
                                          if k != "sortie_energies"}
                                         for a in reserve_audits],
        "reserve_summary": reserve_summary,
        "reserve_rows": reserve_rows,
        "empirical_summary": empirical_summary,
        "empirical_rows": empirical_rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"boundary_base": boundary_summary["idle0.0_hover0.0_tol0.0"],
                      "boundary_stress": boundary_summary["idle0.5_hover1.0_tol0.1"],
                      "empirical": empirical_summary}, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
