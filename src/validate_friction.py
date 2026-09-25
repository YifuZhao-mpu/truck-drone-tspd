"""Independent validation gates for the operational-friction split decoder.

This script is intentionally separate from the experiment driver.  It checks (1) an
exact zero-friction regression to the submitted kernel, (2) an exhaustive pure-Python
enumeration with friction enabled, and (3) the all-ineligible truck-only identity.
"""
from functools import lru_cache
import json
import os

import numpy as np

from problem import (dist_matrix, gen_instance, tspd_cost, tspd_cost_friction,
                     tspd_split, tspd_split_friction)
from alns import alns
from revision_utils import git_commit


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "experiments", "VALIDATION", "results",
                   "friction_validation.json")


def exhaustive_fixed_order(order, Dtr, Ddr, alpha, endurance, max_span,
                           t_service, t_launch, t_recover, eligible):
    """Enumerate every feasible operation sequence for one fixed visiting order."""
    seq = list(map(int, order))
    L = len(seq) - 1

    @lru_cache(None)
    def rec(i):
        if i == L:
            return 0.0
        k = i + 1
        base = Dtr[seq[i], seq[k]] + (t_service if seq[k] != 0 else 0.0)
        ans = base + rec(k)
        for k in range(i + 2, min(L, i + max_span) + 1):
            for j in range(i + 1, k):
                customer = seq[j]
                if not eligible[customer]:
                    continue
                flight = Ddr[seq[i], customer] + Ddr[customer, seq[k]]
                if flight > endurance:
                    continue
                kept = [p for p in range(i, k + 1) if p != j]
                truck = sum(Dtr[seq[kept[a]], seq[kept[a + 1]]]
                            for a in range(len(kept) - 1))
                truck += t_service * sum(seq[p] != 0 for p in kept[1:])
                drone = flight / alpha + t_service
                op = t_launch + max(truck, drone) + t_recover
                ans = min(ans, op + rec(k))
        return ans

    return rec(0)


def evaluate_operations(order, ops, Dtr, Ddr, alpha, t_service, t_launch, t_recover):
    """Independent event evaluation of an explicitly reconstructed operation list."""
    total = 0.0
    for op in ops:
        if op[0] == "truck":
            k = op[2]
            total += Dtr[order[op[1]], order[k]]
            total += t_service if order[k] != 0 else 0.0
            continue
        i, j, k = op[1], op[2], op[3]
        kept = [p for p in range(i, k + 1) if p != j]
        truck = sum(Dtr[order[kept[a]], order[kept[a + 1]]]
                    for a in range(len(kept) - 1))
        truck += t_service * sum(order[p] != 0 for p in kept[1:])
        drone = (Ddr[order[i], order[j]] + Ddr[order[j], order[k]]) / alpha
        drone += t_service
        total += t_launch + max(truck, drone) + t_recover
    return total


def arrival_relations(order, D, alpha, t_service):
    """Return whether candidate sorties cover drone arrival before/during/after service."""
    seen = {"before": False, "during": False, "after": False}
    L = len(order) - 1
    for i in range(L):
        for k in range(i + 2, L + 1):
            for j in range(i + 1, k):
                kept = [p for p in range(i, k + 1) if p != j]
                travel = sum(D[order[kept[a]], order[kept[a + 1]]]
                             for a in range(len(kept) - 1))
                services = t_service * sum(order[p] != 0 for p in kept[1:])
                drone_arrival = (D[order[i], order[j]] + D[order[j], order[k]]) / alpha
                if drone_arrival <= travel:
                    seen["before"] = True
                elif drone_arrival < travel + services:
                    seen["during"] = True
                else:
                    seen["after"] = True
    return seen


def main():
    rng = np.random.default_rng(20260901)
    zero_checks = []
    for q in range(240):
        n = (10, 20, 50)[q % 3]
        inst = gen_instance(n, seed=1000 + q)
        D = dist_matrix(inst["coords"])
        order = [0] + list(map(int, rng.permutation(np.arange(1, n + 1)))) + [0]
        alpha = (1.0, 1.5, 2.0, 2.5, 3.0)[q % 5]
        endurance = (0.5, 1.0, 2.0, np.inf)[q % 4]
        span = (4, 6, 12, n + 1)[q % 4]
        old, old_ops = tspd_split(order, D, alpha, endurance, span)
        new, new_ops = tspd_split_friction(
            order, D, alpha, endurance, span, t_service=0.0, t_launch=0.0,
            t_recover=0.0, eligible=np.ones(n + 1, dtype=bool))
        fast = tspd_cost_friction(
            order, D, alpha, endurance, span, t_service=0.0, t_launch=0.0,
            t_recover=0.0, eligible=np.ones(n + 1, dtype=bool))
        assert old == new == fast
        assert old_ops == new_ops
        zero_checks.append(abs(old - new))

    exhaustive_checks = []
    relation_coverage = {"before": False, "during": False, "after": False}
    friction_cases = (
        (0.0, 0.02, 0.0),       # launch only
        (0.0, 0.0, 0.025),      # recovery only
        (0.0, 0.011, 0.037),    # asymmetric handling
        (0.05, 0.0, 0.0),       # service only
        (0.03, 0.007, 0.021),   # joint, asymmetric
        (0.07, 0.02, 0.02),     # joint, symmetric
    )
    for q in range(360):
        n = 4 + q % 6
        inst = gen_instance(n, seed=4000 + q)
        D = dist_matrix(inst["coords"])
        order = [0] + list(map(int, rng.permutation(np.arange(1, n + 1)))) + [0]
        alpha = (1.0, 1.5, 2.0, 3.0)[q % 4]
        endurance = (0.5, 0.8, 1.0, np.inf)[q % 4]
        span = min(n + 1, (4, 6, 12)[q % 3])
        t_service, t_launch, t_recover = friction_cases[q % len(friction_cases)]
        eligible = rng.random(n + 1) < (0.5, 0.75, 1.0)[q % 3]
        eligible[0] = True
        ref = exhaustive_fixed_order(
            order, D, D, alpha, endurance, span, t_service, t_launch,
            t_recover, eligible)
        got, _ = tspd_split_friction(
            order, D, alpha, endurance, span, t_service=t_service,
            t_launch=t_launch, t_recover=t_recover, eligible=eligible)
        fast = tspd_cost_friction(
            order, D, alpha, endurance, span, t_service=t_service,
            t_launch=t_launch, t_recover=t_recover, eligible=eligible)
        err = max(abs(ref - got), abs(ref - fast))
        assert err < 1e-9, (q, ref, got, fast)
        exhaustive_checks.append(err)
        rel = arrival_relations(order, D, alpha, max(t_service, 0.03))
        for key in relation_coverage:
            relation_coverage[key] = relation_coverage[key] or rel[key]
    assert all(relation_coverage.values()), relation_coverage

    monotonic_checks = []
    for q in range(120):
        n = 5 + q % 5
        inst = gen_instance(n, seed=7000 + q)
        D = dist_matrix(inst["coords"])
        order = [0] + list(map(int, rng.permutation(np.arange(1, n + 1)))) + [0]
        priority = rng.permutation(np.arange(1, n + 1))
        masks = []
        for fraction in (1.0, 0.75, 0.5):
            mask = np.zeros(n + 1, dtype=bool)
            mask[0] = True
            mask[priority[:int(np.ceil(fraction * n))]] = True
            masks.append(mask)
        zero = tspd_cost_friction(order, D, 2.0, 1.0, n + 1, eligible=masks[0])
        friction = tspd_cost_friction(order, D, 2.0, 1.0, n + 1,
                                      t_service=0.03, t_launch=0.01,
                                      t_recover=0.02, eligible=masks[0])
        elig75 = tspd_cost_friction(order, D, 2.0, 1.0, n + 1, eligible=masks[1])
        elig50 = tspd_cost_friction(order, D, 2.0, 1.0, n + 1, eligible=masks[2])
        assert friction >= zero - 1e-12
        assert elig75 >= zero - 1e-12 and elig50 >= elig75 - 1e-12
        monotonic_checks.append(min(friction - zero, elig75 - zero, elig50 - elig75))

    ineligible_checks = []
    for q in range(90):
        n = (8, 20, 50)[q % 3]
        inst = gen_instance(n, seed=8000 + q)
        D = dist_matrix(inst["coords"])
        order = [0] + list(map(int, rng.permutation(np.arange(1, n + 1)))) + [0]
        service = (0.0, 0.02, 0.08)[q % 3]
        eligible = np.zeros(n + 1, dtype=bool)
        got, ops = tspd_split_friction(
            order, D, 2.0, 1.0, n + 1, t_service=service,
            t_launch=0.03, t_recover=0.04, eligible=eligible)
        ref = sum(D[order[a], order[a + 1]] for a in range(len(order) - 1)) + n * service
        err = abs(got - ref)
        assert err < 1e-9
        assert all(op[0] == "truck" for op in ops)
        ineligible_checks.append(err)

    full_path_checks = []
    for q in range(18):
        n = 8 + q % 3
        inst = gen_instance(n, seed=12000 + q)
        D = dist_matrix(inst["coords"])
        eligible = np.ones(n + 1, dtype=bool)
        eligible[1 + q % n] = False
        kwargs = {"t_service": 0.03, "t_launch": 0.011,
                  "t_recover": 0.027, "eligible": eligible}
        result = alns(inst, D, 2.0, endurance=1.0, max_span=6, final_span=n + 1,
                      iters=250, seed=q % 3, **kwargs)
        order = result["order"]
        search_fast = tspd_cost_friction(order, D, 2.0, 1.0, 6, **kwargs)
        search_full, search_ops = tspd_split_friction(order, D, 2.0, 1.0, 6, **kwargs)
        final_fast = tspd_cost_friction(order, D, 2.0, 1.0, n + 1, **kwargs)
        final_full, final_ops = tspd_split_friction(order, D, 2.0, 1.0, n + 1, **kwargs)
        search_event = evaluate_operations(order, search_ops, D, D, 2.0, 0.03, 0.011, 0.027)
        final_event = evaluate_operations(order, final_ops, D, D, 2.0, 0.03, 0.011, 0.027)
        values = (search_fast, search_full, search_event, result["search_objective"])
        assert max(values) - min(values) < 1e-9, values
        values = (final_fast, final_full, final_event, result["makespan"])
        assert max(values) - min(values) < 1e-9, values
        full_path_checks.append(max(abs(search_fast - search_event),
                                    abs(final_fast - final_event)))

    out = {
        "status": "PASS",
        "git_commit": git_commit(os.path.join(HERE, "..")),
        "rng_seed": 20260901,
        "zero_friction": {"n": len(zero_checks), "max_abs_error": max(zero_checks)},
        "exhaustive_friction": {"n": len(exhaustive_checks),
                                "max_abs_error": max(exhaustive_checks),
                                "arrival_relation_coverage": relation_coverage,
                                "case_types": ["launch-only", "recovery-only",
                                               "asymmetric handling", "service-only",
                                               "joint", "partial eligibility"]},
        "monotonicity": {"n": len(monotonic_checks),
                         "minimum_nonnegative_contrast": min(monotonic_checks)},
        "all_ineligible": {"n": len(ineligible_checks),
                             "max_abs_error": max(ineligible_checks)},
        "full_solver_path": {"n": len(full_path_checks),
                             "max_abs_error": max(full_path_checks)},
        "semantics": "service charged once to serving vehicle; rendezvous service may overlap flight; "
                     "launch/recovery bracket synchronization; endurance remains distance based",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
