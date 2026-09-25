"""Revision figures built only from archived v3 results.

The visual design follows the major-revision requests: colour-vision-deficiency-safe
palettes, common scales, uncertainty shown at the instance level, no per-instance
spaghetti curves, and a strict separation between the route-conditioned energy
boundary and service-level choices on an archived-order Pareto set.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FormatStrFormatter, NullFormatter
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
EXPERIMENTS = os.path.join(ROOT, "experiments")
OUTPUT_DIRS = (os.path.join(ROOT, "drafts", "figures"),
               os.path.join(ROOT, "drafts", "latex", "figures"))

# Okabe--Ito colours. Cividis is used for continuous heat maps.
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
BLACK = "#222222"
GREY = "#6B7280"
LIGHT_GREY = "#E5E7EB"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_json(*parts):
    path = os.path.join(EXPERIMENTS, *parts)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"required locked result is not ready: {os.path.relpath(path, ROOT)}")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(fig, stem):
    """Save a vector master and a 600-dpi raster submission copy in both trees."""
    for directory in OUTPUT_DIRS:
        os.makedirs(directory, exist_ok=True)
        fig.savefig(os.path.join(directory, f"{stem}.pdf"), bbox_inches="tight",
                    metadata={"Creator": "src/figures_v3.py"})
        fig.savefig(os.path.join(directory, f"{stem}.png"), bbox_inches="tight",
                    dpi=600, facecolor="white")
    plt.close(fig)
    print(f"saved {stem}.pdf/.png")


def asymmetric_error(center, interval):
    return np.asarray([[center - interval[0]], [interval[1] - center]])


def design_space():
    data = load_json("H2-design-space", "results", "h2_v3.json")
    alphas = data["config"]["alphas"]
    endurance = data["config"]["endurance"]
    matrices = {}
    values = []
    for n in (20, 50):
        for fleet in (1, 2, 3):
            matrix = np.asarray([
                [data["cells"][str(n)][f"a{alpha}_E{limit}_m{fleet}"]
                 ["all_seed_mean_saving"] for limit in endurance]
                for alpha in alphas
            ])
            matrices[(n, fleet)] = matrix
            values.extend(matrix.ravel())

    vmin = np.floor(min(values) / 5.0) * 5.0
    vmax = np.ceil(max(values) / 5.0) * 5.0
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.0), sharex=True, sharey=True,
                             constrained_layout=True)
    for row_index, n in enumerate((20, 50)):
        for column_index, fleet in enumerate((1, 2, 3)):
            ax = axes[row_index, column_index]
            matrix = matrices[(n, fleet)]
            image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="cividis",
                              vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(endurance)), endurance)
            ax.set_yticks(range(len(alphas)), [f"{x:g}" for x in alphas])
            ax.set_title(f"{fleet} drone{'s' if fleet > 1 else ''}")
            if column_index == 0:
                ax.set_ylabel(f"$n={n}$\nDrone-to-truck speed ratio, $\\alpha$")
            if row_index == 1:
                ax.set_xlabel("Endurance, $E$")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    normalized = (matrix[i, j] - vmin) / max(vmax - vmin, 1e-12)
                    colour = "white" if normalized < 0.53 else BLACK
                    ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                            fontsize=8.2, color=colour, fontweight="semibold")
    colourbar = fig.colorbar(image, ax=axes, shrink=0.88, pad=0.02)
    colourbar.set_label("Mean delivery-time saving (%)")
    fig.suptitle("Capability factorial under coordinated batch dispatch", fontsize=12)
    save_figure(fig, "design_space_v3")


def fleet_interactions():
    data = load_json("H2-design-space", "results", "h2_v3.json")
    alphas = np.asarray(data["config"]["alphas"], dtype=float)
    styles = [
        ("1.0", "m1_to_2", BLUE, "o", "-", "Second drone, $E=1$"),
        ("inf", "m1_to_2", BLUE, "o", "--", "Second drone, $E=\\infty$"),
        ("1.0", "m2_to_3", VERMILION, "s", "-", "Third drone, $E=1$"),
        ("inf", "m2_to_3", VERMILION, "s", "--", "Third drone, $E=\\infty$"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.35), sharey=True,
                             constrained_layout=True)
    for ax, n in zip(axes, (20, 50)):
        for endurance, transition, colour, marker, line, label in styles:
            centers, lower, upper = [], [], []
            for alpha in alphas:
                cell = data["interactions"][str(n)][f"a{alpha}_E{endurance}"]
                center = cell[f"{transition}_points"]
                interval = cell[f"{transition}_ci95"]
                centers.append(center)
                lower.append(center - interval[0])
                upper.append(interval[1] - center)
            ax.errorbar(alphas, centers, yerr=np.asarray([lower, upper]), color=colour,
                        marker=marker, linestyle=line, linewidth=1.8, markersize=5,
                        capsize=2.5, elinewidth=1, label=label)
        ax.axhline(0, color=GREY, linewidth=0.8)
        ax.set_title(f"$n={n}$")
        ax.set_xlabel("Drone-to-truck speed ratio, $\\alpha$")
        ax.set_xticks(alphas)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Marginal delivery-time saving (percentage points)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="outside lower center", frameon=False)
    fig.suptitle("Speed changes the marginal value of additional drones", fontsize=12)
    save_figure(fig, "fleet_interactions_v3")


def _front_at_caps(points, caps):
    points = sorted({(float(t), float(e)) for t, e in points})
    t0 = points[0][0]
    e0 = min(e for t, e in points if t <= t0 * (1.0 + 1e-12))
    out = []
    for cap in caps:
        feasible = [e for t, e in points if t <= t0 * cap * (1.0 + 1e-12)]
        out.append(100.0 * (1.0 - min(feasible) / e0))
    return np.asarray(out)


def pareto_summary():
    frontier = load_json("H3-time-energy", "results", "h3_frontier_v3.json")
    caps = np.linspace(1.0, 1.30, 61)
    fig, axes = plt.subplots(1, 2, figsize=(10.9, 4.35),
                             gridspec_kw={"width_ratios": [1.55, 1]},
                             constrained_layout=True)

    ax = axes[0]
    for n, colour in ((20, BLUE), (50, VERMILION)):
        curves = np.asarray([_front_at_caps(points, caps)
                             for points in frontier["combined_fronts"][str(n)].values()])
        median = np.median(curves, axis=0)
        q25, q75 = np.percentile(curves, (25, 75), axis=0)
        ax.plot((caps - 1.0) * 100.0, median, color=colour, linewidth=2,
                label=f"$n={n}$ median")
        ax.fill_between((caps - 1.0) * 100.0, q25, q75, color=colour, alpha=0.18,
                        linewidth=0, label=f"$n={n}$ IQR")
        summary = frontier["summary"][str(n)]
        x = summary["knee_dms_median"]
        y = summary["knee_de_median"]
        x_iqr = summary["knee_dms_iqr"]
        y_iqr = summary["knee_de_iqr"]
        ax.errorbar(x, y,
                    xerr=asymmetric_error(x, x_iqr),
                    yerr=asymmetric_error(y, y_iqr),
                    fmt="D", color=colour, markerfacecolor="white", markersize=5,
                    capsize=2, linewidth=1.1, zorder=5)
    ax.set_xlim(0, 30)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Allowed increase over pool minimum time (%)")
    ax.set_ylabel("Best available total-energy reduction (%)")
    ax.set_title("(a) Median archived-pool trade-off")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = axes[1]
    levels = ("5", "10", "20")
    x = np.arange(len(levels))
    width = 0.34
    for offset, n, colour in ((-width / 2, "20", BLUE),
                              (width / 2, "50", VERMILION)):
        centers, lower, upper = [], [], []
        for level in levels:
            cell = frontier["summary"][n]["reduction_pct"][level]
            centers.append(cell["median"])
            lower.append(cell["median"] - cell["iqr"][0])
            upper.append(cell["iqr"][1] - cell["median"])
        ax.bar(x + offset, centers, width=width, color=colour, alpha=0.88,
               yerr=np.asarray([lower, upper]), capsize=3, label=f"$n={n}$")
    ax.set_xticks(x, [f"+{level}%" for level in levels])
    ax.set_xlabel("Service-time cap")
    ax.set_ylabel("Total-energy reduction (%)")
    ax.set_title("(b) Fixed service-level choices")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.suptitle("Time\u2013energy choices conditional on the archived order pool", fontsize=12)
    save_figure(fig, "pareto_v3")


def _instance_medians(rows, filters):
    """Replicate-first, instance-first: mean over the solver replicates of each
    (instance, capability cell), then the median over cells within each instance."""
    cells = defaultdict(list)
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            cells[(row["n"], row["instance_seed"], row["ai"], row["ei"],
                   row.get("fleet"))].append(float(row["rstar"]))
    groups = defaultdict(list)
    for (n, seed, _ai, _ei, _fleet), values in cells.items():
        if all(np.isfinite(v) for v in values):
            groups[(n, seed)].append(float(np.mean(values)))
    return np.asarray([np.median(values) for values in groups.values()], dtype=float)


def energy_boundary():
    energy = load_json("H2-design-space", "results", "energy_extensions_v3.json.gz")
    h2 = load_json("H2-design-space", "results", "h2_v3.json")
    rows = energy["boundary_rows"]
    alphas = np.asarray(h2["config"]["alphas"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.45),
                             gridspec_kw={"width_ratios": [1.15, 1]},
                             constrained_layout=True)

    ax = axes[0]
    for fleet, colour, marker in ((1, BLUE, "o"), (2, ORANGE, "s"),
                                   (3, GREEN, "^")):
        medians, q25s, q75s = [], [], []
        for ai, _alpha in enumerate(alphas):
            values = _instance_medians(rows, {
                "ai": ai, "ei": 1, "fleet": fleet, "idle_fraction": 0.0,
                "hover_fraction": 0.0,
                "surcharge_fraction_of_median_sortie": 0.0,
            })
            medians.append(float(np.median(values)))
            q25s.append(float(np.percentile(values, 25)))
            q75s.append(float(np.percentile(values, 75)))
        medians = np.asarray(medians)
        ax.errorbar(alphas, medians,
                    yerr=np.asarray([medians - q25s, np.asarray(q75s) - medians]),
                    marker=marker, color=colour, linewidth=1.9, markersize=5,
                    capsize=2.5, label=f"$m={fleet}$")
    ax.set_yscale("log")
    ax.set_yticks([1.5, 2, 3, 4, 6, 8])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(alphas)
    ax.set_xlabel("Drone-to-truck speed ratio, $\\alpha$")
    ax.set_ylabel("Boundary ratio, $r^*$ (median and IQR)")
    ax.set_title("(a) Nominal boundary at $E=1$")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, title="Drones")

    configurations = [
        ("Nominal", "idle0.0_hover0.0_tol0.0", "boundary_summary"),
        ("Idle 0.25", "idle0.25_hover0.0_tol0.0", "boundary_summary"),
        ("Hover 0.5", "idle0.0_hover0.5_tol0.0", "boundary_summary"),
        ("Sortie +10%", "idle0.0_hover0.0_tol0.1", "boundary_summary"),
        ("Moderate all", "idle0.25_hover0.5_tol0.1", "boundary_summary"),
        ("High all", "idle0.5_hover1.0_tol0.1", "boundary_summary"),
        ("20% reserve\n(finite $E$, $m=1$)", "idle0.0_hover0.0_tol0.0",
         "reserve_fixed_order_summary"),
    ]
    ax = axes[1]
    centers, lower, upper = [], [], []
    for label, key, source in configurations:
        cell = energy[source][key]["instance_median_rstar"]
        centers.append(cell["median"])
        lower.append(cell["median"] - cell["iqr"][0])
        upper.append(cell["iqr"][1] - cell["median"])
    y = np.arange(len(configurations))
    ax.errorbar(centers, y, xerr=np.asarray([lower, upper]), fmt="o", color=PURPLE,
                markerfacecolor="white", markeredgewidth=1.5, capsize=2.5,
                linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xticks([2.5, 3, 3.5, 4])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(y, [item[0] for item in configurations])
    ax.invert_yaxis()
    ax.set_xlabel("Boundary ratio, $r^*$ (median and IQR)")
    ax.set_title("(b) Accounting and reserve sensitivity")
    ax.grid(axis="x", alpha=0.25, which="both")
    fig.suptitle("The energy boundary is route- and accounting-dependent", fontsize=12)
    save_figure(fig, "energy_boundary_v3")


def robustness():
    data = load_json("P0-robustness", "results", "p0_v3.json")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5),
                             gridspec_kw={"width_ratios": [1.45, 1]},
                             constrained_layout=True)
    ax = axes[0]
    combinations = (("uniform", "euclidean", "Uniform\nEuclidean"),
                    ("clustered", "euclidean", "Clustered\nEuclidean"),
                    ("uniform", "manhattan", "Uniform\nrectilinear"),
                    ("clustered", "manhattan", "Clustered\nrectilinear"))
    x = np.arange(len(combinations))
    width = 0.19
    arms = ((50, "center", BLUE, 1.0), (50, "corner", SKY, 1.0),
            (100, "center", VERMILION, 1.0), (100, "corner", ORANGE, 1.0))
    for index, (n, depot, colour, alpha) in enumerate(arms):
        centers, lower, upper = [], [], []
        for kind, metric, _label in combinations:
            cell = data["deconfound_cells"][f"{kind}_{depot}_{metric}_n{n}"]
            center = cell["all_seed_mean_saving"]
            centers.append(center)
            lower.append(center - cell["all_seed_ci95"][0])
            upper.append(cell["all_seed_ci95"][1] - center)
        offset = (index - 1.5) * width
        ax.bar(x + offset, centers, width, color=colour, alpha=alpha,
               yerr=np.asarray([lower, upper]), capsize=2,
               label=f"$n={n}$, {depot} depot")
    ax.set_xticks(x, [item[2] for item in combinations])
    ax.set_ylabel("Mean delivery-time saving (%)")
    ax.set_title("(a) Crossed demand, depot and truck metric")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=7.8, loc="upper left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)      # headroom so the legend clears the bars

    alphas = (1.0, 1.5, 2.0, 2.5, 3.0)
    endurance = ("0.5", "1.0", "2.0", "inf")
    matrix = np.asarray([
        [data["surface_n100"][f"a{alpha}_E{limit}"]["all_seed_mean_saving"]
         for limit in endurance] for alpha in alphas])
    ax = axes[1]
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="cividis")
    ax.set_xticks(range(len(endurance)), endurance)
    ax.set_yticks(range(len(alphas)), [f"{x:g}" for x in alphas])
    ax.set_xlabel("Endurance, $E$")
    ax.set_ylabel("Speed ratio, $\\alpha$")
    ax.set_title("(b) Exploratory fixed-area surface, $n=100$")
    vmin, vmax = float(np.min(matrix)), float(np.max(matrix))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            norm = (matrix[i, j] - vmin) / max(vmax - vmin, 1e-12)
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                    fontsize=8, color="white" if norm < 0.53 else BLACK)
    cbar = fig.colorbar(image, ax=ax, shrink=0.86)
    cbar.set_label("Mean saving (%)")
    fig.suptitle("Synthetic robustness checks", fontsize=12)
    save_figure(fig, "robustness_v3")


def sensitivity():
    operations = load_json("P0-robustness", "results", "operational_sensitivity_v3.json")
    density = load_json("P0-robustness", "results", "density_control_v3.json")
    energy = load_json("H2-design-space", "results", "energy_extensions_v3.json.gz")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.65), constrained_layout=True)

    operation_cells = (
        ("base", "Zero friction"),
        ("service_0.25tau", "Service 0.25$\\tau$"),
        ("service_0.50tau", "Service 0.50$\\tau$"),
        ("handling_0.25tau_each", "Launch/recovery 0.25$\\tau$"),
        ("handling_0.50tau_each", "Launch/recovery 0.50$\\tau$"),
        ("joint_0.25tau", "Joint 0.25$\\tau$"),
        ("eligible_75pct", "75% eligible"),
        ("eligible_50pct", "50% eligible"),
        ("reserve_20pct", "20% reserve"),
    )
    centers, lower, upper = [], [], []
    for key, _label in operation_cells:
        cell = operations["summary"][key]
        center = cell["all_seed_mean_saving_pct"]
        centers.append(center)
        lower.append(center - cell["all_seed_ci95"][0])
        upper.append(cell["all_seed_ci95"][1] - center)
    y = np.arange(len(operation_cells))
    ax = axes[0]
    ax.errorbar(centers, y, xerr=np.asarray([lower, upper]), fmt="o", color=BLUE,
                markerfacecolor="white", capsize=2, linewidth=1)
    ax.set_yticks(y, [label for _key, label in operation_cells])
    ax.invert_yaxis()
    ax.set_xlabel("Adjusted delivery-time saving (%)")
    ax.set_title("(a) Operational friction")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    n_values = np.asarray((20, 50, 100))
    for fleet, colour, marker in ((1, BLUE, "o"), (2, ORANGE, "s"), (3, GREEN, "^")):
        means, lows, highs = [], [], []
        for n in n_values:
            cell = density["endurance_penalties"][f"n{n}_E0.5_m{fleet}"]
            means.append(cell["mean_endurance_penalty_points"])
            lows.append(means[-1] - cell["ci95"][0])
            highs.append(cell["ci95"][1] - means[-1])
        ax.errorbar(n_values, means, yerr=np.asarray([lows, highs]), color=colour,
                    marker=marker, capsize=2, linewidth=1.7, label=f"$m={fleet}$")
    ax.set_xticks(n_values)
    ax.set_xlabel("Customers at fixed density")
    ax.set_ylabel("Penalty of $E=0.5$ vs $E=\\infty$ (points)")
    ax.set_title("(b) Fixed-density endurance test")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    empirical = energy["empirical_summary"]
    empirical_keys = (
        ("rodrigues_quadcopter_electric_van", "Rodrigues\nEV van"),
        ("rodrigues_quadcopter_diesel_van", "Rodrigues\ndiesel van"),
        ("stolaroff_quadcopter_class4_ev", "Stolaroff\nClass-4 EV"),
        ("stolaroff_quadcopter_class4_diesel", "Stolaroff\nClass-4 diesel"),
    )
    centers, lower, upper, fractions = [], [], [], []
    for key, _label in empirical_keys:
        cell = empirical[key]
        center = cell["median_instance_energy_saving_among_green_and_fast_pct"]
        interval = cell["median_instance_energy_saving_ci95"]
        centers.append(center)
        lower.append(center - interval[0])
        upper.append(interval[1] - center)
        fractions.append(100.0 * cell["fraction_faster_and_lower_energy"])
    x = np.arange(len(empirical_keys))
    ax = axes[2]
    ax.bar(x, centers, color=(BLUE, VERMILION, SKY, ORANGE),
           yerr=np.asarray([lower, upper]), capsize=3, alpha=0.9)
    ax.text(0.5, 0.97, f"{min(fractions):.0f}% of routes faster and lower-energy\n"
            "under every coefficient pair", transform=ax.transAxes, ha="center",
            va="top", fontsize=8, color=GREY)
    ax.set_xticks(x, [label for _key, label in empirical_keys], rotation=0)
    ax.set_ylim(0, max(c + u for c, u in zip(centers, upper)) * 1.3)
    ax.set_ylabel("Median energy saving vs truck-only (%)")
    ax.set_title("(c) Published movement coefficients")
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Sensitivity analyses delimit the primary estimates", fontsize=12)
    save_figure(fig, "sensitivity_v3")


def h1_validation():
    data = load_json("H1-alns-backbone", "results", "h1_v3.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.25), constrained_layout=True)

    ax = axes[0]
    sizes = sorted(int(n) for n in data["benchmark"])
    for position, n in enumerate(sizes):
        rows = [row for row in data["benchmark_rows"] if row["n"] == n]
        gaps = np.asarray([rep["gap_pct"] for row in rows for rep in row["replicates"]])
        jitter = np.linspace(-0.11, 0.11, len(gaps))
        ax.scatter(position + jitter, gaps, s=9, color=SKY, alpha=0.38,
                   edgecolors="none")
        center = data["benchmark"][str(n)]["all_seed_mean_gap_pct"]
        interval = data["benchmark"][str(n)]["all_seed_mean_gap_ci95"]
        ax.errorbar(position, center, yerr=asymmetric_error(center, interval), fmt="o",
                    color=BLUE, markerfacecolor="white", capsize=2.5, zorder=5)
    ax.axhline(0, color=GREY, linewidth=0.8)
    ax.set_xticks(range(len(sizes)), sizes)
    ax.set_xlabel("Customers")
    ax.set_ylabel("Gap to published optimum (%)")
    ax.set_title("(a) Exact references on small instances")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    sizes = (10, 20, 50)
    methods = (("truck", "Truck-only", GREY), ("greedy", "Constructive", ORANGE),
               ("alns", "ALNS, all seeds", BLUE))
    x = np.arange(len(sizes))
    width = 0.24
    for offset, (field, label, colour) in zip((-width, 0, width), methods):
        centers = []
        for n in sizes:
            rows = [row for row in data["synthetic_rows"] if row["n"] == n]
            if field == "alns":
                values = [np.mean([p["makespan"] for p in row["sortie_replicates"]])
                          for row in rows]
            else:
                values = [row[field] for row in rows]
            centers.append(float(np.mean(values)))
            ax.scatter(np.full(len(values), x[len(centers) - 1] + offset), values,
                       s=7, color=colour, alpha=0.20, edgecolors="none", zorder=2)
        ax.bar(x + offset, centers, width, color=colour, alpha=0.82, label=label,
               zorder=1)
    ax.set_xticks(x, sizes)
    ax.set_xlabel("Customers")
    ax.set_ylabel("Makespan")
    ax.set_title("(b) Constructive-reference diagnostics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.suptitle("Evidence for the computational instrument", fontsize=12)
    save_figure(fig, "h1_validation_v3")


def decoder_strength():
    data = load_json("P0-robustness", "results", "decoder_strength_v3.json")
    decoders = ("greedy", "dp4", "dp6", "dp12")
    labels = ("Greedy", "DP span 4", "DP span 6", "DP span 12")
    x = np.arange(len(decoders))
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for operator, colour, marker, label in (("sortie", BLUE, "o", "Sortie-aware operators"),
                                             ("generic", VERMILION, "s", "Generic operators")):
        values = [data["arms"][f"{decoder}_{operator}"]["all_seed_mean_saving_pct"]
                  for decoder in decoders]
        lower, upper = [], []
        for decoder, center in zip(decoders, values):
            interval = data["arms"][f"{decoder}_{operator}"]["all_seed_mean_saving_ci95"]
            lower.append(center - interval[0])
            upper.append(interval[1] - center)
        ax.errorbar(x, values, yerr=np.asarray([lower, upper]), marker=marker,
                    color=colour, linewidth=1.9, capsize=2.5, label=label)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean delivery-time saving (%)")
    ax.set_xlabel("Decoder registered in the shared ALNS loop")
    ax.set_title("Decoder-strength curve at equal iteration budget ($n=50$)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "decoder_strength_v3")


def solver_framework():
    """A compact vector workflow for the Supplementary Information."""
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, width, height, text, face, edge, fontsize=9.2):
        patch = FancyBboxPatch((x, y), width, height,
                               boxstyle="round,pad=0.03,rounding_size=0.09",
                               facecolor=face, edgecolor=edge, linewidth=1.4)
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
                fontsize=fontsize, color=BLACK, linespacing=1.25)
        return patch

    def arrow(x1, y1, x2, y2, style="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, linewidth=1.25,
                                     linestyle=style, color=GREY))

    ax.text(0.1, 5.52, "SEARCH LAYER", color=BLUE, fontsize=10, fontweight="bold")
    ax.text(0.1, 2.57, "DECODING AND AUDIT LAYER", color=GREEN, fontsize=10,
            fontweight="bold")
    ax.axhline(2.9, color=LIGHT_GREY, linewidth=1.2)

    box(0.25, 3.75, 2.0, 1.05, "Instance +\ncapability cell", "#E8F3FA", BLUE)
    box(2.85, 3.75, 2.1, 1.05, "Seeded initial\ncustomer order", "#E8F3FA", BLUE)
    box(5.55, 3.55, 2.35, 1.45,
        "Adaptive destroy / repair\n+ local order moves\n+ annealing acceptance",
        "#E8F3FA", BLUE, fontsize=8.8)
    box(8.55, 3.75, 2.35, 1.05, "Best archived order\nfor this solver seed",
        "#E8F3FA", BLUE)
    arrow(2.25, 4.275, 2.85, 4.275)
    arrow(4.95, 4.275, 5.55, 4.275)
    arrow(7.90, 4.275, 8.55, 4.275)
    arrow(6.45, 3.55, 5.55, 2.31, style="--")
    ax.text(6.65, 3.02, "each candidate", fontsize=7.8, color=GREY, ha="center")

    box(0.55, 1.0, 2.25, 1.15, "Customer order +\nmodel parameters", "#E9F6F1", GREEN)
    box(3.45, 0.85, 2.65, 1.45,
        "Fixed-order split\n(single-drone exact; batch\nunder stated span/model)",
        "#E9F6F1", GREEN, fontsize=8.7)
    box(6.8, 1.0, 2.05, 1.15, "Feasible synchronized\noperations", "#E9F6F1", GREEN)
    box(9.5, 0.85, 2.15, 1.45,
        "Archive route, time,\nenergy, seed, runtime\n+ independent checks",
        "#E9F6F1", GREEN, fontsize=8.7)
    arrow(2.8, 1.575, 3.45, 1.575)
    arrow(6.1, 1.575, 6.8, 1.575)
    arrow(8.85, 1.575, 9.5, 1.575)
    arrow(9.35, 3.75, 6.0, 2.31, style="--")
    ax.text(8.35, 2.87, "final re-decode", fontsize=7.8, color=GREY, ha="center")
    ax.text(6.0, 0.18,
            "Exactness applies to fixed-order decoding in the stated model; order search remains heuristic.",
            ha="center", fontsize=8.8, color=GREY, style="italic")
    save_figure(fig, "solver_framework_v3")


def energy_sensitivity():
    """SI figure: H3 payload / truck-coefficient sensitivity from the all-seed rerun.

    Each solver seed's 8-lambda weighted-sum front is summarized with the archived
    run_h3_v2.summarize (normalized per instance by that seed's lambda=0 point); the
    three seed-level mean curves are then averaged, so no seed is selected.
    """
    import run_h3_v2 as legacy
    data = load_json("H3-time-energy", "results", "h3_v3.json")
    lambdas = data["config"]["lambdas_sens"]
    arms = {
        "nominal": lambda r: r["tag"] == "main" and r["n"] == 50 and r["instance_seed"] < 15
                             and r["lam"] in lambdas,
        "payload_0.5": lambda r: r["tag"] == "payload" and r["cl"] == 0.5,
        "payload_2.0": lambda r: r["tag"] == "payload" and r["cl"] == 2.0,
        "truckcoef_0.15": lambda r: r["tag"] == "truckcoef" and r["te"] == 0.15,
        "truckcoef_0.6": lambda r: r["tag"] == "truckcoef" and r["te"] == 0.6,
    }
    curves, knees = {}, {}
    for name, keep in arms.items():
        per_seed = []
        for solver_seed in data["config"]["solver_seeds"]:
            rows = []
            for r in data["rows"]:
                if not keep(r):
                    continue
                rep = [x for x in r["replicates"] if x["solver_seed"] == solver_seed][0]
                rows.append({"n": r["n"], "seed": r["instance_seed"], "lam": r["lam"],
                             "makespan": rep["makespan"], "e_total": rep["e_total"],
                             "e_drone": rep["e_drone"], "e_truck": rep["e_truck"],
                             "truck_dist": rep["truck_dist"]})
            per_seed.append(legacy.summarize(rows, lambdas)["50"])
        curves[name] = [
            {"norm_makespan": float(np.mean([s["mean_curve"][i]["norm_makespan"] for s in per_seed])),
             "norm_e_total": float(np.mean([s["mean_curve"][i]["norm_e_total"] for s in per_seed]))}
            for i in range(len(lambdas))]
        knees[name] = (float(np.mean([s["knee_dms_pct"]["median"] for s in per_seed])),
                       float(np.mean([s["knee_de_pct"]["median"] for s in per_seed])))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True, constrained_layout=True)
    panels = [(axes[0], [("nominal", "payload 1.0, $t_e=0.3$ (matched nominal)"),
                         ("payload_0.5", "payload 0.5"), ("payload_2.0", "payload 2.0")],
               "(a) Drone payload coefficient"),
              (axes[1], [("nominal", "$t_e=0.3$ (matched nominal)"),
                         ("truckcoef_0.15", "$t_e=0.15$"), ("truckcoef_0.6", "$t_e=0.6$")],
               "(b) Truck energy coefficient")]
    for ax, series, title in panels:
        for (name, label), colour, marker in zip(series, (BLUE, ORANGE, GREEN), ("o", "s", "^")):
            c = curves[name]
            kt, ke = knees[name]
            ax.plot([p["norm_makespan"] for p in c], [p["norm_e_total"] for p in c],
                    marker=marker, color=colour, linewidth=1.8, markersize=5,
                    label=f"{label}: knee +{kt:.0f}% / $-${ke:.0f}%")
        ax.set_xlabel("Makespan (normalized to the time endpoint)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Total movement energy (normalized)")
    fig.suptitle("Energy-coefficient sensitivity of the weighted-sum sweep ($n=50$, all seeds)",
                 fontsize=12)
    save_figure(fig, "energy_sensitivity_v3")


FIGURES = {
    "design": design_space,
    "fleet": fleet_interactions,
    "pareto": pareto_summary,
    "boundary": energy_boundary,
    "robustness": robustness,
    "sensitivity": sensitivity,
    "h1": h1_validation,
    "decoder": decoder_strength,
    "solver": solver_framework,
    "energysens": energy_sensitivity,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("figures", nargs="*", choices=tuple(FIGURES) + ("all",),
                        default=["all"])
    args = parser.parse_args()
    selected = tuple(FIGURES) if "all" in args.figures else args.figures
    failures = []
    for name in selected:
        try:
            FIGURES[name]()
        except FileNotFoundError as error:
            failures.append(str(error))
            print(f"skip {name}: {error}")
    if failures:
        raise SystemExit("one or more locked result files are still pending")


if __name__ == "__main__":
    main()
