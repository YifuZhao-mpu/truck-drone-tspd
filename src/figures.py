"""Publication figures for the truck-drone paper. Reads experiment JSONs -> drafts/figures/*.png"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "..", "drafts", "figures")
RES = os.path.join(HERE, "..", "experiments")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 300})


def fig_design_space():
    d = json.load(open(f"{RES}/H2-design-space/results/h2.json"))
    alphas = d["alphas"]; elabels = d["endurance"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, n in zip(axes, [20, 50]):
        M = np.array([[d["surface_saving_vs_truckonly"][str(n)][f"alpha_{a}"][el]["mean_saving"]
                       for el in elabels] for a in alphas])
        im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xticks(range(len(elabels))); ax.set_xticklabels(elabels)
        ax.set_yticks(range(len(alphas))); ax.set_yticklabels(alphas)
        ax.set_xlabel("drone endurance (max flight distance)")
        ax.set_ylabel("drone/truck speed ratio α")
        ax.set_title(f"n = {n} customers")
        for i in range(len(alphas)):
            for j in range(len(elabels)):
                ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center",
                        color="white" if M[i, j] < M.max() * 0.6 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, label="makespan saving vs truck-only (%)")
    fig.suptitle("When do drones help? Design-space map of delivery-time savings", y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/design_space_heatmap.png", bbox_inches="tight"); plt.close(fig)
    print("saved design_space_heatmap.png")


def fig_drones():
    d = json.load(open(f"{RES}/H2-design-space/results/h2.json"))
    da = d["drones_axis_alpha2"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ms = [1, 2, 3]
    for el, mk in zip(da.keys(), ["o-", "s-"]):
        sv = [da[el][f"m_{m}"]["mean_saving"] for m in ms]
        st = [da[el][f"m_{m}"]["std"] for m in ms]
        ax.errorbar(ms, sv, yerr=st, fmt=mk, capsize=3, label=f"endurance = {el}")
    ceil = [(1 - 1 / (2 * m + 1)) * 100 for m in ms]
    ax.plot(ms, ceil, "k--", alpha=0.6, label="worst-case ceiling 1−1/(αm+1), α=2")
    ax.set_xticks(ms); ax.set_xlabel("number of drones m (α=2)")
    ax.set_ylabel("makespan saving vs truck-only (%)")
    ax.set_title("Diminishing returns in fleet size"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/drones_curve.png", bbox_inches="tight"); plt.close(fig)
    print("saved drones_curve.png")


def fig_pareto():
    d = json.load(open(f"{RES}/H3-time-energy/results/h3.json"))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    offs = {20: (18, 22), 50: (-105, -95)}
    for n, mk, col in [(20, "o-", "C0"), (50, "s-", "C1")]:
        c = d["by_n"][str(n)]["pareto"]
        x = [p["norm_makespan"] for p in c]; y = [p["norm_energy"] for p in c]
        ax.plot(x, y, mk, color=col, label=f"n = {n}")
        k = d["by_n"][str(n)]["knee"]
        ax.scatter([k["norm_makespan"]], [k["norm_energy"]], s=160, facecolors="none",
                   edgecolors=col, linewidths=2, zorder=5)
        ax.annotate(f"knee (n={n}): +{k['makespan_increase_pct']:.0f}% time\n→ −{k['energy_reduction_pct']:.0f}% drone energy",
                    (k["norm_makespan"], k["norm_energy"]),
                    textcoords="offset points", xytext=offs[n], fontsize=8, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1))
    ax.set_xlabel("makespan (normalized to time-optimal)")
    ax.set_ylabel("drone energy (normalized to time-optimal)")
    ax.set_title("Time–energy trade-off (Pareto front)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/pareto.png", bbox_inches="tight"); plt.close(fig)
    print("saved pareto.png")


def fig_h1():
    d = json.load(open(f"{RES}/H1-alns-backbone/results/h1.json"))
    bg = d["benchmark_gap_to_optimal"]; sy = d["synthetic"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ns = sorted(int(k) for k in bg)
    axes[0].bar([str(n) for n in ns], [bg[str(n)]["mean_gap"] for n in ns],
                yerr=[[0.0] * len(ns),                       # one-sided whisker: mean -> worst instance
                      [bg[str(n)]["max_gap"] - bg[str(n)]["mean_gap"] for n in ns]],
                capsize=3, color="C2")
    axes[0].set_xlabel("instance size (nodes incl. depot)")
    axes[0].set_ylabel("gap to published exact optimum (%)")
    axes[0].set_title("(a) ALNS validation vs exact optima"); axes[0].grid(alpha=0.3, axis="y")
    sizes = [10, 20, 50]
    ref = json.load(open(f"{RES}/TSPREF/results/tspref_cache.json"))
    truck = [np.mean([ref[f"uniform-n{n}-seed{s}-center-euclidean"]["length"]
                      for s in range(30)]) for n in sizes]
    greedy = [sy[str(n)]["greedy_mean"] for n in sizes]
    alns_ = [sy[str(n)]["sortie_mean"] for n in sizes]
    x = np.arange(len(sizes)); w = 0.26
    axes[1].bar(x - w, truck, w, label="truck-only (LKH reference)")
    axes[1].bar(x, greedy, w, label="greedy TSP-LS")
    axes[1].bar(x + w, alns_, w, label="ALNS (ours)")
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"n={n}" for n in sizes])
    axes[1].set_ylabel("mean makespan"); axes[1].set_title("(b) ALNS vs baselines")
    axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{FIG}/h1_validation.png", bbox_inches="tight"); plt.close(fig)
    print("saved h1_validation.png")


def fig_route_example():
    """Re-solve one instance and draw the truck route + drone sorties."""
    from problem import gen_instance, dist_matrix
    from alns import alns
    inst = gen_instance(20, seed=7); Dt = dist_matrix(inst["coords"]); co = inst["coords"]
    r = alns(inst, Dt, 2.0, endurance=1.0, max_span=12, iters=12000, sortie_aware=True, seed=0)
    seq = r["order"]; ops = r["ops"]
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.scatter(co[1:, 0], co[1:, 1], c="steelblue", s=30, zorder=3, label="customer")
    ax.scatter([co[0, 0]], [co[0, 1]], c="black", marker="s", s=90, zorder=4, label="depot")
    for op in ops:
        if op[0] == "truck":
            i, k = op[1], op[2]
            ax.plot([co[seq[i], 0], co[seq[k], 0]], [co[seq[i], 1], co[seq[k], 1]], "-", color="gray", lw=1.6, zorder=2)
        elif op[0] == "sortie":
            i, j, k = op[1], op[2], op[3]
            kept = [p for p in range(i, k + 1) if p != j]
            for a in range(len(kept) - 1):
                ax.plot([co[seq[kept[a]], 0], co[seq[kept[a+1]], 0]],
                        [co[seq[kept[a]], 1], co[seq[kept[a+1]], 1]], "-", color="gray", lw=1.6, zorder=2)
            ax.plot([co[seq[i], 0], co[seq[j], 0]], [co[seq[i], 1], co[seq[j], 1]], "--", color="crimson", lw=1.4, zorder=2)
            ax.plot([co[seq[j], 0], co[seq[k], 0]], [co[seq[j], 1], co[seq[k], 1]], "--", color="crimson", lw=1.4, zorder=2)
            ax.scatter([co[seq[j], 0]], [co[seq[j], 1]], c="crimson", marker="^", s=55, zorder=4)
    ax.plot([], [], "-", color="gray", label="truck route")
    ax.plot([], [], "--", color="crimson", label="drone sortie")
    ax.set_title(f"Example TSP-D solution (n=20, α=2, makespan={r['makespan']:.2f})")
    ax.legend(loc="upper right", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(f"{FIG}/route_example.png", bbox_inches="tight"); plt.close(fig)
    print("saved route_example.png")


def fig_robustness():
    d = json.load(open(f"{RES}/P0-robustness/results/p0.json"))
    alphas = d["config"]["alphas"]; elabels = d["config"]["endurance"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    # (a) cross-setting bars at alpha=2, E=1.0
    cs = d["cross_setting_cell_a2_e1"]
    labels = ["uniform\nEuclid", "clustered\nEuclid", "uniform\nManhattan", "clustered\nManhattan"]
    keys = ["uniform_euclidean", "clustered_euclidean", "uniform_manhattan", "clustered_manhattan"]
    x = np.arange(len(labels)); w = 0.38
    for off, n, c in [(-w/2, 50, "C0"), (w/2, 100, "C1")]:
        vals = [cs[f"{k}_n{n}"]["mean_saving"] for k in keys]
        err = [cs[f"{k}_n{n}"]["std"] for k in keys]
        axes[0].bar(x + off, vals, w, yerr=err, capsize=3, label=f"n={n}", color=c)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("makespan saving vs truck-only (%)")
    axes[0].set_title("(a) Saving across demand × road model (α=2, E=1.0)")
    axes[0].legend(); axes[0].grid(alpha=0.3, axis="y")
    # (b,c) heatmaps: clustered+manhattan n=50, uniform n=100
    for ax, key, ttl in [(axes[1], "clustered_manhattan_n50", "(b) Clustered + road network, n=50"),
                         (axes[2], "uniform_euclidean_n100", "(c) Uniform, n=100")]:
        surf = d["surfaces"][key]
        M = np.array([[surf[f"alpha_{a}"][el]["mean_saving"] for el in elabels] for a in alphas])
        im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xticks(range(len(elabels))); ax.set_xticklabels(elabels)
        ax.set_yticks(range(len(alphas))); ax.set_yticklabels(alphas)
        ax.set_xlabel("endurance"); ax.set_ylabel("speed ratio α"); ax.set_title(ttl)
        for i in range(len(alphas)):
            for j in range(len(elabels)):
                ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center",
                        color="white" if M[i, j] < M.max() * 0.6 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, label="saving (%)")
    fig.suptitle("Robustness of the design-space map to realistic demand, road-network distance, and scale", y=1.03)
    fig.tight_layout(); fig.savefig(f"{FIG}/robustness.png", bbox_inches="tight"); plt.close(fig)
    print("saved robustness.png")


def fig_payload():
    d = json.load(open(f"{RES}/P0-robustness/results/p0.json"))
    ep = d["energy_payload"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for pl, mk, col in [("0.5", "o-", "C0"), ("1.0", "s-", "C1"), ("2.0", "^-", "C2")]:
        c = ep[f"payload_{pl}"]["curve"]
        x = [p[0] for p in c]; y = [p[1] for p in c]
        ax.plot(x, y, mk, color=col, label=f"payload coeff = {pl}")
        k = ep[f"payload_{pl}"]["knee"]
    ax.set_xlabel("makespan (normalized to time-optimal)")
    ax.set_ylabel("drone energy (normalized to time-optimal)")
    ax.set_title("Time–energy trade-off is robust across payload"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/payload_pareto.png", bbox_inches="tight"); plt.close(fig)
    print("saved payload_pareto.png")


def fig_overview():
    # NOTE (2026-07-23): superseded as manuscript Fig. 1 by the author-drawn vector
    # schematic drafts/latex/figures/solver_framework.pdf; kept for provenance.
    """Schematic overview: (a) TSP-D problem, (b) ALNS + exact DP split solver,
    (c) the three measurements. Pure schematic — no experiment data needed."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    NAVY, BODY, MUTE = "#12263a", "#3b4a5a", "#6b7a8a"
    PANEL_EC = "#dfe3e8"
    BLUE_F, BLUE_E, BLUE_T = "#eef3fb", "#c9d8f0", "#1e40af"
    GREEN_F, GREEN_E, GREEN_T = "#eaf7ee", "#bfe3c8", "#166534"
    AMBER_F, AMBER_E, AMBER_T, TAG_BG = "#fcf4dd", "#d9a520", "#7c5c10", "#836000"
    PURP_F, PURP_E, PURP_T = "#f3eefc", "#d8c9f0", "#5b21b6"
    CHIP_F, CHIP_E, CHIP_T = "#e7f5ec", "#b7e0c4", "#1a7f37"
    ORANGE, TRUCK = "#e0820c", "#2c3e50"

    fig = plt.figure(figsize=(16.2, 9.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x0, y0, x1, y1, fc="white", ec=PANEL_EC, lw=1.2, r=0.9, z=1):
        b = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                           fc=fc, ec=ec, lw=lw, zorder=z)
        ax.add_patch(b); return b

    def arrow(p0, p1, color="#5a6b7c", lw=1.6, style="-|>", ms=14, z=3, con=None):
        a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color,
                            lw=lw, zorder=z, connectionstyle=con or "arc3,rad=0")
        ax.add_patch(a); return a

    # ================= panel frames =================
    box(2, 52.5, 35.5, 97.5); box(37.5, 52.5, 98, 97.5); box(2, 2, 98, 48.5)
    ax.text(3.6, 94.6, "a", fontsize=14, fontweight="bold", color=NAVY)
    ax.text(5.8, 94.6, "Problem — TSP-D (min makespan)", fontsize=13, fontweight="bold", color=NAVY)
    ax.text(39.1, 94.6, "b", fontsize=14, fontweight="bold", color=NAVY)
    ax.text(41.3, 94.6, "Solver — ALNS over visiting orders + exact split decoding",
            fontsize=13, fontweight="bold", color=NAVY)
    ax.text(3.6, 45.3, "c", fontsize=14, fontweight="bold", color=NAVY)
    ax.text(5.8, 45.3, "One validated solver, three measurements", fontsize=13, fontweight="bold", color=NAVY)

    # ================= (a) problem sketch =================
    ax.text(3.8, 92.2, "the drone launches from the truck, serves one customer, rejoins downstream;\n"
                       "whoever reaches the rendezvous first waits · objective: all customers served,\n"
                       "both vehicles back at the depot",
            fontsize=8.6, color=MUTE, va="top", linespacing=1.45)

    def A(x, y):  # panel-a local coords (0-1) -> global
        return 3.5 + x * 30.5, 56.5 + y * 28.5
    dep = A(0.05, 0.42)
    t1, t2, t3, t4 = A(0.26, 0.62), A(0.40, 0.44), A(0.58, 0.68), A(0.74, 0.50)
    b1, b2 = A(0.30, 0.16), A(0.55, 0.14)
    d1, d2 = A(0.38, 0.95), A(0.93, 0.20)
    tour = [dep, t1, t2, t3, t4, b2, b1, dep]
    ax.plot(*zip(*tour), color=TRUCK, lw=2.2, zorder=4, solid_capstyle="round")
    for p0, pm, p1 in [(t1, d1, t3), (t4, d2, b2)]:
        ax.plot(*zip(*[p0, pm, p1]), color=ORANGE, lw=1.9, ls=(0, (5, 3)), zorder=3)
    ax.scatter(*dep, marker="s", s=110, color="#111827", zorder=6)
    for p in [t1, t2, t3, t4, b1, b2]:
        ax.scatter(*p, s=64, color="#3d4f63", zorder=6)
    for p in [d1, d2]:
        ax.scatter(*p, s=74, color=ORANGE, zorder=6)
    ax.annotate("drone sortie", A(0.38, 1.04), color=ORANGE, fontsize=8.8, ha="center")
    ax.annotate("depot", (dep[0] - 0.2, dep[1] + 1.6), color=NAVY, fontsize=8.8, ha="center")
    # legend row
    ly = 54.6
    ax.scatter(4.6, ly, marker="s", s=70, color="#111827"); ax.text(5.5, ly, "depot", fontsize=8.2, color=BODY, va="center")
    ax.scatter(9.9, ly, s=52, color="#3d4f63"); ax.text(10.8, ly, "truck customer", fontsize=8.2, color=BODY, va="center")
    ax.scatter(18.9, ly, s=56, color=ORANGE); ax.text(19.8, ly, "drone customer", fontsize=8.2, color=BODY, va="center")
    ax.plot([27.7, 29.5], [ly, ly], color=TRUCK, lw=2.2); ax.text(29.9, ly, "truck", fontsize=8.2, color=BODY, va="center")
    ax.plot([32.3, 34.1], [ly, ly], color=ORANGE, lw=1.9, ls=(0, (4, 2.5))); ax.text(34.4, ly, "drone", fontsize=8.2, color=BODY, va="center")

    # ================= (b) solver flowchart =================
    # initial order
    box(43, 87, 68, 92.3, fc="#fbfcfe", ec="#d5dbe3")
    ax.text(44.2, 89.65, "Initial order", fontsize=9.6, fontweight="bold", color=NAVY, va="center")
    ax.text(51.9, 89.65, "— greedy TSP + local search", fontsize=9, color=MUTE, va="center")
    # ALNS box
    box(43, 66, 68, 84.5, fc=BLUE_F, ec=BLUE_E)
    ax.text(44.2, 82.1, "ALNS — destroy → repair → local move", fontsize=9.8, fontweight="bold", color=BLUE_T)
    ax.text(44.2, 79.3, "destroy: random · worst-detour · Shaw", fontsize=8.6, color=BODY)
    ax.text(44.2, 76.8, "repair: greedy · regret · adaptive roulette weights", fontsize=8.6, color=BODY)
    ax.text(44.2, 74.3, "local:", fontsize=8.6, color=BODY)
    ax.text(46.6, 74.3, "sortie-aware moves", fontsize=8.6, fontweight="bold", color="#9a5b00")
    ax.text(55.2, 74.3, "(relocate · swap · reversal)", fontsize=8.6, color=BODY)
    ax.text(44.2, 71.4, "decoder×operator factorial: exact-vs-greedy split +4.3%,", fontsize=8.2, color=MUTE)
    ax.text(44.2, 69.1, "sortie-vs-generic moves +0.5% — decoder dominates 9×", fontsize=8.2, color=MUTE)
    # SA box
    box(43, 58.5, 68, 63.6, fc=GREEN_F, ec=GREEN_E)
    ax.text(44.2, 61.05, "SA acceptance", fontsize=9.6, fontweight="bold", color=GREEN_T, va="center")
    ax.text(52.4, 61.05, "— cooling 0.994 · double-bridge restarts", fontsize=8.8, color=MUTE, va="center")
    # DP decoder box (dominant lever)
    box(71.5, 66.5, 96.5, 86, fc=AMBER_F, ec=AMBER_E, lw=1.8)
    tag = FancyBboxPatch((84.6, 85), 11.4, 2.3, boxstyle="round,pad=0,rounding_size=1.1",
                         fc=TAG_BG, ec="none", zorder=6)
    ax.add_patch(tag)
    ax.text(90.3, 86.15, "THE DOMINANT LEVER", fontsize=7.6, fontweight="bold", color="white",
            ha="center", va="center", zorder=7)
    ax.text(72.7, 83.2, "Exact DP “operations split” decoder", fontsize=10, fontweight="bold", color=AMBER_T)
    ax.text(72.7, 80.0, r"$D[k]\;=\;\min_{i<k}\,\left(\,D[i]+c(i,k)\,\right)$", fontsize=10.5, color=NAVY)
    ax.text(72.7, 77.0, "truck leg, or sortie i→j→k costing max(truck, drone) time", fontsize=8.4, color=BODY)
    ax.text(72.7, 74.7, "· flight ≤ E", fontsize=8.4, color=BODY)
    ax.text(72.7, 72.1, "· every order → its optimal feasible schedule", fontsize=8.4, fontweight="bold", color=BODY)
    ax.text(72.7, 69.6, "· JIT ≈ 40 µs (n=50)", fontsize=8.4, color=BODY)
    # lambda box
    box(71.5, 55.5, 96.5, 62.5, fc=PURP_F, ec=PURP_E)
    ax.text(72.7, 60.3, "λ objective:", fontsize=8.8, fontweight="bold", color=PURP_T)
    ax.text(78.2, 60.3, "time + λ·total energy → sweep λ ⇒", fontsize=8.8, color=PURP_T)
    ax.text(72.7, 57.7, "time–energy Pareto frontier", fontsize=8.8, color=PURP_T)
    arrow((84, 62.5), (84, 66.5), color="#8b5cf6", lw=1.6)
    # flow arrows
    arrow((55.5, 87), (55.5, 84.5))
    arrow((55.5, 66), (55.5, 63.6))
    arrow((68, 77.5), (71.5, 77.5), color="#a97d10", lw=1.8)
    ax.text(69.75, 78.3, "order", fontsize=7.8, color="#a97d10", ha="center")
    arrow((71.5, 71.8), (68, 71.8), color="#a97d10", lw=1.8)
    ax.text(69.75, 69.9, "cost", fontsize=7.8, color="#a97d10", ha="center")
    arrow((43, 61), (40.8, 61), lw=1.4, style="-")
    arrow((40.8, 61), (40.8, 75.2), lw=1.4, style="-")
    arrow((40.8, 75.2), (43, 75.2), lw=1.4)
    ax.text(40.2, 68.6, "next\niteration", fontsize=7.6, color=MUTE, ha="right", va="center", linespacing=1.3)
    # panel-b -> panel-c
    arrow((50, 52.5), (50, 48.5), lw=1.8)
    ax.text(51, 50.3, "best decoded plans", fontsize=8.4, color=MUTE, va="center")

    # ================= (c) three measurement cards =================
    box(4, 6.5, 33, 43, fc="#fafbfc", ec="#e3e7ec")
    box(35.5, 6.5, 64.5, 43, fc="#fafbfc", ec="#e3e7ec")
    box(67, 6.5, 96, 43, fc="#fafbfc", ec="#e3e7ec")
    ax.text(5.6, 40.2, "Design-space map", fontsize=11, fontweight="bold", color="#2457a8")
    ax.text(37.1, 40.2, "Time–energy frontier", fontsize=11, fontweight="bold", color="#6d28d9")
    ax.text(68.6, 40.2, "Robustness", fontsize=11, fontweight="bold", color="#15803d")

    # --- card 1 mini plot: saturating curves + ceiling
    cx0, cx1, cy0, cy1 = 7, 30, 20.5, 37.5
    xs = np.linspace(0, 1, 60)
    ceil_y = cy0 + 0.92 * (cy1 - cy0)
    ax.plot([cx0, cx1], [ceil_y, ceil_y], ls=(0, (5, 3)), color="#9aa7b4", lw=1.4)
    ax.text(cx1, ceil_y + 1.0, "worst-case ceiling 1−1/(αm+1)", fontsize=7.6, color="#8593a2", ha="right")
    for plateau, c in [(0.86, "#1e4f9c"), (0.66, "#4f7fc4"), (0.46, "#9db9e0")]:
        ys = cy0 + plateau * (cy1 - cy0) * (1 - np.exp(-4.6 * xs))
        ax.plot(cx0 + xs * (cx1 - cx0), ys, color=c, lw=2.0)
    ax.plot([cx0, cx1], [cy0, cy0], color="#8593a2", lw=1.0)
    ax.plot([cx0, cx0], [cy0, cy1], color="#8593a2", lw=1.0)
    ax.text(cx0 - 0.6, cy1 + 0.6, "saving", fontsize=7.8, color="#8593a2")
    ax.text((cx0 + cx1) / 2, cy0 - 1.6, "α · E · m →", fontsize=7.8, color="#8593a2", ha="center")
    ax.text(5.6, 17.4, "full α × E × m factorial: steep rise → saturation; ≈34%\n"
                       "(1 drone) → ≈52% (3 drones), below the ceiling; speed and\n"
                       "fleet size are complements",
            fontsize=8.4, color=BODY, va="top", linespacing=1.5)

    # --- card 2 mini plot: pareto knee
    px0, px1, py0, py1 = 38.5, 61.5, 20.5, 37.5
    xs = np.linspace(0, 1, 80)
    ys = py0 + (py1 - py0) * (1 - xs) ** 2.4
    ax.plot(px0 + xs * (px1 - px0), ys, color="#7c3aed", lw=2.2)
    ax.plot([px0, px1], [py1, py0], ls=(0, (4, 3)), color="#c9c2d8", lw=1.2)
    kx = 0.30; ky = (1 - kx) ** 2.4
    ax.scatter(px0 + kx * (px1 - px0), py0 + ky * (py1 - py0), s=64, color="#e0820c", zorder=6)
    ax.text(px0 + kx * (px1 - px0) + 1.2, py0 + ky * (py1 - py0) + 1.2, "knee", fontsize=8.2, color="#b45309")
    ax.text(px0 + 2.0, py1 - 1.8, "time-optimal", fontsize=7.6, color="#8593a2")
    ax.text(px1 - 0.4, py0 + 2.6, "truck-only →", fontsize=7.6, color="#8593a2", ha="right")
    ax.plot([px0, px1], [py0, py0], color="#8593a2", lw=1.0)
    ax.plot([px0, px0], [py0, py1], color="#8593a2", lw=1.0)
    ax.text(px0 - 0.6, py1 + 0.6, "total energy", fontsize=7.8, color="#8593a2")
    ax.text((px0 + px1) / 2, py0 - 1.6, "makespan →", fontsize=7.8, color="#8593a2", ha="center")
    ax.text(37.1, 17.4, "per-instance fronts in time + λ·total energy: ≈3% more\n"
                        "time ⇒ −15% total energy; completed-front knee +18–20% ⇒ −43–47%",
            fontsize=8.4, color=BODY, va="top", linespacing=1.5)

    # --- card 3: chips + text
    def chip(x, y, w, label):
        box(x, y, x + w, y + 3.4, fc=CHIP_F, ec=CHIP_E, r=1.4)
        ax.text(x + w / 2, y + 1.7, label, fontsize=8.2, color=CHIP_T, ha="center", va="center")
    chip(68.6, 34.2, 11.6, "clustered demand")
    chip(81.2, 34.2, 14.2, "rectilinear (L1) truck metric")
    chip(68.6, 29.9, 9.4, "scale n = 100")
    chip(78.8, 29.9, 8.6, "depot position")
    chip(88.0, 29.9, 7.4, "energy coeffs")
    ax.text(68.6, 26.2, "savings persist in all 16 deconfounded cells (26–39%);\n"
                        "demand pattern, not depot placement, drives the variation;\n"
                        "endurance stops mattering at n=100; the energy knee\n"
                        "survives halving/doubling both energy coefficients",
            fontsize=8.4, color=BODY, va="top", linespacing=1.5)

    ax.text(50, 4, "solver validated first: mean 0.13% gap to published exact optima (80% solved exactly) · "
                   "best-found LKH truck-only references · matched-budget match vs both public implementations",
            fontsize=8.6, color=MUTE, ha="center", style="italic")

    fig.savefig(f"{FIG}/overview.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved overview.png")


# ============================ v2 figures (post-review) ============================
# Read the v2 result files (LKH denominators, one solver config, per-instance fronts).

def fig_design_space_v2():
    d = json.load(open(f"{RES}/H2-design-space/results/h2_v2.json"))
    alphas = d["config"]["alphas"]; elabels = d["config"]["endurance"]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8))
    vmin, vmax = 1e9, -1e9
    Ms = {}
    for n in (20, 50):
        for m in (1, 2, 3):
            M = np.array([[d["cells"][str(n)][f"a{a}_E{el}_m{m}"]["mean_saving"]
                           for el in elabels] for a in alphas])
            Ms[(n, m)] = M
            vmin, vmax = min(vmin, M.min()), max(vmax, M.max())
    for r, n in enumerate((20, 50)):
        for c, m in enumerate((1, 2, 3)):
            ax = axes[r][c]; M = Ms[(n, m)]
            im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower",
                           vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(elabels))); ax.set_xticklabels(elabels, fontsize=9)
            ax.set_yticks(range(len(alphas))); ax.set_yticklabels(alphas, fontsize=9)
            if r == 1: ax.set_xlabel("endurance (max flight distance)")
            if c == 0: ax.set_ylabel(f"n = {n}\nspeed ratio α")
            ax.set_title(f"m = {m} drone{'s' if m > 1 else ''}", fontsize=10)
            for i in range(len(alphas)):
                for j in range(len(elabels)):
                    ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center",
                            color="white" if M[i, j] < vmax * 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=axes, label="makespan saving vs best-found LKH truck-only reference (%)",
                 shrink=0.85)
    fig.suptitle("Design-space map: full speed × endurance × fleet-size factorial", y=0.98)
    fig.savefig(f"{FIG}/design_space_v2.png", bbox_inches="tight"); plt.close(fig)
    print("saved design_space_v2.png")


def fig_fleet_interactions_v2():
    d = json.load(open(f"{RES}/H2-design-space/results/h2_v2.json"))
    alphas = d["config"]["alphas"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, n in zip(axes, (20, 50)):
        g = d["interactions"][str(n)]["drone_marginal_gain"]
        for el, ls in (("1.0", "-"), ("inf", "--")):
            ax.plot(alphas, [g[f"a{a}_E{el}"]["m1_to_2"] for a in alphas], "o" + ls,
                    color="C0", label=f"2nd drone, E={el}")
            ax.plot(alphas, [g[f"a{a}_E{el}"]["m2_to_3"] for a in alphas], "s" + ls,
                    color="C1", label=f"3rd drone, E={el}")
        ax.set_xlabel("speed ratio α"); ax.set_title(f"n = {n}")
        ax.set_ylabel("marginal saving of added drone (percentage points)")
        ax.grid(alpha=0.3)
        if n == 20:
            ax.legend(fontsize=8)
    fig.suptitle("Fleet-size interactions: what each added drone is worth, by speed and endurance")
    fig.tight_layout(); fig.savefig(f"{FIG}/fleet_interactions_v2.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fleet_interactions_v2.png")


def _fronts_by_seed(raw, tag, n, cl=1.0, te=0.3):
    outs = {}
    for r in raw:
        if r["tag"] == tag and r["n"] == n and abs(r["cl"] - cl) < 1e-9 \
                and abs(r["te"] - te) < 1e-9:
            outs.setdefault(r["seed"], []).append(r)
    return outs


def fig_pareto_v2():
    """COMPLETED per-instance fronts (weighted-sum sweep + refinement + exact decode-level
    Pareto pass, archived in h3_frontier.json) with completed-front knee medians."""
    fr = json.load(open(f"{RES}/H3-time-energy/results/h3_frontier.json"))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    for ax, n in zip(axes, (20, 50)):
        for sd, front in fr["combined_fronts"][str(n)].items():
            t0, e0 = front[0]
            ax.plot([t / t0 for t, e in front], [e / e0 for t, e in front],
                    "-", color="gray", alpha=0.22, lw=0.7, zorder=1)
        s = fr["summary"][str(n)]
        km, ke = s["knee_dms_median"], s["knee_de_median"]
        ax.scatter([1 + km / 100], [1 - ke / 100], s=170, facecolors="none",
                   edgecolors="crimson", linewidths=2, zorder=5)
        ax.annotate(f"median knee (completed fronts):\n+{km:.0f}% time → −{ke:.0f}% total energy\n"
                    f"(IQR +{s['knee_dms_iqr'][0]:.0f}..{s['knee_dms_iqr'][1]:.0f}%, "
                    f"−{s['knee_de_iqr'][1]:.0f}..−{s['knee_de_iqr'][0]:.0f}%)",
                    (1 + km / 100, 1 - ke / 100),
                    textcoords="offset points", xytext=(14, 8), fontsize=8, color="crimson")
        npts = [len(f) for f in fr["combined_fronts"][str(n)].values()]
        ax.set_xlabel("makespan (normalized to time-optimal)")
        ax.set_ylabel("TOTAL energy, drone + truck (normalized)")
        ax.set_title(f"n = {n} (thin: completed per-instance fronts, {min(npts)}–{max(npts)} points)")
        ax.grid(alpha=0.3)
    fig.suptitle("Time–energy trade-off under the total-system energy objective (comparable-power regime)")
    fig.tight_layout(); fig.savefig(f"{FIG}/pareto_v2.png", bbox_inches="tight"); plt.close(fig)
    print("saved pareto_v2.png")


def fig_sensitivity_v2():
    d = json.load(open(f"{RES}/H3-time-energy/results/h3_v2.json"))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    # matched nominal control: same 15 seeds and same 8-lambda grid as the perturbed runs
    lams = d["config"]["lambdas_sens"]
    import run_h3_v2 as H
    nom_rows = [r for r in d["raw"] if r["tag"] == "main" and r["n"] == 50
                and r["seed"] < 15 and r["lam"] in lams]
    nom = H.summarize(nom_rows, lams)["50"]
    panels = [(axes[0], [(nom, "1.0/0.3 (nominal, matched)"),
                         (d["sensitivity"]["payload_0.5"]["50"], "payload 0.5 (light)"),
                         (d["sensitivity"]["payload_2.0"]["50"], "payload 2.0 (heavy)")],
               "drone payload coefficient"),
              (axes[1], [(nom, "te 0.3 (nominal, matched)"),
                         (d["sensitivity"]["truckcoef_0.15"]["50"], "te 0.15 (efficient truck)"),
                         (d["sensitivity"]["truckcoef_0.6"]["50"], "te 0.6 (thirsty truck)")],
               "truck energy coefficient")]
    for ax, series, ttl in panels:
        for (src, lbl), col, mk in zip(series, ("C0", "C1", "C2"), ("o-", "s-", "^-")):
            c = src["mean_curve"]
            ax.plot([p["norm_makespan"] for p in c], [p["norm_e_total"] for p in c],
                    mk, color=col, label=(lbl + f"  (knee +{src['knee_dms_pct']['median']:.0f}%"
                                          f"/−{src['knee_de_pct']['median']:.0f}%)"))
        ax.set_xlabel("makespan (normalized)"); ax.set_ylabel("total energy (normalized)")
        ax.set_title(f"Sensitivity: {ttl} (n=50)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/sensitivity_v2.png", bbox_inches="tight")
    plt.close(fig)
    print("saved sensitivity_v2.png")


def fig_robustness_v2():
    d = json.load(open(f"{RES}/P0-robustness/results/p0_v2.json"))
    alphas = [1.0, 1.5, 2.0, 2.5, 3.0]; elabels = ["0.5", "1.0", "2.0", "inf"]
    fig = plt.figure(figsize=(14.5, 4.6))
    ax0 = fig.add_axes([0.06, 0.14, 0.52, 0.72])
    combos = [("uniform", "euclidean"), ("clustered", "euclidean"),
              ("uniform", "manhattan"), ("clustered", "manhattan")]
    labels = ["uniform\nEuclidean", "clustered\nEuclidean",
              "uniform\nrectilinear (L1)", "clustered\nrectilinear (L1)"]
    x = np.arange(len(combos))
    w = 0.2
    for i, (n, depot) in enumerate([(50, "center"), (50, "corner"),
                                    (100, "center"), (100, "corner")]):
        vals, errs = [], []
        for kind, metric in combos:
            cell = d["deconfound_cells"][f"{kind}_{depot}_{metric}_n{n}"]
            vals.append(cell["mean_saving"])
            errs.append([cell["mean_saving"] - cell["ci95"][0],
                         cell["ci95"][1] - cell["mean_saving"]])
        ax0.bar(x + (i - 1.5) * w, vals, w, yerr=np.array(errs).T, capsize=2,
                label=f"n={n}, {depot} depot",
                color=f"C{0 if n == 50 else 1}", alpha=1.0 if depot == "center" else 0.55)
    ax0.set_xticks(x); ax0.set_xticklabels(labels, fontsize=9)
    ax0.set_ylabel("saving vs LKH truck-only reference (%)")
    ax0.set_title("(a) Demand × road model × depot position, deconfounded (α=2, E=1)")
    ax0.legend(fontsize=8); ax0.grid(alpha=0.3, axis="y")
    ax1 = fig.add_axes([0.65, 0.14, 0.30, 0.72])
    M = np.array([[d["surface_n100"][f"a{a}_E{el}"]["mean_saving"] for el in elabels]
                  for a in alphas])
    im = ax1.imshow(M, aspect="auto", cmap="viridis", origin="lower")
    ax1.set_xticks(range(len(elabels))); ax1.set_xticklabels(elabels)
    ax1.set_yticks(range(len(alphas))); ax1.set_yticklabels(alphas)
    ax1.set_xlabel("endurance E (max sortie flight distance)"); ax1.set_ylabel("speed ratio α")
    ax1.set_title("(b) Speed × endurance surface, n=100, one drone")
    for i in range(len(alphas)):
        for j in range(len(elabels)):
            ax1.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center",
                     color="white" if M[i, j] < M.max() * 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax1, label="saving (%)")
    fig.savefig(f"{FIG}/robustness_v2.png", bbox_inches="tight"); plt.close(fig)
    print("saved robustness_v2.png")


if __name__ == "__main__":
    import sys
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "robust"):
        try: fig_robustness(); fig_payload()
        except FileNotFoundError: print("P0 results not ready")
    if what in ("all", "h1"): fig_h1()
    if what in ("all", "design"): fig_design_space()
    if what in ("all", "drones"): fig_drones()
    if what in ("all", "pareto"): fig_pareto()
    if what in ("all", "route"): fig_route_example()
    if what in ("all", "overview"): fig_overview()
    if what in ("v2", "all2"):
        for fn in (fig_design_space_v2, fig_fleet_interactions_v2, fig_pareto_v2,
                   fig_sensitivity_v2, fig_robustness_v2):
            try:
                fn()
            except FileNotFoundError as e:
                print(f"skip {fn.__name__}: {e}")


def fig_calibrated():
    """Two-regime figure: total energy of time-optimal and energy-minimal truck-drone
    plans relative to the truck-only plan, across the truck-to-drone power ratio te."""
    import json as _json
    cal = _json.load(open(f"{RES}/H3-time-energy/results/h3_calibrated.json"))
    h3 = _json.load(open(f"{RES}/H3-time-energy/results/h3_v2.json"))
    cache = _json.load(open(f"{RES}/TSPREF/results/tspref_cache.json"))
    tes = [0.3, 1.0, 6.0, 30.0, 60.0]
    to_r, em_r = [], []
    rows = [r for r in h3["raw"] if r["tag"] == "main" and r["n"] == 50 and r["seed"] < 15]
    t_, e_ = [], []
    for sd in range(15):
        pts = sorted([r for r in rows if r["seed"] == sd], key=lambda p: p["lam"])
        ref_e = 0.3 * cache[f"uniform-n50-seed{sd}-center-euclidean"]["length"]
        t_.append(pts[0]["e_total"] / ref_e)
        e_.append(min(p["e_total"] for p in pts) / ref_e)
    to_r.append(np.mean(t_)); em_r.append(np.mean(e_))
    for te in ("1.0", "6.0", "30.0", "60.0"):
        s = cal["by_te"][te]
        to_r.append(s["time_opt_vs_truck_only_energy"]["mean"])
        em_r.append(s["energy_min_vs_truck_only_energy"]["mean"])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(1.0, color="gray", lw=1.2, ls="--")
    ax.text(0.32, 1.03, "truck-only energy", fontsize=8, color="gray")
    ax.plot(tes, to_r, "o-", color="C3", lw=2, label="time-optimal truck–drone plan")
    ax.plot(tes, em_r, "s-", color="C0", lw=2, label="best-found energy minimum (pooled, incl. truck-only)")
    ax.set_xscale("log")
    ax.set_xticks(tes); ax.set_xticklabels(["0.3\n(model\nunits)", "1\n(parity)",
                                            "6\n(light\nelectric)", "30\n(EV van)",
                                            "60\n(diesel\nvan)"], fontsize=8)
    ax.axvspan(0.25, 2.2, color="C3", alpha=0.06)
    ax.axvspan(2.2, 75, color="C0", alpha=0.06)
    ax.text(0.55, 3.1, "trade-off regime:\ndrones buy time\nwith energy", fontsize=8.5,
            color="C3", ha="center")
    ax.text(14, 1.55, "win–win regime:\ndrones save time\nAND total energy", fontsize=8.5,
            color="C0", ha="center")
    ax.set_xlabel("truck-to-drone power ratio  $t_e$  (truck power / drone empty power)")
    ax.set_ylabel("total energy relative to truck-only")
    ax.set_title("Whether drones cost or save energy depends on what they ride with")
    ax.legend(fontsize=8.5, loc="upper right"); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(f"{FIG}/calibrated_regimes.png", bbox_inches="tight")
    plt.close(fig)
    print("saved calibrated_regimes.png")


def fig_phase_chart():
    """Operating chart: (a) two-regime energy curve vs te (as fig_calibrated panel);
    (b) the exact green-and-fast boundary r* across the capability grid, with
    vehicle-class bands; (c) energy attainable under hard service-time caps on the
    completed fronts. Replaces calibrated_regimes as one display item."""
    import json as _json
    cal = _json.load(open(f"{RES}/H3-time-energy/results/h3_calibrated.json"))
    h3 = _json.load(open(f"{RES}/H3-time-energy/results/h3_v2.json"))
    cache = _json.load(open(f"{RES}/TSPREF/results/tspref_cache.json"))
    rst = _json.load(open(f"{RES}/H2-design-space/results/rstar_v2.json"))
    svc = _json.load(open(f"{RES}/H3-time-energy/results/service_level_v2.json"))

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3))

    # (a) two-regime curve; the te=0.3 endpoint uses the COMPLETED per-instance fronts
    # (tie-cleaned time-optimal energy), matching the pooled semantics of the te>=1 points
    h3f = _json.load(open(f"{RES}/H3-time-energy/results/h3_frontier.json"))
    ax = axes[0]
    tes = [0.3, 1.0, 6.0, 30.0, 60.0]
    to_r, em_r = [], []
    t_, e_ = [], []
    for sd in range(15):
        ref_e = 0.3 * cache[f"uniform-n50-seed{sd}-center-euclidean"]["length"]
        front = sorted(set(tuple(p) for p in h3f["combined_fronts"]["50"][str(sd)]))
        t0 = front[0][0]
        e_timeopt = min(e for t, e in front if t <= t0 * (1 + 1e-12))
        t_.append(e_timeopt / ref_e)
        e_.append(min(e for _, e in front) / ref_e)
    to_r.append(np.mean(t_)); em_r.append(np.mean(e_))
    for te in ("1.0", "6.0", "30.0", "60.0"):
        s = cal["by_te"][te]
        to_r.append(s["time_opt_vs_truck_only_energy"]["mean"])
        em_r.append(s["energy_min_vs_truck_only_energy"]["mean"])
    ax.axhline(1.0, color="gray", lw=1.2, ls="--")
    ax.text(0.32, 1.04, "truck-only energy", fontsize=8, color="gray")
    ax.plot(tes, to_r, "o-", color="C3", lw=2, label="time-optimal truck–drone plan")
    ax.plot(tes, em_r, "s-", color="C0", lw=2, label="best-found energy minimum")
    ax.set_xscale("log")
    ax.set_xticks(tes)
    ax.set_xticklabels(["0.3\n(model)", "1\n(parity)", "6\n(light\nelectric)",
                        "30\nEV\nvan", "60\ndiesel\nvan"], fontsize=7.5)
    ax.axvspan(0.25, 2.2, color="C3", alpha=0.06)
    ax.axvspan(2.2, 75, color="C0", alpha=0.06)
    ax.set_xlabel("truck-to-drone power ratio $t_e$")
    ax.set_ylabel("total energy relative to truck-only")
    ax.set_title("(a) two regimes of the time–energy relation", fontsize=10)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3, which="both")

    # (b) exact boundary r* across the capability grid (E=1), pooled n=20+50
    ax = axes[1]
    alphas = [1.0, 1.5, 2.0, 2.5, 3.0]
    recs = rst["per_record"]
    for m, color, mk in ((1, "C0", "o"), (2, "C1", "s"), (3, "C2", "^")):
        med = [np.median([r["rstar"] for r in recs
                          if r["ai"] == ai and r["ei"] == 1 and r["m"] == m])
               for ai in range(5)]
        ax.plot(alphas, med, mk + "-", color=color, lw=2, label=f"$m$={m}")
    ax.set_yscale("log")
    ax.axhspan(1, 2, color="C3", alpha=0.08)
    ax.axhspan(5, 10, color="gray", alpha=0.10)
    ax.axhspan(20, 60, color="C0", alpha=0.08)
    ax.text(1.02, 1.35, "drone-comparable", fontsize=7.5, va="center")
    ax.text(3.02, 7.1, "light electric", fontsize=7.5, va="center")
    ax.text(1.02, 35, "van-class", fontsize=7.5, va="center")
    ax.set_xlim(0.93, 3.75); ax.set_ylim(1, 70)
    ax.set_xticks(alphas)
    ax.set_yticks([1, 2, 5, 10, 30, 60]); ax.set_yticklabels([1, 2, 5, 10, 30, 60])
    ax.set_xlabel(r"drone speed ratio $\alpha$")
    ax.set_ylabel("median boundary ratio $r^{*}$")
    ax.set_title("(b) green-and-fast boundary $r^{*}=E_D/(L_0-D_T)$", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", title="drones", title_fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # (c) energy attainable under hard service-time caps (completed fronts)
    ax = axes[2]
    caps = ["5", "10", "20"]
    x = np.arange(3)
    for off, n, color in ((-0.19, "20", "C0"), (0.19, "50", "C3")):
        med = [svc["per_n"][n]["completed_reduction_pct"][c]["median"] for c in caps]
        lo = [med[i] - svc["per_n"][n]["completed_reduction_pct"][c]["iqr"][0]
              for i, c in enumerate(caps)]
        hi = [svc["per_n"][n]["completed_reduction_pct"][c]["iqr"][1] - med[i]
              for i, c in enumerate(caps)]
        ax.bar(x + off, med, width=0.36, color=color, alpha=0.85,
               yerr=[lo, hi], capsize=3, label=f"$n$={n}")
    ax.set_xticks(x)
    ax.set_xticklabels(["+5%", "+10%", "+20%"])
    ax.set_xlabel("allowed increase over minimum delivery time")
    ax.set_ylabel("total-energy reduction (%)")
    ax.set_title("(c) energy recovered under service-time caps", fontsize=10)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(f"{FIG}/phase_chart_v2.png", bbox_inches="tight")
    plt.close(fig)
    print("saved phase_chart_v2.png")
