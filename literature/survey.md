# Literature Survey — Truck–Drone Collaborative Last-Mile Delivery Routing

**Project:** A reproducible sortie-aware metaheuristic for the TSP-with-Drone, and a systematic
characterization of *when drones help* (speed × endurance × fleet) including the time–energy trade-off.
**Target venue:** Scientific Reports (Nature Portfolio; CAS 3区).
**Date:** 2026-06-27. **Method stance:** pure metaheuristic / OR (no deep RL — deliberate non-overlap with the local `evrp-drl` DRL project).

---

## 1. Review protocol

- **Research area:** collaborative truck + drone routing for last-mile delivery (FSTSP / TSP-D / VRP-D family).
- **Key questions:** (i) what is the canonical, benchmark-backed formulation a metaheuristic should target? (ii) what is the strongest current solver and what gaps remain? (iii) how do delivery-time savings depend on operating parameters, and is this systematically mapped? (iv) is the time–energy trade-off characterized? (v) reproducibility status.
- **Databases / sources:** Google Scholar, arXiv, ScienceDirect (Transportation Research C/D/E, EJOR, C&OR, Omega), Springer (Transportation Science, Networks, Optimization Letters, J. Heuristics), IEEE, MDPI (Drones), Nature (Sci Rep, Nat Commun).
- **Inclusion:** peer-reviewed or strong preprints on truck-drone routing formulations, exact/heuristic/metaheuristic algorithms, savings/energy/emissions analyses, surveys 2015–2026.
- **Exclusion:** drone-only path planning, pure facility location, non-routing UAV applications.
- **Screening:** 4 parallel thematic sweeps (formulations/exact; metaheuristics/benchmarks; savings/energy/Pareto; recent advances/surveys/reproducibility). ~70+ papers identified; ~40 retained as directly relevant; details below. Items the agents could not verify verbatim are flagged *(unverified)* and must be checked against primary PDFs before final citation.

---

## 2. Evidence map (by theme)

### Theme A — Formulations & exact methods
- **FSTSP & PDSTSP** — Murray & Chu (2015), *Transp. Res. C* 54:86–109. Founding paper. One truck + one drone; drone serves **one** customer per sortie and rejoins the truck at a **distinct** node; **objective = minimize makespan** (all served and both vehicles back at depot). PDSTSP = depot-launched parallel drones + one truck TSP. The FSTSP MILP is weak (struggles at ~10 customers).
- **TSP-D** — Agatz, Bouman & Schmidt (2018), *Transp. Sci.* 52(4):965–981. The "clean" reference model: same min-makespan objective, but **more relaxed** than FSTSP — drone may launch/land at the **same** node (truck waits), route may revisit nodes, **all** customers drone-eligible, **endurance = ∞** in the base model, zero launch/recovery times. Establishes the **speed-ratio convention α ∈ {1,2,3}** (drone speed = α × truck), the public instance generator, and worst-case approximation ratios. **This is the recommended target formulation** (best benchmark + exact-optimal ecosystem).
- **Min-cost TSP-D** — Ha et al. (2018), *Transp. Res. C* 86:597–621. Different objective: transport cost + waiting cost. GRASP + optimal **split** of a TSP tour into a TSP-D solution.
- **mFSTSP** — Murray & Raj (2020), *Transp. Res. C* 110:368–398. One truck + **multiple heterogeneous drones**, makespan, with service times + endurance; 3-phase heuristic to ~100 customers + 4 drones.
- **VRP-D** — Wang, Poikonen & Golden (2017), *Optim. Lett.* 11:679–697; Poikonen et al. (2017), *Networks* 70:34–43. Multiple trucks each with multiple drones; **worst-case savings bound 1 − 1/(αk+1)** for k drones at speed ratio α.
- **Exact SOTA** — **Roberti & Ruthmair (2021), *Transp. Sci.* 55(2):315–335**: branch-and-price on set-partitioning + ng-route → **optimal up to 39 customers** (the record). Bouman et al. (2018), *Networks* 72:528–542: exact **DP** for TSP-D (~10–20). Dell'Amico et al. (2021), *Optim. Lett.* / arXiv:2107.13275: **published optimal solutions** for 9 settings (validation oracle). Dell'Amico et al. (2021/2022): exact FSTSP B&B (~20).

**Standard assumptions (locked for this work):** drone speed = α × truck (α∈{1,2,3}); Euclidean distances for both (drone "as the crow flies"); **one customer per sortie**; launch/land at **nodes only**; first-arriver **waits** (synchronization); finite endurance per sortie (∞ allowed as a limiting case).

### Theme B — Metaheuristics & benchmarks
- **Strongest metaheuristic:** **HGA-TAC** — Mahmoudinazlou & Kwon (2024), *EJOR* 318(3):719–739. Hybrid GA with **type-aware chromosomes** + **DP decoding** of optimal rendezvous + local search. **New best-knowns on 538/920 TSP-D and 74/132 FSTSP**; beats Ha-HGA (2020, *J. Heuristics* 26:219–247) and Agatz TSP-LS in quality and time. Code lineage: `github.com/chkwon/TSPDrone.jl` (Julia).
- **ALNS precedent (VRP-D):** Sacramento, Pisinger & Røpke (2019), *Transp. Res. C* 102:289–315 — random/worst/Shaw removal + greedy/regret insertion; scales to 200 customers. **ALNS is the standard for multi-truck VRP-D but barely used for single-vehicle TSP-D/FSTSP** (GA/VNS dominate there).
- **VNS/Tabu:** Freitas, Penna & Toffolo (2023), *EURO J. Transp. Logist.* 12:100094 — hybrid GVNS+Tabu + stronger MIP; improved >80% of best-knowns on 1415 instances; **open code** `github.com/tuliotoffolo/fstsp`.
- **Emerging bar:** Xu & Carlsson (2026), arXiv:2602.20310 — LKH-based 3-phase unified framework over FSTSP/TSP-mD/VRP-D (consolidates 7 instance sets). Ren et al. (2024), *EJOR* 318(2) — decomposition over 4 VRP-D variants, 189 new best-knowns, public instances.
- **Public benchmark instances (with optima/best-known):** Agatz/Bouman TSP-D instances `github.com/pcbouman-eur/TSP-D-Instances` (uniform / single-center / double-center; α=1,2,3; n=10..100; incl. exact small-n); Murray&Chu FSTSP (72 × 10-cust); Poikonen (100 × 9-cust); Ponza (≤200, in tuliotoffolo/fstsp); Dell'Amico **optima** at or.unimore.it.
- **Operator gap:** generic TSP/VRP moves (2-opt, Or-opt, relocate) + a make-drone/make-truck toggle + DP rendezvous re-optimization are near-universal; **explicit launch/rendezvous reassignment and drone-customer swap as *first-class, named* ALNS destroy/repair operators are largely absent.**

### Theme C — When/why drones help; energy; Pareto
- **Speed ratio (most-studied axis):** Agatz et al. (2018) — savings increase with α but **diminishing returns**. Worst-case ceiling **1−1/(αk+1)** (Wang/Poikonen/Golden). Continuous "horsefly" efficiency ∝ **√(speed ratio)** (Carlsson & Song 2018, *Manag. Sci.* 64:4052–4069). Asymptotic constant: at **α=2, one drone ⇒ makespan ≈ 70% of truck-only (~30% typical savings)** (Lee, Hwang & Kwon 2026, arXiv:2603.00328) — far below the worst-case envelope, quantifying the best-case/typical gap.
- **Endurance / drops:** savings **saturate** beyond a moderate endurance threshold; "faster + longer-endurance is not always best" (Tamke & Buscher 2021, *C&OR* 137:105540; Schaumann et al. 2024, arXiv:2403.18091). Most savings come from **parallelizing the truck's work / cutting driver time**, not drone energy per se.
- **# drones:** diminishing returns; the worst-case bound is concave in k; truck route becomes the binding (Amdahl-style) constraint.
- **Spatial:** drones help most in **sparse/rural/large** areas, least in **dense urban** cores.
- **Energy models (implementable):** affine power-in-payload **E_leg = (a·m_payload + b)·(d/v)** (Dorling 2017); cruise **P = g·m_total·v/(370·η·r) + p_avionics** (D'Andrea 2014); induced **P ∝ (m·g)^{1.5}/√(2ρA)** (Stolaroff 2018). Emissions = energy × grid intensity (drone) + distance × fuel factor (truck); Figliozzi (2017) lifecycle CO₂e. Comparison: Zhang et al. (2021), *Transp. Res. D* 90:102668.
- **Emissions evidence (incl. a Sci Rep paper):** Stolaroff et al. (2018), *Nat. Commun.* 9:409 — small-package drone ~54% less CO₂e than truck in clean grids, worse for heavy packages/dirty grids. **Raghunatha et al. (2023), *Scientific Reports* 13:11814** — drones beat diesel rurally but lose in cities; electric trucks dominate. (Confirms SciRep publishes this topic with a sustainability framing.)

### Theme D — Surveys, open problems, reproducibility
- **Surveys:** Otto et al. (2018, *Networks*); Chung et al. (2020, *C&OR* 123:105004); Macrina et al. (2020, *Transp. Res. C* 120:102762); Moshref-Javadi & Winkenbach (2021, *ESWA* 177:114854); Liang & Luo (2022, *JORSC* 10:343–377); Dang et al. (2024, *Drones* 8(10):550); Zhou et al. (2025, *Swarm Evol. Comput.* 92:101780); Cengiz et al. (2025, *RAIRO-OR* 59:3169–3205).
- **Most-cited open problems (deduplicated):** (1) multi-truck/multi-drone at scale; (2) stochastic/dynamic/real-time; (3) realistic nonlinear energy/payload; (4) scalable exact/metaheuristics; (5) **standardized benchmarks + reproducibility + real-world validation**; (6) multi-objective/sustainability.
- **Reproducibility:** explicitly fragmented — "no universally accepted benchmark … fragmented evaluations." Code release is the exception; among recent classical papers only a handful (Mahmoudinazlou&Kwon 2024; Ren 2024; Freitas/Toffolo 2023; Xu&Carlsson 2026) release code/instances.

---

## 3. PRISMA-style audit (informal)

```
Records identified (4 thematic web sweeps):     ~70+
Retained as directly relevant:                  ~40
Foundational/formulation:                        8
Algorithmic (metaheuristic/exact):              ~15
Savings/energy/Pareto:                          ~12
Surveys:                                          8
```

## 4. Open problems (→ research tree)
1. No **unified, consistent-solver empirical design-space map** of truck-drone delivery savings across **speed ratio × endurance × #drones**; findings are fragmented across papers, objectives, and solvers.
2. The **time-vs-drone-energy Pareto frontier** is not well-characterized under a credible energy model + strong solver (existing multi-objective work proxies truck CO₂ by route length, treats drone energy crudely).
3. **Sortie-specific operators** (launch/rendezvous reassignment, drone-customer swap) are not first-class in metaheuristics.
4. **Reproducibility/standardization**: few open, code-backed, statistically-rigorous baselines for TSP-D.
5. Multi-truck/multi-drone scalability; stochastic/dynamic; nonlinear energy (broader, mostly out of scope here).

## 5. Underexplored areas (→ hypotheses)
- A reproducible, open **ALNS with first-class sortie-aware operators** for single-truck TSP-D (ALNS is under-used here vs GA/VNS).
- A **3-axis savings surface** (σ × endurance × m) under ONE solver, anchored to the 1−1/(αk+1) ceiling and the ~30% (α=2) asymptotic reference.
- A standardized **makespan-vs-drone-energy Pareto** characterization with a simple payload/speed energy model and decision rules (saturation thresholds, switch-over points).

## 6. Honest novelty assessment
- H1 (sortie-aware ALNS) is **incremental** on its own — HGA-TAC is near-optimal and SOTA-chasing is risky. Its value is as a **reproducible enabling baseline** + an operator-level contribution + the engine for H2/H3.
- H2 (design-space map) is the **headline**: defensible as *unification + consistency + an explicit 3-axis surface with decision rules*, NOT as the first study of any single axis (must cite Agatz/Wang/Lee and frame as consolidation).
- H3 (time-energy Pareto) is a **genuinely underexplored** secondary contribution with strong SciRep sustainability fit.
- Combination H1+H2+H3 = a coherent, reproducible, application-relevant paper well-matched to Scientific Reports' taste for clear empirical findings + open data + sustainability.
