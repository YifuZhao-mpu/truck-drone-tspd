# When truck-paired drones save time and energy — capability interactions and a power-ratio boundary (TSP-D)

A reproducible, optimality-validated solver for the **min-makespan Travelling Salesman Problem
with Drone (TSP-D)**, used as a consistent instrument to map (i) *when* a drone helps across
speed ratio × endurance × fleet size, and (ii) the **time–energy trade-off** of truck–drone
delivery. Pure metaheuristic / OR (no deep learning). Target venue: *Scientific Reports*.

The manuscript is under review at *Scientific Reports*; the citation will be added here on
publication (see `CITATION.cff`). `research-tree.yaml` / `research-log.md` document the full
research record, including the internal adversarial review rounds; `MANIFEST.md` identifies
this release. Manuscript sources will be added on publication.

## Key results (v2, post-review; savings vs best-found LKH truck-only references)
- **Solver:** an ALNS over the visiting order with an exact dynamic-programming sortie split;
  reproduces published exact optima exactly, **mean gap 0.13%** (CI 0.04–0.23, 80% solved
  exactly, brute-force-exact at n=8). **Controlled decoder×operator factorial:** replacing the
  exact split with a greedy decoder costs **+4.3%** (p<1e-8); replacing sortie-aware moves
  with generic ones costs **+0.5%** (p=0.0009); no interaction — the decoder is the dominant
  lever by ~9×, and survives an equal-wall-clock comparison.
- **Design-space map (full 5×4×3 factorial):** saving is a diminishing-returns surface; α=2,E=1:
  **34.3/45.4/51.4%** (n=20) and **34.6/45.8/52.0%** (n=50) for m=1/2/3, below the ceiling
  1−1/(αm+1); endurance reversals ≤0.04pt (solver noise). **Interactions:** speed and fleet
  size are complements (2nd drone +7.3–9.4pt at α=1 → +12.5–14pt at α=3); short endurance gates
  fleet value only in sparse, small instances (fixed-area n-dependent pattern); at n=100, E=0.5 ≈ unlimited endurance.
- **Robustness (16 deconfounded cells):** saving persists 25.7–38.6% across
  {uniform,clustered}×{center,corner depot}×{L2,L1} at n=50/100; demand pattern dominates
  (−7pt@n=50 shrinking to −2.5pt@n=100), depot placement nearly irrelevant (−0.5–1.9pt),
  rectilinear truck metric helps drones (+1.9–4pt).
- **Time–energy trade-off (total system energy = drone + truck):** per-instance non-dominated
  fronts are non-degenerate on 60/60 instances; the time-optimal plan uses **~3.6× the total
  energy** of the energy-minimal (≈truck-only) plan; completed-front median **knee +20.1%/−47.2%**
  (n=20) and **+18.1%/−43.0%** (n=50; sampled-grid values +19.4/−45.1 and +15.6/−43.5); knees survive halving/doubling payload and truck
  energy coefficients.
- **Exact green-and-fast boundary (v8):** under the affine model a plan saves total energy
  iff the truck-to-drone power ratio exceeds **r\* = E_D/(L0 − D_T)**; over all 3,600
  archived factorial plans median r\* = **2.98** (range 1.2–19.8), 84.7% green at r=6,
  100% at r≥30; faster drones lower the boundary (5.6→1.7 across α), added drones raise it
  (2.36/2.90/3.26) — `experiments/H2-design-space/results/rstar_v2.json`.
- **Service-time caps on completed fronts (v8):** allowing +5/+10/+20% over minimum delivery
  time recovers a median 16.8/24.6/39.3% (n=20) and 17.9/29.0/43.1% (n=50) of total energy;
  the exact decoding-level completion adds a median 3.3/2.0pt (max 23.0/13.9) at the +10%
  cap over refined weighted sums — `experiments/H3-time-energy/results/service_level_v2.json`.
- **Street-network transfer as prediction error (v8):** fleet-increment errors all <1pt with
  bootstrap CIs covering 0 (Manhattan −0.38 [−1.06,0.31] / +0.37; Paris +0.31 / +0.20);
  absolute saving matches the L1-circuity proxy in Manhattan (+0.9 [−0.6,2.5]) but exceeds
  it in Paris (+3.7 [2.2,5.2]) — `experiments/REALNET/results/transfer_test_v2.json`.

## Environment
```
./.venv (pinned, see requirements-lock.txt): python 3.13, numpy 2.1, numba 0.66, scipy, matplotlib
LKH-3.0.9 (built from source, binary at ~/.local/bin/LKH) for best-found TSP references
CPU only (40 cores used)
```

## Reproduce (v2 pipeline, post-review)
```bash
cd src && P=../.venv/bin/python
$P run_tspref.py         # best-found LKH truck-only references, all instance sets -> TSPREF/results/
$P run_validation_v2.py  # benchmark gap + DP==sim + brute force + span sensitivity + timing -> VALIDATION/results/
$P run_h3_v2.py          # time-energy: consistent total-energy objective, per-instance fronts -> h3_v2.json
$P run_h2_v2.py          # FULL alpha x endurance x m factorial, symmetric decoding -> h2_v2.json
$P run_p0_v2.py          # robustness: depot/demand deconfound + n=100 surface + ablation + budget diag -> p0_v2.json
$P run_ops_factorial.py  # decoder x operator 2x2 causal factorial -> ops_factorial_v2.json
$P run_multi_exact.py    # multi-drone exact validation 120/120 -> VALIDATION/results/
$P run_h3_calibrated.py  # physically calibrated two-regime energy analysis -> h3_calibrated.json
$P run_h3_frontier.py    # frontier completion (refinement + exact decode Pareto) -> h3_frontier.json
$P run_realnet.py        # real OSM street-network case (2 districts) -> REALNET/results/
$P analyze_rstar.py      # exact green-and-fast boundary r* from archived artifacts -> rstar_v2.json
$P analyze_service_level.py  # energy under +5/10/20% service-time caps -> service_level_v2.json
$P analyze_transfer.py   # street-network transfer prediction-error test -> transfer_test_v2.json
$P verify_frontier_fields.py  # provenance: re-decode 10,455/10,546 front points, value-match the rest
$P rebuild_calibrated_endpoints.py  # reproduce pooled by_te endpoints to <=7e-5 from archive
$P count_words.py        # committed SciRep word-cap counter (title/abstract/main text/captions)
$P figures.py v2         # v2 figures -> drafts/figures/*_v2.png (dir auto-created)
$P -c "from figures import fig_phase_chart; fig_phase_chart()"  # phase chart (Fig. 6)
```
v1 drivers (run_h1/h2/h3/p0/p1.py) are retained for provenance; their known defects and all
protocol deviations are documented in `experiments/protocol-deviations.md` and
`RESEARCH_REVIEW_REPORT_2026-07-16.md`.

## Layout
```
src/         problem.py (instances, JIT DP splits: single / multi-drone / lambda-aware), exact.py
             (independent checker + brute-force oracle), baselines.py, alns.py (ONE ALNS for all
             experiments), energy.py (Dorling model + evaluate_solution audit), tsp_ref.py (LKH),
             benchmark.py (Agatz/Bouman loader), run_*_v2.py drivers, figures.py
experiments/ H1-alns-backbone/ (protocol.md = LOCKED pre-registration), H2-design-space/,
             H3-time-energy/, P0-robustness/, TSPREF/, VALIDATION/, protocol-deviations.md
literature/  survey.md, references.bib
research-tree.yaml, research-log.md, RESEARCH_REVIEW_REPORT_2026-07-16.md (cross-model review)
MANIFEST.md  release identity: source tag/commit, canonical artifacts, known limitations
```

## Benchmark data
The Agatz/Bouman TSP-D instances are not vendored. Fetch them with:
`git clone https://github.com/pcbouman-eur/TSP-D-Instances` then copy `uniform/` into
`benchmarks/external/uniform/`. Our evaluator reproduces their published exact optima exactly.

## Validation (stored artifacts: experiments/VALIDATION/results/validation_v2.json)
- DP-split makespan == independent event-simulator to 2e-14 (randomized instances/orders/E/m).
- ALNS == brute-force optimum over all orders on 10/10 instances at n=8.
- Mean 0.125% gap (95% CI 0.045–0.231) to published exact optima (n=11–17), 80% solved exactly.
- Span-cap sensitivity: caps >= 10 lossless vs unrestricted decode at every m; cap 6 loses up to 5%.
- Savings denominators: best-found LKH-3 references, no optimality certificate (never worse than the old internal heuristic
  on 540/540 instances; that heuristic averaged +3.1%, max +13.3% — see TSPREF/results/).

## Matched wall-clock comparison vs BOTH public TSP-D implementations
Neither package is vendored; both pinned (isolated Julia 1.10.4 depot, env status archived):
- TSPDroneHGATAC.jl (Sasanm88; the HGA-TAC of Mahmoudinazlou & Kwon) @ 8f1f345c93ed...
- TSPDrone.jl (chkwon; the DPS/TSP-ep-all of Bogyrbayeva et al. — NOTE: this package is
  DPS, not HGA-TAC; the pre-v7.6 manuscript mislabeled it, caught by targeted audit) @ f42d27c0...
Protocol: identical per-instance wall budgets (30/35/140/430s = our standard best-of-3 measured
time), every side single-threaded + JIT-warmed, complete default-settings runs until deadline,
HARD cutoff applied symmetrically in analysis, ALL replicates archived. Drivers:
`src/run_sota_matched.py` (ours), `src/run_sota_matched.jl` (DPS), `src/run_sota_hgatac.jl`
(HGA-TAC), `src/analyze_sota_matched.py` -> `experiments/SOTA/results/sota_matched.json`.
RESULT (hard cutoff): vs HGA-TAC 25W/1L/19T (benchmark gaps ours 0.00-0.20% vs 0.00-0.43%;
synthetic means 1.7-3.1% lower); vs DPS 27W/0L/18T (gaps vs 0.28-0.83%; 1.6-3.4% lower).
Compute matched, tuning effort not; instance families = ours + public benchmark. The v1
single-default-run DPS comparison is kept in sota_compare.json for history.
