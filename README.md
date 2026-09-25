# Drone capabilities jointly shape truck–drone time savings, but energy is saved only above a route-specific truck-to-drone energy ratio (TSP-D)

A reproducible computational study of the **min-makespan Travelling Salesman Problem
with Drone (TSP-D)**, using one open solver to map (i) how speed, endurance and fleet
size interact and (ii) when a delivery-time saving is also a movement-energy saving.
Fixed-order decoding and small instances have exact checks; the larger-instance
solutions are heuristic estimates and carry no optimality or near-optimality claim.

**Development repository:** https://github.com/YifuZhao-mpu/truck-drone-tspd. The
submitted `v1.0-submission` tag remains available for provenance. The fully reproducible
`v1.1-revision` release is archived at https://doi.org/10.5281/zenodo.22297604.

See `drafts/latex/main.tex` for the authoritative manuscript (`drafts/paper.md` is a pointer
with key numbers) and `research-tree.yaml` / `research-log.md` for the research record.

## Major revision (v1.1, 2026-09): every solver seed retained, plus six new experiments

The round-1 revision reran every archived experiment with **all three solver replicates,
routes and runtimes retained** (`src/run_h1_v3.py`, `run_h2_v3.py`, `run_h3_v3.py`,
`run_p0_v3.py`; outputs `*_v3.json` + resumable `*_v3.jsonl` checkpoints) and added the
experiments pre-registered in `revision/revision_protocol.md` (locked before any
run; post-lock amendments only as rows 16–19 of `experiments/protocol-deviations.md`).
`src/check_rerun_reproduction.py` re-verifies that the v3 best-of-three replicate equals
the archived v2 value on **every stored field and route** (H2: 21,600 fields + 3,600
routes exact; H3: 8,820 + 1,260 exact; P0: 3,480 fields exact; H1: 750 fields within
1e-12). Run-by-run outcomes and gate results: `revision/run_ledger.md`.

- **All-seed restatement:** central cell 34.3% (best-of-3 34.6%); best-of-3 selection
  advantage 0.15 pt pooled over instances (cells 0.00–0.61); robustness cells 25.5–38.1%
  (best-of-3 25.7–38.6%); H1 exact-reference gap 0.15% (best-of-3 0.13%).
- **Speed–fleet interaction (`h2_v3.json`):** second-drone increment +5.5 pt (n=20) and
  +4.8 pt (n=50) larger at α=3 than α=1 (CIs exclude 0); third-drone +2.9 pt at n=50 but
  +0.7 pt with CI −0.2–1.5 at n=20 → reported as suggestive, removed from the abstract.
- **Fixed-density control (`run_density_control.py` → `density_control_v3.json`):** the
  short-range penalty S(∞)−S(0.5) at α=2, m=1 is 16.8/12.3/11.9 pt at n=20/50/100 under
  fixed density vs 16.8/3.1/0.3 under fixed area — the fixed-area fade is mostly density.
- **Operational friction (`run_operational_sensitivity.py`; decoder options `t_service`,
  `t_launch`, `t_recover`, `eligible` validated by `validate_friction.py`):** launch +
  recovery of 0.25τ/0.50τ each cut the saving to 19.9%/9.0%; service time raises the
  adjusted saving (36–38%) while hybrid makespan rises 19–37% (both metrics reported);
  eligibility 75%/50% → 30.7%/23.1%; 20% reserve → 34.1%.
- **Execution-noise stress test (`run_execution_noise.py`):** fixed routes under mean-one
  lognormal leg times, CV 0.10/0.20 → realized saving 33.4%/31.5% vs 34.3% nominal.
- **Decoder-strength curve (`run_decoder_strength.py`):** greedy / span-4 / span-6 /
  span-12 decoders in one ALNS loop at equal iterations: 31.2/33.9/34.2/34.4% saving;
  operator effect 0.2–0.7%, baseline-dependent (the former "≈9×" ratio is withdrawn).
- **Energy extensions (`analyze_energy_extensions.py` → `energy_extensions_v3.json.gz`):**
  generalized boundary r\*(i,h,c) over all 10,800 routes (nominal median 2.93; harshest
  accounting 3.86; 20% reserve re-decode); four published coefficient pairs (Rodrigues
  2022, Stolaroff 2018): 100% of time-oriented routes also save energy, median 32–39%.
- **Held-out comparator cross-check (`run_heldout_comparators.py` + pinned Julia env in
  `experiments/SOTA/julia/`):** 25 never-used instances, 4× budgets, 3 seeded restart loops
  per solver — ours equals the pooled best-found on 25/25; HGA-TAC median gap
  0.0/3.1/4.7% and DPS 1.9/4.2/4.4% by size (defaults; tuning unmatched; not a bound).
- **Two-district capability cross (`run_realnet_capability.py`):** Manhattan 39.1%,
  Paris 41.9% at (α,E)=(2,1); identical ordering of the speed/endurance levers.

Reproduce (36-core machine, ~26 wall-hours in total; every driver is checkpointed):
```bash
cd src && P=../.venv/bin/python
$P run_h2_v3.py; $P run_h1_v3.py; $P run_p0_v3.py; $P run_h3_v3.py   # §A all-seed reruns
$P check_rerun_reproduction.py                                         # all four gates
$P run_decoder_strength.py; $P run_execution_noise.py; $P analyze_energy_extensions.py
$P run_density_control.py; $P run_operational_sensitivity.py; $P run_realnet_capability.py
$P run_heldout_comparators.py --workers 24   # needs Julia 1.10.4 + experiments/SOTA/julia
$P figures_v3.py all                          # revision figures -> drafts/latex/figures/*_v3
```

## Environment
```
./.venv (pinned, see requirements-lock.txt): python 3.13, numpy 2.1, numba 0.66, scipy, matplotlib
LKH-3.0.9 (built from source, binary at ~/.local/bin/LKH) for best-found TSP references
CPU only (40 cores used)
```

## Legacy v2 pipeline (retained for provenance; superseded by v1.1)
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
$P figures.py v2         # v2 figures -> drafts/figures/*_v2.png
$P -c "from figures import fig_phase_chart; fig_phase_chart()"  # phase chart (Fig. 6)
```
v1 drivers (run_h1/h2/h3/p0/p1.py) are retained for provenance; their known defects and
all protocol deviations are documented in `experiments/protocol-deviations.md`.

## Layout
```
src/         problem.py (instances, JIT DP splits: single / multi-drone / lambda-aware), exact.py
             (independent checker + brute-force oracle), baselines.py, alns.py (ONE ALNS for all
             experiments), energy.py (Dorling model + evaluate_solution audit), tsp_ref.py (LKH),
             benchmark.py (Agatz/Bouman loader), run_*_v2.py drivers, figures.py
experiments/ H1-alns-backbone/ (protocol.md = LOCKED pre-registration), H2-design-space/,
             H3-time-energy/, P0-robustness/, TSPREF/, VALIDATION/, protocol-deviations.md
literature/  survey.md, references.bib
revision/    locked revision protocol, run ledger and point ledger (release copies)
literature/  bibliography and source-positioning notes
MANIFEST.md + SHA256SUMS are generated in the DOI release tree
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
