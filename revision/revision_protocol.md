# Revision experiment protocol (round 1)

**STATUS: LOCKED v3 upon commit of this revision.** Drafted and rewritten 2026-09-01 after an
adversarial pre-registration review and verification (traces
`.aris/traces/research-review/2026-09-01_run01/002-protocol-review.response.md` and
`003-protocol-verify.response.md`; findings and newly detected conflicts addressed
before lock). The commit introducing this locked text is the scientific lock; its
hash is inserted by one immediate administrative follow-up commit with no design or
code changes: `LOCK COMMIT: 592708c7778930231eb5e97a393aa4c1b8f5ee27`. No result used as revision evidence is generated
before lock. One pre-lock implementation dry-run of `validate_friction.py` and
interrupted H2/P0 canaries were used to debug the drivers; their artifacts are marked
pre-lock, invalid as evidence, and replaced by post-lock runs of the locked code.
Amendments after lock go to the repository's protocol-deviation table. The invalid
development outputs are preserved as `*.prelock.*`; the validator and every canary
are rerun after lock before any production chain.

**Lineage.** Solver code, iteration budgets (9,000/16,000/16,000 at n=20/50/100),
span cap 12, instance seeds 0–29, and the truck-only reference protocol are inherited
from the **frozen v2 amended execution protocol** — the original lock *as amended by
the disclosed deviation table* (`experiments/protocol-deviations.md`, notably rows 9
and 15). The truck-only references are **best-found LKH-3 tours with no optimality
certificate**; the deviation table's row-9 phrase "certified reference" is a wording
defect and a correction row is appended as part of §A.

**Global analysis contract (every section).** Sampling unit = instance;
replicate-first-then-instance aggregation as specified in §A applies to B–H as well.
All bootstrap CIs resample whole instances (paired across arms sharing instances;
20,000 resamples, seed recorded). Optimization drivers expected to run longer than an
hour write per-job JSONL checkpoints and assemble final JSON at the end (a crash loses
one job, not the run); deterministic post-processing scripts write one final artifact.
Every optimization artifact records config, per-job wall time, and the git commit of
the producing code.

## A. All-seed reruns (R5.10, R5.13)

Drivers `run_h2_v3.py`, `run_p0_v3.py`, `run_h3_v3.py`, `run_h1_v3.py` re-execute the
archived designs with **identical** jobs, budgets, and seeds, retaining **every solver
replicate** (makespan, energy audit, route, wall time, solver seed) instead of the
winner only. Outputs go to new files (`*_v3.json` + `.jsonl` checkpoints); the v2
artifacts stay frozen. H1's validation/ablation is included because it too discarded
replicates and still supports revised claims; the decoder×operator factorial is not
rerun here — §E replaces it wholesale.

- **Reproduction gate (pre-committed)**: for every job, the v3 best-of replicate
  (selected exactly as the archived driver selected: H2/P0/H1 strict-< on makespan,
  ascending seeds; H3 strict-< on the scalarized `objective`) must equal the archived
  v2 raw row on **every stored field** — makespan, saving_pct, e_drone, e_truck,
  e_total, truck_dist, n_sorties, objective, and the route as an exact integer
  sequence — bit-identical floats, not makespan alone. Field scope = whatever the
  archive stores per driver: H2/H3 archives include routes, so their routes are
  compared; the P0 and H1 archives store numeric fields only, so their gate is all
  archived numeric fields (disclosed residual: a same-cost different route would pass
  there — with strict-< selection over bit-identical makespans this cannot change any
  reported number).
  `check_rerun_reproduction.py` verifies 100% of rows; any mismatch halts the
  revision and is investigated before anything else runs.
- **Canary gate**: an ~8-job subset per driver (covering both n sizes, m>1, λ>0, and
  clustered/manhattan arms) runs and passes the full-field gate before the chain
  launches.
- **New reported quantities** (pre-committed definitions):
  - per-cell **all-seed mean** saving: per instance, mean over the 3 replicates → cell
    mean over 30 instances (same bootstrap CI method as v2);
  - per-cell **worst-seed** saving (per-instance max makespan replicate);
  - **best-of-3 selection advantage** (the R5.10 quantity; not called "bias"): per
    instance i, G_i = S_i^best − (1/3)Σ_s S_i,s in saving points; report its mean,
    distribution and paired instance-bootstrap CI, per cell and pooled;
  - per-instance replicate spread (max−min makespan, % of best).
  - For H3, the historical best-of-3 front is kept as a separately reported object
    (the reproduction target), never merged with the seed-specific fronts.
- The bootstrap resampling unit remains the **instance**, not an individual algorithm
  replicate. Replicates are first averaged within instance. For the energy sweep we
  additionally construct one sampled frontier for each solver seed (holding that seed
  fixed across the lambda grid) and report the between-seed spread of the endpoint and
  knee summaries. Completed archive-conditional fronts remain based on the union of
  all retained orders and are never described as global fronts.
- **Decision rule**: wording branches are governed by the §I claims matrix (A's
  general principle — all-seed values are the primary restatement, best-of-3 the
  disclosed historical estimand — is instantiated there; where A and §I could be read
  differently, §I wins).
- **Repo wording fixes bundled with A** (R5.13): append a correction row to
  `experiments/protocol-deviations.md` replacing the row-9 "certified reference"
  phrase with "best-found LKH-3 reference (no optimality certificate)"; correct the
  `run_h3_v2.py` docstring's "for EVERY (lambda, seed) point" (that "seed" is the
  instance index; solver replicates were discarded — v3 states this history); mark
  the stale `h2_v2.log`/`p0_v2.log` outputs explicitly as retained v1 provenance and
  exclude them from revision evidence.

## B. Operational friction (R5.3, feeds R5.2)

New optional single-drone decoder parameters, all defaulting to the frictionless model:
- `t_service` — service duration per customer, incurred by the serving vehicle (truck
  legs: at the arrival customer; sortie operations: on the truck side at every
  truck-served customer after the launch node, including a customer rendezvous node,
  and on the drone side at the drone-served customer). Service at the launch node is
  charged to the preceding operation. Depot service is zero. Thus every customer is
  charged exactly once. Service at a customer rendezvous may overlap the drone's
  flight, but recovery occurs only after both sides have arrived and completed service.
- `t_launch`, `t_recover` — per-sortie: operation cost = `t_launch` + max(truck side,
  drone side) + `t_recover` (both vehicles engaged during launch and recovery).
- `eligible` — per-node boolean; the drone may serve customer j only if eligible.
  Masks: per instance, ONE permutation of the customers drawn from
  `default_rng((instance_seed, 0xE119))` — domain-separated from the coordinate RNG so
  masks are independent of geometry; the 75% mask = first ceil(0.75·n)=38 customers of
  that permutation, the 50% mask = first 25; nested by construction; every mask
  archived with the results.
- Endurance stays a **flight-distance** budget — deliberately the Agatz-compatible
  range convention, NOT a Murray–Chu-style battery clock coupled to elapsed sortie
  time; service/handling times do not consume it. Stated as such in Methods.
- Truck-only reference under friction: LKH length + n·`t_service` (launch/recovery do
  not apply; the truck serves every customer; equal additive service cannot change the
  distance-minimizing tour). Best-found reference, no certificate.

**Full-path propagation requirement.** t_L, t_R, t_S and `eligible` propagate through
the fast cost kernel, the full split (ops reconstruction), both ALNS closures, the
final re-decode at final_span, and the stored makespan. For **every** nonzero-friction
returned route, the driver checks in-run: fast cost = reconstructed split cost =
independent event-simulator makespan = stored makespan, within 1e-9, at both the
search span and the final span. A frictionless path anywhere in the loop cannot
survive this check.

**Validation gates (pre-committed, must pass before the experiment runs)**:
1. friction kernel at (0, 0, 0, all-eligible) equals the original kernel **exactly**
   on ≥200 random orders across n∈{10, 20, 50}, all archived parameter combinations;
2. brute-force oracle on n≤9: an independent pure-Python enumerator over **all**
   fixed-order partitions and drone-customer choices (not a re-evaluation of
   DP-selected operations) agrees with the friction DP to <1e-9 on ≥300
   (order, friction) cases with friction ON — the case set must include launch-only
   (t_L>0=t_R), recovery-only, **asymmetric t_L≠t_R**, service-only, joint friction,
   partial eligibility, consecutive sorties, customer vs depot rendezvous, and drone
   arrival before/during/after the rendezvous customer's service;
3. monotonicity: nonnegative friction and nested eligibility restriction never
   decrease the fixed-order optimum, over the whole oracle case set;
4. eligibility: with a mask forbidding all customer nodes, the decode equals the
   **same fixed order's** truck travel time plus `n*t_service` exactly (a kernel
   check, not a claim that the fixed order is the globally best truck tour);
5. after §A lands: the B base cell's three replicates reproduce the corresponding
   **H2-v3 replicates** (not merely the archived winner) bit-for-bit.

**Design** (n=50, α=2, E=1, m=1, uniform/center/euclidean, 30 instances, all 3 seeds
retained): friction levels expressed as fractions of τ̃ = the median over the pooled
multiset of consecutive-stop truck leg lengths — depot legs included — across the 30
archived LKH reference tours at n=50 uniform/center/euclidean (truck speed 1, so leg
length = leg time; one number, computed once, recorded in the output config):
- service alone: t_S ∈ {0, 0.25, 0.5}·τ̃ at t_L=t_R=0 (3 cells incl. the base);
- launch/recovery alone: t_L=t_R ∈ {0.25, 0.5}·τ̃ at t_S=0 (2 cells);
- joint: (t_S, t_L=t_R) = (0.25, 0.25)·τ̃ (1 cell);
- eligibility ∈ {75%, 50%} at zero friction (2 cells; 100% is the base).
- battery reserve: a 20% usable-range reserve, implemented as E=0.8 instead of E=1.0,
  at otherwise zero friction (1 cell).
Total 9 analyzed cells = 8 new (720 solver runs) + the base cell reused from H2-v3
(90 observations; gate 5's reproduction, not a new run).
- **Report, per cell (instance-first)**: BOTH the friction-adjusted saving vs the
  adjusted truck-only reference AND the absolute + relative hybrid-makespan change vs
  the zero-friction base — the dual metric prevents a growing denominator from
  concealing absolute deterioration. No pre-set threshold: the deliverable is the
  sensitivity curve; §I row I10 governs the wording.

## C. Execution-noise Monte Carlo (R5.2 — explicitly partial compliance)

Post-hoc stress test on **fixed routes** (no re-optimization; stated as such).
- **Routes**: all three §A H2-v3 replicates of the a2.0_E1.0_m1 n=50 cell (zero
  friction — the manuscript's nominal model) and their LKH truck-only reference tours.
- **Noise**: independent multiplicative lognormal per (vehicle, directed leg), mean 1:
  σ² = ln(1+CV²), μ = −σ²/2, CV ∈ {0.10, 0.20}. **Common random numbers**: each leg's
  multiplier is keyed by (instance_seed, realization_index, vehicle_type, from_node,
  to_node). Thus any directed leg shared between the hybrid truck route and the
  truck-only tour gets the same multiplier within a realization.
- 10,000 realizations per route; per realization, operation times recomputed from the
  fixed structure (operation cost = max of noisy sides; waits re-derived). Realized
  saving = (T_truckonly − T_hybrid)/T_truckonly.
- **Report** (per CV; replicate-first, then instance-first): median realized saving,
  5th/95th percentiles, and Pr(realized saving > 0). **Decision rule**: if the median
  realized saving drops by more than 3 percentage points vs nominal, the Results
  sensitivity subsection reports it prominently; the claim language restricts to
  deterministic travel times either way. The response letter states plainly that
  multi-customer sorties, off-node launch and asynchronous fleets are different
  formulations, not tested, with no direction of bias asserted (R5.2 partial).

## D. Fixed-density control (R5.11)

New driver `run_density_control.py`. Anchor: n=20 on the unit square (density
20/unit²). For n∈{50, 100}: scale all coordinates by s=√(n/20) (area n/20, density
constant), truck/drone distances scale by s; LKH references reuse the cached unit-square
tours × s (exact under uniform scaling). Physical endurance held fixed at the n=20
values E∈{0.5, 1, ∞} (so in unit-square terms E_eff = E/s — the point of the control).
α=2, m∈{1,2,3}, 30 instances, 3 seeds, all replicates retained. Iteration budgets: the
standard per-n budgets (16,000 at both 50 and 100). Cells: 18 new (n∈{50,100} ×
E∈{0.5,1,∞} × m); the 9 n=20 baseline cells come from **H2-v3** (same all-seed
estimand — never legacy best-of-3 against new all-seed). A degenerate s=1 identity
check (one n=20 job through the density driver) must reproduce its H2-v3 job
bit-for-bit before the grid runs. Archived fixed-area n=100 results keep their
exploratory label; D is a preregistered reviewer-requested sensitivity, not
independent confirmation.
- The pre-specified estimand is the paired endurance penalty
  Delta_E(n, m, E) = S(E=inf) - S(E) at each (n, m, finite E), using the
  within-instance all-seed mean saving S; mean + instance-bootstrap 95% CI per cell.
- **Fade contrast (pre-specified)**: fade(m, E) = Delta_E(20, m, E) −
  Delta_E(100, m, E) on the fixed-density grid, with paired instance-bootstrap CIs.
  For fixed-area continuity, H2 supplies n=20→50 contrasts for m=1–3 and P0 supplies
  the n=20→100 contrast for m=1; no fixed-area n=100 multi-drone cells are invented.
  Shared instance seeds pair every compared cell.
- **Wording (unconditional, aligned with §I row I3)**: the manuscript's densification
  wording becomes fixed-area-specific in every case; the fixed-density fade CIs are
  reported beside it and the causal attribution (density vs scale) is written from
  the two contrasts as measured — no binary survives/dies gate.

## E. Decoder-strength ablation (R5.9)

Locked configuration: uniform/center/euclidean, n=50, α=2, **E=∞**, m=1 — matching
the archived `ops_factorial_v2` arms — 30 instances, 3 seeds retained, budgets 16,000.
Arms (8): decoder ∈ {greedy, span-4 DP, span-6 DP, span-12 DP} × operators ∈
{sortie-aware, generic}. All four decoder strengths are rerun (the archive lacks
per-replicate routes), inside **one ALNS framework** — the greedy decoder becomes a
decode option of the same search loop, replacing the archived duplicated loop. The
archived greedy best-of-3 values serve as reproduction references after the loop is
unified. The archived span-12 values do not: v2 silently used an unrestricted final
re-decode, whereas the new curve deliberately matches search and final spans. Every
arm uses the same span for search and final
decoding (no silent upgrade); all replicates, routes and runtimes retained. The
comparison is equal-iteration (the causal control); runtime is reported rather than
treated as matched, and any efficiency statement carries that scope caveat.
- **Pre-committed**: the "≈9×" decoder-vs-operator ratio is removed from the
  manuscript regardless of outcome; the replacement is the decoder-strength curve
  (savings vs decoder span) with the operator effect overlaid.

## F. Held-out long-budget comparator cross-check (R5.7 — renamed)

- **Pre-F gate**: the original Julia depot lived in a since-cleaned scratchpad
  (`julia_env_status.txt` points at a dead path; `julia` is not on PATH). Reinstall
  Julia 1.10.4; pin TSPDrone.jl @f42d27c and TSPDroneHGATAC.jl @8f1f345 in a
  **committed** Project.toml/Manifest.toml inside the repo; prescribe and archive
  comparator RNG seeds, statuses, costs and runtimes; smoke-test all three solvers on
  2 instances before the full run. If the environment cannot be restored faithfully,
  F is reported as **not run** and the archived matched-budget comparison stands
  as-is — no silent substitution.
- **Instances**: held-out seeds 30–39 at n∈{20, 50}, 30–34 at n=100
  (uniform/center/euclidean; never used in any prior run).
- **Budgets**: 4× the v2 matched per-size wall budgets, single-threaded; the v2 hard
  cutoff applied symmetrically (over-budget replicates discarded on every side);
  comparators at default settings, ours at the study configuration — disclosed as an
  unmatched-tuning comparison (R5.8 applies here identically).
- **Replicates & seeds**: 3 replicates per (solver, instance), replicate seeds
  {1, 2, 3} passed explicitly to each solver's RNG (Julia global RNG seeded per
  (solver, instance, replicate); ours seeded the same way); per-solver per-instance
  value = best of its own completed replicates (symmetric); per-replicate values,
  routes/solution structures, statuses and runtimes all archived.
- **Failure policy**: a crash or no-feasible-result replicate is recorded as a
  failure; if a solver has no completed replicate on an instance, that instance is
  excluded from that solver's pairwise comparisons and the exclusion count is
  reported.
- **Report**: per instance, each solver's relative gap to the pooled best-found value;
  pairwise win/tie/loss counts (tie tolerance 1e-6 relative — the v2 value);
  per-solver median/IQR/max gap and per-replicate spread. **Language**: descriptive
  cross-check only — the pooled best is not a bound and includes each solver itself;
  no binary "agreement" claim, no ranking, no optimality inference. **Adverse
  branch**: divergence, whatever its size, is reported as measured and the
  manuscript's large-n instrument argument is written from those numbers (§I row I5)
  — never patched or reframed.

## G. Energy-model extensions (R5.4, R5.5)

1. **Generalized boundary post-processing** (no solver runs): for every H2-v3
   solver-replicate plan, re-derive truck waiting time W_T and aggregate drone waiting time
   W_D from its operations. With truck idle-power fraction i, drone hover-power
   fraction h, and per-sortie takeoff/landing surcharge c, the correct boundary is

       r*(i,h,c) = (E_D + h W_D + c N_s) / (L_0 - D_T - i W_T).

   A non-positive denominator is reported as no finite boundary. We report the full
   stated grid i in {0,0.25,0.5}, h in {0,0.5,1}, and
   c in {0, 0.1·median(E_s)}, where median(E_s) is the archived median nominal sortie
   energy. **Battery reserve** (ρ_res = 0.2): every archived **m=1 finite-E**
   solver-replicate
   order is **re-decoded** (fixed-order DP) at usable endurance E·(1−ρ_res) and r*
   recomputed on the re-decoded plans — an archive-conditional, fixed-order analysis,
   labelled as such. Scope: m>1 is excluded (a batch re-decode at reduced E can leave
   the archived operation structure infeasible) and E=∞ has no reserve to bind; §B's
   single re-optimized E=0.8 cell complements this by measuring what re-optimization
   (not just re-decoding) recovers at the central cell. Aggregation:
   instance-first over plans, per the global contract; G.1 involves no new solver
   replicates.
2. **Empirically parameterized accounting check**: evaluate every time-oriented
   factorial plan under two named, traceable coefficient sets rather than calling the
   study's nominal coefficients calibrated: (a) Rodrigues et al.'s flight-tested small
   quadcopter intensity and electric/diesel van intensities, and (b) Stolaroff et al.'s
   measured/model-validated small-quadcopter intensity and Class-4 electric/diesel
   truck intensities. The evaluated plans are all H2-v3 λ=0 solver-replicate
   **best-found time-oriented plans** (best found within each run; no optimality
   implied, per I5). Replicates are averaged within instance before inference; the
   historical v2 winner-only value is also reported descriptively for continuity.
   Report, separately for each
   of the **four combinations** (2 drone parameterizations × {electric, diesel} ground
   vehicle), plan total energy relative to the corresponding truck-only route. This is
   a route-accounting validation, not fleet-specific calibration or life-cycle
   analysis.
   **Language gate:** "calibrated" is used only for a model fitted to data from the
   actual deployment; these literature-derived checks are "empirically parameterized".
   **Executability gate:** before any G.2 computation, every equation and coefficient
   is transcribed into a committed `experiments/REVISION/energy_models.md` with its
   exact source (paper table/page or repo file + version), together with the
   unit-square→physical conversion used (unit length in km, truck speed, drone speed
   = α × truck speed, payload per sortie) — G.2 runs only after that commit, so the
   model set cannot be adjusted after seeing results. The four combinations are
   reported separately, never blended or averaged.
- **Decision rule**: "green-and-fast" = simultaneously faster and lower total energy
  than the truck-only reference, per plan. Per combination, report (a) the fraction of
  λ=0 best-found plans with the property + instance-bootstrap CI, and (b) the
  **magnitude estimand**: median % total-energy saving of those plans vs truck-only,
  with CI — (b) is what carries or replaces the manuscript's "34–36%" figure. The
  van-class claim survives only for combinations whose fraction-CI **lower bound
  exceeds 50%**, and is then attributed to the named parameterization ("under the
  Stolaroff-parameterized model…"); otherwise it is narrowed to the combinations where
  it holds, stated as such in Results and Discussion (§I row I6).

## H. Street-network capability contrasts (R5.12 — partial compliance)

Cached districts (Manhattan, Paris), the **same archived 30 draws** per district, m=1,
**five** capability points: the four added (α, E) ∈ {(1,1), (3,1), (2,0.5), (2,∞)}
plus a rerun of the archived (2,1) arm with replicates retained — its best-of-3 must
reproduce the archived realnet values (reproduction gate) and it serves as the paired
baseline. **Estimand**: paired per-draw differences vs the (2,1) arm, per district,
instance-first bootstrap. **Scope statements,
unconditional**: the analysis is a two-district road-geometry stress test; the
speed×fleet interaction remains untested on road networks (added arms are m=1 only);
transfer-to-deployment, real-demand, traffic and regulatory generalizations leave the
manuscript regardless of results.

## I. Claims matrix (pre-committed wording branches)

| # | Manuscript claim (sites) | Governing estimand | Branch |
|---|---|---|---|
| I1 | "one moderate drone ≈34%; two/three 46%/52%" (abstract, Results, Discussion) | §A all-seed cell means | Restated unconditionally on all-seed means with the R2.1 instance-family qualifier; historical best-of-3 values move to SI. Expected m1<m2<m3 ordering: if it changes, the fleet claim is rewritten from the all-seed table. |
| I2 | "speed and fleet size are complements (≈ half again)" | §A all-seed contrast, per fleet step k∈{1→2, 2→3} at E=1: C_k = mean_i[g_i(k; α=3) − g_i(k; α=1)], where g_i(k; α) is instance i's all-seed marginal saving gain of the step at speed α; paired instance-bootstrap CI | CI excludes zero in the positive direction for both steps → keep, restate the magnitude from the all-seed table. Either CI includes zero → downgrade to "suggestive", remove from the abstract. |
| I3 | "endurance matters mainly in sparse, small instances / range constraints fade as instances densify" | §D fade(m, E) contrasts on the fixed-density and fixed-area grids | Wording becomes fixed-area-specific in every case; both fade CIs are reported and the causal attribution (density vs scale) is written from them as measured. |
| I4 | "decoder ≈9× the sortie operators" (Results) | §E curve | Removed unconditionally; replaced by the decoder-strength curve (all-seed mean saving per arm, with CIs) with operator effects overlaid — the curve's values speak; no ≈-style convergence judgment is pre-baked. |
| I5 | "optimality-validated" (abstract, intro, Results heading, Fig. 2 caption, Discussion) | §A + §F | Removed unconditionally (R2.4): small-n exact evidence and fixed-order decoder exactness named as such; large-n = stability diagnostic + §F descriptive cross-check, explicitly no bound. §F divergence, if any, is reported and weakens the instrument argument in Discussion. |
| I6 | "van-class pairings cut ~36% time and 34–36% energy / green-and-fast" | §G.2 property fraction per arm; §G.1 r* shifts | Survives only per G.2's decision rule, attributed to the named arm(s); §G.1's shifts qualify the boundary's robustness in the same paragraph; wind and speed-dependent power stay in Limitations regardless. |
| I7 | "savings persist 26–39% across demand/depot/metric" | §A P0 all-seed cells | Range restated on all-seed means; if it widens below 26%, the new range replaces it everywhere, abstract included. |
| I8 | "completed fronts / knee region" | §A H3 seed-specific fronts | "Archive-conditional" qualifier at every mention, unconditionally (R5.6); knee statistics restated on the historical front with the between-seed spread beside them. |
| I9 | street-network transfer sentences | §H | Rewritten to two-district geometry-stress-test scope unconditionally. |
| I10 | frictionless / deterministic model scope | §B + §C | New Results sensitivity subsection reports B's dual metric and C's realized-saving distribution; the abstract keeps the deterministic-model scope wording. |

## J. Sequencing, budget, authorization

Order: (1) the cheap gates — §A canaries, §B gates 1–4, §E greedy reproduction probe,
and §F Julia gate; (2) §A H2, because its all-seed artifact is the common input to
B–D and G; (3) the remaining §A chain (P0→H3→H1), while headline-moving §E pilots can
run; (4) full §E and §D grids, §B grid, §C, §G, §H and any restorable §F run; (5) the
§I rewrite starts only after the runs governing its rows land. Section D's s=1
identity runs inside its locked driver immediately before its new grid.

Budget: ~730–800 scheduled core-hours; with 20% contingency 875–960 core-hours ≈
**24.3–26.7 ideal wall-hours at 36 workers** (730×1.2/36 to 800×1.2/36). This corrected
estimate is disclosed in the working session before the chain launches. The user's
instruction to continue the complete revision authorizes these in-scope,
checkpointed local computations; any paid infrastructure or materially broader run
still requires separate approval.

**Global-contract exemptions** (stated so the contract stays true): §G.1 is
deterministic post-processing rather than a solver run, but it processes all H2-v3
replicates and aggregates replicate-first/instance-first. §F's estimand is defined in its own section (per-solver
best-of-replicates, symmetric), not by §A's replicate-mean rule — its replicate
spread is reported alongside.
