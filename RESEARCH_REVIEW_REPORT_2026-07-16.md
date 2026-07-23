# ARIS Research Review Report — 2026-07-16

**Paper**: "When drones help: a systematic map of the savings and time–energy trade-off in
truck–drone last-mile delivery" (`drafts/latex/main.tex`, submission-v5 + LaTeX port)
**Target venue**: Scientific Reports
**Reviewer**: Codex MCP, `gpt-5.6-sol`, reasoning effort `ultra`, read-only sandbox, fresh thread
(threadId `019f6bc6-f799-7403-9cde-e3eb3da26f22`)
**Protocol**: ARIS `research-review` (cross-model adversarial cold read; reviewer verified raw
artifacts itself). Trace: `.aris/traces/research-review/2026-07-16_run01/`.
**Executor verification**: Claude re-checked every load-bearing reviewer claim against source
and result JSONs before accepting this report (see §3).

---

## 1. Verdict

**Score 3/10 — reject in current form** (confidence: 5/5 on the computational audit, 4/5 on
novelty judgment). Recommendation: substantial computational and narrative rebuild, then
resubmit. Previous internal assessment ("sufficient for Scientific Reports", research-log #8)
is superseded: the review found defects the earlier self-critique missed because it audited
*claims vs. summaries*, not *claims vs. code and raw data*.

The three killers, in order of severity:

1. **H3 (time–energy) is internally inconsistent and must be rerun.** The searched objective,
   the seed-selection objective, the plotted quantity, and the claimed objective are four
   different things. The plotted "frontiers" are not Pareto fronts (no nondominance filtering;
   dominated endpoints; the λ=0 "time-optimum" is dominated on 5/20 (n=20) and 6/12 (n=50)
   instances because H3 uses a weaker, non-adaptive solver than the validated ALNS).
2. **The truck-only "optimum" (denominator of every saving) is provably suboptimal.** The
   project's own artifacts contain zero-drone routes that beat it on 12/12 n=50 instances.
   All headline savings are inflated by roughly 1–2 points.
3. **The advertised three-axis (α×E×m) design-space map was never run as such.** α×E only at
   m=1; the m sweep only at α=2, E∈{1,∞}, n=20 — and the multi-drone arm is algorithmically
   weaker (span-6 cap, generic moves), mechanically depressing the marginal value of drones.

Additionally, several Methods statements are factually false as written (instance counts,
seed counts, "one consistent solver", "bootstrap CIs", monotonicity).

## 2. What survived the audit (genuinely solid)

- Benchmark evaluator reproduces published exact optima with difference exactly 0.0.
- Mean benchmark gap 0.13% (bootstrap CI [0.043, 0.230] replicated independently); 56/70 exact,
  66/70 within 1%.
- ALNS-vs-greedy medians 14–19%, Wilcoxon p ≤ 1.3e-6 — verified.
- SOTA table arithmetic (2.579/2.661, 3.717/3.856, 5.178/5.357; wins 6/10, 10/10, 5/5) — verified.
- Operator-ablation counts at n=10/20/50 and the n=100 30-pair result (21/9, p=0.021, in
  `p0.json`) — verified.
- Openness/auditability of the repo, clarity of writing, willingness to publish a negative
  result — explicitly praised.

## 3. Executor verification of the reviewer's claims (all CONFIRMED)

| # | Reviewer claim | Independent check | Result |
|---|---|---|---|
| 1 | H3 search objective includes truck energy (te=0.3) but returns/selects on drone-only energy | `src/alns.py:280` (`te=0.3` default, `obj_cost_lambda` in search, returns `(ms, en)` drone-only); `src/run_h3.py:28` (`obj = ms + lam * e`) | CONFIRMED |
| 2 | H3 uses a weaker non-adaptive solver | `alns_energy` body: uniform-random operator choice (no adaptive weights), no sortie moves, single random segment reversal instead of iterate-to-local-optimum, cooling 0.9995 | CONFIRMED |
| 3 | Truck-only denominator suboptimal | H3 λ=8 routes (drone energy = 0.0 exactly) beat the H2 truck-only value on **12/12** n=50 instances; means 5.76542 vs 5.92655 | CONFIRMED |
| 4 | Savings inflated ~2 pts | Recomputing α=2, E=1, n=50 with the better zero-drone denominator: 36.98% → 35.23% (reviewer got 35.09% by a slightly different route; same conclusion) | CONFIRMED |
| 5 | m-axis only at n=20 | `h2.json` `raw_drone` contains n=20 rows only | CONFIRMED |
| 6 | "30 instances at n=20 and n=50" false | `h2.json` config: `n20_inst: 20, n50_inst: 12`; `nseed: 2` (paper says best-of-3) | CONFIRMED |
| 7 | P0 robustness uses a different solver config | `src/run_p0.py`: `sortie_aware=False`, nseed=1 for grid cells, different budgets — contradicts "The dynamic-programming split and ALNS are unchanged" | CONFIRMED |
| 8 | `ablation_n100_v2.json` stale | Contains the OLD 10-pair run (8/2, p=0.055); the paper's 21/9 p=0.021 lives in `p0.json.ablation_n100` | CONFIRMED |
| 9 | Robustness CIs are normal-approx, not bootstrap | `p1.json.saving_ci.clustered_manhattan_n50/n100`, `uniform_manhattan_n50` carry `"method": "normal_approx"`; paper's Statistical analysis claims 20k-resample bootstrap for all CIs | CONFIRMED |
| 10 | Endurance monotonicity literally false | 13 aggregate reversals across H2 + P0 surfaces (largest: n=100, α=3, E1→E2: 38.12→36.98) — heuristic noise, since feasible sets are nested | CONFIRMED |
| 11 | Multi-drone decode capped at span 6 with generic moves | `src/alns.py:101` (`min(max_span, 6)` in `tspd_cost_multi` path) | CONFIRMED |

No reviewer claim checked out false; no pushback round was warranted. Reviewer-independence
protocol respected (fresh thread, path-only brief, reviewer read artifacts itself).

## 4. Full criticism inventory (beyond the three killers)

- **Sustainability logic gap**: drone-energy-only frontier ignores truck energy/emissions; the
  "energy optimum" under a drone-only metric is trivially the no-drone route. No total-system
  energy, no calibrated units, no grid mix. "Greener logistics" framing unsupported.
- **"Every candidate decoded exactly"** overstated: span-truncation during search (span-12 H3,
  span-6 multi-drone) with no experiment establishing "negligible loss".
- **Convergence "bound" is not a bound** — budget-sensitivity diagnostic of the same biased
  search; also best-of-2 vs best-of-3 confound. And "no exact solver scales n≥20" contradicts
  the paper's own citation of exact methods to ~39 customers (Roberti–Ruthmair).
- **Operator conclusion causally unsupported**: sortie-aware calls do ~3× more split
  evaluations per call; "equal iterations" ≠ equal compute; local search capped at 3 stochastic
  cycles (not to local optimum as claimed); no decoder×operator factorial. Defensible narrow
  claim: ~0.4% (n=50) / ~1.0% (n=100) mean improvement under unequal per-iteration work.
  n=100 sample expanded after a borderline result → not clean confirmatory inference.
- **Clustered demand confounded with depot position** (uniform=central, clustered=corner);
  Manhattan metric ≠ road network; "exactly the geometry of real cities" unjustified.
- **"~20 µs split", "DP==simulator to 1e-15", "brute-force check"**: no stored artifacts —
  not reproducible claims as published.
- **"Comparable runtime" vs TSPDrone.jl**: our runtimes not saved; theirs single-call vs our
  best-of-3.
- **Reproducibility**: per-instance P0/P1 raw data partially missing; routes not saved;
  TSPDrone.jl version unpinned; no dependency lock; current conda env has a NumPy/Numba
  incompatibility that breaks `import`; locked protocol deviates from execution (sample sizes,
  seeds, m grid, endurance definition, operators, H3 objective) with no amendment table.

## 5. Consensus claims matrix (what can be claimed under what evidence)

| Claim | Status now | Allowed wording now | To claim as headline |
|---|---|---|---|
| Solver near-optimal at n≤17 | Solid | as written | — |
| "On par with public SOTA" | Weak | "limited external sanity check vs one unmodified public implementation at default settings" | matched-wall-clock, multi-run, pinned-version comparison |
| Savings map (α×E) | Inflated denominator, monotonicity overclaim | descriptive surface w/ heuristic-noise caveat, corrected denominator | exact TSP denominators (Concorde/LKH) + full factorial |
| m-axis effect | Narrow slice, asymmetric solver | "exploratory, n=20, α=2 only" | symmetric multi-drone solver, full α×E×m |
| Time–energy knee | **Invalid** | do not claim | rebuild H3: one objective, per-instance nondominated fronts, total-system energy, per-instance knees + uncertainty |
| Decoder-dominance ("split does the heavy lifting") | Causally unsupported | "these specific operators add little at equal iteration count" | decoder×operator factorial at equal wall-clock/evaluations |
| Robustness band 27–39% | Aggregates OK, CIs mislabeled, config inconsistent | fix CI description, disclose per-experiment configs | single config rerun + saved per-instance raws |

## 6. Prioritized fix plan (reviewer's ranking, 40-core CPU estimates)

1. **Rebuild H3** with one consistent objective; store truck distance/energy, drone energy,
   total energy; per-instance nondominated fronts (1–4 h) — *essential*.
2. **Exact/certified TSP denominators** (Concorde/LKH) for every saving (<1 h) — *very high*.
3. **Per-instance knees + uncertainty**; compare knee definitions; denser adaptive λ or
   ε-constraint (<1 h after #1) — *very high*.
4. **True α×E×m factorial**, symmetric decoding, ≥30 instances, ≥3 seeds (3–10 h) — *very high*.
5. **Exact TSP-D references at n=20–30** + independent bounds at n=50/100 (6–48 h) — *very high*.
6. Matched-budget, pinned-version TSPDrone.jl comparison (2–12 h) — *high*.
7. Decoder×operator 2×2 factorial at equal wall-clock and equal split-evaluations (2–8 h) — *high*.
8. Operational realism: launch/recovery time, eligibility, reserve range, speed–range coupling
   (4–16 h) — *high*.
9. Deconfound depot position from demand pattern; real road-network shortest-path matrices
   (<4 h compute) — *very high for SciRep*.
10. Scale n=200–500 (1–3 days) — *moderate*.
+ Zero-CPU: novelty-comparison table vs prior work; Methods corrections (instance/seed counts,
  CI method, per-experiment solver configs, protocol-deviation table); refresh or delete the
  stale `ablation_n100_v2.json`; pin environments; save per-instance raws and routes.

## 7. Narrative recommendation

Reframe around a defensible empirical phenomenon — *capability interactions and total-system
energy trade-offs in coordinated truck–drone delivery* — with the solver demoted to a validated
instrument in Methods/SI. The current four co-equal contributions (solver, negative operator
result, design map, sustainability) leave all four underdeveloped. Switching to an OR methods
journal would not help: the solver novelty is incremental and the operator evidence is not
fair enough for a methods paper.

## 8. Reviewer thread

- threadId `019f6bc6-f799-7403-9cde-e3eb3da26f22` (resumable via `mcp__codex__codex-reply`
  for follow-ups within this review session; per ARIS reviewer-independence, any future
  *re-scoring* after fixes must use a **fresh** thread).
- Round 1 full text: `.aris/traces/research-review/2026-07-16_run01/001-round-1-review.response.md`.
- Review brief: `RESEARCH_REVIEW_REQUEST.md`.

---

# ROUND 2 addendum — 2026-07-17

**Reviewer**: fresh Codex thread `019f71e1-3528-7e33-95f9-2c14bc01da20` (gpt-5.6-sol, ultra,
read-only; trace `.aris/traces/research-review/2026-07-17_run01/`).
**Verdict: 5/10 — MAJOR REVISION** (up from 3/10). The reviewer independently re-decoded all
3,600 factorial and 1,200 energy stored routes — every route-level makespan and energy
component reproduced exactly — and confirmed the round-1 killers were substantively fixed
(11-item table: 2 verified fixed, 9 partially fixed, 0 cosmetic-only, 0 unfixed).

## New findings (all executor-verified, all fixed same day)

1. **λ=0 domination counter bug** (new in v2): the survival check accepted any same-makespan
   front point. Correct accounting: strictly dominated 0/30 (n=20) + 3/30 (n=50); displaced
   from the front incl. same-time ties 5/30 + 7/30. Fixed in code + JSON + text (both
   numbers now reported with their meanings).
2. **"bit-for-bit 60/60" false** — 29/60 IEEE-exact, all ≤2.7e-15. Reworded to machine
   precision.
3. **LKH "certified" overclaim** — best-found tours, no lower bounds. Global rewording to
   "best-found LKH reference"; residual-gap bias direction disclosed.
4. **Stale figure** — h1_validation.png panel (b) still plotted the rejected heuristic
   truck baseline. Regenerated from the LKH cache.
5. **Methods 20µs vs measured 39µs** — fixed.
6. **Robustness ranges understated** — full paired contrasts recomputed
   (`p0_v2_contrasts.json`): demand 2.5–9.5pt, depot 0.5–2.4pt, metric 1.9–4.0pt,
   demand×metric interaction +1.6–2.0pt at n=50 (~0 at n=100). Text now reports full ranges
   + interaction + "demand ≈ metric at n=100" nuance.
7. **Pareto/knee framing** — weighted sums give sampled supported frontiers; knee is
   grid-sensitive (+15.3% on 12-λ vs +24.3% on 8-λ, same instances). Reframed as a "broad
   efficient region +13–24% / −31–47%"; sensitivity now uses a matched control (same 15
   instances, same 8-λ grid; figure regenerated).
8. **Scoping fixes** — "one config for every experiment"→primary experiments w/ disclosed
   exceptions; interaction "negligible"→"not detected (CI cannot exclude operator-sized
   effects)"; equal-wall-clock→post-hoc runtime observation; decoder 9×→"for these
   instantiations"; Table 1 "(no exact reference available)"; budget diagnostic N=15;
   routes-release claim scoped to factorial+energy experiments.

## Acknowledged, deferred (the reviewer's remaining majors)

- Physically calibrated energy coefficients (common units, truck idling, speed-dependent
  drone power) — needed if operator-facing sustainability claims are to strengthen.
- ε-constraint or adaptive-λ frontier method (recover unsupported nondominated points).
- Independent multi-drone validation (small-n exact or strong external reference).
- Replicate-level archiving (all 3 seeds, decoded schedules for every run).
- One realistic road-network/demand case.

These are the round-3 work package if a further uplift is pursued.

---

# ROUND 3 addendum — 2026-07-17 (submission-v6.1 → v6.2)

**Reviewer**: fresh thread `019f71ff-e48b-7772-ae68-9fae42747f4b`; **5/10, major revision**;
trace `.aris/traces/research-review/2026-07-17_run02/`. Spot audit: all headline arithmetic
reproduced (validation, 540 LKH tours, 120 factorial cells, robustness, ablation, nominal H3).

**Decisive finding (executor-verified, REAL)**: the round-2 "matched 8-λ sensitivity control"
was not matched — the nominal grid lacked λ=0.35, so the claimed control had only 7 points.
**Fixed by running the missing arm** (60 jobs; main grid now 13-λ ⊇ sensitivity grid): true
matched-control knee +23.6%/−45.2% (15/15 knees); dense-grid medians unchanged
(+19.4/−45.1, +15.6/−43.5); all band claims corrected to +13–24% / −31–45%.

Other round-3 findings, all fixed in v6.2: residual "certified"/"road-network" wording in 3
figures + 3 text locations (figures regenerated); span-losslessness overclaim replaced by
measured residuals (cap-12 search losses 25/1800 multi-drone winners ≤1.8%; reported cap-14
solutions residual ≤0.002%); λ0 counter fixed in the primary driver (not just postprocessor);
n=100 exploratory tags at point of use; multi-drone batch-model restriction and
best-of-three-only archiving added to Limitations.

Deferred majors (unchanged): physical energy calibration; ε-constraint frontiers; exact
multi-drone references; replicate-level re-runs with full archiving; real road/demand case.

---

# ROUND 4 addendum (FINAL) — 2026-07-17 (v6.2 → v6.3)

**Reviewer**: fresh thread `019f7221-fcc1-7e40-b010-a1b83e271b9e`; **6/10, confidence 5/5** —
the review-fix loop's target score. Verbatim: "**No new result-invalidating defect found.**
Independent route decoding, matched-control reconstruction, and all 2,400 multi-drone decoder
cross-checks support the numerical results. The remaining defects concern claims, denominators,
assets, and reproducibility — not the main arithmetic."

Remaining round-4 items were packaging-consistency fixes, all applied in **v6.3**: the three
figures whose regenerations were missed after the wording patch (design_space_v2,
robustness_v2, overview — final PDF now contains zero instances of "certified"); the primary
H3 driver natively 13-λ; span denominator corrected to 25/2,400; exactness wording qualified
(single-drone exact / m>1 capped batch) at every unqualified site; n=100 exploratory tag in
the robustness caption; "equal solver quality" replaced by measured-residual phrasing;
denominator metadata strings in artifacts and drivers.

## Loop summary

| Round | Version | Score | Verdict |
|---|---|---|---|
| 1 | submission-v5 | 3/10 | reject in current form (three killers) |
| 2 | v6 | 5/10 | major revision (killers fixed; new bugs found) |
| 3 | v6.1 | 5/10 | major revision (unmatched sensitivity control) |
| 4 | v6.2 | **6/10** | major revision = packaging only; **no result-invalidating defects** |
| — | **v6.3** | — | all round-4 packaging items applied |

## Remaining roadmap (deferred majors, disclosed as limitations)

1. Physically calibrated energy coefficients (common units; truck idling; speed-dependent power).
2. ε-constraint / adaptive-λ frontier method (recover unsupported nondominated points).
3. Exact or independently strong multi-drone references at small n.
4. Replicate-level archiving (all seeds + decoded schedules) — requires re-runs.
5. One realistic road-network / real-demand case.
6. Optional: matched-budget, pinned-version TSPDrone.jl comparison.

---

# ROUNDS 5–7 addendum (FINAL) — 2026-07-17/18 (v7 → v7.3)

Three further evidence packages were added post-loop (multi-drone exact validation 120/120;
physical calibration → the TWO-REGIME result; frontier completion via exact decoding-level
Pareto pass), then hardened through three more fresh-thread rounds:

| Round | Version | Score | Key finding → fix |
|---|---|---|---|
| 5 | v7 | 6/10 | calibrated "energy-min" was a sample min, not a minimum; frontier stats mis-attributed ("≤0.9%" metric blind to between-point unsupported solutions; true ε-gap mean 8.2%) → energy arm + honest ε audit (v7.1) |
| 6 | v7.1 | 6/10 | energy arm still selected by scalarized objective; text/figure inconsistencies; λ=0 endpoints dominated → POOLED endpoints (all replicates + exact re-decodings + truck-only), completed-front figures (v7.2) |
| 7 | v7.2 | **7/10, MINOR REVISION — "at the Scientific Reports submission bar"** | residual polish: cross-te order pool (emin 0.5468/0.5220, matching reviewer's own recomputation), Pareto-cleaned ties, "best-found" legend, knees recomputed from the archived fronts as single source of truth (v7.3) |

**Final headline results (v7.3):**
- Factorial map with interactions (unchanged, verified rounds 2–7).
- Two-regime energy result: comparable-power trade-off regime (time-opt ≈3.6× energy;
  completed-front knees +20.1%/−47.2% (n=20), +18.1%/−43.0% (n=50)); van-class win–win
  regime (time-opt cuts ~36% time AND 34–36% total energy vs truck-only ≈ 75–76% of the
  best-found max saving at zero time cost; best-found emin 11–12pts deeper at +60–64% time).
- Multi-drone batch solver exact on 120/120 small-instance references.
- Frontier completeness: exact decode pass closes an ε-gap of mean 8.2%/6.7%; completed
  fronts (44–393 pts/instance) archived and are the single source for all knee statistics.

**Score arc: 3 → 5 → 5 → 6 → 6 → 6 → 7 (minor revision).** Remaining disclosed limitations
(no fleet-measured energy coefficients, synthetic instances, uncertified large-n optima,
restricted batch multi-drone model, pre-round-4 runs archived winners-only, single
default-run TSPDrone.jl check) are judged by the round-7 reviewer as limiting, not
invalidating.

---

# ROUND 8 addendum (CLOSING) — 2026-07-18 (v7.4 → v7.5)

**Reviewer**: fresh thread `019f7307-91e8-7333-904b-e54d8b17e300`; **7/10, confidence 5/5,
minor revision — "no rerun required"**. The real-street-network package was verified sound:
the reviewer regenerated all 60 road matrices, reproduced all 180 winning-route makespans
to 5e-15, verified all 540 replicates and best-of-three selections, and reproduced every
LKH reference exactly. **No new result-invalidating defect.**

v7.5 applies the round-8 editorial items: increment digit 10.9→10.8; transfer claim scoped
to the tested (α=2, E=1) slice; OSM/OSMnx citations + circuity definition; osmnx in the
lockfile; LKH explicit-matrix tours archived (60/60 match stored values); replicate-archiving
limitation reworded to the accurate partial statement.

## Final state

**Score arc across 8 fresh-thread rounds: 3 → 5 → 5 → 6 → 6 → 6 → 7 → 7.**
Two consecutive 7/10 verdicts; round 7: "at the Scientific Reports submission bar";
round 8: "no rerun required", residue purely editorial (applied). Outstanding items before
actual submission are administrative: author names/affiliations/funding, repository URL +
Zenodo DOI, and (optional) a matched-budget pinned-version TSPDrone.jl comparison and
fleet-measured energy coefficients if reviewers of record request them.

---

# TARGETED AUDIT + ROUND 9 addendum (CLOSING) — 2026-07-18/19 (v7.6 → v7.8)

**Targeted SOTA audit** (thread `019f733a`): found the deepest error of the project, present
since v4 and missed by 8 full rounds — chkwon/TSPDrone.jl implements **DPS/TSP-ep-all
(Bogyrbayeva et al.)**, not the HGA-TAC it was cited as. Also: launch-deadline ≠ hard cap
(ours had +5.7% elapsed), tie/denominator mislabels, missing env archive. **All fixed**: the
real TSPDroneHGATAC.jl (pinned 8f1f345) run under the identical protocol; analyzer v2 with
symmetric HARD cutoff; three-way result (vs HGA-TAC **25W/1L/19T**, benchmark 0.00–0.20% vs
0.00–0.43%, mean paired reduction 1.7–3.1%; vs DPS 27W/0L/18T); Table 1 three solvers
correctly attributed.

**Round 9 full-manuscript verdict** (fresh thread `019f7370`): **7/10, confidence 5/5, minor
revision, no rerun** — "the SOTA fix lands correctly; raw recomputation gives exactly
25W/1L/19T; **no remaining result-invalidating defect found anywhere**; the manuscript is
scientifically sound, unusually well audited, appropriately qualified." Final wording residue
(paired-reduction labels, 'two strong public implementations', sampled-vs-completed front
consistency in Fig. 1 and Results) applied in **v7.8**.

## FINAL STATE

**Score arc over 9 full rounds + 1 targeted audit: 3 → 5 → 5 → 6 → 6 → 6 → 7 → 7 → 7.**
Round-9 quote: submission-ready once author metadata and the repository URL are filled.
Outstanding items are exclusively user-supplied: authors/affiliations/funding, public repo
URL + Zenodo DOI. Optional externals: fleet-measured energy coefficients.
