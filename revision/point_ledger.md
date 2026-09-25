# Round-1 revision — point ledger

One row per review point. `Class`: text (edit only) / experiment (new computation) /
figure / rebuttal (respond, no change or partial) / decision (author must decide) /
diagnose (establish the fact first). `Status`: ⬜ open → 🔧 in progress → ✅ resolved
(with the commit) / 💬 answered in response letter only.

The verbatim sources are in `reviews/`. The response letter will be built from this
table; keep the two in sync — every row eventually needs a response-letter paragraph.

## Hard constraints the plan must respect

- Scientific Reports **mandates** Intro→Results→Discussion→Methods for Articles — R4's
  requested reordering (and half of R3.1's force) is answered by journal format, not by
  compliance. The fix that *is* ours to make: subsection system, definitions, narrative
  rebalance inside Results.
- Main text ≤ 4,500 words (final counter: 4,310); ≤ 8 display items (six figures and
  one table in the main text).
- The reviewers read the 41-page **Word build** — whatever we resubmit, regenerate both
  builds and verify the table/figures render in LibreOffice too (R4.7).
- COPE: add only citations that genuinely position the work (R5.0 names five literatures);
  no citation stuffing.

## Editor

| ID | Point | Class | Plan | Status |
|---|---|---|---|---|
| E1 | Structure/clarity/presentation of the core contribution | text | Delivered via R3.1–R3.4 + R4 rows | ✅ resolved |
| E2 | Strengthen validation, sensitivity, statistics, reproducibility, assumptions/limitations | experiment+text | Delivered via R5 rows | ✅ resolved |
| E3 | Justify optimality/scalability/practicality claims | text | Delivered via R2.4, R5.6, R5.7, R5.12 | ✅ resolved |
| E4 | COPE — only scientifically relevant citations | text | Governs R5.0; note in response letter | ✅ resolved |
| E5 | **Custom code must be deposited in a DOI repository (e.g. Zenodo), linked from Methods or Code availability** | release/compliance | Mandatory (journal requirement supersedes the 2026-07-26 "no DOI" decision, made when it was optional). Deposit the **revision release** (v1.1-revision: revised code + drivers + all-seed artifacts + manifest + env), cite the *version DOI* in Code availability. Version DOI `10.5281/zenodo.22297604` is reserved and inserted; final ZIP upload, publication and resolution check remain | 🔧 in progress |

## Reviewer 2 (positive; minor)

| ID | Point | Class | Plan | Status |
|---|---|---|---|---|
| R2.1 | Abstract savings numbers need context ("on typical test instances") | text | Added tested-family and capability-cell qualifier; all-seed estimates used | ✅ resolved |
| R2.2 | sortie vs operation used interchangeably | text | Distinct definitions added and usage audited | ✅ resolved |
| R2.3 | Fig. 5 clarity + colour contrast | figure | Rebuilt as revised Fig. 1 with accessible palette and shared scales | ✅ resolved |
| R2.4 | "optimality-validated" → near-optimality; budget-sensitivity shows stability, not optimality | text | Umbrella label removed; exact scope and large-instance diagnostics separated | ✅ resolved |
| R2.5 | Fig. 6 dense → reorganize | figure | Split by scientific purpose across revised Figs. 3, 4 and 6 | ✅ resolved |

## Reviewer 3 (structural rewrite)

| ID | Point | Class | Plan | Status |
|---|---|---|---|---|
| R3.1 | Solver validation overshadows mechanism findings; separate instrument from discovery | text | Mechanisms open Results; validation is the final Results subsection | ✅ resolved |
| R3.2 | Nested parentheticals, long sentences | text | Full sentence-level rewrite completed | ✅ resolved |
| R3.3 | Paragraph structure: one info-class per paragraph, topic sentences, consolidate limitations | text | Paragraphs rebuilt; limitations synthesized with claim-local cautions retained | ✅ resolved |
| R3.4 | r* patterns shown before its definition | text | Boundary identity now precedes every estimate | ✅ resolved |

## Reviewer 4 (organization/format; reviewed the .docx)

| ID | Point | Class | Plan | Status |
|---|---|---|---|---|
| R4.1 | Section order should be I→M→R→D | rebuttal | Journal order retained; navigation issue addressed substantively | 💬 answered in response letter |
| R4.2 | Subsection system missing | text | Descriptive Results and Methods subsections added | ✅ resolved |
| R4.3 | Undefined concepts (decoder, operator, …) | text | Terminology paragraph and first-use definitions added | ✅ resolved |
| R4.4 | Long inline formulas | text | Energy model and boundary promoted to displayed equations | ✅ resolved |
| R4.5 | Bold abuse; bold run-ins should be subsections or definitions | text | Narrative bold removed; real subheadings used | ✅ resolved |
| R4.6 | Fig. 1 hard to read; formal algorithm definition belongs in Methods | figure+text | Solver graphics moved to SI; recurrence in Methods and Algorithm S1 in SI | ✅ resolved |
| R4.7 | "Table 1 doesn't occur in .docx" | diagnose | Word-export style/OMML interaction fixed; final DOCX rendered and inspected in LibreOffice | ✅ resolved (`9a3b925`) |
| R4.8 | Grammar slips ("small instances is quantified", p. 15) | text | Final grammar/readability pass completed | ✅ resolved |

## Reviewer 5 (technical)

| ID | Point | Class | Plan | Status |
|---|---|---|---|---|
| R5.0 | Literature: connect to multi-drone VRP, bi-objective, energy-aware, variable-speed, exact TSP-D | text | Five literatures positioned with only directly relevant citations | ✅ resolved |
| R5.1 | Common-launch/common-rendezvous batch decoder restricts the multi-drone model → effect on speed×fleet complementarity | text | Claim scoped to coordinated batch dispatch; no direction-of-bias assertion | ✅ resolved |
| R5.2 | Idealized assumptions (sync, single-customer sorties, node-only ops, deterministic times) → quantify impact on savings | experiment+text | Feasibility check: stochastic flight-time perturbation and launch/recovery overheads are addable to the decoder; quantify on a subgrid | ✅ §C (noise) + §B (friction) landed 2026-09-02; explicitly partial (no multi-customer sorties / off-node / async) |
| R5.3 | Sensitivity: launch/recovery time, service time, drone eligibility | experiment | **New decoder options** (nothing of this exists in the solver — the v8 "service-time caps" are delivery-time deterioration caps, a different thing): launch/recovery + customer-service durations + nested eligibility masks; 6–10 cells, 11–18 core-h; zero-friction arm must reproduce the submitted kernel bit-for-bit | ✅ §B landed 2026-09-02: base reproduces H2-v3 exactly; handling 0.25τ/0.50τ → saving 19.9%/9.0% (+22%/+39% makespan); service raises adjusted saving but +19–37% makespan; eligibility 75/50% → 30.7/23.1%; reserve 34.1% |
| R5.4 | Energy model omissions (hover, TOL, idling, wind, reserve, speed-dependent power) | text±experiment | Explicit omissions paragraph plus idle/hover/surcharge and reserve sensitivities | ✅ resolved |
| R5.5 | r* needs empirically calibrated energy models | experiment+text | The vehicle-power anchoring data already in Methods → push through the r* pipeline as a calibration check | ✅ §G.2 landed 2026-09-02: all four named pairings 100% green-and-fast, 32–39% median energy saving |
| R5.6 | Pareto knees not global optima | text | Pool rebuilt from every retained order; all claim sites state pool-conditional scope | ✅ resolved |
| R5.7 | Solution quality at n≥20 unproven | text | Merge with R2.4 global rename + strengthen the budget-diagnostic caveat | ✅ §F landed 2026-09-02: 25 held-out instances, 4× budgets, 3 seeded loops/solver; ours = pooled best on 25/25, HGA-TAC median gap 0.0/3.1/4.7% (n=20/50/100), DPS 1.9/4.2/4.4%; descriptive, no bound, tuning unmatched |
| R5.8 | Comparator fairness: defaults for others, tuned for ours | text (comply) | Table 1 recast as a descriptive default-configuration check; unmatched tuning stated | ✅ resolved |
| R5.9 | 9× decoder-vs-operator claim vs deliberately weak greedy decoder | experiment | Span-4/6 restricted-DP decoders as intermediate baselines (existing kernel, `final_span` matched — the one trap), crossed with both operator sets, ~6–9 core-h. **Drop the "≈9×" ratio regardless of outcome**; report a baseline-effect curve | ✅ §E landed 2026-09-02: decoder penalty greedy +4.85%, span-4 +0.60%, span-6 +0.09% vs span-12; operator effect 0.2–0.7% (baseline-dependent); "≈9×" removed |
| R5.10 | Best-of-3 seeds → optimistic bias; report all seeds | experiment (EXPENSIVE) | Per-seed values are **not** archived (drivers keep the winner only — verified in `run_h2_v2.py`/`run_h3_v2.py`/`run_p0_v2.py`). Patch drivers to retain every replicate, rerun H2+P0+H3 (+H1/ablation): ~600–620 core-h ≈ 17 h wall. Report all-seed variability + selection gain | ✅ all four reruns landed 2026-09-02 (H2 exact, H1 1e-12, P0 exact, H3 exact); selection advantage 0.15 pt / 0.15% pooled; all-seed values in text, SI Table S1 |
| R5.11 | Scale vs density confounded (fixed area) | experiment | Fixed-density companion grid (area grows with n) at the key cells | ✅ §D landed 2026-09-02: fixed-density fade n20→n100 4.8 pt (CI 2.8–6.8) vs fixed-area 16.5 pt → fade is mostly density; wording fixed-area-specific (I3) |
| R5.12 | Two districts, one capability setting → temper transfer claims | text+experiment | Rename to "two-district road-geometry stress test" (adding convenient cities would not validate deployment); add 4 capability contrasts on the cached districts: (α,E)=(1,1),(3,1),(2,0.5),(2,∞) at m=1, ~40 core-h | ✅ §H landed 2026-09-02: central cell exact; Manhattan 39.1% / Paris 41.9%; paired lever contrasts identical in ordering (α=1 −13, E=0.5 −4, E=∞ +0, α=3 +4 pt); two-district stress-test wording |
| R5.13 | "Partially reproducible archive" → make it fully reproducible | text+release | Missing seeds/routes and stale labels fixed; release builder adds manifest and hashes | ✅ content complete; DOI publication remains E5 |

## Cross-cutting sequencing

1. All scientific, writing, figure and reproducibility work is complete.
2. The Zenodo version DOI is reserved and inserted in all submission sources (E5).
3. Rebuild the release and submission bundle, upload it to the saved Zenodo draft,
   publish the record, verify DOI resolution, and then upload the journal revision.
