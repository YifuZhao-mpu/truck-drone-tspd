# Protocol deviations and amendments

The pre-registered protocol (`H1-alns-backbone/protocol.md`, locked 2026-06-27 at commit #1)
was followed in design but deviated from in several execution details. The external review of
2026-07-16 (`RESEARCH_REVIEW_REPORT_2026-07-16.md`) flagged the undisclosed deviations; this
table discloses all of them and records the v2 amendments (2026-07-16/17) that either restore
the protocol or supersede it with a documented, stricter choice.

| # | Item | Locked protocol | v1 execution (submission-v5) | v2 execution (current) | Status / reason |
|---|------|-----------------|------------------------------|------------------------|-----------------|
| 1 | Endurance definition | max drone **time** per sortie | max **flight distance** (Agatz "maxradius" convention) | same as v1 | AMENDED at implementation (2026-06-27, research-log #5): distance budget matches the public benchmark so published optima remain comparable. Disclosed in Methods. |
| 2 | Sortie-aware operator set | launch/rendezvous reassignment; drone-customer swap; mode flip | drone-relocate; drone/truck swap; sortie-window reversal | same as v1 | AMENDED: the implemented set is order-space-compatible (the DP split re-decides modes, making "mode flip" meaningless under an optimal decoder). Disclosed in Methods. |
| 3 | Iteration budget | 20k–40k iters or wall-clock cap, identical across methods | 7k/9k/16k (n=10/20/50), 6–7k in P0, 6–8k in H3 | 9k/16k/16k (n=20/50/100) everywhere, identical across all arms and experiments | v2 UNIFIED. Budget-sensitivity vs 2.5x budget reported as a diagnostic. |
| 4 | Instances per size | 30 per size | H1: 30; **H2: 20 (n=20), 12 (n=50)**; H3: 20/12; P0: 12 | 30 everywhere | v2 RESTORED to protocol. v1 Methods text claiming 30 was wrong (review finding). |
| 5 | Solver seeds | >= 3 seeds | H1: 3; **H2/H3: 2; P0 grid: 1** | 3 everywhere (best-of-3) | v2 RESTORED to protocol. |
| 6 | One consistent solver | one solver, identical budgets across methods | P0 ran sortie_aware=False; H3 used a separate simplified non-adaptive search | one code path (`alns()`) for every experiment incl. lambda objective; sortie moves on everywhere | v2 RESTORED. |
| 7 | H3 objective | makespan + lambda * drone_energy | decoder optimized makespan + lambda*(drone + 0.3*truck-distance) energy, seed selection and plots used drone-only (inconsistent; review-invalidating) | makespan + lambda * E_total (drone Dorling + truck 0.3/dist) used consistently in search, selection, and reporting; drone/truck components stored separately; per-instance nondominated fronts | AMENDED (stricter than protocol): drone-only energy makes the energy optimum trivially the no-drone plan; total-system energy is the meaningful sustainability objective. Protocol's "truck energy ~ distance" coefficient retained (te=0.3), with te in {0.15, 0.6} sensitivity. |
| 8 | H2 grid | full alpha x E x m | alpha x E at m=1 only; m in {1,2,3} only at (alpha=2, E in {1,inf}, n=20); multi-drone decode span-capped at 6 with generic moves only (asymmetric) | full 5 x 4 x 3 factorial at n in {20,50}, symmetric decoding (same span 12, same sortie-aware moves for all m) | v2 RESTORED to protocol. Multi-drone remains a heuristic extension of the split (protocol already flagged this). |
| 9 | Savings denominator | truck-only TSP via NN + 2-opt + Or-opt | as declared -- but review showed this heuristic averages 3.1% above certified optima (max 13.3%), inflating every saving | LKH-3 certified reference (8 runs, float-metric re-evaluation, local-search polish; never worse than the internal heuristic on 540/540 instances) | AMENDED (stricter): the declared denominator was itself the flaw; savings are now measured against `experiments/TSPREF/results/tspref_cache.json`. |
| 10 | n=100 | exploratory | presented with confirmatory-style p-values after expanding the sample from 10 to 30 pairs when the first result was borderline | rerun from scratch: 30 pre-set paired instances, one config, best-of-3 both arms; still labeled exploratory-extended | v2 RERUN; the stale intermediate artifact `P0-robustness/results/ablation_n100_v2.json` (10-pair pilot, p=0.055) is deleted -- superseded by `p0_v2.json:ablation.100` (git history preserves the pilot). |
| 11 | Optimality evidence n>=20 | (not pre-declared) | 2.5-3x budget self-comparison presented as an optimality bound ("effectively converged") | same comparison reported strictly as a budget-sensitivity diagnostic; NOT a bound | Reframed per review. |
| 12 | CI method | (not pre-declared) | paper said 20k bootstrap; robustness CIs were actually normal-approx | bootstrap (20k resamples) computed in-driver for every reported CI | v2 FIXED. |
| 13 | Depot/demand confound | (not considered) | clustered demand always used a corner depot; uniform always central -- effects confounded | full {uniform,clustered} x {center,corner} x {L2,L1} deconfound grid at n in {50,100} | v2 ADDED per review. |
| 14 | Stored evidence | results JSON | several claims (DP==sim 1e-15, brute force, 20us timing) had no stored artifacts; P0/P1 lacked per-instance raws | `VALIDATION/results/validation_v2.json` stores simulator/algebra checks, span sensitivity, brute-force check, timing; all v2 drivers store per-instance raws incl. routes | v2 FIXED. |
| 15 | Environment | "CPU only, 40 cores" | no dependency lock; base env import broken (NumPy 2.5 vs Numba) | `.venv` + `requirements-lock.txt` (numpy 2.1.3, numba 0.66.0, scipy 1.18.0); LKH-3.0.9 built from source | v2 FIXED. TSPDrone.jl version pin still TODO for any future SOTA rerun. |

Decision rules and falsification criteria of the locked protocol are unchanged. H2's
pre-declared "monotone non-decreasing" decision rule is evaluated up to solver noise:
nested feasible sets imply true optima are monotone, so small observed reversals
(documented in v2 results) measure residual heuristic error rather than a real effect;
reversal magnitudes are reported alongside the surfaces.
