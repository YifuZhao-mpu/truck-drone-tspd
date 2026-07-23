# Release manifest

- **Release**: initial public release accompanying the *Scientific Reports* submission.
- **Source**: private research repository at tag `submission-v8.4`, commit `55ef6ba`
  (fresh-history snapshot; the private repo's full history is summarized in
  `research-log.md`).
- **Manuscript**: under review; citation and manuscript sources will be added on
  publication. Authors and ORCIDs: see `CITATION.cff`.

## Canonical result artifacts (single source of truth for every number in the paper)

| Claim family | Artifact |
|---|---|
| Solver validation (benchmark gap, DP==sim, brute force, span caps, timing) | `experiments/VALIDATION/results/validation_v2.json` |
| Multi-drone exact validation (120/120) | `experiments/VALIDATION/results/multi_exact_v2.json` |
| Capability factorial (5x4x3) | `experiments/H2-design-space/results/h2_v2.json` |
| Green-and-fast boundary r* | `experiments/H2-design-space/results/rstar_v2.json` |
| Time-energy sweep + completed fronts | `experiments/H3-time-energy/results/h3_v2.json`, `h3_frontier.json` |
| Service-time caps | `experiments/H3-time-energy/results/service_level_v2.json` |
| Calibrated two-regime analysis | `experiments/H3-time-energy/results/h3_calibrated.json` |
| Frontier/endpoint provenance checks | `frontier_provenance_check.json`, `calibrated_endpoints_check.json` |
| Robustness deconfound + ablation + budget diagnostic | `experiments/P0-robustness/results/p0_v2.json`, `ops_factorial_v2.json` |
| Street-network case + transfer test | `experiments/REALNET/results/realnet.json`, `transfer_test_v2.json` |
| Truck-only references (best-found LKH) | `experiments/TSPREF/results/` |
| Matched wall-clock external comparison | `experiments/SOTA/results/sota_matched.json` (+ replicates) |

## Known limitations of the archive (disclosed in the paper)

- Early experiments (factorial, deconfound grid, main energy sweep) archive best-of-three
  winners only; later experiments archive all replicates and seeds.
- Completed-front provenance: 10,455 of 10,546 points re-decode from archived routes; the
  remaining 91 match stored solution values (`verify_frontier_fields.py`, zero violations).
- Truck-only references are best-found LKH tours, not certified optima.
- The public Agatz-Bouman benchmark is fetched from its original repository, not
  redistributed (see README).
