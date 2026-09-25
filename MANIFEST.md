# Reproducibility release manifest

- Release: v1.1-revision
- Source commit: `4d73704935714c50008d9d56a54407f9768fd41d`
- Version DOI: 10.5281/zenodo.22297604
- License: MIT
- Files before manifest/checksums: 204

## Contents

- `src/`: solver, validation, experiment, analysis and figure-generation code.
- `experiments/`: raw seed-level rows, routes, resumable checkpoints, summaries,
  validation artifacts, comparator environment and run logs.
- `benchmarks/realnet/`: cached OpenStreetMap extracts used in the two-district test.
- `figures/`: vector PDF and 600-dpi PNG revision figures generated from the results.
- `revision/`: locked revision protocol, disclosed deviations, run ledger and point ledger.
- `requirements-lock.txt` and `experiments/SOTA/julia/Manifest.toml`: pinned Python
  and Julia environments.
- `SHA256SUMS`: SHA-256 digest of every other file in this release.

Workstation-specific absolute home prefixes in console logs are normalized to
`<author-home>` in this public copy; numerical result files are unchanged.

The public Agatz--Bouman benchmark is not redistributed; README.md gives the original
source and retrieval instructions. The version DOI above is read from CITATION.cff; a
PENDING value marks a preview archive and must not be published.
