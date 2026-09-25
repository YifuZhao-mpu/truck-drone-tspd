#!/usr/bin/env bash
# Post-§A pipeline (revision_protocol.md §J step 4), launched once the §A P0->H3 chain and
# the §E decoder-strength run are in flight.  Usage: run_post_chain.sh <chain_pid>
#
#   waits for   the P0->H3 chain process (<chain_pid>) and run_decoder_strength.py
#   1. check_rerun_reproduction.py   independent §A gates on H1/H2/H3/P0 v3 — any
#                                    mismatch halts the pipeline (protocol §A)
#   2. run_heldout_comparators.py    §F on the then-quiet machine (wall-clock budgets;
#                                    24 single-threaded processes)
#   3. run_density_control.py        §D fixed-density grid          (36 workers)
#   4. run_operational_sensitivity.py §B friction/eligibility/reserve grid (36 workers)
#   5. run_realnet_capability.py     §H two-district capability cross (36 workers)
#
# Each stage logs to experiments/REVISION/logs/<stage>.log; pipeline.log carries one line
# per START/DONE/FAILED event and the pipeline stops at the first failure.  Every driver
# is checkpointed, so a halted pipeline resumes from where it stopped when re-run.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/experiments/REVISION/logs"
mkdir -p "$LOG"
CHAIN_PID="${1:?usage: run_post_chain.sh <chain_pid>}"
cd "$ROOT/src"

note() { echo "$(date -Is) $*" >> "$LOG/pipeline.log"; }

note "WAITING for chain pid $CHAIN_PID and run_decoder_strength.py"
while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 60; done
while pgrep -f 'run_decoder_strength.py' >/dev/null; do sleep 60; done
note "chain and §E processes have exited"

for f in "$ROOT/experiments/P0-robustness/results/p0_v3.json" \
         "$ROOT/experiments/H3-time-energy/results/h3_v3.json" \
         "$ROOT/experiments/P0-robustness/results/decoder_strength_v3.json"; do
  [ -f "$f" ] || { note "MISSING $f — a run did not complete; pipeline halted"; exit 1; }
done

stage() {
  local name=$1; shift
  note "START $name"
  if "$@" > "$LOG/$name.log" 2>&1; then
    note "DONE $name"
  else
    note "FAILED $name (exit $?) — pipeline halted"
    exit 1
  fi
}

stage check_rerun_reproduction "$PY" check_rerun_reproduction.py
stage heldout_comparators      "$PY" run_heldout_comparators.py --workers 24
stage density_control          "$PY" run_density_control.py --workers 36
stage operational_sensitivity  "$PY" run_operational_sensitivity.py --workers 36
stage realnet_capability       "$PY" run_realnet_capability.py --workers 36
note "PIPELINE COMPLETE"
