#!/usr/bin/env bash
# The remaining confirmatory matrix (docs/confirmatory_study_plan.md, amendment 3, A3.5), one step at a time on one GPU.
# Every step's exit code is checked and nothing is piped: the script stops at the first failing step.
#
#   bash scripts/confirmatory_matrix.sh <python> <log dir> [<model-1 screening log>]
#
# Run it from a clean checkout of the commit that is to be recorded. Steps:
#   0. wait for model 1's seeds 1-2 screening (if a log is given) and require its exit code 0
#   1. the full test suite, recorded in ~/cap-prune-runs/_tests/<date>_<commit>/ (GPU idle)
#   2. model 1: freeze `main`; confirm `pilot` (amendment 3 re-confirmation) and `main`; report both
#   3. model 2: screen seeds 0-2 (all three arms); freeze `main`; confirm; report
#   4. the study summary over both models
set -u
PY="$1"
LOGS="$2"
M1_SCREEN_LOG="${3:-}"
M1=configs/confirmatory_smollm3.yaml
M2=configs/confirmatory_granite.yaml
mkdir -p "$LOGS"

note() { echo "$(date '+%F %T') $*" >> "$LOGS/matrix.log"; }

step() {
  local name="$1"; shift
  note "START $name: $*"
  "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  note "END   $name exit=$rc"
  if [ "$rc" -ne 0 ]; then
    note "STOPPED at $name"
    exit "$rc"
  fi
}

note "matrix driver: commit $(git rev-parse HEAD), clean tree: $([ -z "$(git status --porcelain)" ] && echo yes || echo NO)"

# 0. model 1 seeds 1-2 screening (already running): wait for its recorded exit code
if [ -n "$M1_SCREEN_LOG" ]; then
  note "waiting for model 1 screening ($M1_SCREEN_LOG)"
  until grep -q "^supervise exit=" "$M1_SCREEN_LOG" 2>/dev/null; do sleep 30; done
  rc=$(sed -n 's/^supervise exit=\([0-9]*\).*/\1/p' "$M1_SCREEN_LOG" | tail -1)
  note "model 1 screening finished: supervise exit=$rc"
  if [ "$rc" != "0" ]; then note "STOPPED: model 1 screening failed"; exit 1; fi
fi

# 1. the full test suite, recorded
COMMIT=$(git rev-parse HEAD)
TD="$HOME/cap-prune-runs/_tests/$(date +%F)_${COMMIT:0:12}"
mkdir -p "$TD"
CLEAN=$([ -z "$(git status --porcelain)" ] && echo true || echo false)
STARTED=$(date -Iseconds)
"$PY" -m pip freeze > "$TD/environment.txt" 2>&1
"$PY" --version >> "$TD/environment.txt" 2>&1
note "START tests -> $TD"
"$PY" -m pytest -rA -v --durations=0 -p no:cacheprovider --junitxml="$TD/junit.xml" --basetemp="$TD/basetemp" > "$TD/pytest.log" 2>&1
TRC=$?
note "END   tests exit=$TRC"
"$PY" -c "
import json, sys
log = open(sys.argv[1], encoding='utf-8', errors='replace').read().strip().split(chr(10))
json.dump({'commit': sys.argv[2], 'working_tree_clean_at_start': sys.argv[3] == 'true', 'exit_code': int(sys.argv[4]),
           'started': sys.argv[5], 'finished': sys.argv[6], 'summary': log[-1].strip('= '),
           'command': 'python -m pytest -rA -v --durations=0 -p no:cacheprovider --junitxml=junit.xml --basetemp=basetemp',
           'files': ['junit.xml', 'pytest.log', 'environment.txt', 'basetemp/']}, open(sys.argv[7], 'w'), indent=1)
" "$TD/pytest.log" "$COMMIT" "$CLEAN" "$TRC" "$STARTED" "$(date -Iseconds)" "$TD/record.json"
if [ "$TRC" -ne 0 ]; then note "STOPPED: the test suite failed (exit $TRC)"; exit "$TRC"; fi

# 2. model 1
step m1-freeze-main     "$PY" -m capprune confirmatory freeze  --config "$M1" --batch main
step m1-confirm-pilot   "$PY" -m capprune confirmatory confirm --config "$M1" --batch pilot
step m1-confirm-main    "$PY" -m capprune confirmatory confirm --config "$M1" --batch main
step m1-report-pilot    "$PY" -m capprune confirmatory report  --config "$M1" --batch pilot
step m1-report-main     "$PY" -m capprune confirmatory report  --config "$M1" --batch main

# 3. model 2
step m2-supervise       "$PY" -m capprune supervise --config "$M2" --set "matrix.seeds=[0,1,2]"
step m2-freeze-main     "$PY" -m capprune confirmatory freeze  --config "$M2" --batch main
step m2-confirm-main    "$PY" -m capprune confirmatory confirm --config "$M2" --batch main
step m2-report-main     "$PY" -m capprune confirmatory report  --config "$M2" --batch main

# 4. both models
step study              "$PY" -m capprune confirmatory study --config configs/confirmatory.yaml
note "matrix complete"
