#!/usr/bin/env bash
# Run evaluation jobs for the given arms and questions against a dataset fileset,
# then download row scores to runs/eval/<arm>.<question>.rows.jsonl.
#
#   scripts/run-evals.sh default/meridian-incidents-test#eval.jsonl A-frontier-no-context A-plus-frontier-with-context
#   QUESTIONS="runbook action" scripts/run-evals.sh <dataset> <arm>...
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
NEMO="${NEMO:-nemo}"
PY="${PY:-python3}"
BASE="${NHX_BASE_URL:-http://localhost:8080}"
DATASET="${1:?dataset fileset ref}"; shift
ARMS=("$@")
QUESTIONS="${QUESTIONS:-runbook action}"
mkdir -p "$HERE/runs/eval" "$HERE/eval/specs"

for q in $QUESTIONS; do
  "$PY" "$HERE/eval/make_eval_specs.py" --dataset "$DATASET" --out "$HERE/eval/specs" --arms "${ARMS[@]}" --question "$q" >/dev/null
done

# macOS ships bash 3 (no associative arrays): keep key/job pairs in a plain list.
PAIRS=()
for arm in "${ARMS[@]}"; do
  for q in $QUESTIONS; do
    out=$("$NEMO" --base-url "$BASE" evaluator evaluate --spec-file "$HERE/eval/specs/$arm.$q.json" 2>&1)
    job=$(echo "$out" | grep -o "nemo-evaluator-[a-z0-9]*" | head -1)
    [ -n "$job" ] || { echo "submit failed for $arm/$q: $out"; exit 1; }
    PAIRS+=("$arm.$q=$job")
    echo "$(date '+%T') submitted $arm $q -> $job"
  done
done

for pair in "${PAIRS[@]}"; do
  key="${pair%%=*}"; job="${pair#*=}"
  "$NEMO" --base-url "$BASE" jobs watch "$job" --workspace default >/dev/null 2>&1 || true
  status=$("$NEMO" --base-url "$BASE" jobs get-status "$job" --workspace default 2>/dev/null | grep -o '"status": "[^"]*"' | head -1)
  "$NEMO" --base-url "$BASE" jobs results download row-scores --job "$job" --workspace default --output-file "$HERE/runs/eval/$key.rows.jsonl" >/dev/null 2>&1 || echo "no row scores for $key"
  "$NEMO" --base-url "$BASE" jobs results download aggregate-scores --job "$job" --workspace default --output-file "$HERE/runs/eval/$key.aggregate.json" >/dev/null 2>&1 || true
  echo "$(date '+%T') $key $job $status -> runs/eval/$key.rows.jsonl"
done
