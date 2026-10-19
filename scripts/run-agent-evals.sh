#!/usr/bin/env bash
# Score deployed agents end to end with the Helix Evals plugin, 50 test incidents per job.
# A whole agent evaluation must finish inside the platform's 1-hour job limit, or every
# row is lost (KNOWN-ISSUES.md), so the test set goes in batches of 50: batch 1 is
# incidents 1 to 50, batch 2 is 51 to 100, and so on. Each batch becomes its own
# fileset, meridian-incidents-test-b<N>.
#
#   scripts/run-agent-evals.sh agent-finetuned agent-frontier-with-context        # batch 1 only
#   BATCHES="1 2 3 4" scripts/run-agent-evals.sh agent-finetuned agent-frontier-with-context   # 200 incidents
#
# Agent names match eval/specs-agent/<name>.json. Row scores land in
# runs/eval/agent/<name>.b<N>.rows.jsonl; read them with eval/analyze_agent.py.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
NEMO="${NEMO:-nemo}"
PY="${PY:-python3}"
BASE="${NHX_BASE_URL:-http://localhost:8080}"
BATCHES="${BATCHES:-1}"
TEST="$HERE/data/meridian-example/test.jsonl"
[ -f "$TEST" ] || gunzip -k "$TEST.gz"
[ "$#" -gt 0 ] || { echo "usage: $0 <agent-spec-name>..." >&2; exit 2; }
OUT="$HERE/runs/eval/agent"
mkdir -p "$OUT"

for b in $BATCHES; do
  dir="$OUT/batches/b$b"; mkdir -p "$dir"
  sed -n "$(( (b - 1) * 50 + 1 )),$(( b * 50 ))p" "$TEST" > "$dir/eval.jsonl"
  "$NEMO" --base-url "$BASE" files filesets create "meridian-incidents-test-b$b" --workspace default \
    --description "Meridian test incidents, batch $b" >/dev/null 2>&1 || true
  "$NEMO" --base-url "$BASE" files upload "$dir/eval.jsonl" "meridian-incidents-test-b$b" --workspace default >/dev/null
done

for name in "$@"; do
  for b in $BATCHES; do
    spec="$OUT/batches/$name.b$b.json"
    "$PY" - "$HERE/eval/specs-agent/$name.json" "$spec" "$b" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
s["dataset"] = f"default/meridian-incidents-test-b{sys.argv[3]}#eval.jsonl"
s["target"]["name"] = f"{s['target']['name']}-b{sys.argv[3]}"
json.dump(s, open(sys.argv[2], "w"), indent=2)
PY
    job=$("$NEMO" --base-url "$BASE" evaluator evaluate --spec-file "$spec" 2>&1 | grep -o "nemo-evaluator-[a-z0-9]*" | head -1)
    [ -n "$job" ] || { echo "submit failed for $name batch $b"; continue; }
    echo "$(date '+%T') $name batch $b -> $job"
    "$NEMO" --base-url "$BASE" jobs watch "$job" --workspace default >/dev/null 2>&1 || true
    "$NEMO" --base-url "$BASE" jobs results download row-scores --job "$job" --workspace default \
      --output-file "$OUT/$name.b$b.rows.jsonl" >/dev/null 2>&1
    echo "$(date '+%T') $name batch $b done: $(wc -l < "$OUT/$name.b$b.rows.jsonl" 2>/dev/null || echo 0) rows"
  done
done
