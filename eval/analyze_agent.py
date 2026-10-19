"""Score the deployed agents end to end from Helix Evals plugin row scores.

The agent answers with the runbook ID on line one and the action it took on line
two. The evaluation's exact-match checks line one; this script checks both, and counts
rows the agent never answered separately (timeouts are not wrong answers).

    python eval/analyze_agent.py runs/eval/agent/*.rows.jsonl
    python eval/analyze_agent.py results/agent/*.rows.jsonl.gz      # our runs

Files for the same agent in batches (agent-finetuned.b1, .b2, ...) are added up
into one line per agent.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ACTIONS = ("auto-remediate", "page-oncall", "ticket-business-hours", "escalate-incident-commander")


def score(paths: list[Path], name: str) -> dict:
    n = answered = rb_ok = act_ok = both_ok = 0
    for path in paths:
        for line in (gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")):
            r = json.loads(line); n += 1
            out = r.get("sample", {}).get("output_text")
            if not out:
                continue
            answered += 1
            lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
            rb = lines[0][:5] if lines else ""
            act = next((a for a in ACTIONS if len(lines) > 1 and lines[1].startswith(a)), "")
            item = r["item"]
            a = rb == item["runbook_id"]; b = act == item["action_id"]
            rb_ok += a; act_ok += b; both_ok += a and b
    return {"agent": name, "rows": n, "answered": answered,
            "runbook_accuracy": round(rb_ok / answered, 4) if answered else None,
            "action_accuracy": round(act_ok / answered, 4) if answered else None,
            "both_right": round(both_ok / answered, 4) if answered else None}


def agent_name(path: Path) -> str:
    """agent-finetuned.b2.rows.jsonl.gz -> agent-finetuned"""
    return path.name.split(".")[0]


if __name__ == "__main__":
    groups: dict[str, list[Path]] = {}
    for p in sys.argv[1:]:
        groups.setdefault(agent_name(Path(p)), []).append(Path(p))
    results = [score(paths, name) for name, paths in groups.items()]
    for r in results:
        print(json.dumps(r))
    out = ROOT / "runs" / "eval" / "agent-summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("wrote", out.relative_to(ROOT))
