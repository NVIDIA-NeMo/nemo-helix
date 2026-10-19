"""The confidence gate as an architecture: the fine-tuned decision model answers when
it is confident; otherwise the incident escalates to the large model with Meridian's
notes. Computed from the evaluation row scores already downloaded, so no extra calls.
Glue: this is the routing rule Switchyard would apply, evaluated offline.

    python eval/cascade.py --small runs/eval/remote/C-qwen-finetuned.action.rows.jsonl \
        --cal runs/eval/remote/cal-C-qwen-finetuned.action.rows.jsonl \
        --large runs/eval/A-plus-frontier-with-context.action.rows.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import analyze_results as ar  # noqa: E402


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")


def large_answers(path: Path, label_field: str, letters: str) -> dict[str, str]:
    out = {}
    for line in (gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")):
        r = json.loads(line)
        txt = (r["sample"].get("output_text") or "").strip()[:1]
        out[r["item"]["prompt"]] = txt if txt in letters and txt else ""
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True)
    ap.add_argument("--cal", required=True)
    ap.add_argument("--large", required=True)
    ap.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9,0.95,0.99")
    args = ap.parse_args()
    action = ".action." in Path(args.small).name
    letters = ar.prompting.ACTION_LETTERS if action else ar.LETTERS
    label_field = "action_completion" if action else "completion"

    temp = ar.fit_temperature(ar.load(Path(args.cal)))
    small_rows_raw = [json.loads(l) for l in _open(args.small)]
    small = ar.load(Path(args.small))
    big = large_answers(Path(args.large), label_field, letters)
    print(f"temperature {temp:.2f} (fit on calibration split)")
    print("threshold  small_handles  escalated  accuracy  small_acc_on_its_share")
    rows_out = []
    for th in [float(x) for x in args.thresholds.split(",")]:
        correct = handled = handled_ok = 0
        for raw, r in zip(small_rows_raw, small):
            conf = ar.confidence(r, temp) or 0.0
            if conf >= th:
                handled += 1
                ok = r["pred"] == r["label"]; handled_ok += ok
            else:
                ok = big.get(raw["item"]["prompt"], "") == r["label"]
            correct += ok
        n = len(small)
        row = {"threshold": th, "small_share": handled / n, "escalated_share": 1 - handled / n,
               "accuracy": correct / n, "small_accuracy": handled_ok / handled if handled else None}
        rows_out.append(row)
        print(f"  {th:.2f}      {row['small_share']:.0%}           {row['escalated_share']:.0%}       {row['accuracy']:.1%}    {row['small_accuracy'] if row['small_accuracy'] is None else format(row['small_accuracy'], '.1%')}")
    out = ROOT / "runs/eval" / f"cascade-{Path(args.small).name.replace('.gz', '').replace('.rows.jsonl', '')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"temperature": temp, "rows": rows_out}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
