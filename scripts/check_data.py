"""Check a prepared dataset before you train on it. Read only, no Helix calls.

Prints row counts per file, label balance per runbook and action, how many test
incidents come from scenario types held out of training, whether any test incident
also appears in training, and a scan for incidents that give away the answer (same
patterns split_dataset.py uses).

    python scripts/check_data.py                          # the packaged data in data/meridian-example
    python scripts/check_data.py runs/datasets/train runs/datasets/test   # your own split_dataset.py output

Reads .jsonl or .jsonl.gz. Exits 1 if a check fails.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "data"))
from mcps import prompting  # noqa: E402

TAX = prompting.load_taxonomy(ROOT / "taxonomy.json")
HOLDOUT = set(TAX["holdout_families"]["families"])
RB_BY_LETTER = {letter: rb["id"] for letter, rb in prompting.runbook_options(TAX)}
ACT_BY_LETTER = {letter: a["id"] for letter, a in zip(prompting.ACTION_LETTERS, TAX["actions"])}


def read(dirs: list[Path], name: str) -> list[dict]:
    for d in dirs:
        for p in (d / f"{name}.jsonl", d / f"{name}.jsonl.gz"):
            if p.exists():
                fh = gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else open(p, encoding="utf-8")
                with fh:
                    return [json.loads(line) for line in fh if line.strip()]
    return []


def incident_from_prompt(prompt: str) -> str:
    body = prompt.split("Incident:\n", 1)[-1]
    return re.split(r"\n\n(?:Runbooks|Actions):\n", body, maxsplit=1)[0].strip()


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]] or [ROOT / "data" / "meridian-example"]
    train = read(dirs, "train")
    validation = read(dirs, "validation")
    calibration = read(dirs, "calibration")
    test = read(dirs, "test") or read(dirs, "eval")
    failed = False

    print("Rows per file")
    print(f"  train        {len(train):>6} records ({len(train) // 2} incidents, two questions each)")
    print(f"  validation   {len(validation):>6} records ({len(validation) // 2} incidents)")
    print(f"  calibration  {len(calibration):>6} incidents")
    print(f"  test         {len(test):>6} incidents")
    if not train or not test:
        print("FAIL: train or test file missing")
        sys.exit(1)

    rb_train = Counter(RB_BY_LETTER.get(r["completion"].strip()) for r in train if "Runbooks:" in r["prompt"])
    act_train = Counter(ACT_BY_LETTER.get(r["completion"].strip()) for r in train if "Actions:" in r["prompt"])
    print("\nTraining incidents per runbook")
    print("  " + "  ".join(f"{k} {v}" for k, v in sorted(rb_train.items())))
    print("Training incidents per action")
    print("  " + "  ".join(f"{k} {v}" for k, v in sorted(act_train.items())))
    missing = [rb["id"] for rb in TAX["runbooks"] if rb_train.get(rb["id"], 0) == 0]
    if missing:
        print(f"FAIL: no training examples for {missing}")
        failed = True

    held = sum(1 for r in test if r.get("scenario_family") in HOLDOUT)
    train_families = {r.get("scenario_family") for r in calibration}
    print(f"\nTest incidents from scenario types held out of training: {held} of {len(test)}")
    if HOLDOUT & train_families:
        print(f"FAIL: held-out scenario types found in the calibration split: {sorted(HOLDOUT & train_families)}")
        failed = True

    train_texts = {incident_from_prompt(r["prompt"]) for r in train}
    overlap = sum(1 for r in test if r["incident_text"].strip() in train_texts)
    print(f"Test incidents that also appear in training: {overlap}")
    if overlap:
        failed = True

    try:
        import incidents  # noqa: PLC0415  (needs the data-designer library from requirements.txt)
        patterns = [re.compile(p) for p in incidents.LEAK_PATTERNS]
    except ImportError as exc:
        print(f"\nSKIP answer scan: {exc}. Install requirements.txt to run it.")
    else:
        def gives_away(text: str) -> bool:
            return any(p.search(text) for p in patterns)
        hits = {
            "train": sum(1 for t in train_texts if gives_away(t)),
            "calibration": sum(1 for r in calibration if gives_away(r["incident_text"])),
            "test": sum(1 for r in test if gives_away(r["incident_text"])),
        }
        print("\nIncidents whose text gives away the answer (runbook ID, team, action or scenario name):")
        print("  " + "  ".join(f"{k} {v}" for k, v in hits.items()))
        if any(hits.values()):
            failed = True

    print("\nFAIL" if failed else "\nPASS: data looks ready to train on")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
