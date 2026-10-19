"""Turn Data Designer job output into the filesets the rest of the pipeline reads. Glue, labeled as such.

What it does, and why it is a script rather than platform config:
- downloads the generated dataset from a Data Designer job (nemo jobs results download)
- drops rows that fail the quality gates: judge label_support < 4, judge no_leak < 4,
  any deterministic leak hint, or a regex hit on the full leak list (team names and
  scenario slugs are too many for the sandboxed expression columns)
- splits the training run into train / validation / calibration, stratified by runbook
- writes JSONL files: train.jsonl and validation.jsonl in TRL prompt/completion form
  (loss lands on the answer letter only), plus eval.jsonl rows that carry both the
  plain prompt and the with-context prompt for the four experiment arms
- uploads them as filesets with `nemo files upload`

Usage:
    python scripts/split_dataset.py --job <train-job> --split train --fileset default/meridian-incidents-train
    python scripts/split_dataset.py --job <test-job>  --split test  --fileset default/meridian-incidents-test
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "data"))
from mcps import prompting  # noqa: E402
import incidents  # noqa: E402  (for LEAK_PATTERNS)

NEMO = os.environ.get("NEMO", "nemo")
BASE_URL = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
TAX = prompting.load_taxonomy(ROOT / "taxonomy.json")
LEAK_RE = [re.compile(p) for p in incidents.LEAK_PATTERNS]  # case handled per pattern


def nemo(*args: str) -> str:
    cmd = [NEMO, "--base-url", BASE_URL, *args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)}\n{res.stdout}\n{res.stderr}")
    return res.stdout


def download(job: str, workdir: Path) -> pd.DataFrame:
    """A Data Designer job exposes two results: `analysis` (JSON column statistics) and
    `artifacts` (a gzipped tar with dataset parquet files and processors-files/)."""
    import tarfile
    archive = workdir / f"{job}-artifacts.tgz"
    nemo("jobs", "results", "download", "artifacts", "--job", job, "--workspace", "default", "--output-file", str(archive))
    outdir = workdir / f"{job}-artifacts"
    outdir.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(outdir, filter="data")
    parts = sorted(p for p in outdir.rglob("*.parquet") if p.parent.name == "parquet-files")
    if not parts:
        raise SystemExit(f"no dataset parquet under {outdir}")
    print("dataset files:", [str(p.relative_to(outdir)) for p in parts])
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def judge_scores(q) -> dict[str, int]:
    if isinstance(q, str):
        try:
            q = json.loads(q)
        except json.JSONDecodeError:
            return {}
    if not isinstance(q, dict):
        return {}
    out = {}
    for k, v in q.items():
        try:
            out[k] = int(v.get("score") if isinstance(v, dict) else v)
        except (TypeError, ValueError):
            pass
    return out


def leaks(text: str) -> list[str]:
    return sorted({m.group(0) for rx in LEAK_RE for m in rx.finditer(text or "")})


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keep, dropped = [], {"judge_support": 0, "judge_leak": 0, "leak_hint": 0, "regex_leak": 0, "realism": 0, "empty": 0}
    for _, r in df.iterrows():
        text = str(r.get("incident_text") or "").strip()
        if len(text) < 40:
            dropped["empty"] += 1; continue
        s = judge_scores(r.get("quality"))
        if s.get("label_support", 5) < 4:
            dropped["judge_support"] += 1; continue
        if s.get("no_leak", 5) < 4:
            dropped["judge_leak"] += 1; continue
        if s.get("realism", 5) < 3:
            dropped["realism"] += 1; continue
        if any(bool(r.get(c)) for c in ("leak_runbook_words", "leak_action_words", "leak_severity_codes")):
            dropped["leak_hint"] += 1; continue
        if leaks(text):
            dropped["regex_leak"] += 1; continue
        keep.append(r)
    return pd.DataFrame(keep), dropped


POLICY = {rb["id"]: rb["policy"] for rb in TAX["runbooks"]}


def relabel_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Meridian's policy is deterministic in (runbook, severity). Rows generated
    before the policy was pinned carry a sampled action; recompute it so every
    split agrees with the taxonomy. The text only ever encoded severity."""
    df = df.copy()
    if "action_id" in df.columns:
        df["sampled_action_id"] = df["action_id"]
    df["action_id"] = [POLICY[rb][sev] for rb, sev in zip(df["runbook_id"], df["severity"])]
    return df


def eval_row(r) -> dict:
    return {
        "incident_text": r["incident_text"],
        "runbook_id": r["runbook_id"],
        "action_id": r["action_id"],
        "severity": r.get("severity"),
        "scenario_family": r["scenario_family"],
        "reporter_style": r["reporter_style"],
        "collision": r.get("collision"),
        "completion": " " + prompting.letter_for_runbook(r["runbook_id"], TAX),
        "action_completion": " " + prompting.letter_for_action(r["action_id"], TAX),
        "prompt": prompting.render_runbook_question(r["incident_text"], TAX),
        "prompt_with_context": prompting.render_runbook_question(r["incident_text"], TAX, with_context=True),
        "action_prompt": prompting.render_action_question(r["incident_text"], r["runbook_id"], TAX),
        "action_prompt_with_context": prompting.render_action_question(r["incident_text"], r["runbook_id"], TAX, with_context=True),
    }


def train_rows(r) -> list[dict]:
    """Two TRL prompt/completion records per incident (loss on the completion only):
    which runbook owns it, and given the runbook, which action Meridian's policy takes.
    The decision model answers both questions at inference."""
    return [
        {"prompt": prompting.render_runbook_question(r["incident_text"], TAX), "completion": " " + prompting.letter_for_runbook(r["runbook_id"], TAX)},
        {"prompt": prompting.render_action_question(r["incident_text"], r["runbook_id"], TAX), "completion": " " + prompting.letter_for_action(r["action_id"], TAX)},
    ]


def write_jsonl(rows, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_split(df: pd.DataFrame, fractions: dict[str, float], seed: int) -> dict[str, pd.DataFrame]:
    rng = random.Random(seed)
    parts = {k: [] for k in fractions}
    for _, group in df.groupby("runbook_id"):
        idx = list(group.index); rng.shuffle(idx)
        n = len(idx); start = 0
        names = list(fractions)
        for i, name in enumerate(names):
            count = n - start if i == len(names) - 1 else int(round(n * fractions[name]))
            parts[name] += idx[start:start + count]; start += count
    return {k: df.loc[v] for k, v in parts.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--fileset", required=True, help="workspace/name, e.g. default/meridian-incidents-train")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    workdir = ROOT / "runs" / "datasets"; workdir.mkdir(parents=True, exist_ok=True)
    df = download(args.job, workdir)
    print(f"downloaded {len(df)} rows")
    df = relabel_actions(df)
    kept, dropped = clean(df)
    print(f"kept {len(kept)} rows; dropped {dropped}")
    print("per runbook:", kept["runbook_id"].value_counts().sort_index().to_dict())

    outdir = workdir / args.split; outdir.mkdir(exist_ok=True)
    if args.split == "train":
        parts = stratified_split(kept, {"train": 0.84, "validation": 0.08, "calibration": 0.08}, args.seed)
        write_jsonl([row for _, r in parts["train"].iterrows() for row in train_rows(r)], outdir / "train.jsonl")
        write_jsonl([row for _, r in parts["validation"].iterrows() for row in train_rows(r)], outdir / "validation.jsonl")
        write_jsonl([eval_row(r) for _, r in parts["calibration"].iterrows()], outdir / "calibration.jsonl")
        print({k: len(v) for k, v in parts.items()})
    else:
        write_jsonl([eval_row(r) for _, r in kept.iterrows()], outdir / "eval.jsonl")
        heldout = kept[kept["scenario_family"].isin(TAX["holdout_families"]["families"])]
        print(f"test rows in held-out families: {len(heldout)} of {len(kept)}")
    kept.to_parquet(outdir / "clean.parquet", index=False)

    if not args.no_upload:
        # `nemo files` takes the bare fileset name; the fileset must exist first.
        workspace, _, fileset = args.fileset.rpartition("/")
        workspace = workspace or "default"
        res = subprocess.run([NEMO, "--base-url", BASE_URL, "files", "filesets", "create", fileset, "--workspace", workspace,
                              "--description", f"Meridian incident routing: {args.split} split"], capture_output=True, text=True)
        print("fileset:", "created" if res.returncode == 0 else "exists or error: " + (res.stderr or res.stdout).strip()[-120:])
        for f in sorted(outdir.glob("*.jsonl")):
            print(nemo("files", "upload", str(f), fileset, "--workspace", workspace).strip()[-160:])
        print(f"uploaded to {workspace}/{fileset}")


if __name__ == "__main__":
    main()
