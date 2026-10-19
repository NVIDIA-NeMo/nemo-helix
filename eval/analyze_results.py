"""Turn evaluation row scores into the comparison table. Glue, labeled as such.

The Helix Evals plugin gives per-row exact-match plus, for decision-model arms, the raw
next-token logprobs captured through trajectory_path. This script adds what
the Helix Evals plugin does not compute natively: macro-F1, per-class confusion, expected
calibration error, Brier score, and the confidence gate (coverage vs error at a
threshold chosen on the calibration split).

Usage:
    nemo jobs results download row-scores --job <job> --output-file runs/eval/<arm>.rows.jsonl
    python eval/analyze_results.py runs/eval/A-frontier-no-context.rows.jsonl runs/eval/C-small-finetuned.rows.jsonl \
        --calibration runs/eval/C-calibration.rows.jsonl --error-budget 0.05
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
from mcps import prompting  # noqa: E402

LETTERS = prompting.LETTERS


def load(path: Path) -> list[dict]:
    letters = prompting.ACTION_LETTERS if ".action." in path.name else LETTERS
    label_field = "action_completion" if ".action." in path.name else "completion"
    rows = []
    for line in (gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")):
        r = json.loads(line)
        item = r["item"]; sample = r.get("sample") or {}
        out = sample.get("output_text")
        pred = (out or "").strip()[:1] or None
        label = item[label_field].strip()
        probs = None
        traj = sample.get("trajectory")
        if isinstance(traj, dict) and traj:
            # top_logprobs: {token: logprob}; keep only option letters, softmax, temperature later
            scores = {}
            for tok, lp in traj.items():
                t = str(tok).strip()
                if len(t) == 1 and t in letters:   # "" is "in" every string: require one letter
                    scores[t] = max(scores.get(t, -1e9), float(lp))
            if scores:
                z = max(scores.values()); den = sum(math.exp(v - z) for v in scores.values())
                probs = {k: math.exp(v - z) / den for k, v in scores.items()}
                # Decision-model readout: the most probable option letter, whatever text was generated.
                pred = max(probs, key=probs.get)
        rows.append({"label": label, "pred": pred, "probs": probs, "family": item.get("scenario_family"),
                     "style": item.get("reporter_style"), "runbook": item.get("runbook_id"), "collision": item.get("collision"),
                     "answered": out is not None})
    return rows


def macro_f1(rows: list[dict]) -> float:
    labels = sorted({r["label"] for r in rows})
    f1s = []
    for c in labels:
        tp = sum(1 for r in rows if r["label"] == c and r["pred"] == c)
        fp = sum(1 for r in rows if r["label"] != c and r["pred"] == c)
        fn = sum(1 for r in rows if r["label"] == c and r["pred"] != c)
        p = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * rc / (p + rc) if p + rc else 0.0)
    return sum(f1s) / len(f1s) if f1s else float("nan")


def confidence(r: dict, temperature: float) -> float | None:
    if not r["probs"]:
        return None
    logits = {k: math.log(v) / temperature for k, v in r["probs"].items()}
    z = max(logits.values()); den = sum(math.exp(v - z) for v in logits.values())
    return math.exp(logits[r["pred"]] - z) / den if r["pred"] in logits else None


def ece_brier(rows: list[dict], temperature: float, bins: int = 10) -> tuple[float, float]:
    pairs = [(confidence(r, temperature), 1.0 if r["pred"] == r["label"] else 0.0) for r in rows]
    pairs = [(c, y) for c, y in pairs if c is not None]
    if not pairs:
        return float("nan"), float("nan")
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inb = [(c, y) for c, y in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if inb:
            ece += len(inb) / len(pairs) * abs(sum(y for _, y in inb) / len(inb) - sum(c for c, _ in inb) / len(inb))
    brier = sum((c - y) ** 2 for c, y in pairs) / len(pairs)
    return ece, brier


def fit_temperature(rows: list[dict]) -> float:
    """Grid-search the temperature that minimises NLL of the labeled letter on the calibration rows."""
    best, best_nll = 1.0, float("inf")
    for t in [x / 20 for x in range(5, 81)]:
        nll, n = 0.0, 0
        for r in rows:
            if not r["probs"] or r["label"] not in r["probs"]:
                continue
            logits = {k: math.log(v) / t for k, v in r["probs"].items()}
            z = max(logits.values()); den = sum(math.exp(v - z) for v in logits.values())
            nll -= math.log(max(math.exp(logits[r["label"]] - z) / den, 1e-12)); n += 1
        if n and nll / n < best_nll:
            best, best_nll = t, nll / n
    return best


def gate(rows: list[dict], temperature: float, threshold: float) -> dict:
    auto = [r for r in rows if (confidence(r, temperature) or 0.0) >= threshold]
    wrong = sum(1 for r in auto if r["pred"] != r["label"])
    return {"threshold": threshold, "automated": len(auto), "to_human": len(rows) - len(auto),
            "coverage": len(auto) / len(rows) if rows else 0, "error_among_automated": wrong / len(auto) if auto else 0.0}


def choose_threshold(cal_rows: list[dict], temperature: float, error_budget: float) -> float:
    best = 1.0
    for th in [x / 100 for x in range(50, 100)]:
        g = gate(cal_rows, temperature, th)
        if g["error_among_automated"] <= error_budget and g["automated"] >= 10:
            best = th; break
    return best


def report(name: str, rows: list[dict], temperature: float | None, threshold: float | None) -> dict:
    answered = [r for r in rows if r["answered"] or r["probs"]]
    acc = sum(1 for r in answered if r["pred"] == r["label"]) / len(answered) if answered else float("nan")
    out = {"arm": name, "rows": len(rows), "unanswered": len(rows) - len(answered), "accuracy": round(acc, 4), "macro_f1": round(macro_f1(answered), 4)}
    by = defaultdict(list)
    for r in answered:
        by[r["style"]].append(r["pred"] == r["label"])
    out["accuracy_by_format"] = {k: round(sum(v) / len(v), 3) for k, v in sorted(by.items())}
    heldout = [r for r in answered if r["family"] in set(prompting.load_taxonomy()["holdout_families"]["families"])]
    if heldout:
        out["accuracy_heldout_families"] = round(sum(1 for r in heldout if r["pred"] == r["label"]) / len(heldout), 4)
    coll = [r for r in answered if r.get("collision") == "yes"]
    if coll:
        out["accuracy_collision_cases"] = round(sum(1 for r in coll if r["pred"] == r["label"]) / len(coll), 4)
    conf = Counter((r["label"], r["pred"]) for r in answered if r["pred"] != r["label"])
    out["top_confusions"] = [f"{a}->{b}: {n}" for (a, b), n in conf.most_common(5)]
    if temperature is not None and any(r["probs"] for r in answered):
        ece, brier = ece_brier(answered, temperature)
        out.update({"temperature": round(temperature, 3), "ece": round(ece, 4), "brier": round(brier, 4)})
        if threshold is not None:
            out["gate"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in gate(answered, temperature, threshold).items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+", help="row-scores JSONL files, one per arm")
    ap.add_argument("--calibration", help="row-scores JSONL for the calibration split of the decision model")
    ap.add_argument("--error-budget", type=float, default=0.05, help="max error rate among automated decisions")
    ap.add_argument("--out", default=str(ROOT / "runs" / "eval" / "summary.json"))
    args = ap.parse_args()

    temperature = threshold = None
    if args.calibration:
        cal = load(Path(args.calibration))
        temperature = fit_temperature(cal)
        threshold = choose_threshold(cal, temperature, args.error_budget)
        print(f"calibration: temperature={temperature:.2f} threshold={threshold:.2f} (error budget {args.error_budget})")

    results = []
    for p in args.rows:
        rows = load(Path(p))
        rep = report(Path(p).stem.replace(".rows", ""), rows, temperature if any(r["probs"] for r in rows) else None, threshold)
        results.append(rep)
        print(json.dumps(rep, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
