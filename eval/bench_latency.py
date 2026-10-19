"""Latency and tokens per routing decision, same incidents, one request at a time.
Glue: Helix Evals plugin rows do not record per-request latency (see WORKAROUNDS.md E6).

    export FRONTIER_MODEL=<large model name on your gateway>    # for the A arms
    export NHX_ACCESS_TOKEN=$(nemo auth token)                   # only if your Helix requires login
    python eval/bench_latency.py --n 50 --arms A-plus-frontier-with-context C-qwen-finetuned
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
from mcps import prompting  # noqa: E402

TAX = prompting.load_taxonomy(ROOT / "taxonomy.json")
BASE = os.environ.get("NHX_BASE_URL", "http://localhost:8080").rstrip("/")
LOCAL_GW = f"{BASE}/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
REMOTE = f"{BASE}/apis/inference-gateway/v2/workspaces/default/provider"
FRONTIER_MODEL = os.environ.get("FRONTIER_MODEL", "REPLACE-your-large-model")
FINETUNED_MODEL = os.environ.get("FINETUNED_MODEL", "default--meridian-router-qwen3.5-4b-seed29")
STRANDS = "http://127.0.0.1:8765/v1/systemone"


def post(url: str, body: dict, token: str | None = None) -> tuple[dict, float]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    t0 = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=180))
    return d, (time.perf_counter() - t0) * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "runs/latency.json"))
    ap.add_argument("--arms", nargs="*", help="subset of arms to run")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds between requests (rate-limited endpoints)")
    args = ap.parse_args()
    token = os.environ.get("NHX_ACCESS_TOKEN")
    test = ROOT / "data/meridian-example/test.jsonl"
    fh = open(test, encoding="utf-8") if test.exists() else gzip.open(f"{test}.gz", "rt", encoding="utf-8")
    rows = [json.loads(l) for l in fh][: args.n]
    rb_criteria = {rb["id"]: f"{rb['name']} ({rb['team']}): {rb['definition']}" for rb in TAX["runbooks"]}

    arms = {
        "A-frontier-no-context": lambda r: post(f"{LOCAL_GW}/chat/completions", {
            "model": FRONTIER_MODEL, "messages": [{"role": "user", "content": r["prompt"]}],
            "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}),
        "A-plus-frontier-with-context": lambda r: post(f"{LOCAL_GW}/chat/completions", {
            "model": FRONTIER_MODEL, "messages": [{"role": "user", "content": r["prompt_with_context"]}],
            "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}),
        "C-qwen-finetuned": lambda r: post(f"{REMOTE}/meridian-qwen/-/v1/completions", {
            "model": FINETUNED_MODEL, "prompt": r["prompt"], "max_tokens": 1,
            "temperature": 0, "logprobs": 20}, token),
        "S-strands-decider-untuned": lambda r: post(STRANDS, {
            "model": "strands-decider-2b", "state": r["incident_text"],
            "questions": {"answer": {"type": "choice", "instructions": "Which Meridian Compute runbook owns this GPU platform incident?", "criteria": rb_criteria}}}),
    }
    results = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    for name, call in arms.items():
        if args.arms and name not in args.arms:
            continue
        lat, ptok, ctok, errs = [], [], [], 0
        try:
            call(rows[0])  # warm-up, not counted
        except Exception:
            pass
        for r in rows:
            time.sleep(args.pause)
            try:
                d, ms = call(r)
            except Exception as e:  # noqa: BLE001
                errs += 1
                continue
            lat.append(ms)
            u = d.get("usage") or {}
            ptok.append(u.get("prompt_tokens") or u.get("input_tokens") or 0)
            ctok.append(u.get("completion_tokens") or u.get("output_tokens") or 0)
        lat.sort()
        results[name] = {
            "n": len(lat), "errors": errs,
            "p50_ms": round(statistics.median(lat)) if lat else None,
            "p95_ms": round(lat[int(0.95 * (len(lat) - 1))]) if lat else None,
            "mean_prompt_tokens": round(statistics.mean(ptok)) if ptok else None,
            "mean_completion_tokens": round(statistics.mean(ctok), 1) if ctok else None,
        }
        print(name, results[name], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
