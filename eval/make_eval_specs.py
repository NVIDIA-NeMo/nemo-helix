"""Write one evaluation job spec per experiment arm. Glue, labeled as such.

Everything the specs contain is Helix Evals plugin configuration:

- dataset: inline rows or a fileset of eval rows (prompt, prompt_with_context,
  completion letter, labels)
- prompt_template: "{{ item.prompt }}" (or the with-context prompt for arm A+)
- target: a GenericAgent pointing at the Helix inference gateway, so the request
  body can switch thinking off for Nemotron reasoning models and, for the
  decision-model arm, ask /completions for next-token logprobs
- metrics: exact-match on the answer letter; the raw response is kept per row via
  trajectory_path so calibration can be computed from the row scores

Usage:
    python eval/make_eval_specs.py --dataset default/meridian-incidents-test#eval.jsonl \
        --out eval/specs
    python eval/make_eval_specs.py --inline-from artifacts/preview_results_x/dataset.parquet --out eval/specs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
from mcps import prompting  # noqa: E402

HELIX = os.environ.get("NHX_BASE_URL", "http://localhost:8080").rstrip("/")
GATEWAY = HELIX + "/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
# The large model to compare against, as named on your gateway (`nemo models list`).
FRONTIER_MODEL = os.environ.get("FRONTIER_MODEL", "nvidia-nemotron-3-ultra-550b-a55b")


def provider_route(deployment: str) -> str:
    """A LoRA adapter only answers on its deployment's provider route, not on the
    OpenAI route (KNOWN-ISSUES.md). The base model answers here too."""
    return f"{HELIX}/apis/inference-gateway/v2/workspaces/default/provider/{deployment}/-/v1"


ARMS = {
    # name: (served model name, with_context, mode, base URL)
    "A-frontier-no-context": (FRONTIER_MODEL, False, "chat", None),
    "A-plus-frontier-with-context": (FRONTIER_MODEL, True, "chat", None),
    # Small models: one vLLM deployment per base serves both the base (B) and the
    # LoRA adapter (C). Check the served names with
    #   nemo inference providers get meridian-qwen -f json | jq -r '.served_models[].served_model_name'
    # The base is served as <workspace>/<model>, an adapter as <workspace>--<output name>.
    "B-qwen-untuned": ("default/qwen3.5-4b-base", False, "logits", provider_route("meridian-qwen")),
    "C-qwen-finetuned": ("default--meridian-router-qwen3.5-4b-seed29", False, "logits", provider_route("meridian-qwen")),
    "B-nemotron-untuned": ("default/nemotron-nano-9b-v2", False, "logits", provider_route("meridian-nemotron")),
    "C-nemotron-finetuned": ("default--meridian-router-nemotron-nano-9b-seed29", False, "logits", provider_route("meridian-nemotron")),
}


def exact_match_metric(name: str, reference: str, candidate: str) -> dict:
    return {
        "bundle_kind": "metric-bundle",
        "bundle_format_version": "v1",
        "metric_type": "exact-match",
        "metadata": {"description": f"{name}: predicted letter equals the labeled letter", "labels": {}},
        # The output name must equal the metric type; `name` only labels the description.
        "outputs": [{"name": "exact-match", "description": None,
                     "value_json_schema": {"description": "Continuous numeric metric value.", "title": "ContinuousScore", "type": "number"}}],
        "secrets": {},
        "payload": {"kind": "inline", "metric": {"type": "exact-match", "description": None, "labels": {},
                                                  "supported_job_types": ["online", "offline"],
                                                  "reference": reference, "candidate": candidate}},
    }


def target(model: str, mode: str, base: str | None = None) -> dict:
    base = base or GATEWAY
    if mode == "chat":
        return {
            "format": "generic",
            "name": f"gateway-chat-{model}",
            "url": f"{base}/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": "{{ prompt }}"}],
                "temperature": 0.0,
                "max_tokens": 8,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            "response_path": "$.choices[0].message.content",
            "trajectory_path": "$.usage",
        }
    # Decision-model arms: one forward pass, read the next-token distribution.
    return {
        "format": "generic",
        "name": f"gateway-logits-{model}",
        "url": f"{base}/completions",
        "body": {"model": model, "prompt": "{{ prompt }}", "temperature": 0.0, "max_tokens": 1, "logprobs": 20},
        "response_path": "$.choices[0].text",
        "trajectory_path": "$.choices[0].logprobs.top_logprobs[0]",
    }


def spec(arm: str, dataset, parallelism: int = 2, question: str = "runbook", base: str | None = None, model: str | None = None) -> dict:
    """question: "runbook" (which runbook owns it) or "action" (what to do, given the runbook)."""
    default_model, with_context, mode, default_base = ARMS[arm]
    model = model or default_model
    base = base or default_base
    if question == "runbook":
        prompt_field, completion_field = ("prompt_with_context" if with_context else "prompt"), "completion"
    else:
        prompt_field, completion_field = ("action_prompt_with_context" if with_context else "action_prompt"), "action_completion"
    return {
        "dataset": dataset,
        "prompt_template": "{{ item." + prompt_field + " }}",
        "target": target(model, mode, base),
        # The public endpoint returns 503s under load; keep concurrency low and retry
        # hard so a dropped request is retried rather than scored as wrong.
        "params": {"parallelism": parallelism, "ignore_request_failure": True, "request_timeout": 180, "max_retries": 8},
        "metrics": [
            exact_match_metric(f"{question}_accuracy", "{{ item." + completion_field + " | trim }}", "{{ (sample.output_text | trim)[:1] }}"),
        ],
    }


STRANDS_URL = "http://127.0.0.1:8765/v1/systemone"


def strands_spec(dataset, question: str, parallelism: int = 1) -> dict:
    """Arm S: Strands Decider 2B, untuned, through its own /v1/systemone server.
    Options are sent per request (runbook or action IDs with their definitions),
    so no prompt letters are involved; the answer is the ID itself."""
    tax = prompting.load_taxonomy()
    if question == "runbook":
        criteria = {rb["id"]: f"{rb['name']} ({rb['team']}): {rb['definition']}" for rb in tax["runbooks"]}
        instructions = "Which Meridian Compute runbook owns this GPU platform incident?"
        reference = "{{ item.runbook_id }}"
    else:
        criteria = {a["id"]: a["definition"] for a in tax["actions"]}
        # Generic-target bodies only see {{ prompt }}, so the owning runbook travels in the state.
        instructions = "The owning runbook is stated at the top. Which response action should Meridian take for this incident?"
        reference = "{{ item.action_id }}"
    state = "{{ item.incident_text }}" if question == "runbook" else "Owning runbook: {{ item.runbook_id }}\n\nIncident:\n{{ item.incident_text }}"
    return {
        "dataset": dataset,
        "prompt_template": state,
        "target": {
            "format": "generic",
            "name": "strands-decider-2b",
            "url": STRANDS_URL,
            "body": {
                "model": "strands-decider-2b",
                "state": "{{ prompt }}",
                "questions": {"answer": {"type": "choice", "instructions": instructions, "criteria": criteria}},
            },
            "response_path": "$.answers.answer.choice",
            "trajectory_path": "$.answers.answer",
        },
        "params": {"parallelism": parallelism, "ignore_request_failure": True, "request_timeout": 180, "max_retries": 3},
        "metrics": [exact_match_metric(f"{question}_accuracy", reference, "{{ sample.output_text | trim }}")],
    }


def rows_from_parquet(path: Path) -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(path)
    tax = prompting.load_taxonomy()
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "incident_text": r["incident_text"],
            "runbook_id": r["runbook_id"],
            "action_id": r["action_id"],
            "scenario_family": r["scenario_family"],
            "reporter_style": r["reporter_style"],
            "completion": " " + prompting.letter_for_runbook(r["runbook_id"], tax),
            "prompt": prompting.render_runbook_question(r["incident_text"], tax),
            "prompt_with_context": prompting.render_runbook_question(r["incident_text"], tax, with_context=True),
            "action_completion": " " + prompting.letter_for_action(r["action_id"], tax),
            "action_prompt": prompting.render_action_question(r["incident_text"], r["runbook_id"], tax),
            "action_prompt_with_context": prompting.render_action_question(r["incident_text"], r["runbook_id"], tax, with_context=True),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="fileset ref, e.g. default/meridian-incidents-test#eval.jsonl")
    ap.add_argument("--inline-from", help="parquet to turn into inline rows (smoke tests)")
    ap.add_argument("--out", default=str(ROOT / "eval" / "specs"))
    ap.add_argument("--arms", nargs="*", default=list(ARMS) + ["S-strands-decider-untuned"])
    ap.add_argument("--question", choices=["runbook", "action"], default="runbook")
    ap.add_argument("--base", help="OpenAI-compatible base URL to call instead of the default for the arm (e.g. an in-cluster vLLM service)")
    ap.add_argument("--model", help="override the served model name for the selected arm(s)")
    args = ap.parse_args()
    if args.inline_from:
        dataset = rows_from_parquet(Path(args.inline_from))
    elif args.dataset:
        dataset = args.dataset  # FilesetRef is a plain string: workspace/fileset#file
    else:
        ap.error("--dataset or --inline-from required")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        p = out / f"{arm}.{args.question}.json"
        body = strands_spec(dataset, args.question) if arm == "S-strands-decider-untuned" else spec(arm, dataset, question=args.question, base=args.base, model=args.model)
        p.write_text(json.dumps(body, indent=2))
        print("wrote", p)


if __name__ == "__main__":
    main()
