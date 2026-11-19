# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run every eval prompt N times against a deployment and score its tool calls from the chat stream."""

import argparse
import asyncio
import collections
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

EVAL_DIR = Path(__file__).parent
TOOL_LINE = re.compile(r"^→ (?P<tool>[\w.-]+): ", re.MULTILINE)
REJECTION_MARKER = "TOOL CALL REJECTED"
REASON_CATEGORIES = [
    ("corrupted or unknown key", re.compile(r"unknown field")),
    ("missing required field", re.compile(r"requires '|is a required property")),
    ("model/provider not on platform", re.compile(r"not registered|not served")),
    ("undeclared model alias", re.compile(r"no model is configured with alias")),
    ("model_configs as a column", re.compile(r"cannot be built from a description")),
    ("bad column reference", re.compile(r"not an earlier column")),
    ("invalid job name", re.compile(r"entity name pattern")),
    ("misplaced or extra property", re.compile(r"Additional properties")),
]


@dataclass
class RunResult:
    prompt_id: str
    attempt: int
    calls: int = 0
    rejected: int = 0
    ended_accepted: bool = False
    final_text: str = ""
    seconds: float = 0.0
    error: str | None = None
    reasons: list[str] = field(default_factory=list)


def categorize(reason: str) -> str:
    for label, pattern in REASON_CATEGORIES:
        if pattern.search(reason):
            return label
    return "other"


def score_reasoning(reasoning: str, result: RunResult) -> None:
    starts = [match.start() for match in TOOL_LINE.finditer(reasoning)]
    for start, end in zip(starts, [*starts[1:], len(reasoning)][: len(starts)], strict=True):
        block = reasoning[start:end]
        result.calls += 1
        is_rejected = REJECTION_MARKER in block
        result.rejected += is_rejected
        result.ended_accepted = not is_rejected
        if is_rejected:
            result.reasons.extend(line.strip()[2:] for line in block.splitlines() if line.strip().startswith("- "))


async def run_one(
    client: httpx.AsyncClient, url: str, prompt_id: str, prompt: str, attempt: int, timeout: float
) -> RunResult:
    result = RunResult(prompt_id=prompt_id, attempt=attempt)
    reasoning: list[str] = []
    content: list[str] = []
    started = time.monotonic()
    body = {"model": "agent", "stream": True, "messages": [{"role": "user", "content": prompt}]}
    try:
        async with client.stream("POST", url, json=body, timeout=timeout) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: {"):
                    continue
                chunk = json.loads(line[len("data: ") :])
                if "error" in chunk:
                    result.error = str(chunk["error"].get("message"))
                    continue
                delta = chunk["choices"][0]["delta"]
                reasoning.append(delta.get("reasoning_content") or "")
                content.append(delta.get("content") or "")
    except (httpx.HTTPError, json.JSONDecodeError) as error:
        result.error = f"{type(error).__name__}: {error}"
    result.seconds = round(time.monotonic() - started, 1)
    result.final_text = "".join(content).strip()[:500]
    score_reasoning("".join(reasoning), result)
    return result


@dataclass
class Rates:
    tool_calls: int
    rejected_calls: int
    tool_error_rate: float
    first_call_accepted_rate: float
    task_success_rate: float
    avg_calls_per_run: float


@dataclass
class Summary:
    label: str
    runs: int
    runs_with_errors: int
    overall: Rates
    rejection_reasons: dict[str, int]
    per_prompt: dict[str, Rates]


def rates(results: list[RunResult]) -> Rates:
    calls = sum(r.calls for r in results)
    rejected = sum(r.rejected for r in results)
    return Rates(
        tool_calls=calls,
        rejected_calls=rejected,
        tool_error_rate=round(rejected / calls, 3) if calls else 0.0,
        first_call_accepted_rate=round(sum(1 for r in results if r.calls and r.rejected == 0) / len(results), 3),
        task_success_rate=round(sum(r.ended_accepted for r in results) / len(results), 3),
        avg_calls_per_run=round(calls / len(results), 2),
    )


def summarize(label: str, results: list[RunResult]) -> Summary:
    by_prompt: dict[str, list[RunResult]] = collections.defaultdict(list)
    for r in results:
        by_prompt[r.prompt_id].append(r)
    reasons = collections.Counter(categorize(reason) for r in results for reason in r.reasons)
    return Summary(
        label=label,
        runs=len(results),
        runs_with_errors=sum(1 for r in results if r.error),
        overall=rates(results),
        rejection_reasons=dict(reasons.most_common()),
        per_prompt={prompt_id: rates(rs) for prompt_id, rs in by_prompt.items()},
    )


def print_summary(summary: Summary) -> None:
    overall = summary.overall
    print(f"\n== {summary.label}: {summary.runs} runs, {overall.tool_calls} tool calls")
    print(f"tool_error_rate          {overall.tool_error_rate:.0%}")
    print(f"first_call_accepted_rate {overall.first_call_accepted_rate:.0%}")
    print(f"task_success_rate        {overall.task_success_rate:.0%}")
    print(f"avg_calls_per_run        {overall.avg_calls_per_run}")
    print(f"runs_with_errors         {summary.runs_with_errors}")
    print("rejection reasons:")
    for reason, count in summary.rejection_reasons.items():
        print(f"  {count:>4}  {reason}")
    print("per prompt:")
    for prompt_id, stats in summary.per_prompt.items():
        print(
            f"  {prompt_id:<20} error {stats.tool_error_rate:.0%}  first-call ok {stats.first_call_accepted_rate:.0%}  "
            f"success {stats.task_success_rate:.0%}  calls {stats.avg_calls_per_run}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--label", required=True, help="Name for this batch, e.g. v1-baseline.")
    parser.add_argument("--runs", type=int, default=4, help="Runs per prompt.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--base-url", default=os.environ.get("NHX_BASE_URL", "http://localhost:8080"))
    args = parser.parse_args()

    url = (
        f"{args.base_url.rstrip('/')}/apis/agents/v2/workspaces/{args.workspace}"
        f"/deployments/{args.deployment}/-/v1/chat/completions"
    )
    prompts = [json.loads(line) for line in (EVAL_DIR / "prompts.jsonl").read_text().splitlines() if line.strip()]
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient() as client:

        async def bounded(prompt: dict[str, str], attempt: int) -> RunResult:
            async with semaphore:
                result = await run_one(client, url, prompt["id"], prompt["prompt"], attempt, args.timeout)
                status = "ok" if result.ended_accepted else "FAILED"
                print(
                    f"{result.prompt_id:<20} #{attempt}  {status:<6} calls={result.calls} "
                    f"rejected={result.rejected} {result.seconds}s{'  ERROR ' + result.error if result.error else ''}",
                    flush=True,
                )
                return result

        results = await asyncio.gather(*(bounded(p, i) for p in prompts for i in range(1, args.runs + 1)))

    summary = summarize(args.label, list(results))
    print_summary(summary)
    out = EVAL_DIR / "results" / f"{args.label}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": asdict(summary), "runs": [asdict(r) for r in results]}, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
