#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate a comparison report for Harbor CLI eval batch runs.

Reads result.json files from a batch directory and produces a markdown report
comparing standard vs easy eval variants on success rate and wallclock time.

Usage:
    python scripts/generate_eval_report.py <batch_dir>
    python scripts/generate_eval_report.py jobs/batch-2026-02-12__10-00-00

Options:
    --format md|csv|json    Output format (default: md)
    --output PATH           Write report to file (default: stdout)
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class TrialResult:
    eval_name: str
    task_name: str
    reward: float | None = None
    started_at: str = ""
    finished_at: str = ""
    agent_start: str = ""
    agent_end: str = ""
    agent_duration_sec: float = 0.0
    total_duration_sec: float = 0.0
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    n_cache_tokens: int = 0
    n_tool_calls: int = 0
    exception: str | None = None
    job_dir: str = ""


@dataclass
class EvalPairComparison:
    base_name: str
    standard: TrialResult | None = None
    easy: TrialResult | None = None


def parse_duration(start_str: str, end_str: str) -> float:
    """Parse ISO timestamps and return duration in seconds."""
    if not start_str or not end_str:
        return 0.0
    # Handle both Z and non-Z formats
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            start = datetime.strptime(start_str, fmt)
            break
        except ValueError:
            continue
    else:
        return 0.0

    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            end = datetime.strptime(end_str, fmt)
            break
        except ValueError:
            continue
    else:
        return 0.0

    return (end - start).total_seconds()


def count_tool_calls(result_file: Path) -> int:
    """Count tool calls from trajectory.json or JSONL session logs."""
    trial_dir = result_file.parent
    agent_dir = trial_dir / "agent"

    # Try trajectory.json first (ATIF format)
    trajectory_file = agent_dir / "trajectory.json"
    if trajectory_file.exists():
        try:
            data = json.loads(trajectory_file.read_text())
            return sum(len(step.get("tool_calls", [])) for step in data.get("steps", []))
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback: count tool_use blocks in JSONL session logs
    sessions_dir = agent_dir / "sessions" / "projects"
    if not sessions_dir.exists():
        return 0
    count = 0
    for jsonl_file in sessions_dir.rglob("*.jsonl"):
        try:
            for line in jsonl_file.read_text().splitlines():
                msg = json.loads(line)
                if msg.get("type") == "assistant":
                    for block in msg.get("message", {}).get("content", []):
                        if block.get("type") == "tool_use":
                            count += 1
        except (json.JSONDecodeError, OSError):
            continue
    return count


def find_trial_results(batch_dir: Path) -> list[TrialResult]:
    """Find and parse all trial result.json files in the batch directory.

    Handles both structures:
      - Single job:  <dir>/<trial_name>/result.json
      - Batch run:   <dir>/<job_name>/<trial_name>/result.json
    """
    results = []

    # Recursively find all result.json files, then filter to trial-level ones
    for result_file in sorted(batch_dir.rglob("result.json")):
        try:
            data = json.loads(result_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        # Must have task_name to be a trial result (not the job-level result)
        if "task_name" not in data:
            continue

        task_name = data["task_name"]

        # Extract reward
        reward = None
        verifier = data.get("verifier_result", {})
        if verifier and "rewards" in verifier:
            reward = verifier["rewards"].get("reward")

        # Extract timing
        agent_exec = data.get("agent_execution", {})
        agent_start = agent_exec.get("started_at", "")
        agent_end = agent_exec.get("finished_at", "")
        agent_duration = parse_duration(agent_start, agent_end)

        total_start = data.get("started_at", "")
        total_end = data.get("finished_at", "")
        total_duration = parse_duration(total_start, total_end)

        # Token usage
        agent_result = data.get("agent_result", {})

        # Exception info
        exception = None
        exc_info = data.get("exception_info")
        if exc_info:
            exception = str(exc_info)

        results.append(
            TrialResult(
                eval_name=task_name,
                task_name=task_name,
                reward=reward,
                started_at=total_start,
                finished_at=total_end,
                agent_start=agent_start,
                agent_end=agent_end,
                agent_duration_sec=agent_duration,
                total_duration_sec=total_duration,
                n_input_tokens=agent_result.get("n_input_tokens") or 0,
                n_output_tokens=agent_result.get("n_output_tokens") or 0,
                n_cache_tokens=agent_result.get("n_cache_tokens") or 0,
                n_tool_calls=count_tool_calls(result_file),
                exception=exception,
                job_dir=str(result_file.parent.parent),
            )
        )

    return results


def match_pairs(results: list[TrialResult]) -> list[EvalPairComparison]:
    """Match standard and easy eval variants into pairs."""
    # Index by task_name
    by_task: dict[str, TrialResult] = {}
    for r in results:
        by_task[r.task_name] = r

    # Known base names
    base_names = [
        "auditor-config-crud-cli",
        "auditor-target-crud-cli",
        "auth-authorization-cli",
        "data-designer-config-cli",
        "entities-basic-cli",
        "files-crud-cli",
        "files-upload-dataset-cli",
        "guardrails-content-safety-cli",
        "inference-chat-completions-cli",
        "inference-provider-reg-cli",
        "secrets-crud-cli",
        "workspace-basic-cli",
    ]

    pairs = []
    for base in base_names:
        pair = EvalPairComparison(base_name=base)
        if base in by_task:
            pair.standard = by_task[base]
        if f"{base}-easy" in by_task:
            pair.easy = by_task[f"{base}-easy"]
        if pair.standard or pair.easy:
            pairs.append(pair)

    return pairs


def fmt_duration(seconds: float) -> str:
    """Format duration as mm:ss."""
    if seconds <= 0:
        return "-"
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes}:{secs:02d}"


def fmt_reward(reward: float | None) -> str:
    if reward is None:
        return "ERROR"
    return "PASS" if reward >= 1.0 else "FAIL"


def fmt_tokens(n: int | None) -> str:
    if not n:
        return "-"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def generate_markdown_report(pairs: list[EvalPairComparison], batch_dir: Path) -> str:
    """Generate a markdown comparison report."""
    lines = []

    # Header
    lines.append("# Harbor CLI Eval Comparison: Standard vs Easy")
    lines.append("")
    lines.append(f"**Batch directory:** `{batch_dir}`")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("**Model:** `aws/anthropic/bedrock-claude-sonnet-4-5-v1`")
    lines.append("**Agent:** `claude-code`")
    lines.append("")

    # Summary stats
    std_pass = sum(1 for p in pairs if p.standard and p.standard.reward is not None and p.standard.reward >= 1.0)
    std_total = sum(1 for p in pairs if p.standard)
    easy_pass = sum(1 for p in pairs if p.easy and p.easy.reward is not None and p.easy.reward >= 1.0)
    easy_total = sum(1 for p in pairs if p.easy)

    lines.append("## Summary")
    lines.append("")
    lines.append("| Variant  | Pass | Total | Rate |")
    lines.append("|----------|------|-------|------|")
    lines.append(f"| Standard | {std_pass} | {std_total} | {std_pass / max(std_total, 1) * 100:.0f}% |")
    lines.append(f"| Easy     | {easy_pass} | {easy_total} | {easy_pass / max(easy_total, 1) * 100:.0f}% |")
    lines.append("")

    # Timing summary
    std_times = [p.standard.agent_duration_sec for p in pairs if p.standard and p.standard.agent_duration_sec > 0]
    easy_times = [p.easy.agent_duration_sec for p in pairs if p.easy and p.easy.agent_duration_sec > 0]

    if std_times or easy_times:
        lines.append("### Timing (Agent Execution)")
        lines.append("")
        lines.append("| Variant  | Mean   | Median | Min    | Max    |")
        lines.append("|----------|--------|--------|--------|--------|")
        if std_times:
            import statistics

            lines.append(
                f"| Standard | {fmt_duration(statistics.mean(std_times))} | {fmt_duration(statistics.median(std_times))} | {fmt_duration(min(std_times))} | {fmt_duration(max(std_times))} |"
            )
        if easy_times:
            import statistics

            lines.append(
                f"| Easy     | {fmt_duration(statistics.mean(easy_times))} | {fmt_duration(statistics.median(easy_times))} | {fmt_duration(min(easy_times))} | {fmt_duration(max(easy_times))} |"
            )
        lines.append("")

    # Detailed pair comparison
    lines.append("## Pair-by-Pair Comparison")
    lines.append("")
    lines.append(
        "| Eval | Std Result | Std Time | # Std Tokens | # Std Tools | Easy Result | Easy Time | # Easy Tokens | # Easy Tools | Time Diff | # Token Diff | # Tool Diff |"
    )
    lines.append(
        "|------|-----------|----------|------------|-----------|-------------|-----------|-------------|------------|-----------|------------|-----------|"
    )

    for pair in pairs:
        name = pair.base_name.replace("-cli", "")

        std_result = fmt_reward(pair.standard.reward) if pair.standard else "N/A"
        std_time = fmt_duration(pair.standard.agent_duration_sec) if pair.standard else "-"
        std_tokens = fmt_tokens(pair.standard.n_input_tokens) if pair.standard else "-"
        std_tools = str(pair.standard.n_tool_calls) if pair.standard and pair.standard.n_tool_calls else "-"

        easy_result = fmt_reward(pair.easy.reward) if pair.easy else "N/A"
        easy_time = fmt_duration(pair.easy.agent_duration_sec) if pair.easy else "-"
        easy_tokens = fmt_tokens(pair.easy.n_input_tokens) if pair.easy else "-"
        easy_tools = str(pair.easy.n_tool_calls) if pair.easy and pair.easy.n_tool_calls else "-"

        # Calculate time difference
        time_diff = "-"
        if pair.standard and pair.easy and pair.standard.agent_duration_sec > 0 and pair.easy.agent_duration_sec > 0:
            diff = pair.standard.agent_duration_sec - pair.easy.agent_duration_sec
            pct = diff / pair.standard.agent_duration_sec * 100
            sign = "+" if diff < 0 else "-"
            time_diff = f"{sign}{abs(pct):.0f}%"

        # Calculate token difference
        token_diff = "-"
        if pair.standard and pair.easy and pair.standard.n_input_tokens > 0 and pair.easy.n_input_tokens > 0:
            tdiff = pair.standard.n_input_tokens - pair.easy.n_input_tokens
            tpct = tdiff / pair.standard.n_input_tokens * 100
            tsign = "+" if tdiff < 0 else "-"
            token_diff = f"{tsign}{abs(tpct):.0f}%"

        # Calculate tool call difference
        tool_diff = "-"
        if pair.standard and pair.easy and pair.standard.n_tool_calls > 0 and pair.easy.n_tool_calls > 0:
            tc_diff = pair.standard.n_tool_calls - pair.easy.n_tool_calls
            tc_pct = tc_diff / pair.standard.n_tool_calls * 100
            tc_sign = "+" if tc_diff < 0 else "-"
            tool_diff = f"{tc_sign}{abs(tc_pct):.0f}%"

        lines.append(
            f"| {name} | {std_result} | {std_time} | {std_tokens} | {std_tools} | {easy_result} | {easy_time} | {easy_tokens} | {easy_tools} | {time_diff} | {token_diff} | {tool_diff} |"
        )

    lines.append("")

    # Errors section
    errors = []
    for pair in pairs:
        if pair.standard and pair.standard.exception:
            errors.append((pair.base_name, "standard", pair.standard.exception))
        if pair.easy and pair.easy.exception:
            errors.append((f"{pair.base_name}-easy", "easy", pair.easy.exception))

    if errors:
        lines.append("## Errors")
        lines.append("")
        for eval_name, variant, exc in errors:
            lines.append(f"### {eval_name} ({variant})")
            lines.append("```")
            lines.append(exc[:500])
            lines.append("```")
            lines.append("")

    # Outcome matrix (visual)
    lines.append("## Outcome Matrix")
    lines.append("")
    lines.append("```")
    lines.append(f"{'Eval':<40} {'Standard':>10} {'Easy':>10}")
    lines.append("-" * 62)
    for pair in pairs:
        name = pair.base_name
        std_icon = {None: "  ERR", 1.0: " PASS", 0.0: " FAIL"}.get(
            pair.standard.reward if pair.standard else None, "  N/A"
        )
        easy_icon = {None: "  ERR", 1.0: " PASS", 0.0: " FAIL"}.get(pair.easy.reward if pair.easy else None, "  N/A")
        # Handle non-1.0 non-0.0 rewards
        if pair.standard and pair.standard.reward is not None and pair.standard.reward not in (0.0, 1.0):
            std_icon = f" {pair.standard.reward:.1f}"
        if pair.easy and pair.easy.reward is not None and pair.easy.reward not in (0.0, 1.0):
            easy_icon = f" {pair.easy.reward:.1f}"

        lines.append(f"{name:<40} {std_icon:>10} {easy_icon:>10}")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def generate_csv_report(pairs: list[EvalPairComparison]) -> str:
    """Generate a CSV comparison report."""
    lines = [
        "eval_name,variant,result,reward,agent_duration_sec,total_duration_sec,input_tokens,cache_tokens,output_tokens,tool_calls,exception"
    ]
    for pair in pairs:
        for variant, trial in [("standard", pair.standard), ("easy", pair.easy)]:
            if trial:
                lines.append(
                    f"{pair.base_name},{variant},{fmt_reward(trial.reward)},"
                    f"{trial.reward},{trial.agent_duration_sec:.1f},"
                    f"{trial.total_duration_sec:.1f},{trial.n_input_tokens},"
                    f"{trial.n_cache_tokens},{trial.n_output_tokens},"
                    f"{trial.n_tool_calls},"
                    f"{trial.exception or ''}"
                )
    return "\n".join(lines)


def generate_json_report(pairs: list[EvalPairComparison]) -> str:
    """Generate a JSON comparison report."""
    data = []
    for pair in pairs:
        entry = {"base_name": pair.base_name}
        for variant, trial in [("standard", pair.standard), ("easy", pair.easy)]:
            if trial:
                entry[variant] = {
                    "result": fmt_reward(trial.reward),
                    "reward": trial.reward,
                    "agent_duration_sec": round(trial.agent_duration_sec, 1),
                    "total_duration_sec": round(trial.total_duration_sec, 1),
                    "input_tokens": trial.n_input_tokens,
                    "cache_tokens": trial.n_cache_tokens,
                    "output_tokens": trial.n_output_tokens,
                    "tool_calls": trial.n_tool_calls,
                    "exception": trial.exception,
                }
        data.append(entry)
    return json.dumps(data, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate Harbor eval comparison report")
    parser.add_argument("batch_dir", type=Path, help="Path to batch results directory")
    parser.add_argument("--format", choices=["md", "csv", "json"], default="md", help="Output format")
    parser.add_argument("--output", type=Path, help="Write report to file (default: stdout)")
    args = parser.parse_args()

    if not args.batch_dir.exists():
        print(f"ERROR: Directory not found: {args.batch_dir}", file=sys.stderr)
        sys.exit(1)

    # Find all trial results
    results = find_trial_results(args.batch_dir)
    if not results:
        print(f"ERROR: No trial results found in {args.batch_dir}", file=sys.stderr)
        print("  Expected structure: <batch_dir>/<job_name>/<trial_name>/result.json", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(results)} trial results", file=sys.stderr)

    # Match into pairs
    pairs = match_pairs(results)
    print(f"Matched {len(pairs)} eval pairs", file=sys.stderr)

    # Generate report
    if args.format == "md":
        report = generate_markdown_report(pairs, args.batch_dir)
    elif args.format == "csv":
        report = generate_csv_report(pairs)
    else:
        report = generate_json_report(pairs)

    # Output
    if args.output:
        args.output.write_text(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
