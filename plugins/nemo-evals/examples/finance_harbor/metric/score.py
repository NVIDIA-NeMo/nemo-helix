#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score one shard's responses with the finance cookbook, as the task's verifier.

Runs inside the task's sandbox. It reads the rows the dataset runner answered (``responses.jsonl``) and the shard's
references, scores every answered row with ``scoring.score`` (the same function the Gym ``finance_qa`` server calls),
and writes three files to the verifier directory:

- ``reward.json``: the mean of each component over the scored rows; ``reward`` is ``task_correct``.
- ``row_scores.jsonl``: one line per reference row, with its components, or why it has none.
- ``summary.json``: each component's sum and count, so a report can pool shards into exact means.

A row the runner couldn't answer, or never answered, is not scored, as the Gym environment doesn't score a failed
rollout; it still counts in ``summary.json``'s ``rows``, so coverage can be reported.

The Gym route reads a chat completion through Gym's ``inference_provider`` model server, which keeps the message's
``content`` and, without a reasoning parser, drops ``reasoning_content``. :func:`gym_response` rebuilds that same
response, so both routes score identical input.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# The Gym example's scorer; make_tasks.py copies it next to this file in the image.
import scoring  # ty: ignore[unresolved-import]

#: Components every scored row has; ``program_syntax_valid`` is added for code tasks.
COMPONENTS = (*scoring.COMPONENTS, "program_syntax_valid")


def gym_response(completion: Mapping[str, Any]) -> dict[str, Any]:
    """The Responses API response Gym's ``inference_provider`` builds from ``completion``, as far as scoring reads it."""
    choices = completion.get("choices") or [{}]
    message = choices[0].get("message") or {}
    usage = completion.get("usage") or {}
    return {
        "output": [{"type": "message", "content": [{"type": "output_text", "text": message.get("content") or ""}]}],
        "usage": {"output_tokens": usage.get("completion_tokens") or 0},
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_shard(
    responses: list[dict[str, Any]], references: list[dict[str, Any]], *, tolerance: float = 0.05
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    """``reward.json``, the ``row_scores.jsonl`` lines and ``summary.json`` for one shard."""
    by_id = {record["id"]: record for record in responses}
    rows: list[dict[str, Any]] = []
    sums: dict[str, float] = dict.fromkeys(COMPONENTS, 0.0)
    counts: dict[str, int] = dict.fromkeys(COMPONENTS, 0)
    for reference in references:
        record = by_id.get(reference["id"])
        row: dict[str, Any] = {
            "id": reference["id"],
            "task": reference["task"],
            "capability": reference["capability"],
        }
        if record is None:
            row.update(status="missing", scores=None, error="the runner wrote no response for this row")
        elif record["status"] != "ok":
            row.update(status=record["status"], scores=None, error=record.get("error"))
        else:
            components = scoring.score(
                task=reference["task"],
                expected_answer=reference["expected_answer"],
                response=gym_response(record["response"]),
                tolerance=tolerance,
            )
            row.update(status="scored", scores=components, error=None)
            for name, value in components.items():
                sums[name] += value
                counts[name] += 1
        rows.append(row)

    rewards = {name: sums[name] / counts[name] for name in COMPONENTS if counts[name]}
    rewards["reward"] = rewards.get("task_correct", 0.0)
    summary = {
        "rows": len(references),
        "scored": sum(1 for row in rows if row["status"] == "scored"),
        "sums": {name: sums[name] for name in COMPONENTS if counts[name]},
        "counts": {name: counts[name] for name in COMPONENTS if counts[name]},
    }
    return rewards, rows, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses", type=Path, default=Path("/logs/artifacts/responses.jsonl"))
    parser.add_argument("--references", type=Path, default=Path("/tests/references.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("/logs/verifier"))
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args(argv)

    responses = read_jsonl(args.responses) if args.responses.is_file() else []
    rewards, rows, summary = score_shard(responses, read_jsonl(args.references), tolerance=args.tolerance)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "row_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Written last: Harbor reads the trial's scores from it, so it only appears once the other files are complete.
    (args.out / "reward.json").write_text(json.dumps(rewards), encoding="utf-8")


if __name__ == "__main__":
    main()
