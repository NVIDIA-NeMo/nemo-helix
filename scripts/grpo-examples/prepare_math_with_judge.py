#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a Gym JSONL dataset for the `math_with_judge` resources server.

Wraps the prep scripts Gym already ships (`prepare_dapo17k.py`, `prepare_aime24.py`) and adds
the one thing they omit: an ``agent_ref`` on every row. Gym infers the agent from the config's
``datasets:`` block when run locally, but the platform path has no such config -- NeMo-RL reads
``row["agent_ref"]["name"]`` directly off each row.

Both sources are public HuggingFace datasets, so nothing here needs the internal GitLab
dataset registry.

    uv run --no-project --with datasets scripts/grpo-examples/prepare_math_with_judge.py --out-dir /tmp/mwj-dataset

Then upload the directory as a FileSet with ``purpose=dataset``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterator

# The `agent_ref.name` is the *config key* under `responses_api_agents:` in
# math_with_judge.yaml, not the server directory name (`simple_agent`).
AGENT_REF = {"type": "responses_api_agents", "name": "math_with_judge_simple_agent"}

SYSTEM_PROMPT = (
    "Your task is to solve a math problem.  Make sure to put the answer (and only the answer) inside \\boxed{}."
)


def _dapo17k_rows() -> Iterator[dict[str, Any]]:
    """DAPO-Math-17k, the train split NeMo-RL's `grpo-dapomath17k-*` recipes use."""
    from datasets import load_dataset

    for example in load_dataset("YouJiacheng/DAPO-Math-17k-dedup", split="train"):
        yield {
            "responses_create_params": {"input": example["prompt"]},
            "question": example["prompt"][0]["content"],
            "expected_answer": example["reward_model"]["ground_truth"],
        }


def _aime24_rows() -> Iterator[dict[str, Any]]:
    """AIME 2024: 30 problems. Repeat at eval time for mean@k."""
    from datasets import load_dataset

    for example in load_dataset("HuggingFaceH4/aime_2024", split="train"):
        yield {
            "responses_create_params": {
                "input": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ]
            },
            "question": example["problem"],
            "expected_answer": str(example["answer"]),
        }


def _write(path: Path, rows: list[dict[str, Any]], repeats: int) -> int:
    """Write rows with agent_ref and a stable task_idx, repeating each `repeats` times.

    task_idx is not read by the resources server, but Gym sorts rollouts by it and the
    platform's dataset validators expect an int, so number the emitted rows rather than
    the source rows.
    """
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            for _ in range(repeats):
                handle.write(json.dumps({**row, "task_idx": written, "agent_ref": AGENT_REF}) + "\n")
                written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--train-size",
        type=int,
        default=-1,
        help="Cap on DAPO17k rows. -1 means all (~17k); 0 writes none. Subsample for a shorter run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-source",
        choices=("holdout", "aime24"),
        default="holdout",
        help=(
            "holdout: a slice of DAPO17k excluded from training -- same distribution, so the "
            "base model scores in a measurable range and uplift is visible. aime24: 30 "
            "competition problems, a publishable benchmark but near-zero for small models."
        ),
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=200,
        help="Held-out prompts when --validation-source=holdout.",
    )
    parser.add_argument(
        "--val-repeats",
        type=int,
        default=1,
        help=(
            "Repeat each validation prompt this many times, e.g. 32 for AIME24 mean@32 during "
            "training. For `gym eval`, leave at 1 and pass +num_repeats=32 instead."
        ),
    )
    args = parser.parse_args()

    # Guard the numeric caps: -1 means "all" and anything below it is meaningless, while a
    # bare `> 0` test would silently treat 0 and negatives as unlimited and write every row.
    if args.train_size < -1:
        parser.error("--train-size must be -1 (all rows), 0 (none), or a positive count")
    if args.validation_size < 0:
        parser.error("--validation-size must be 0 or a positive count")
    if args.val_repeats < 1:
        parser.error("--val-repeats must be at least 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(_dapo17k_rows())
    random.Random(args.seed).shuffle(rows)

    if args.validation_source == "holdout":
        # Sliced off before the train cap so a held-out prompt can never also be trained on.
        val_rows, rows = rows[: args.validation_size], rows[args.validation_size :]
    else:
        val_rows = list(_aime24_rows())

    train_rows = rows if args.train_size == -1 else rows[: args.train_size]

    n_train = _write(args.out_dir / "training.jsonl", train_rows, repeats=1)
    n_val = _write(args.out_dir / "validation.jsonl", val_rows, repeats=args.val_repeats)

    print(
        json.dumps(
            {
                "training_jsonl": str(args.out_dir / "training.jsonl"),
                "training_rows": n_train,
                "validation_jsonl": str(args.out_dir / "validation.jsonl"),
                "validation_rows": n_val,
                "validation_source": args.validation_source,
                "val_repeats": args.val_repeats,
                "agent_ref": AGENT_REF["name"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
