#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build Gym rollout rows for the CLEVR-CoGenT resources server.

The splits and the answer extraction match NeMo-RL ``CLEVRCoGenTDataset``
(``nemo_rl/data/datasets/response_datasets/clevr.py``). The prompt text is copied
from ``examples/prompts/clevr_cogent_cot.txt`` and
``examples/prompts/clevr_cogent_cot_nemotron_omni.txt`` at NeMo-RL
``a659df9ba91cc50d09c523aeef25de34c82d2b1d``.

A positive ``--train-size`` or ``--validation-size`` keeps the first N rows of that
split, then shuffles them with ``--seed``. That avoids downloading the rest of the
70k training set for a smoke. ``-1`` loads the whole split.

    uv run --no-project --with datasets --with pillow \
      scripts/grpo-examples/prepare_clevr_cogent.py \
      --out-dir /tmp/clevr-data --train-size 32 --validation-size 16

Then upload the directory as a FileSet with ``purpose=dataset``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gym_rows import image_message, pil_to_data_uri, write_rows  # noqa: E402

# Config key under responses_api_agents in configs/clevr_cogent.yaml.
AGENT_REF = {"type": "responses_api_agents", "name": "clevr_cogent_simple_agent"}

TRAIN_DATASET = "MMInstruction/Clevr_CoGenT_TrainA_70K_Complex"
VALIDATION_DATASET = "MMInstruction/Clevr_CoGenT_ValA"

# examples/prompts/clevr_cogent_cot.txt
QWEN_PROMPT = (
    "Think step-by-step to solve the following problem, and answer in the following format: "
    "<think> step-by-step thought process </think> <answer> final answer </answer>\n"
    "Note that your final answer must only contain a single numerical output (e.g. 2, 12, 45) for "
    'numerical solutions, and only the strings "yes" or "no" for yes/no type questions.\n'
    "\n"
    "Let's think step-by-step: \n"
    "{}"
)

# examples/prompts/clevr_cogent_cot_nemotron_omni.txt
NEMOTRON_OMNI_PROMPT = (
    "Solve the following problem step by step. When you are done thinking, provide your final "
    "answer wrapped in <answer> </answer> tags.\n"
    "Your final answer must only contain a single numerical output (e.g. 2, 12, 45) for numerical "
    'solutions, and only the strings "yes" or "no" for yes/no type questions.\n'
    "\n"
    "{}"
)

PROMPTS = {"qwen": QWEN_PROMPT, "nemotron-omni": NEMOTRON_OMNI_PROMPT}


def format_answer_fromtags(answer: str) -> str:
    """Extract the text between ``<answer>`` tags. Otherwise return the stripped string."""
    match = re.search(r"<answer>(.*?)</answer>", answer)
    return match.group(1).strip() if match else answer.strip()


def _load_rows(repo: str, size: int) -> list[dict[str, Any]]:
    from datasets import Dataset, load_dataset

    spec = "train" if size < 0 else f"train[:{size}]"
    dataset = load_dataset(repo, split=spec)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"{repo} split {spec} is {type(dataset).__name__}, expected a Dataset")
    return list(dataset)


def _rollout_row(example: dict[str, Any], prompt: str) -> dict[str, Any]:
    problem = str(example["problem"])
    return {
        "responses_create_params": {
            "input": [image_message(prompt.format(problem), [pil_to_data_uri(example["image"])])]
        },
        "question": problem,
        "expected_answer": format_answer_fromtags(str(example["solution"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--train-size",
        type=int,
        default=-1,
        help="First N training rows, then shuffled. -1 loads the whole train split.",
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=-1,
        help="First N validation rows, then shuffled. -1 loads the whole valA split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", choices=tuple(PROMPTS), default="qwen")
    args = parser.parse_args()
    if args.train_size < -1:
        parser.error("--train-size must be -1 (all rows) or a non-negative count")
    if args.validation_size < -1:
        parser.error("--validation-size must be -1 (all rows) or a non-negative count")

    prompt = PROMPTS[args.prompt]
    rng = random.Random(args.seed)
    train = [_rollout_row(example, prompt) for example in _load_rows(TRAIN_DATASET, args.train_size)]
    validation = [_rollout_row(example, prompt) for example in _load_rows(VALIDATION_DATASET, args.validation_size)]
    rng.shuffle(train)
    rng.shuffle(validation)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_train = write_rows(args.out_dir / "training.jsonl", train, AGENT_REF)
    n_val = write_rows(args.out_dir / "validation.jsonl", validation, AGENT_REF)
    print(
        json.dumps(
            {
                "training_jsonl": str(args.out_dir / "training.jsonl"),
                "training_rows": n_train,
                "validation_jsonl": str(args.out_dir / "validation.jsonl"),
                "validation_rows": n_val,
                "prompt": args.prompt,
                "agent_ref": AGENT_REF["name"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
