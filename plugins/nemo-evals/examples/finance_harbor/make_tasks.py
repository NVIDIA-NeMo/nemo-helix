#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn the finance Gym rows into a Harbor dataset: one task per shard of rows, scored in the task's sandbox.

The input is the rows ``../finance_gym/convert_rows.py`` writes, one file per capability, so the Gym and Harbor
routes answer the same requests. Each shard of ``--shard-size`` rows becomes one task directory:

- ``instruction.md``: the rows as a ``nemo.dataset-shard/v1`` document for ``DatasetRunnerAgent``. Each row's
  Responses API request becomes the chat-completions request Gym's ``inference_provider`` sends for it.
- ``environment/``: the metric image, identical in every task, so the dataset has one image build.
- ``tests/``: ``test.sh`` and the shard's ``references.jsonl``, which Harbor uploads only when it verifies.
- ``task.toml``: no network, and no ``[verifier.environment]``, so the verifier runs in the same sandbox.

    python3 make_tasks.py rows/visual_ocr.jsonl rows/reasoning.jsonl rows/code_generation.jsonl \\
      --shard-size 100 --output finance-harbor
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

SHARD_SCHEMA = "nemo.dataset-shard/v1"
HERE = Path(__file__).resolve().parent
#: Every file the metric image needs: the wrapper here, and the Gym example's scorer, unchanged.
METRIC_FILES = (
    HERE / "metric" / "Dockerfile",
    HERE / "metric" / "score.py",
    HERE.parent / "finance_gym" / "environment" / "resources_servers" / "finance_qa" / "cookbook.py",
    HERE.parent / "finance_gym" / "environment" / "resources_servers" / "finance_qa" / "scoring.py",
)
#: Harbor reads ``instruction.md`` up to 1 MiB; leave room for the JSON envelope.
MAX_INSTRUCTION_BYTES = 1024 * 1024 - 4096

TEST_SH = """\
#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
python3 /opt/metric/score.py --responses /logs/artifacts/responses.jsonl \\
  --references /tests/references.jsonl --out /logs/verifier
"""

TASK_TOML = """\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

version = "1.0"

[task]
name = "{name}"

[metadata]
category = "dataset-eval"
tags = ["finance", "{capability}"]
rows = {rows}

[agent]
timeout_sec = {agent_timeout_sec}

[verifier]
timeout_sec = 600.0

[environment]
network_mode = "no-network"
cpus = 1
memory_mb = 1024
storage_mb = 2048
"""


def chat_content(content: Any) -> Any:
    """A Responses API input message's content as chat-completions content, as Gym's converter maps it."""
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for part in content:
        kind = part.get("type")
        if kind == "input_text":
            parts.append({"type": "text", "text": part["text"]})
        elif kind == "input_image":
            url = part["image_url"]["url"] if isinstance(part["image_url"], dict) else part["image_url"]
            parts.append({"type": "image_url", "image_url": {"url": url, "detail": part.get("detail", "auto")}})
        else:
            raise ValueError(f"unsupported content part type {kind!r}")
    return parts


def shard_row(gym_row: dict[str, Any]) -> dict[str, Any]:
    """One Gym row as a dataset-runner row: the chat request Gym's ``inference_provider`` sends."""
    request = gym_row["responses_create_params"]
    params: dict[str, Any] = {}
    if request.get("temperature") is not None:
        params["temperature"] = request["temperature"]
    if request.get("max_output_tokens") is not None:
        params["max_tokens"] = request["max_output_tokens"]
    messages = [{"role": message["role"], "content": chat_content(message["content"])} for message in request["input"]]
    return {"id": gym_row["row_id"], "messages": messages, "params": params}


def reference(gym_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": gym_row["row_id"],
        "expected_answer": gym_row["expected_answer"],
        "task": gym_row["task"],
        "capability": gym_row["capability"],
    }


def write_task(
    task_dir: Path, *, name: str, capability: str, rows: list[dict[str, Any]], agent_timeout_sec: float
) -> None:
    instruction = json.dumps({"schema": SHARD_SCHEMA, "rows": [shard_row(row) for row in rows]}, ensure_ascii=False)
    if len(instruction.encode()) > MAX_INSTRUCTION_BYTES:
        raise ValueError(f"{name}: {len(rows)} rows don't fit in one instruction; lower --shard-size")
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")
    (task_dir / "task.toml").write_text(
        TASK_TOML.format(name=name, capability=capability, rows=len(rows), agent_timeout_sec=agent_timeout_sec),
        encoding="utf-8",
    )
    for file in METRIC_FILES:
        shutil.copy2(file, task_dir / "environment" / file.name)
    test_sh = task_dir / "tests" / "test.sh"
    test_sh.write_text(TEST_SH, encoding="utf-8")
    test_sh.chmod(0o755)
    with (task_dir / "tests" / "references.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(reference(row), ensure_ascii=False) + "\n")


def make_tasks(
    sources: list[Path], output: Path, *, shard_size: int, prefix: str, agent_timeout_sec: float
) -> list[Path]:
    """Write one task per shard of each source's rows; shards never mix capabilities."""
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for source in sources:
        with source.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        capabilities = {row["capability"] for row in rows}
        if len(capabilities) != 1:
            raise ValueError(f"{source}: expected one capability per file, found {sorted(capabilities)}")
        (capability,) = capabilities
        for index, start in enumerate(range(0, len(rows), shard_size)):
            name = f"{prefix}-{capability.replace('_', '-')}-{index:04d}"
            task_dir = output / name
            write_task(
                task_dir,
                name=f"nemo/{name}",
                capability=capability,
                rows=rows[start : start + shard_size],
                agent_timeout_sec=agent_timeout_sec,
            )
            written.append(task_dir)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", type=Path, nargs="+", help="Gym rows files from convert_rows.py, one per capability")
    parser.add_argument("--output", type=Path, required=True, help="The Harbor dataset directory to write")
    parser.add_argument("--shard-size", type=int, default=100, help="Rows per task")
    parser.add_argument("--prefix", default="finance", help="Task name prefix")
    parser.add_argument(
        "--agent-timeout-sec", type=float, default=3600.0, help="Limit on answering one shard, all rows included"
    )
    args = parser.parse_args(argv)
    if args.shard_size < 1:
        sys.exit("--shard-size must be at least 1")
    try:
        written = make_tasks(
            args.sources,
            args.output,
            shard_size=args.shard_size,
            prefix=args.prefix,
            agent_timeout_sec=args.agent_timeout_sec,
        )
    except ValueError as error:
        sys.exit(str(error))
    print(f"{len(written)} tasks -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
