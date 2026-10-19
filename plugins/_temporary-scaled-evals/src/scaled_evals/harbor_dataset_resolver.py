# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve Harbor dataset members with the selected Harbor virtualenv.

This file is intentionally executable with ``/opt/harbor/<version>/.venv/bin/python``.
Keep Harbor imports inside the async entry point: the scaled-evals application
environment does not install Harbor directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tomllib
from pathlib import Path
from typing import Any


def _docker_images(value: Any) -> set[str]:
    images: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "docker_image" and isinstance(child, str) and child.strip():
                images.add(child.strip())
            else:
                images.update(_docker_images(child))
    elif isinstance(value, list):
        for child in value:
            images.update(_docker_images(child))
    return images


async def resolve(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from harbor.models.job.config import DatasetConfig

    try:
        from harbor.job_plan import JobPlan  # ty: ignore[unresolved-import]

        cache_tasks = JobPlan.cache_tasks
    except ImportError:
        from harbor.job import Job

        cache_tasks = Job._cache_tasks

    configs = [DatasetConfig.model_validate(item) for item in datasets]
    task_configs = []
    for dataset in configs:
        task_configs.extend(await dataset.get_task_configs())
    downloads = await cache_tasks(task_configs)

    members: list[dict[str, Any]] = []
    for config in task_configs:
        task_id = config.get_task_id()
        result = downloads.get(task_id)
        task_dir = (result.path if result is not None else config.get_local_path()).resolve()
        task_toml = task_dir / "task.toml"
        payload = tomllib.loads(task_toml.read_text())
        images = sorted(_docker_images(payload))
        if not images:
            raise ValueError(f"Harbor dataset task has no docker_image: {task_dir}")
        members.append(
            {
                "name": task_id.get_name(),
                "path": str(task_dir),
                "images": images,
                "content_hash": result.content_hash if result is not None else None,
                "resolved_git_commit_id": (result.resolved_git_commit_id if result is not None else None),
                "source": config.source,
                "ref": config.ref,
            }
        )
    return members


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    members = asyncio.run(resolve(payload["datasets"]))
    args.output.write_text(json.dumps({"members": members}, sort_keys=True))


if __name__ == "__main__":
    main()
