# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic on-disk layout for a single agent-eval task run.

A run produces an agent-log dir and a workspace dir under a run dir, plus a
written instruction file. Callers that need extra directories (e.g. preserved
platform state) add them on top of :class:`RunLayout`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    """Filesystem layout for one task run."""

    run_dir: Path
    agent_log_dir: Path
    workspace_dir: Path
    instruction_path: Path


def resolve_run_dir(output_dir: str | Path | None, default_factory: Callable[[], Path]) -> Path:
    """Resolve the run dir to an absolute path.

    An explicit ``output_dir`` must be made absolute: run-dir subpaths are used as
    Docker bind-mount sources, and Docker treats a relative ``-v`` source as a
    (slash-free) named volume rather than a host directory.
    """
    if output_dir is not None:
        return Path(output_dir).resolve()
    return default_factory()


def prepare_run_layout(
    run_dir: str | Path,
    instruction_text: str,
    *,
    agent_subdir: str = "agent",
    workspace_subdir: str = "workspace",
    instruction_name: str = "instruction.md",
) -> RunLayout:
    """Create the agent/workspace dirs under ``run_dir`` and write the instruction."""
    run_dir = Path(run_dir)
    agent_log_dir = run_dir / agent_subdir
    workspace_dir = run_dir / workspace_subdir
    agent_log_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    instruction_path = agent_log_dir / instruction_name
    instruction_path.write_text(instruction_text, encoding="utf-8")

    return RunLayout(
        run_dir=run_dir,
        agent_log_dir=agent_log_dir,
        workspace_dir=workspace_dir,
        instruction_path=instruction_path,
    )
