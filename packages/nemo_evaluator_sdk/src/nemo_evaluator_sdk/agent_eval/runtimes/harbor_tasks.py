# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK-owned local Harbor tasks and their repeatable collection."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import overload

from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from pydantic import Field, field_validator

# Task metadata recording where a discovered Harbor task lives on the local filesystem.
HARBOR_TASK_DIR_KEY = "harbor_task_dir"
HARBOR_DATASET_PATH_KEY = "harbor_dataset_path"


class HarborAgentEvalTask(AgentEvalTask):
    """Executable Harbor task whose source directory is authoritative for publication."""

    source_dir: Path = Field(frozen=True)

    @field_validator("source_dir")
    @classmethod
    def absolute_source(cls, value: Path) -> Path:
        """Make paths absolute without hiding symlinks."""
        return value.absolute()


class HarborTaskCollection(Sequence[HarborAgentEvalTask]):
    """Fixed membership with configurable scoring on each task; slices retain the type."""

    def __init__(self, tasks: Iterable[HarborAgentEvalTask]) -> None:
        self._tasks = tuple(tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[HarborAgentEvalTask]:
        return iter(self._tasks)

    @overload
    def __getitem__(self, index: int) -> HarborAgentEvalTask: ...

    @overload
    def __getitem__(self, index: slice) -> HarborTaskCollection: ...

    def __getitem__(self, index: int | slice) -> HarborAgentEvalTask | HarborTaskCollection:
        if isinstance(index, slice):
            return HarborTaskCollection(self._tasks[index])
        return self._tasks[index]
