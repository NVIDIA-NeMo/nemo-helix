# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalized spec for an agent optimization run.

Every agent-optimize job takes the same inputs: an agent entity, a bundle
holding a plugin-specific config, and the name for the optimized agent it
produces.  ``strategy`` lives only on the router's spec, because a strategy
job already knows which strategy it is.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath

from nemo_platform_plugin.refs import ENTITY_REF_PATTERN
from pydantic import BaseModel, Field, model_validator


def is_fileset_relative(value: str) -> bool:
    """True when *value* stays inside a fileset root once joined to it.

    Checked in both host flavours: the submitting client may be on Windows
    while the task host is Linux.  ``~`` and bare drive letters are rejected
    because they anchor to client-side state the task host does not have.
    """
    if value.startswith("~"):
        return False
    flavours = (PurePosixPath(value), PureWindowsPath(value))
    return not any(path.is_absolute() or ".." in path.parts or path.drive for path in flavours)


class AgentOptimizeSpec(BaseModel):
    """What every agent-optimize job consumes."""

    agent: str = Field(
        min_length=1,
        description="Platform agent to optimize: 'name' or 'workspace/name'.",
    )
    config_fileset: str = Field(
        min_length=1,
        description="Fileset holding the optimization bundle: the config named by "
        "'config' plus every asset it references. Stage one with "
        "`nemo agents optimize prepare-fileset`.",
    )
    config: str = Field(
        min_length=1,
        description="Path to the strategy configuration YAML, relative to the fileset root.",
    )
    output_agent: str = Field(
        min_length=1,
        description="Name for the new, optimized agent entity this run creates.",
    )
    workspace: str = Field(default="default", description="Workspace for every entity in the run.")

    @model_validator(mode="after")
    def _validate(self) -> AgentOptimizeSpec:
        if not re.match(ENTITY_REF_PATTERN, self.config_fileset):
            raise ValueError(
                f"config_fileset must be 'name' or 'workspace/name'; got {self.config_fileset!r}."
            )
        if not is_fileset_relative(self.config):
            raise ValueError(
                "config must be a path relative to the fileset root (no leading '/', no '..' "
                f"segments); got {self.config!r}."
            )
        return self


class OptimizeSpec(AgentOptimizeSpec):
    """Router spec: the normalized fields plus strategy selection."""

    strategy: str = Field(min_length=1, description="Installed optimization strategy, e.g. 'nat'.")


class OptimizeSubmitSpec(BaseModel):
    """Client-submitted router fields (workspace is added server-side)."""

    strategy: str = Field(min_length=1, description="Installed optimization strategy, e.g. 'nat'.")
    agent: str = Field(min_length=1, description="Platform agent to optimize.")
    config_fileset: str = Field(min_length=1, description="Fileset holding the optimization bundle.")
    config: str = Field(min_length=1, description="Config YAML path, relative to the fileset root.")
    output_agent: str = Field(min_length=1, description="Name for the new optimized agent entity.")
