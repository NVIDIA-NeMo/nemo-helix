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

    Checked in both host flavours — the submitting client may be on Windows while the task host
    is Linux, so a POSIX-only check would let ``C:\\bundle\\optimize.yaml`` through, and a
    POSIX-only ``..`` scan would miss ``..\\escape.yaml``. ``~`` is rejected too: it is not
    absolute to ``PurePath``, but it expands to a client home directory that does not exist on
    the task host. A bare drive letter (``D:optimize.yml``) is rejected too: ``PureWindowsPath``
    treats it as drive-relative rather than absolute, but it is still anchored to a drive's
    current directory on the client host, not the fileset root.
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
    optimize_config_fileset: str = Field(
        min_length=1,
        description="Fileset holding the optimization bundle: the config named by "
        "'optimize_config' plus every asset it references. Stage one with "
        "`nemo agents optimize prepare-fileset`.",
    )
    optimize_config: str = Field(
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
        if not re.match(ENTITY_REF_PATTERN, self.optimize_config_fileset):
            raise ValueError(
                f"optimize_config_fileset must be 'name' or 'workspace/name'; got {self.optimize_config_fileset!r}."
            )
        if not is_fileset_relative(self.optimize_config):
            raise ValueError(
                "optimize_config must be a path relative to the fileset root (no leading '/', no '..' "
                f"segments); got {self.optimize_config!r}."
            )
        return self


class OptimizeSpec(AgentOptimizeSpec):
    """Router spec: the normalized fields plus strategy selection."""

    strategy: str = Field(min_length=1, description="Installed optimization strategy, e.g. 'nat'.")


class OptimizeSubmitSpec(BaseModel):
    """Client-submitted router fields (workspace is added server-side)."""

    strategy: str = Field(min_length=1, description="Installed optimization strategy, e.g. 'nat'.")
    agent: str = Field(min_length=1, description="Platform agent to optimize.")
    optimize_config_fileset: str = Field(min_length=1, description="Fileset holding the optimization bundle.")
    optimize_config: str = Field(min_length=1, description="Config YAML path, relative to the fileset root.")
    output_agent: str = Field(min_length=1, description="Name for the new optimized agent entity.")
