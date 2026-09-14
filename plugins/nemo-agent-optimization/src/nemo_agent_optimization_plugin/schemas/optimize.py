# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strategy-agnostic spec for an Agents optimization run (``nemo agents optimize``).

This is a fresh, minimal spec for the ``nemo-agent-optimization`` plugin — it is not shared
with ``nemo_optimization.schemas.optimize``, which stays HPO-shaped and untouched. ``strategy``
is a plain ``str`` rather than an enum because strategies are discovered at runtime through the
``nemo.optimization-strategy`` entry-point group (see ``strategies.py``); the set of valid values
is not known statically at import time.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath

from nemo_platform_plugin.refs import ENTITY_REF_PATTERN
from pydantic import BaseModel, Field, model_validator

FILESET_REQUIRED = (
    "optimize_config_fileset is required when submitting an optimize run remotely: the job runs "
    "on the platform and cannot read the submitting client's filesystem.  Stage the bundle first "
    "with `nemo agents optimize prepare-fileset --source <dir> --optimize-config <file> "
    "--fileset <name>`, then launch with the fileset ref it prints.  (Absolute-path configs remain "
    "available for co-located programmatic local runs.)"
)


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


class OptimizeSubmitSpec(BaseModel):
    """Client-submitted fields for ``nemo agents optimize`` (workspace added server-side)."""

    strategy: str = Field(
        min_length=1,
        description="Installed nemo.optimization-strategy name, e.g. 'nat' or 'prompt-master'.",
    )
    optimize_config: str = Field(
        min_length=1,
        description="Location of the strategy configuration YAML. With optimize_config_fileset "
        "set — required for remote submission — this is a path relative to the fileset root. "
        "Without it (programmatic local runs only) it is an absolute path on the host running "
        "the job.",
    )
    optimize_config_fileset: str | None = Field(
        default=None,
        description="Fileset holding the optimization bundle: the config named by optimize_config "
        "plus every asset it references. Stage one with `nemo agents optimize prepare-fileset`. "
        "Required for remote submissions, where the job has no access to the client's filesystem.",
    )
    agent: str | None = Field(
        default=None,
        min_length=1,
        description="Agent source: a platform reference ('name' or 'workspace/name') or a local "
        "nemo-agents-spec-v1 agent.yaml path. Whether it is required is decided by the selected "
        "strategy, which rejects the omission from its own validate_config.",
    )
    output: str | None = Field(
        default=None,
        description="Where to publish the optimization artifacts once the run succeeds — either a "
        "local directory or a NeMo Platform fileset reference ('name' or 'workspace/name'). "
        "Classification of which is handled by this plugin's own job, not by this spec.",
    )


class OptimizeSpec(OptimizeSubmitSpec):
    """Server-side spec — adds ``workspace``."""

    workspace: str = Field(default="default", description="Workspace used to fetch a platform agent.")

    @model_validator(mode="after")
    def _validate(self) -> OptimizeSpec:
        if self.optimize_config_fileset is None:
            return self
        if not re.match(ENTITY_REF_PATTERN, self.optimize_config_fileset):
            raise ValueError(
                f"optimize_config_fileset must be 'name' or 'workspace/name'; got {self.optimize_config_fileset!r}."
            )
        if not is_fileset_relative(self.optimize_config):
            raise ValueError(
                "optimize_config must be a path relative to the fileset root (no leading '/', no '..' "
                f"segments) when optimize_config_fileset is set; got {self.optimize_config!r}."
            )
        return self
