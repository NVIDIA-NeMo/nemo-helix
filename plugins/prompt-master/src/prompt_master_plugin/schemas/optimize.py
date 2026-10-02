# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spec for the ``prompt-master`` optimization strategy.

The router (``nemo agents optimize run-strategy``) forwards every submitted field but
``strategy`` to this schema, so what is required and what each field means is decided
here, not there.  The shared fields keep the router's names so they arrive through its
``--optimize-config`` / ``--optimize-config-fileset`` / ``--agent`` / ``--output`` flags.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath

from nemo_helix_plugin.refs import ENTITY_REF_PATTERN, FilesetRef, OutputTarget
from pydantic import BaseModel, Field, ValidationInfo, model_validator

FILESET_REQUIRED = (
    "optimize_config_fileset is required when submitting a prompt-master run: the job runs on "
    "the platform and cannot read the submitting client's filesystem.  Stage the config first "
    "(`nemo files filesets create <name>`, then `nemo files upload <config.yaml> <name>`) and "
    "pass the fileset as `--optimize-config-fileset <workspace>/<name>`.  (Absolute-path configs "
    "remain available for co-located programmatic local runs.)"
)


class PromptMasterOptimizeSpec(BaseModel):
    """Canonical spec -- the shape ``compile`` and ``run`` see."""

    optimize_config: str = Field(
        min_length=1,
        description="Location of the Prompt Master config YAML (the optimizer model, an optional "
        "prompt_override, timeout_seconds).  With optimize_config_fileset set -- required for remote "
        "submission -- this is a path relative to the fileset root.  Without it (programmatic local "
        "runs only) it is an absolute path on the host running the job.",
    )
    optimize_config_fileset: FilesetRef | None = Field(
        default=None,
        description="Fileset holding the config named by optimize_config.  Required for remote "
        "submissions, where the job has no access to the client's filesystem.",
    )
    agent: str = Field(
        min_length=1,
        pattern=ENTITY_REF_PATTERN,
        description="Platform agent whose system prompt is optimized ('name' or 'workspace/name').  "
        "The stored agent is read, never modified.",
    )
    output: OutputTarget | None = Field(
        default=None,
        description="Where to publish the optimized agent config and the run summary, in addition to "
        "the job's own results -- either a local directory (path-shaped: starts with '/', './', "
        "'../', '~/') or a NeMo Helix fileset reference ('name' or 'workspace/name').  Filesets are "
        "created on demand if missing.",
    )
    workspace: str = Field(
        default="default",
        description="Workspace the agent and the config fileset are resolved in when their references name none.",
    )

    @model_validator(mode="after")
    def _validate_config_location(self) -> PromptMasterOptimizeSpec:
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


class PromptMasterOptimizeSubmitSpec(PromptMasterOptimizeSpec):
    """Submitter-facing spec: a remote submission must stage its config in a fileset.

    The router validates the forwarded fields against this schema without a context, so a
    remote submission naming no fileset is refused at submit time with :data:`FILESET_REQUIRED`
    instead of failing inside the task.  The local scheduler validates with
    ``context={"is_local": True}`` and may use an absolute host path instead.
    """

    @model_validator(mode="after")
    def _require_remote_fileset(self, info: ValidationInfo) -> PromptMasterOptimizeSubmitSpec:
        is_local = bool(info.context and info.context.get("is_local"))
        if not is_local and self.optimize_config_fileset is None:
            raise ValueError(FILESET_REQUIRED)
        return self


def is_fileset_relative(config_path: str) -> bool:
    """True when *config_path* stays inside a fileset root once joined to it.

    Checked in both host flavours -- the submitting client may be on Windows while the task
    host is Linux, so a POSIX-only check would let ``C:\\bundle\\prompt-master.yaml`` through.
    ``~`` and bare drive letters are rejected too: they are anchored to the client host, not
    to the fileset root.  (The same rule the ``legacy`` strategy applies; copied rather than
    imported because a strategy plugin must not depend on a sibling strategy, and the router
    deliberately holds no opinion on config locations.)
    """
    if config_path.startswith("~"):
        return False
    flavours = (PurePosixPath(config_path), PureWindowsPath(config_path))
    return not any(path.is_absolute() or ".." in path.parts or path.drive for path in flavours)
