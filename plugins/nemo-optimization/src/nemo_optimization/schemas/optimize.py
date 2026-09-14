# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical optimize study spec."""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from nemo_platform_plugin.refs import ENTITY_REF_PATTERN, FilesetRef, OutputTarget
from pydantic import BaseModel, Field, ValidationInfo, model_validator

FILESET_REQUIRED = (
    "optimize_config_fileset is required when submitting an optimize study: the job runs on the "
    "platform and cannot read the submitting client's filesystem.  Stage the bundle first with "
    "`nemo agents optimize prepare-fileset --source <dir> --optimize-config <file> --fileset <name>`, "
    "then launch with the fileset ref it prints.  (Absolute-path configs remain available for "
    "co-located programmatic local runs.)"
)


class OptimizeSpec(BaseModel):
    """Spec for an Agents optimize study (``nemo agents optimize``)."""

    optimize_config: str = Field(
        min_length=1,
        description="Location of the Fabric-native optimization YAML.  With optimize_config_fileset "
        "set — required for remote submission — this is a path relative to the fileset root.  Without it "
        "(programmatic local runs only) it is an absolute path on the host running the job.",
    )
    optimize_config_fileset: FilesetRef | None = Field(
        default=None,
        description="Fileset holding the optimization bundle: the config named by optimize_config plus "
        "every asset it references (Agent under Test package, dataset, eval.fabric.base_dir tree, "
        "hooks and MCP configs).  Stage one with `nemo agents optimize prepare-fileset`. Required "
        "for remote submissions, where the job has no access to the client's filesystem.",
    )
    workspace: str = Field(
        default="default",
        description="Workspace used to fetch a platform agent and for VirtualModel preflight.",
    )
    agent: str | None = Field(
        default=None,
        min_length=1,
        description="Optional platform agent reference ('name' or 'workspace/name'). "
        "When omitted, the optimization config must include an inline Fabric agent package.",
    )
    output: OutputTarget | None = Field(
        default=None,
        description="Where to publish the study artifacts (optimized config, trials dataframe, "
        "pareto plots, ATIF evidence) once the study succeeds — either a local directory "
        "(path-shaped: starts with '/', './', '../', '~/') or a NeMo Platform fileset "
        "reference ('name' or 'workspace/name').  Filesets are created on demand if missing.  "
        "This is in addition to the per-job artifacts that ``ctx.results.save`` always "
        "registers; it gives remote clients a stable, addressable location to read from.",
    )

    @model_validator(mode="after")
    def _validate_config_location(self) -> "OptimizeSpec":
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


class OptimizeSubmitSpec(OptimizeSpec):
    """Submitter-facing optimize spec for remote platform submissions."""

    optimize_config_fileset: FilesetRef | None = Field(
        default=None,
        description="Fileset holding the optimization bundle: the config named by optimize_config plus "
        "every asset it references (Agent under Test package, dataset, eval.fabric.base_dir tree, "
        "hooks and MCP configs).  Stage one with `nemo agents optimize prepare-fileset`. Required "
        "for remote submissions, where the job has no access to the client's filesystem. "
        "Programmatic local runs may omit it and use an absolute host path for optimize_config.",
    )

    @model_validator(mode="before")
    @classmethod
    def _allow_local_missing_fileset(cls, data: Any, info: ValidationInfo) -> Any:
        if isinstance(data, dict) and _is_local(info) and "optimize_config_fileset" not in data:
            return {**data, "optimize_config_fileset": None}
        return data

    @model_validator(mode="after")
    def _require_remote_fileset(self, info: ValidationInfo) -> "OptimizeSubmitSpec":
        if not _is_local(info) and self.optimize_config_fileset is None:
            raise ValueError(FILESET_REQUIRED)
        return self


def is_fileset_relative(config_path: str) -> bool:
    """True when *config_path* stays inside a fileset root once joined to it.

    Checked in both host flavours — the submitting client may be on Windows while
    the task host is Linux, so a POSIX-only check would let ``C:\\bundle\\optimize.yaml``
    through, and a POSIX-only ``..`` scan would miss ``..\\escape.yaml``.  ``~`` is rejected
    too: it is not absolute to ``PurePath``, but it expands to a client home directory
    that does not exist on the task host.  A bare drive letter (``D:optimize.yml``) is
    rejected too: ``PureWindowsPath`` treats it as drive-relative rather than absolute, but
    it is still anchored to a drive's current directory on the client host, not the fileset
    root.
    """
    if config_path.startswith("~"):
        return False
    flavours = (PurePosixPath(config_path), PureWindowsPath(config_path))
    return not any(path.is_absolute() or ".." in path.parts or path.drive for path in flavours)


def _is_local(info: ValidationInfo) -> bool:
    return bool(info.context and info.context.get("is_local"))
