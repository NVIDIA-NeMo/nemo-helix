# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spec for ``nemo agents optimize run-strategy``.

These are the router's fields: ``strategy`` picks the job that does the work,
and the rest is the shape every agent optimization run needs — an agent under
test, a bundle holding the strategy's own config, and somewhere to publish the
artifacts.

Deliberately thin on validation: the router forwards everything but
``strategy`` to the selected strategy job's own ``spec_schema``, which is the
thing that actually knows whether a given config location or output target
makes sense for it.  Re-stating those rules here would mean two schemas to
keep in step and a router with opinions about a strategy it has never seen.
"""

from __future__ import annotations

from nemo_helix_plugin.refs import FilesetRef, OutputTarget
from pydantic import BaseModel, Field


class RunStrategySubmitSpec(BaseModel):
    """Client-submitted fields; ``workspace`` is added server-side."""

    strategy: str = Field(
        min_length=1,
        description="Installed optimization strategy to run, e.g. 'nat'. "
        "List what this platform has with `nemo agents optimize list-strategies`.",
    )
    optimize_config: str = Field(
        min_length=1,
        description="Location of the strategy's configuration YAML.  With optimize_config_fileset "
        "set — required for remote submission — this is a path relative to the fileset root.  "
        "Without it (programmatic local runs only) it is an absolute path on the host running the job.",
    )
    optimize_config_fileset: FilesetRef | None = Field(
        default=None,
        description="Fileset holding the optimization bundle: the config named by optimize_config "
        "plus every asset it references.  Stage one with `nemo agents optimize prepare-fileset`.  "
        "Required for remote submissions, where the job has no access to the client's filesystem.",
    )
    agent: str | None = Field(
        default=None,
        min_length=1,
        description="Platform agent to optimize ('name' or 'workspace/name'). Strategies that can "
        "take the agent under test from the config itself may leave this unset.",
    )
    output: OutputTarget | None = Field(
        default=None,
        description="Where the strategy should publish its artifacts once the run succeeds — either "
        "a local directory (path-shaped: starts with '/', './', '../', '~/') or a NeMo Helix "
        "fileset reference ('name' or 'workspace/name').  Filesets are created on demand if missing.",
    )


class RunStrategySpec(RunStrategySubmitSpec):
    """Canonical stored spec — the submitted fields plus the resolved workspace."""

    workspace: str = Field(
        default="default",
        description="Workspace for the agent, the bundle fileset, and any entity the strategy resolves.",
    )
