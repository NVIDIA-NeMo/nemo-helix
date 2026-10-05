# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spec for the ``prompt-master`` optimization strategy.

The router (``nemo agents optimize run-strategy``) forwards every submitted field but
``strategy`` to this schema, so what is required and what each field means is decided
here, not there.  The shared fields keep the router's names so they arrive through its
``--optimize-config`` / ``--optimize-config-fileset`` / ``--agent`` / ``--output`` flags.
"""

from __future__ import annotations

from typing import Self

from nemo_helix_plugin.refs import ENTITY_REF_PATTERN, FilesetRef, OutputTarget
from pydantic import BaseModel, Field, model_validator


class PromptMasterOptimizeSpec(BaseModel):
    agent: str = Field(
        min_length=1,
        pattern=ENTITY_REF_PATTERN,
        description="Platform agent whose system prompt is optimized ('name' or 'workspace/name').  "
        "The stored agent is read, never modified.",
    )
    optimize_config: str | None = Field(
        default=None,
        description="Optional overrides for the optimizer agent: a partial agent.yaml (nemo-agents-spec-v1) "
        "whose fields replace the bundled defaults, as a path relative to the root of optimize_config_fileset.",
    )
    optimize_config_fileset: FilesetRef | None = Field(
        default=None,
        pattern=ENTITY_REF_PATTERN,
        description="Fileset holding optimize_config ('name' or 'workspace/name').  The job runs on the "
        "platform and cannot read the submitting client's filesystem.",
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
    def _config_and_fileset_together(self) -> Self:
        if (self.optimize_config is None) != (self.optimize_config_fileset is None):
            raise ValueError(
                "optimize_config and optimize_config_fileset must be given together: stage the config in a "
                "fileset (`nemo files upload <config.yaml> <fileset>`) and pass both."
            )
        return self
