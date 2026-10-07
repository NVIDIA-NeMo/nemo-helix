# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spec for the ``prompt-master`` optimization strategy.

The router (``nemo agents optimize run-strategy``) forwards every submitted field but
``strategy`` to this schema, so what is required and what each field means is decided
here, not there.  The shared fields keep the router's names so they arrive through its
``--optimize-config`` / ``--optimize-config-fileset`` / ``--agent`` flags.  Anything else
the router forwards -- its ``--output``, ``--spec`` extras -- is refused rather than
silently dropped: the artifacts only ever land in the job's own results.
"""

from __future__ import annotations

from typing import Self

from nemo_helix_plugin.refs import ENTITY_REF_PATTERN, FilesetRef
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PromptMasterOptimizeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    workspace: str = Field(
        default="default",
        description="Workspace the agent and the config fileset are resolved in when their references name none.",
    )

    @model_validator(mode="after")
    def _config_and_fileset_together(self) -> Self:
        # Without a fileset, resolve_staged_config reads the config as a path on the task host,
        # so a bare --optimize-config would only fail at run time with a file-not-found.
        if (self.optimize_config is None) != (self.optimize_config_fileset is None):
            raise ValueError(
                "optimize_config and optimize_config_fileset must be given together: stage the config in a "
                "fileset (`nemo files upload <config.yaml> <fileset>`) and pass both."
            )
        return self
