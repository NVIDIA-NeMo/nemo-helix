# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spec for the ``eval-author.first-eval`` job."""

from __future__ import annotations

from typing import Self

from nemo_helix_plugin.refs import ENTITY_REF_PATTERN, FilesetRef
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FirstEvalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str = Field(
        min_length=1,
        pattern=ENTITY_REF_PATTERN,
        description="Platform agent whose first evals are authored ('name' or 'workspace/name').  "
        "The stored agent is read, never modified.",
    )
    agent_fileset: FilesetRef | None = Field(
        default=None,
        pattern=ENTITY_REF_PATTERN,
        description="Fileset staged into the author's workspace as the agent's repository (ETHOS.md, docs).  "
        "Defaults to the agent's '<name>-ethos' fileset when it exists.",
    )
    output: FilesetRef | None = Field(
        default=None,
        pattern=ENTITY_REF_PATTERN,
        description="Fileset receiving .eval-author/** and ETHOS.md; created when missing.  "
        "Defaults to '<name>-evals' in the agent's workspace.",
    )
    author_config: str | None = Field(
        default=None,
        description="Optional overrides for the author agent: a partial agent.yaml (nemo-agents-spec-v1) "
        "whose fields replace the bundled defaults, as a path relative to the root of author_config_fileset.",
    )
    author_config_fileset: FilesetRef | None = Field(
        default=None,
        pattern=ENTITY_REF_PATTERN,
        description="Fileset holding author_config ('name' or 'workspace/name').",
    )
    workspace: str = Field(
        default="default",
        description="Workspace the agent and filesets are resolved in when their references name none.",
    )

    @model_validator(mode="after")
    def _config_and_fileset_together(self) -> Self:
        # Without a fileset, resolve_staged_config reads the config as a path on the task host,
        # so a bare --author-config would only fail at run time with a file-not-found.
        if (self.author_config is None) != (self.author_config_fileset is None):
            raise ValueError(
                "author_config and author_config_fileset must be given together: stage the config in a "
                "fileset (`nemo files upload <config.yaml> <fileset>`) and pass both."
            )
        return self
