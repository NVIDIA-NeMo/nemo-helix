# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spec for the ``strands-harness-optimizer`` optimization strategy.

The router (``nemo agents optimize run-strategy``) forwards every submitted field but
``strategy`` here, so this schema decides what is required.  The shared fields keep the
router's names so they arrive through its ``--agent`` / ``--optimize-config`` /
``--optimize-config-fileset`` flags; anything else, including the router's ``--output``, is
refused: the artifacts only ever land in the job's own results.
"""

from __future__ import annotations

from nemo_helix_plugin.refs import ENTITY_REF_PATTERN, FilesetRef
from pydantic import BaseModel, ConfigDict, Field


class StrandsHarnessOptimizeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str = Field(
        min_length=1,
        pattern=ENTITY_REF_PATTERN,
        description="Platform agent whose system prompt is optimized ('name' or 'workspace/name').  "
        "The stored agent is read, never modified.",
    )
    optimize_config: str = Field(
        min_length=1,
        description="Strategy config YAML (dataset, epochs, batch size, ...) as a path relative to the root of "
        "optimize_config_fileset.",
    )
    optimize_config_fileset: FilesetRef = Field(
        pattern=ENTITY_REF_PATTERN,
        description="Fileset holding optimize_config and the dataset it names ('name' or 'workspace/name').  "
        "The job runs on the platform and cannot read the submitting client's filesystem.",
    )
    workspace: str = Field(
        default="default",
        description="Workspace the agent and the bundle fileset are resolved in when their references name none.",
    )
