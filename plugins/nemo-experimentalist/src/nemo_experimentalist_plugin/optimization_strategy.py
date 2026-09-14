# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``experimentalist`` implementation of ``nemo agents optimize --strategy``.

Registered so the strategy name resolves and appears in
``nemo agents optimization-strategies list``. The Experimentalist's
harness-improvement loop (``nemo_experimentalist_plugin.experimentalist.runner``)
is not wired up yet: its config shape — an Insight reference versus a
Harbor-compatible evaluation config — is still an open design question, and
guessing it here would bake in the wrong contract.
"""

from __future__ import annotations

from typing import Any, ClassVar

from nemo_platform import NeMoPlatform
from nemo_platform_plugin.job_context import JobContext


class ExperimentalistOptimizationStrategy:
    """Reserves the ``experimentalist`` strategy name; execution is not implemented yet."""

    name: ClassVar[str] = "experimentalist"

    def validate_config(self, config: dict[str, Any], *, agent: str | None) -> None:
        del config, agent

    def run(
        self,
        *,
        agent_config: dict[str, Any] | None,
        source_agent_config: dict[str, Any] | None = None,
        config: dict[str, Any],
        ctx: JobContext,
        workspace: str,
        sdk: NeMoPlatform | None = None,
    ) -> dict[str, Any]:
        del agent_config, source_agent_config, config, ctx, workspace, sdk
        raise NotImplementedError(
            "The experimentalist optimization strategy is not implemented yet. "
            "Run the Experimentalist directly with `nemo agents experimentalist` instead."
        )
