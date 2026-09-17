# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated: ``FabricContainerRuntime`` is now ``FabricAgentRuntime(config, sandbox=provider)``.

Kept importable for one release so existing callers keep working; construction warns.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

from nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime import FabricAgentRuntime
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill
from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.base import SandboxProvider
from nemo_evaluator_sdk.values.common import SecretRef

_DEPRECATION_MSG = (
    "FabricContainerRuntime is deprecated and will be removed in a future release; "
    "use FabricAgentRuntime(config, sandbox=provider, image=..., secrets=..., skills=...) instead."
)


class FabricContainerRuntime(FabricAgentRuntime):
    """Deprecated alias for :class:`FabricAgentRuntime` in sandbox mode."""

    def __init__(
        self,
        config: Mapping[str, Any] | Any,
        *,
        provider: SandboxProvider,
        secrets: Mapping[str, SecretRef] | None = None,
        image: str | None = None,
        skills: Sequence[AgentSkill] | None = None,
    ) -> None:
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        super().__init__(config, sandbox=provider, image=image, secrets=secrets, skills=skills)
