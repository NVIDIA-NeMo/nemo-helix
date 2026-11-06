# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the victim from a pre-built image instead of a local ``docker build``.

The image is built elsewhere from the context ``agent-hardener stage`` writes
(:func:`~nemo_agent_hardener_plugin.jobs.execution.stage_context`), then handed to every command that creates
the sandbox. agent-hardener folds the ref into its ``--reuse`` fingerprint, so ``up`` and the ``run --reuse``
that follows must receive the identical ref.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VictimImage:
    """A pre-built victim image ref and whether a staged-context mismatch only warns."""

    ref: str
    allow_context_drift: bool = False

    @classmethod
    def from_spec(cls, config: Mapping[str, Any]) -> VictimImage | None:
        """Read ``victim_image`` / ``allow_context_drift`` from a validated job spec; ``None`` builds locally."""
        ref = config.get("victim_image")
        if not ref:
            return None
        return cls(str(ref), bool(config.get("allow_context_drift")))

    def args(self) -> list[str]:
        """The agent-hardener flags selecting this image."""
        args = ["--victim-image", self.ref]
        if self.allow_context_drift:
            args.append("--allow-context-drift")
        return args


def victim_image_args(image: VictimImage | None) -> list[str]:
    """The agent-hardener flags for *image*, or none to keep the local docker build."""
    if image is None:
        return []
    return image.args()
