# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ray actor that owns a :class:`SandboxedGymSession` (raw Gym results, no RL postprocess)."""

from __future__ import annotations

from typing import Any

import ray

from sandboxed_gym.orchestrator import (
    SandboxedGymOrchestrator,
    SandboxedGymSession,
    install_job_sandbox_reaper,
    install_termination_cleanup,
)
from sandboxed_gym.serve_config import SandboxedGymServeConfig

GYM_ACTOR_FQN = "sandboxed_gym.ray.gym_actor.SandboxedGymActor"


# Deliberately no max_restarts, as on SandboxEpisodeBrokerActor: the host handle lives only in this
# process, so a restarted actor provisions a second sandbox and leaks the first until its ttl_s. A
# crash should fail the job. Restarts need label-based reconciliation of the job id on the spec.
@ray.remote
class SandboxedGymActor:
    """Trusted Ray proxy: broker + Gym host; returns raw ``/rollouts/run`` results."""

    def __init__(self, cfg: SandboxedGymServeConfig | dict[str, Any]) -> None:
        self._cfg = cfg if isinstance(cfg, SandboxedGymServeConfig) else SandboxedGymServeConfig.model_validate(cfg)
        self._session: SandboxedGymSession | None = None

    def spinup(self) -> dict[str, Any]:
        # Before start(), so a cancel during create still reaps this job's sandboxes.
        # start() arms the same hook; the second call is a no-op.
        install_job_sandbox_reaper(
            self._cfg.job_id,
            host_provider=self._cfg.host_provider,
            host_provider_options=self._cfg.sandbox.host_provider_options,
        )
        self._session = SandboxedGymOrchestrator().start(self._cfg)
        # After start(), so a spinup that failed leaves nothing registered to destroy.
        install_termination_cleanup(self.shutdown)
        return self._session.descriptor().model_dump(mode="json")

    def run_rollouts(self, examples: list[dict[str, Any]]) -> list[Any]:
        if self._session is None:
            raise RuntimeError("SandboxedGymActor.spinup has not completed")
        return self._session.run_rollouts(examples)

    def shutdown(self) -> None:
        if self._session is not None:
            self._session.shutdown()
            self._session = None
