# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ray actor wrapping :class:`sandboxed_gym.broker.EpisodeBrokerServer`."""

from __future__ import annotations

import logging
from typing import Any

import ray

from sandboxed_gym.broker import EpisodeBrokerServer
from sandboxed_gym.config import BrokerEndpoint, EpisodeBrokerConfig

LOGGER = logging.getLogger(__name__)

BROKER_ACTOR_FQN = "sandboxed_gym.ray.broker_actor.SandboxEpisodeBrokerActor"


# Deliberately no max_restarts: a silently restarted broker would lose the handle map.
@ray.remote
class SandboxEpisodeBrokerActor:
    """Ray placement wrapper around the in-process episode broker server."""

    def __init__(self, config: EpisodeBrokerConfig | dict[str, Any]) -> None:
        self._server = EpisodeBrokerServer(config)

    def start(self) -> BrokerEndpoint:
        return self._server.start()

    def get_endpoint(self) -> BrokerEndpoint:
        return self._server.get_endpoint()

    def shutdown(self) -> None:
        self._server.shutdown()


def start_episode_broker(
    config: EpisodeBrokerConfig | dict[str, Any],
    *,
    node_id: str | None = None,
    extra_ray_options: dict[str, Any] | None = None,
) -> tuple[Any, BrokerEndpoint]:
    """Create and start the broker Ray actor; return ``(handle, endpoint)``."""
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    options: dict[str, Any] = {}
    if node_id is not None:
        options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
    if extra_ray_options:
        options.update(extra_ray_options)

    actor = SandboxEpisodeBrokerActor.options(**options).remote(config)
    endpoint = ray.get(actor.start.remote())
    return actor, endpoint


class RayEpisodeBroker:
    """Runs the broker in its own Ray actor, for :meth:`SandboxedGymOrchestrator.start`.

    Worth choosing when a job's environments open episodes often: the broker serves every episode
    ``exec``, and hosting it here keeps that traffic off the calling process's GIL.

    Ray supplies placement and lifecycle only. The HTTP server still runs on a background thread
    inside the actor, so requests never serialize through Ray actor dispatch -- which is what
    makes this a placement decision rather than a throughput one.
    """

    def __init__(
        self,
        config: EpisodeBrokerConfig | dict[str, Any],
        *,
        node_id: str | None = None,
        extra_ray_options: dict[str, Any] | None = None,
    ) -> None:
        self._config = config
        self._node_id = node_id
        self._extra_ray_options = extra_ray_options
        self._actor: Any | None = None

    def start(self) -> BrokerEndpoint:
        self._actor, endpoint = start_episode_broker(
            self._config, node_id=self._node_id, extra_ray_options=self._extra_ray_options
        )
        return endpoint

    def shutdown(self) -> None:
        """Stop the actor. Logged rather than raised: a caller is already tearing down."""
        if self._actor is None:
            return
        try:
            ray.get(self._actor.shutdown.remote())
        except Exception:
            LOGGER.exception("Failed to shut down the episode broker actor")
        finally:
            self._actor = None
