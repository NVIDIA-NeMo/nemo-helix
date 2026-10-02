# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example plugin controller — registered under ``nemo.controllers``.

Demonstrates the :class:`~nemo_helix_plugin.controller.NemoController` pattern:
implement :meth:`list_objects` and :meth:`reconcile_one`, use
:meth:`on_startup` to load configuration via :class:`~nemo_helix_plugin.config.NemoConfig`,
and let the platform manage the loop lifecycle.

The platform runner instantiates this class, calls :meth:`on_startup` once,
then calls :meth:`reconcile` (which calls :meth:`list_objects` + :meth:`reconcile_one`)
every :attr:`interval_seconds`.  Graceful shutdown invokes :meth:`on_shutdown`
after the final reconcile cycle completes.
"""

from __future__ import annotations

import logging

from nemo_helix_plugin.controller import NemoController

logger = logging.getLogger(__name__)

# A trivial in-memory "object store" the controller reconciles.
_OBJECTS: list[str] = ["alpha", "beta", "gamma"]


class ExampleController(NemoController):
    """A reference controller showing config loading and reconcile loop patterns.

    On startup, loads :class:`~nemo_example_plugin.config.ExampleConfig` and
    logs the active configuration values.  This demonstrates that config is
    shared naturally across plugin components (service + controller) without
    any explicit wiring — both call ``ExampleConfig.get()`` independently and
    receive the same cached singleton.

    In a real plugin, replace :attr:`_OBJECTS` and :meth:`reconcile_one` with
    entity client calls and state-machine transitions.
    """

    name = "example-controller"

    @property
    def interval_seconds(self) -> float:
        # Short interval for demo purposes; real controllers derive this from config.
        return 30.0

    async def on_startup(self) -> None:
        from nemo_example_plugin.config import ExampleConfig

        config = ExampleConfig.get()
        logger.info(
            "ExampleController starting up — greeting_style=%r  log_requests=%r",
            config.greeting_style,
            config.log_requests,
        )

    async def on_shutdown(self) -> None:
        logger.info("ExampleController shutting down.")

    async def list_objects(self) -> list:
        """Return the set of objects to reconcile this cycle."""
        return list(_OBJECTS)

    async def reconcile_one(self, obj: object) -> None:
        """Reconcile a single object — here we just log it."""
        logger.debug("ExampleController reconciling object: %s", obj)
