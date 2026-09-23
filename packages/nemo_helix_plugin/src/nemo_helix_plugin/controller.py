# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin controller interface — what plugin authors implement for reconcile loops.

Plugin authors subclass :class:`NemoController` and register a class under the
``nemo.controllers`` entry-point group.  The platform wraps each class in a
:class:`NemoControllerAdapter` at startup so it runs as a first-class singleton
background controller alongside the built-in models/entities/jobs controllers.

Example::

    # my_plugin/controller.py
    from nemo_helix_plugin.controller import NemoController

    class MyController(NemoController):
        name = "my-controller"
        dependencies = ["entities"]

        async def list_objects(self) -> list:
            return await self._entity_client.list(MyEntity, workspace="-")

        async def reconcile_one(self, obj: object) -> None:
            ...  # drive state machine for a single object

    # pyproject.toml:
    # [project.entry-points."nemo.controllers"]
    # my-controller = "my_plugin.controller:MyController"

Lifecycle
---------

The framework manages the reconcile loop:

1. Declared ``dependencies`` are polled via ``/status`` until each service is ready.
2. ``on_startup()`` is called once before the first reconcile cycle.
3. ``reconcile()`` is called repeatedly at ``interval_seconds`` intervals.
   The default implementation calls :meth:`list_objects` once per cycle and
   :meth:`reconcile_one` for each result with per-item error isolation.
4. On SIGINT/SIGTERM the platform sets a stop signal, waits for the current
   ``reconcile()`` call to complete, then calls ``on_shutdown()``.
   Controllers that override :meth:`reconcile` may call :meth:`stop_requested`
   between phases or per-item iterations to exit early during shutdown.

Plugin authors do **not** manage signals or threads — implement cleanup in
:meth:`on_shutdown` and the framework guarantees it is called after the last
reconcile cycle completes.
"""

from __future__ import annotations

import logging
import threading
from abc import abstractmethod
from typing import ClassVar

from nemo_helix_plugin._base import _NamedPlugin

logger = logging.getLogger(__name__)


class NemoController(_NamedPlugin):
    """Abstract base class for plugin-contributed reconcile-loop controllers.

    Subclasses declare their identity via class variables and implement at
    minimum :meth:`list_objects` and :meth:`reconcile_one`.  The platform
    instantiates and wraps them at startup — plugin authors do not call the
    platform directly.

    Class variables:

    .. attribute:: name
        :type: str

        Unique kebab-case controller name.  Must match the ``nemo.controllers``
        entry-point key.

    .. attribute:: dependencies
        :type: list[str]

        Names of platform services that must be ready before this controller
        starts (e.g. ``["entities"]``).  Defaults to ``[]``.
    """

    name: ClassVar[str]
    dependencies: ClassVar[list[str]] = []
    _stop_signal: threading.Event | None = None

    def set_stop_signal(self, stop_signal: threading.Event | None) -> None:
        """Register the platform shutdown signal for cooperative cancellation."""
        self._stop_signal = stop_signal

    def stop_requested(self) -> bool:
        """Return True when the platform has requested shutdown."""
        return self._stop_signal is not None and self._stop_signal.is_set()

    async def reconcile(self) -> None:
        """Run one full reconciliation cycle.

        The default implementation calls :meth:`list_objects` once, then
        :meth:`reconcile_one` for each result with per-item error isolation
        so a failure on one object does not abort the rest.

        Override this method entirely for controllers with complex multi-phase
        reconciliation logic.  In that case stub :meth:`list_objects` and
        :meth:`reconcile_one` with ``raise NotImplementedError``.
        """
        for obj in await self.list_objects():
            if self.stop_requested():
                return
            try:
                await self.reconcile_one(obj)
            except Exception:
                logger.exception("Failed to reconcile %s", obj)

    @abstractmethod
    async def list_objects(self) -> list:
        """Return the objects to reconcile this cycle (e.g. list entities).

        Called once per cycle by the default :meth:`reconcile` implementation.
        """

    @abstractmethod
    async def reconcile_one(self, obj: object) -> None:
        """Reconcile a single object.

        Errors are caught and logged by the default :meth:`reconcile`
        implementation so that one failing object does not abort the cycle.
        """

    async def on_startup(self) -> None:
        """Called once before the reconcile loop begins.

        Override for initialisation that requires async (e.g. connecting to
        an entity client, seeding a backend registry).  The default
        implementation does nothing.
        """

    async def on_shutdown(self) -> None:
        """Called once after the reconcile loop ends.

        Override for cleanup (e.g. closing connections, draining queues).
        The default implementation does nothing.
        """

    @property
    def is_healthy(self) -> bool:
        """Return ``True`` if the controller considers itself healthy.

        The platform health endpoint aggregates this across all controllers.
        Override to implement custom liveness logic.
        """
        return True

    @property
    def interval_seconds(self) -> float:
        """Seconds between reconcile cycles.

        Defined as a property (not a ``ClassVar``) so subclasses can derive
        the interval from config loaded at runtime in :meth:`on_startup`.
        Defaults to 10.0.
        """
        return 10.0
