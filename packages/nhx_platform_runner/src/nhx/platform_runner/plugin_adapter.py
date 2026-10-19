# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters that wrap Nemo plugin types in the platform's internal contracts.

:class:`NemoServiceAdapter`
    Wraps a :class:`~nemo_helix_plugin.service.NemoService` as a platform-native
    :class:`~nhx.common.service.Service`.

:class:`NemoControllerAdapter`
    Wraps a :class:`~nemo_helix_plugin.controller.NemoController` (async) as a
    platform-native :class:`~nhx.common.controller.Controller` (sync /
    thread-based).  Use :func:`make_controller_run_func` to create the
    ``run(stop_signal)`` callable expected by :func:`~nhx.platform_runner.server.create_app`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable

from fastapi import FastAPI
from nemo_helix_plugin.controller import NemoController
from nemo_helix_plugin.service import NemoService
from nhx.common.config import get_platform_config
from nhx.common.controller import Controller, ControllerManager, Loop, TimedLoopWaiter, TrackLastExecutionTime
from nhx.common.service import RouterConfig, Service
from nhx.common.service.api.health import wait_for_service_ready

logger = logging.getLogger(__name__)

_SHUTDOWN_TIMEOUT_SECONDS: float = 10.0


class NemoServiceAdapter(Service):
    """Wrap a plugin service as a platform-native service."""

    def __init__(self, plugin: NemoService) -> None:
        super().__init__(name=plugin.name, module_name=type(plugin).__module__)
        self._plugin = plugin
        self._dependencies = list(getattr(type(plugin), "dependencies", []))

    def get_routers(self) -> list[RouterConfig]:
        return [
            RouterConfig(
                router=spec.router,
                tag=spec.tag or self._plugin.name,
                description=spec.description,
                prefix=spec.prefix,
            )
            for spec in self._plugin.get_routers()
        ]

    def create_app(self) -> FastAPI:
        app = super().create_app()
        for exc_type, handler in self._plugin.get_exception_handlers().items():
            app.add_exception_handler(exc_type, handler)
        return app

    async def on_startup(self) -> None:
        logger.debug("Starting plugin service %s", self._plugin.name)
        await super().on_startup()
        try:
            await self._plugin.on_startup()
        except Exception:
            await super().on_shutdown()
            raise

    async def on_shutdown(self) -> None:
        logger.debug("Shutting down plugin service %s", self._plugin.name)
        await self._plugin.on_shutdown()
        await super().on_shutdown()


class NemoControllerAdapter(Controller):
    """Wrap a NemoController (async) as a platform-native Controller (sync/thread).

    Creates a dedicated asyncio event loop for the controller so that async
    reconcile cycles can run from within the platform's thread-based
    :class:`~nhx.common.controller.Loop`.

    The event loop is owned by this adapter and closed in :meth:`shutdown`,
    which the :class:`~nhx.common.controller.Loop` calls as its
    ``shutdown_func`` after the reconcile loop exits.
    """

    def __init__(self, plugin: NemoController) -> None:
        self._plugin = plugin
        self._loop = asyncio.new_event_loop()

    def step(self) -> None:
        """Run one reconcile cycle synchronously on the adapter's event loop."""
        self._loop.run_until_complete(self._plugin.reconcile())

    @property
    def is_healthy(self) -> bool:
        return self._plugin.is_healthy

    def shutdown(self) -> None:
        """Call ``on_shutdown()`` on the plugin, then close the event loop.

        Enforces a ``_SHUTDOWN_TIMEOUT_SECONDS`` timeout so a misbehaving
        ``on_shutdown()`` implementation cannot hang the platform shutdown.
        """
        if self._loop.is_closed():
            return
        try:
            self._loop.run_until_complete(
                asyncio.wait_for(
                    self._plugin.on_shutdown(),
                    timeout=_SHUTDOWN_TIMEOUT_SECONDS,
                )
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Plugin controller %r on_shutdown() did not complete within %ss — proceeding.",
                self._plugin.name,
                _SHUTDOWN_TIMEOUT_SECONDS,
            )
        except Exception:
            logger.exception(
                "Error in plugin controller %r on_shutdown().",
                self._plugin.name,
            )
        finally:
            self._loop.close()


def _wait_for_controller_dependencies(
    controller_cls: type[NemoController],
    stop_signal: threading.Event,
) -> bool:
    """Poll until declared service dependencies are ready.

    Returns False when shutdown is requested during the wait; True otherwise
    (including when a dependency times out — the caller may proceed anyway).
    """
    dependencies = list(controller_cls.dependencies)
    if not dependencies:
        return True

    platform_config = get_platform_config()
    for dep in dependencies:
        if wait_for_service_ready(platform_config, dep, stop_signal):
            continue
        if stop_signal.is_set():
            logger.info(
                "Shutdown requested before %r dependency %r became ready",
                controller_cls.name,
                dep,
            )
            return False
        logger.warning(
            "Plugin controller %r dependency %r did not become ready in time — starting anyway",
            controller_cls.name,
            dep,
        )
    return True


def make_controller_run_func(controller_cls: type[NemoController]) -> Callable[[threading.Event], None]:
    """Return a ``run(stop_signal)`` function that drives a :class:`NemoController`.

    The returned callable matches the signature expected by
    :func:`~nhx.platform_runner.server.create_app` (same as core controller
    run functions registered in ``AVAILABLE_CONTROLLERS``).

    Lifecycle inside the returned function:

    1. Instantiate *controller_cls*.
    2. Wait for each service in ``dependencies`` via ``wait_for_service_ready``.
    3. Call ``on_startup()`` on the adapter's event loop.
    4. Wrap in :class:`~nhx.common.controller.Loop` with a
       :class:`~nhx.common.controller.TimedLoopWaiter` honouring *stop_signal*.
    5. Start the loop thread and join it (blocking until stop).
    6. The loop's ``shutdown_func`` calls :meth:`NemoControllerAdapter.shutdown`
       which runs ``on_shutdown()`` and closes the event loop.

    Args:
        controller_cls: The :class:`NemoController` subclass to run.

    Returns:
        A callable ``run(stop_signal: threading.Event) -> None``.
    """

    def run(stop_signal: threading.Event) -> None:
        controller = controller_cls()
        controller.set_stop_signal(stop_signal)
        adapter = NemoControllerAdapter(controller)

        if not _wait_for_controller_dependencies(controller_cls, stop_signal):
            adapter._loop.close()
            return

        try:
            adapter._loop.run_until_complete(controller.on_startup())
        except Exception:
            logger.exception(
                "Plugin controller %r on_startup() failed — controller will not run.",
                controller.name,
            )
            adapter.shutdown()
            raise

        monitored = TrackLastExecutionTime(adapter)
        waiter = TimedLoopWaiter(
            sleep_secs=controller.interval_seconds,
            stop_signal=stop_signal,
        )
        loop = Loop(
            waiter=waiter,
            controller=monitored,
            shutdown_func=adapter.shutdown,
            stop_signal=stop_signal,
        )
        loop.name = f"controller-plugin-{controller.name}"
        try:
            ControllerManager.get_instance().register(loop.name, loop)
        except Exception:
            logger.exception(
                "Plugin controller %r failed to register its control loop — controller will not run.",
                controller.name,
            )
            adapter.shutdown()
            raise
        try:
            loop.start()
        except Exception:
            _unregister_quietly(loop.name, controller.name)
            adapter.shutdown()
            raise
        loop.join()

    return run


def _unregister_quietly(loop_name: str, controller_name: str) -> None:
    """Best-effort unregister that never masks the caller's in-flight exception.

    Called from an ``except Exception:`` block after ``loop.start()`` fails. A
    bare ``ControllerManager.unregister(...)`` there would let a second
    exception replace the original failure on the ``raise`` that follows,
    hiding the real cause from logs/health status.
    """
    try:
        ControllerManager.get_instance().unregister(loop_name)
    except Exception:
        logger.exception(
            "Plugin controller %r failed to unregister its control loop after loop.start() failed",
            controller_name,
        )
