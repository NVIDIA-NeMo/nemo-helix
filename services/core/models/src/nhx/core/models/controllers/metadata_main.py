# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import threading

import httpx
from nhx.common.client_factory import get_async_nemo_client
from nhx.common.config import get_platform_config
from nhx.common.controller import (
    Controller,
    ControllerManager,
    HeartbeatMixin,
    Loop,
    TimedLoopWaiter,
    TrackLastExecutionTime,
)
from nhx.common.service.api.health import wait_for_service_ready
from nhx.core.models.config import config as models_config
from nhx.core.models.metadata.controller import ModelMetadataController, build_http_client, load_plugin_sources

logger = logging.getLogger(__name__)

CONTROLLER_NAME = "model-metadata"
_REQUIRED_SERVICES = ("models", "inference-gateway")


class _MetadataLoopStep(HeartbeatMixin, Controller):
    """Runs one async reconcile pass per loop step on a private event loop."""

    def __init__(self, controller: ModelMetadataController, http: httpx.AsyncClient) -> None:
        self._controller = controller
        self._http = http
        self._event_loop = asyncio.new_event_loop()

    def step(self) -> None:
        self._event_loop.run_until_complete(self._controller.reconcile(on_progress=self.emit_heartbeat))

    def shutdown(self) -> None:
        if self._event_loop.is_closed():
            return
        self._event_loop.run_until_complete(self._http.aclose())
        self._event_loop.close()


def run(parent_stop_signal: threading.Event | None = None) -> None:
    stop_signal = parent_stop_signal or threading.Event()
    config = models_config.metadata
    platform_config = get_platform_config()
    for service in _REQUIRED_SERVICES:
        if not wait_for_service_ready(platform_config, service, stop_signal) and stop_signal.is_set():
            return

    http = build_http_client(config)
    controller = ModelMetadataController(
        config=config,
        client=get_async_nemo_client(as_service=CONTROLLER_NAME, internal=True),
        http=http,
        plugin_sources=load_plugin_sources(),
    )
    logger.info("Model metadata controller started with sources: %s", controller.source_names)
    step = _MetadataLoopStep(controller, http)
    loop = Loop(
        TimedLoopWaiter(config.interval_seconds, stop_signal=stop_signal),
        TrackLastExecutionTime(step),
        shutdown_func=step.shutdown,
        stop_signal=stop_signal,
    )
    ControllerManager.get_instance().register(CONTROLLER_NAME, loop)
    loop.start()
    try:
        while not stop_signal.is_set():
            stop_signal.wait(timeout=1)
    finally:
        loop.stop()
        loop.join(timeout=10)
