# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import signal
import threading

from nhx.common.config import get_platform_config, get_service_config
from nhx.common.controller import ControllerManager, Loop, TimedLoopWaiter, TrackLastExecutionTime
from nhx.common.sdk_factory import get_async_platform_sdk
from nhx.common.service.api.health import wait_for_service_ready
from nhx.core.entities.app.repository import (
    SQLAlchemyWorkspaceRepository,
    get_async_session_maker,
    initialize_async_engine,
)
from nhx.core.entities.config import EntitiesConfig
from nhx.core.entities.controllers.workspace_cleanup import WorkspaceCleanup

logger = logging.getLogger(__name__)

stop_signal = threading.Event()


def handle_signal(signum, frame):
    logger.info("Received shutdown signal, stopping entities controller...")
    stop_signal.set()


def run(parent_stop_signal: threading.Event | None = None):
    platform_config = get_platform_config()
    logger.info("Starting entities controller")

    if parent_stop_signal is None:
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        local_stop_signal = stop_signal
    else:
        local_stop_signal = parent_stop_signal

    nhx_sdk = get_async_platform_sdk(
        as_service="entities",
        internal=True,
    )

    # Create a single event loop that will be shared for DB init and the cleanup controller,
    # so SQLAlchemy's async pool is bound to the same loop that later runs queries.
    entities_config = get_service_config(EntitiesConfig)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(initialize_async_engine(entities_config))
    session_maker = loop.run_until_complete(get_async_session_maker())

    workspace_repository = SQLAlchemyWorkspaceRepository(session_maker)

    if not wait_for_service_ready(platform_config, "entities", local_stop_signal):
        if local_stop_signal.is_set():
            logger.info("Shutdown requested before server became ready")
            return
        logger.warning("Server did not become ready in time, starting loops anyway")

    cleanup_controller = WorkspaceCleanup(
        nhx_sdk=nhx_sdk,
        workspace_repository=workspace_repository,
        stop_signal=local_stop_signal,
        loop=loop,
    )
    cleanup_controller_monitored = TrackLastExecutionTime(cleanup_controller)

    cleanup_loop = Loop(
        TimedLoopWaiter(entities_config.workspace_cleanup_interval, stop_signal=local_stop_signal),
        cleanup_controller_monitored,
        stop_signal=local_stop_signal,
    )

    controller_manager = ControllerManager.get_instance()
    controller_manager.register("workspace_cleanup", cleanup_loop)

    cleanup_loop.start()
    logger.info("Entities controller started successfully")

    try:
        while not local_stop_signal.is_set():
            local_stop_signal.wait(timeout=1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping entities controller")
    finally:
        cleanup_loop.stop()
        cleanup_loop.join(timeout=10)
        if cleanup_loop.is_alive():
            logger.warning("Workspace cleanup loop did not stop in time")
        logger.info("Entities controller stopped")


if __name__ == "__main__":
    run()
