# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import signal
import threading

from nhx.common.config import get_platform_config
from nhx.common.controller import ControllerManager, Loop, TimedLoopWaiter, TrackLastExecutionTime
from nhx.common.sdk_factory import get_async_platform_sdk
from nhx.common.service.api.health import wait_for_service_ready
from nhx.core.models.config import backends
from nhx.core.models.config import config as models_config
from nhx.core.models.controllers.backends.registry import BackendRegistry
from nhx.core.models.controllers.models_controller import ModelsController

stop_signal = threading.Event()

# Global reference to monitor control loop health
models_controller_monitored = None


def get_health_status() -> dict:
    """Get the health status of the models controller."""
    if models_controller_monitored:
        if models_controller_monitored.is_healthy:
            return {"status": "ready"}
    return {"status": "not ready"}


def handle_sighup(signum, frame):
    """Handle SIGHUP, SIGINT, and SIGTERM signals for graceful shutdown."""
    logger = logging.getLogger(__name__)
    logger.info("Received shutdown signal, stopping Models Controller...")
    stop_signal.set()


def run(parent_stop_signal: threading.Event | None = None):
    """Run the Models Controller with its control loop."""
    global models_controller_monitored

    # Create logger after configuration is set up
    logger = logging.getLogger(__name__)
    logger.debug("Starting models controller")

    # Use provided stop signal or create our own
    if parent_stop_signal is None:
        # Register the handler for SIGHUP, SIGINT, and SIGTERM only if running standalone
        signal.signal(signal.SIGHUP, handle_sighup)
        signal.signal(signal.SIGINT, handle_sighup)
        signal.signal(signal.SIGTERM, handle_sighup)
        local_stop_signal = stop_signal
    else:
        local_stop_signal = parent_stop_signal

    # Initialize NeMo Helix SDK (used for all API interactions including secrets)
    nhx_sdk = get_async_platform_sdk(as_service="models", internal=True)

    # Initialize backend registry from configuration
    logger.info("Initializing backend registry...")
    logger.debug(f"Models backend configs: {backends}")
    backend_registry = BackendRegistry.from_config(
        nhx_sdk=nhx_sdk,
        backend_configs=backends,
        huggingface_model_puller=models_config.huggingface_model_puller,
    )
    logger.info(f"Backend registry initialized with: {', '.join(backend_registry.list_backends())}")

    # Wait for the models service to be ready before starting control loops (polls /status so we can start once models is ready)
    platform_config = get_platform_config()
    if not wait_for_service_ready(platform_config, "models", local_stop_signal):
        if local_stop_signal.is_set():
            logger.info("Shutdown requested before server became ready")
            return
        logger.warning("Server did not become ready in time, starting loops anyway")

    # Create the Models Controller
    models_controller = ModelsController(backend_registry=backend_registry, stop_signal=local_stop_signal)
    models_controller_monitored = TrackLastExecutionTime(models_controller)

    # Create the control loop with configured interval.
    # shutdown_func ensures controller resources (event loop, backends) are cleaned up
    # inside the loop thread after it exits, avoiding race conditions with step().
    models_controller_loop = Loop(
        TimedLoopWaiter(models_config.controller.interval_seconds, stop_signal=local_stop_signal),
        models_controller_monitored,
        shutdown_func=models_controller.shutdown,
        stop_signal=local_stop_signal,
    )

    # Register loop with ControllerManager
    controller_manager = ControllerManager.get_instance()
    controller_manager.register("models_controller", models_controller_loop)

    # Start the control loop
    logger.debug("Starting Models Controller control loop...")
    models_controller_loop.start()
    logger.debug("Models controller started successfully")

    try:
        # Wait for stop signal or control loop to finish
        while not local_stop_signal.is_set():
            local_stop_signal.wait(timeout=1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping models controller")
    finally:
        # Tiered shutdown: graceful wait → force cancel → forced cleanup
        models_controller_loop.stop()
        models_controller_loop.join(timeout=10)
        if models_controller_loop.is_alive():
            logger.warning("Models controller step did not finish in 10s, cancelling...")
            models_controller.cancel_step()
            models_controller_loop.join(timeout=5)
        if models_controller_loop.is_alive():
            logger.warning("Models controller loop did not stop, forcing cleanup")
            models_controller.shutdown()
        logger.info("Models controller stopped")


if __name__ == "__main__":
    run()
