# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import signal
import threading
import time

from nhx.common.config import get_platform_config
from nhx.common.controller import ControllerManager, Loop, TimedLoopWaiter, TrackLastExecutionTime
from nhx.common.sdk_factory import get_platform_sdk
from nhx.common.service.api.health import wait_for_service_ready
from nhx.core.jobs.config import config as jobs_config
from nhx.core.jobs.config import profiles
from nhx.core.jobs.controllers.backends.registry import BackendRegistry
from nhx.core.jobs.controllers.reconciler import JobReconciler
from nhx.core.jobs.controllers.scheduler import JobScheduler

stop_signal = threading.Event()
logger = logging.getLogger(__name__)


def handle_sighup(signum, frame):
    stop_signal.set()


def run(parent_stop_signal: threading.Event | None = None):
    # Create logger after configuration is set up
    logger.info("Starting jobs controller")

    # Use provided stop signal or create our own
    if parent_stop_signal is None:
        # Register the handler for SIGINT and SIGTERM only if running standalone
        signal.signal(signal.SIGINT, handle_sighup)
        signal.signal(signal.SIGTERM, handle_sighup)
        local_stop_signal = stop_signal
    else:
        local_stop_signal = parent_stop_signal

    # Initialize components
    # Use service principal for controller - runs in background thread without user context
    nhx_sdk = get_platform_sdk(as_service="jobs", internal=True)
    logger.debug("Platform SDK initialized successfully.")

    # from_config also prunes the shared ``profiles`` list so advertised executors
    # match backends that actually registered (e.g. Docker skipped when unavailable).
    backend_registry = BackendRegistry.from_config(nhx_sdk=nhx_sdk, profiles=profiles)
    logger.info("Executor backends registry initialized successfully.")

    # Wait for the jobs service to be ready before starting control loops (polls /status so we can start once jobs is ready)
    platform_config = get_platform_config()
    if not wait_for_service_ready(platform_config, "jobs", local_stop_signal):
        if local_stop_signal.is_set():
            logger.info("Shutdown requested before server became ready")
            return
        logger.warning("Server did not become ready in time, starting loops anyway")

    # Job scheduling loop
    job_scheduler = JobScheduler(backend_registry, nhx_sdk, stop_signal=local_stop_signal)
    job_scheduler_monitored = TrackLastExecutionTime(job_scheduler)
    job_scheduler_loop = Loop(
        TimedLoopWaiter(jobs_config.schedule_interval_seconds, stop_signal=local_stop_signal),
        job_scheduler_monitored,
        stop_signal=local_stop_signal,
    )

    # Job reconciler loop
    job_reconciler = JobReconciler(backend_registry, nhx_sdk, stop_signal=local_stop_signal)
    job_reconciler_monitored = TrackLastExecutionTime(job_reconciler)
    job_reconciler_loop = Loop(
        TimedLoopWaiter(jobs_config.reconcile_interval_seconds, stop_signal=local_stop_signal),
        job_reconciler_monitored,
        shutdown_func=backend_registry.shutdown_all_backends,
        stop_signal=local_stop_signal,
    )

    # Register loops with ControllerManager
    controller_manager = ControllerManager.get_instance()
    controller_manager.register("job_scheduler", job_scheduler_loop)
    controller_manager.register("job_reconciler", job_reconciler_loop)

    # Start control loops
    job_scheduler_loop.start()
    job_reconciler_loop.start()
    logger.info("Jobs controller started successfully")

    # Main loop
    while not local_stop_signal.is_set():
        time.sleep(1)

    logger.info("Shutting down control loops...")
    job_scheduler_loop.stop()
    job_reconciler_loop.stop()

    for loop in [job_scheduler_loop, job_reconciler_loop]:
        loop.join()

    logger.info("All jobs control loops have been shut down.")
