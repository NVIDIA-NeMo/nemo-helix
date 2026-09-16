# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task to write a greeting message to the file API."""

import os
import signal
import threading
import time
import traceback
import types

from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nhx.common.jobs.config import get_job_id, get_task_config, get_workspace
from nhx.common.sdk_factory import get_platform_sdk
from nhx.hello_world.api.v2.jobs.schemas import HelloWorldJobConfig

DEFAULT_FILE_PATH = "message.txt"
BUSY_LOOP_DURATION_SECONDS = float(os.environ.get("BUSY_LOOP_DURATION_SECONDS", "10"))
BUSY_LOOP_CHECK_INTERVAL_SECONDS = float(os.environ.get("BUSY_LOOP_CHECK_INTERVAL_SECONDS", "0.5"))

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum: int, frame: types.FrameType | None) -> None:
    """Handle termination signals gracefully."""
    global shutdown_requested
    shutdown_requested = True
    print(f"\nReceived signal {signal.Signals(signum).name}, shutting down gracefully...\n")


def register_signal_handlers() -> None:
    """Register handlers for common termination signals.

    Only registers handlers when running in the main thread, as signal handlers
    can only be set from the main thread.
    """
    if threading.current_thread() is not threading.main_thread():
        return

    signals_to_handle = [
        signal.SIGTERM,
        signal.SIGINT,
    ]

    # SIGHUP is not available on Windows
    if hasattr(signal, "SIGHUP"):
        signals_to_handle.append(signal.SIGHUP)

    for sig in signals_to_handle:
        signal.signal(sig, signal_handler)


def busy_loop(duration_seconds: float) -> bool:
    """
    Simulate busy work with periodic checks for shutdown signals.

    Returns:
        True if completed without interruption, False if shutdown was requested.
    """
    print(f"Starting busy loop for {duration_seconds} seconds...")
    start_time = time.time()
    elapsed = 0.0

    while elapsed < duration_seconds:
        if shutdown_requested:
            print(f"Busy loop interrupted after {elapsed:.1f} seconds")
            return False

        time.sleep(BUSY_LOOP_CHECK_INTERVAL_SECONDS)
        elapsed = time.time() - start_time

    print(f"Busy loop completed after {duration_seconds} seconds")
    return True


def run(*, sdk: NeMoHelix | None = None) -> int:
    """Execute the task to write the configured message to the file API.

    Args:
        sdk: Optional SDK instance for dependency injection (for testing).
            If None, uses get_platform_sdk().

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        # Register signal handlers for graceful shutdown
        # TODO: Should these be part of the common task setup?
        register_signal_handlers()

        config = get_task_config(HelloWorldJobConfig)
        job_id = get_job_id()
        workspace = get_workspace()
        fileset_name = f"hello-world-{job_id}"

        print(f"Writing message to {workspace}/{fileset_name}/{DEFAULT_FILE_PATH}")

        sdk = sdk or get_platform_sdk()

        # Busy loop to simulate work and allow for signal testing
        if not busy_loop(BUSY_LOOP_DURATION_SECONDS):
            print("Task interrupted during busy loop, exiting gracefully")
            return 0

        # Upload the message (creates fileset if it doesn't exist)
        files = client_from_platform(sdk, FilesClient)
        files.create_fileset(body=CreateFilesetRequest(name=fileset_name), workspace=workspace, exist_ok=True)
        files.upload_file(
            content=config.message,
            path=DEFAULT_FILE_PATH,
            name=fileset_name,
            workspace=workspace,
        )

        print(f"Successfully wrote message: {config.message}")
        return 0
    except KeyboardInterrupt:
        print("Task interrupted by KeyboardInterrupt, exiting gracefully")
        return 0
    except Exception as e:
        print(f"Task failed: {e}")
        traceback.print_exc()
        return 1
