# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Python equivalent of run-ray.sh for Ray cluster bootstrap.

This module provides a Python implementation of Ray cluster bootstrapping
on Volcano-provisioned pods, replacing the shell script approach for
better integration and error handling.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType

from nhx.rl.tasks.training.errors.exceptions import format_exception_string
from nhx.rl.tasks.training.errors.parser import (
    MAX_OUTPUT_LINES,
    read_subprocess_output,
)

logger = logging.getLogger(__name__)

# Timeouts (seconds) for Ray CLI subprocesses so a hung `ray` invocation can't
# block the bootstrap — or its non-daemon cleanup thread — indefinitely.
# `ray start` may pull images / initialize for a while; `ray status` / `ray
# memory` are quick queries.
RAY_START_TIMEOUT_SECONDS = 300
RAY_STATUS_TIMEOUT_SECONDS = 60


def _pause(seconds: float) -> None:
    time.sleep(seconds)


@dataclass
class RayPortConfig:
    """Port configuration for Ray cluster services.

    All ports are configurable via environment variables with sensible defaults.
    Head nodes use port+1 offset for manager ports to avoid conflicts with workers.
    """

    node_manager_port: int = field(default_factory=lambda: int(os.getenv("NODE_MANAGER_PORT", "53001")))
    object_manager_port: int = field(default_factory=lambda: int(os.getenv("OBJECT_MANAGER_PORT", "53003")))
    runtime_env_agent_port: int = field(default_factory=lambda: int(os.getenv("RUNTIME_ENV_AGENT_PORT", "53005")))
    dashboard_agent_grpc_port: int = field(default_factory=lambda: int(os.getenv("DASHBOARD_AGENT_GRPC_PORT", "53007")))
    metrics_export_port: int = field(default_factory=lambda: int(os.getenv("METRICS_EXPORT_PORT", "53009")))
    gcs_port: int = field(default_factory=lambda: int(os.getenv("GCS_PORT", "6379")))
    ray_client_server_port: int = field(default_factory=lambda: int(os.getenv("RAY_CLIENT_SERVER_PORT", "10001")))
    dashboard_port: int = field(default_factory=lambda: int(os.getenv("DASHBOARD_PORT", "8265")))
    dashboard_agent_listen_port: int = field(
        default_factory=lambda: int(os.getenv("DASHBOARD_AGENT_LISTEN_PORT", "52365"))
    )
    min_worker_port: int = field(default_factory=lambda: int(os.getenv("MIN_WORKER_PORT", "54001")))
    max_worker_port: int = field(default_factory=lambda: int(os.getenv("MAX_WORKER_PORT", "54257")))


@dataclass
class RayClusterBootstrap:
    """Bootstrap Ray cluster on Volcano-provisioned pods.

    This class handles starting Ray head nodes and worker nodes in a
    distributed training environment managed by Volcano job scheduler.

    The bootstrap process:
    - Head (rank 0): Start Ray head -> wait for workers -> run driver -> cleanup
    - Worker (rank > 0): Start Ray worker -> monitor for ENDED file -> exit

    Attributes:
        rank: The rank of this node (0 = head, >0 = worker)
        world_size: Total number of nodes in the cluster
        master_addr: IP address of the head node
        gpus_per_node: Number of GPUs per node
        log_dir: Directory for logs and coordination files. For multi-node clusters,
            this MUST be a shared filesystem (e.g., NFS) accessible by all nodes.
            The head node writes an ENDED marker file here that workers poll for
            graceful shutdown coordination. Set via BASE_LOG_DIR environment variable.
        ports: Port configuration for Ray services
        num_retries: Number of retries for Ray start commands
        retry_sleep: Seconds to sleep between retries
        driver_python: Python executable for running driver scripts (allows using
            a different virtual environment). Defaults to DRIVER_PYTHON env var
            or current Python interpreter.
        driver_extra_pythonpath: Additional paths to append to PYTHONPATH when running
            driver scripts. Useful for accessing packages from other environments.
            Defaults to DRIVER_EXTRA_PYTHONPATH env var.

    Example:
        Basic usage with environment variables (recommended for Volcano jobs)::

            # Environment variables set by Volcano: RANK, WORLD_SIZE, MASTER_ADDR
            bootstrap = create_bootstrap_from_env()
            exit_code = bootstrap.run_with_driver(
                driver_script="/path/to/dpo_driver.py",
                driver_args=["--config", "/path/to/config.yaml", "--id", "job-123"],
            )
            sys.exit(exit_code)

        Manual configuration for testing::

            bootstrap = RayClusterBootstrap(
                rank=0,              # Head node
                world_size=2,        # 2-node cluster
                master_addr="10.0.0.1",
                gpus_per_node=8,
            )

            # Option 1: Start cluster and run driver script
            exit_code = bootstrap.run_with_driver(
                driver_script="train_dpo.py",
                driver_args=["--config", "config.yaml"],
            )

            # Option 2: Just start the cluster (for workers or manual control)
            bootstrap.start()

        Using a different Python virtual environment for the driver::

            # Via environment variables
            os.environ["DRIVER_PYTHON"] = "/opt/nemo-venv/bin/python"
            os.environ["DRIVER_EXTRA_PYTHONPATH"] = "/opt/venv/lib/python3.12/site-packages"
            bootstrap = create_bootstrap_from_env()

            # Or via direct configuration
            bootstrap = RayClusterBootstrap(
                rank=0,
                world_size=1,
                master_addr="127.0.0.1",
                driver_python="/opt/nemo-venv/bin/python",  # Custom venv
                driver_extra_pythonpath="/opt/venv/lib/python3.12/site-packages",  # Extra packages
            )

        Command-line invocation::

            # Start cluster and run driver
            python -m nhx.rl.tasks.training.backends.nemo_rl.ray_bootstrap \\
                /path/to/driver.py --config config.yaml --id job-123

            # Just start cluster node (head or worker based on RANK env var)
            python -m nhx.rl.tasks.training.backends.nemo_rl.ray_bootstrap
    """

    rank: int
    world_size: int
    master_addr: str
    gpus_per_node: int = field(default_factory=lambda: int(os.getenv("GPUS_PER_NODE", "1")))
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("BASE_LOG_DIR", "/tmp")) / "logs")
    ports: RayPortConfig = field(default_factory=RayPortConfig)
    num_retries: int = 3
    retry_sleep: int = 20
    attempt_id: str = field(default_factory=lambda: os.getenv("NEMO_JOB_ATTEMPT_ID", "attempt-0"))
    """Job attempt id, used to scope the ENDED coordination marker so a stale
    marker left by a previous attempt cannot short-circuit a retry's startup."""
    driver_python: str = field(default_factory=lambda: os.getenv("DRIVER_PYTHON", sys.executable))
    """Python executable path for running driver scripts.

    This allows running driver scripts in a different virtual environment.
    Can be set via DRIVER_PYTHON environment variable or passed directly.
    Defaults to sys.executable (current Python interpreter).
    """

    ray_executable: str = field(default="")
    """Path to the ray executable. If empty, derived from driver_python's directory."""

    driver_extra_pythonpath: str = field(default_factory=lambda: os.getenv("DRIVER_EXTRA_PYTHONPATH", ""))
    """Additional paths to append to PYTHONPATH when running driver scripts.

    Multiple paths can be separated by colons (Unix) or semicolons (Windows).
    Can be set via DRIVER_EXTRA_PYTHONPATH environment variable or passed directly.
    Example: "/opt/venv/lib/python3.12/site-packages:/other/path"
    """

    # Internal state
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _driver_output: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_OUTPUT_LINES), repr=False)
    _driver_process: subprocess.Popen | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize the bootstrap environment."""
        # Warn if multi-node setup might have coordination issues
        if self.world_size > 1 and not os.getenv("BASE_LOG_DIR"):
            logger.warning(
                "Multi-node Ray cluster detected (world_size=%d) but BASE_LOG_DIR not set. "
                "The ENDED coordination file requires a shared filesystem across all nodes. "
                "Workers may not detect graceful termination if log_dir (%s) is not shared.",
                self.world_size,
                self.log_dir,
            )

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Derive ray executable from driver_python if not specified
        if not self.ray_executable:
            # Get the bin directory from driver_python path
            driver_bin_dir = Path(self.driver_python).parent
            self.ray_executable = str(driver_bin_dir / "ray")

        # Disable proxy environment variables for local pod communication
        self._unset_proxy_env()

    def _unset_proxy_env(self) -> None:
        """Unset proxy environment variables for local pod communication."""
        proxy_vars = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]
        for var in proxy_vars:
            os.environ.pop(var, None)

    @property
    def driver_output(self) -> deque[str]:
        """Rolling buffer of recent driver output lines for error extraction."""
        return self._driver_output

    def terminate_driver(self, signum: int = signal.SIGTERM, timeout: int = 30) -> None:
        """Terminate the driver subprocess if it is running.

        Sends the specified signal to the driver process and waits for it to exit.
        If it doesn't exit within the timeout, it is forcefully killed.

        Args:
            signum: Signal to send (default SIGTERM).
            timeout: Seconds to wait for graceful exit before killing.
        """
        process = self._driver_process
        if process is None or process.poll() is not None:
            return
        logger.warning(f"Terminating driver process (pid={process.pid}) with signal {signum}")
        try:
            process.send_signal(signum)
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning(f"Driver process did not exit within {timeout}s, killing")
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Killed driver process did not terminate within 5s")

    @property
    def ended_file(self) -> Path:
        """Path to the ENDED coordination file.

        Scoped by the job attempt id: the head writes this marker during cleanup,
        and on shared (multi-node) storage it would otherwise persist into the
        next attempt and immediately short-circuit the retry's head/worker
        startup. A new attempt id yields a fresh path with no stale marker.
        """
        return self.log_dir / f"ENDED.{self.attempt_id}"

    def _clear_ended_marker(self) -> None:
        """Remove a lingering ENDED marker before the head starts a fresh run.

        The marker is attempt-scoped so a normal retry already gets a new path;
        this additionally covers attempt-id reuse (e.g. local runs / resume) where
        a marker from the prior run could otherwise linger on shared storage.
        """
        try:
            self.ended_file.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"Failed to clear stale ENDED marker {self.ended_file}: {e}")

    @property
    def expected_worker_units(self) -> int:
        """Total expected worker units (nodes * GPUs per node)."""
        return self.world_size * self.gpus_per_node

    @property
    def max_wait_seconds(self) -> int:
        """Maximum wait time for workers to connect.

        Multi-node clusters get 40 minutes to accommodate slow image downloads.
        Single-node clusters only need 4 minutes.
        """
        return 2400 if self.world_size > 1 else 240

    def start(self) -> None:
        """Start Ray head (rank 0) or worker (rank > 0)."""
        if self.rank == 0:
            self._run_as_head()
        else:
            self._run_as_worker()

    def run_with_driver(self, driver_script: str, driver_args: list[str]) -> int:
        """Start Ray cluster and run driver script on head node.

        This is the main entry point for executing training with Ray.

        Args:
            driver_script: Path to the Python driver script
            driver_args: Arguments to pass to the driver script

        Returns:
            Exit code from the driver script (0 for success)
        """
        if self.rank == 0:
            return self._run_head_with_driver(driver_script, driver_args)
        else:
            return self._run_as_worker()

    def _run_as_head(self) -> None:
        """Run as head node: start head, wait for workers, then exit."""
        self._clear_ended_marker()
        if not self._start_head_background():
            raise RuntimeError("Failed to start Ray head node")
        self._wait_for_workers()

    def _run_head_with_driver(self, driver_script: str, driver_args: list[str]) -> int:
        """Run as head node with driver execution.

        Args:
            driver_script: Path to the Python driver script
            driver_args: Arguments to pass to the driver script

        Returns:
            Exit code from driver execution
        """
        exit_code = 1
        try:
            self._clear_ended_marker()
            if not self._start_head_background():
                raise RuntimeError("Failed to start Ray head node")

            # Wait a bit for Ray to fully initialize before checking status
            print("[ray_bootstrap] Waiting for Ray to initialize...", flush=True)
            _pause(5)

            print("[ray_bootstrap] Waiting for workers to connect...", flush=True)
            self._wait_for_workers()

            logger.info("--- All workers connected! ---")
            print("[ray_bootstrap] --- All workers connected! ---", flush=True)
            self._log_ray_status()

            logger.info("--- Starting driver ---")
            print(f"[ray_bootstrap] --- Starting driver: {driver_script} ---", flush=True)
            exit_code = self._run_driver(driver_script, driver_args)
            logger.info(f"Driver completed with exit code: {exit_code}")
            print(f"[ray_bootstrap] Driver completed with exit code: {exit_code}", flush=True)

        except Exception as e:
            logger.exception(f"Error in head node execution: {e}")
            print(f"[ray_bootstrap] Error in head node execution: {e}", flush=True)
            # Capture the exception message into the output buffer so the
            # backend's parse_error_from_output can surface it to the user.
            self._driver_output.append(format_exception_string(e))
            exit_code = 1
        finally:
            self._cleanup_with_timeout()

        return exit_code

    def _run_as_worker(self) -> int:
        """Run as worker node: start worker and monitor for termination.

        Returns:
            Exit code (0 for graceful termination, non-zero otherwise)
        """
        if not self._start_worker_background():
            return 1
        return self._monitor_for_termination()

    def _start_head_background(self) -> bool:
        """Start Ray head node with retry logic.

        Since ray start returns immediately (without --block), we run this
        synchronously with retries rather than in a background thread.

        Returns:
            True if head started successfully, False otherwise
        """
        for attempt in range(self.num_retries):
            if self._stop_event.is_set() or self.ended_file.exists():
                logger.info("Head node stopping due to termination signal")
                return False

            logger.info(f"Launching Head Node (attempt {attempt + 1}/{self.num_retries})")
            print(
                f"[ray_bootstrap] Launching Head Node (attempt {attempt + 1}/{self.num_retries})",
                flush=True,
            )
            try:
                result = self._start_head_process()
                if result is not None:
                    return True
                logger.warning(f"Head start failed, attempt {attempt + 1}/{self.num_retries}")
            except Exception as e:
                logger.exception(f"Head node error: {e}")
                print(f"[ray_bootstrap] Head node error: {e}", flush=True)

            if not self._stop_event.is_set() and not self.ended_file.exists():
                _pause(self.retry_sleep)

        logger.error("Head Node failed to start after all retries")
        print("[ray_bootstrap] Head Node failed to start after all retries", flush=True)
        return False

    def _start_worker_background(self) -> bool:
        """Start Ray worker node with retry logic.

        Since ray start returns immediately (without --block), we run this
        synchronously with retries rather than in a background thread.

        Returns:
            True if worker started successfully, False otherwise
        """
        for attempt in range(self.num_retries):
            if self._stop_event.is_set() or self.ended_file.exists():
                logger.info("Worker node stopping due to termination signal")
                return False

            logger.info(f"Launching Worker Node (attempt {attempt + 1}/{self.num_retries})")
            print(
                f"[ray_bootstrap] Launching Worker Node (attempt {attempt + 1}/{self.num_retries})",
                flush=True,
            )
            try:
                result = self._start_worker_process()
                if result is not None:
                    return True
                logger.warning(f"Worker start failed, attempt {attempt + 1}/{self.num_retries}")
            except Exception as e:
                logger.exception(f"Worker node error: {e}")
                print(f"[ray_bootstrap] Worker node error: {e}", flush=True)

            if not self._stop_event.is_set() and not self.ended_file.exists():
                _pause(self.retry_sleep)

        logger.error("Worker Node failed to start after all retries")
        print("[ray_bootstrap] Worker Node failed to start after all retries", flush=True)
        return False

    def _start_head_process(self) -> subprocess.CompletedProcess | None:
        """Start the Ray head process.

        Note: Unlike the bash script which uses --block, we don't need it here
        because ray start returns immediately and Ray continues running in the
        background. The bash script needed --block to keep the background job alive.

        Returns:
            The CompletedProcess result from ray start, or None on failure
        """
        p = self.ports
        cmd = [
            self.ray_executable,
            "start",
            "--head",
            "--disable-usage-stats",
            "--include-dashboard=false",
            f'--resources={{"worker_units": {self.gpus_per_node}}}',
            f"--node-ip-address={self.master_addr}",
            f"--port={p.gcs_port}",
            f"--ray-client-server-port={p.ray_client_server_port}",
            f"--dashboard-port={p.dashboard_port}",
            # Head uses port+1 offset to avoid conflicts
            f"--node-manager-port={p.node_manager_port + 1}",
            f"--object-manager-port={p.object_manager_port + 1}",
            f"--runtime-env-agent-port={p.runtime_env_agent_port + 1}",
            f"--dashboard-agent-grpc-port={p.dashboard_agent_grpc_port + 1}",
            f"--dashboard-agent-listen-port={p.dashboard_agent_listen_port + 1}",
            f"--metrics-export-port={p.metrics_export_port + 1}",
        ]
        logger.info(f"Starting head: {' '.join(cmd)}")
        print(f"[ray_bootstrap] Starting head: {' '.join(cmd)}", flush=True)
        try:
            result = subprocess.run(cmd, check=False, timeout=RAY_START_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning(f"Ray head start timed out after {RAY_START_TIMEOUT_SECONDS}s")
            print(f"[ray_bootstrap] Ray head start timed out after {RAY_START_TIMEOUT_SECONDS}s", flush=True)
            return None
        if result.returncode != 0:
            print(f"[ray_bootstrap] Head start failed with code {result.returncode}", flush=True)
            return None
        print("[ray_bootstrap] Head node started successfully", flush=True)
        return result

    def _start_worker_process(self) -> subprocess.CompletedProcess | None:
        """Start the Ray worker process.

        Note: Unlike the bash script which uses --block, we don't need it here
        because ray start returns immediately and Ray continues running in the
        background.

        Returns:
            The CompletedProcess result from ray start, or None on failure
        """
        p = self.ports
        cmd = [
            self.ray_executable,
            "start",
            f"--address={self.master_addr}:{p.gcs_port}",
            "--disable-usage-stats",
            f'--resources={{"worker_units": {self.gpus_per_node}}}',
            f"--min-worker-port={p.min_worker_port}",
            f"--max-worker-port={p.max_worker_port}",
            f"--node-manager-port={p.node_manager_port}",
            f"--object-manager-port={p.object_manager_port}",
            f"--runtime-env-agent-port={p.runtime_env_agent_port}",
            f"--dashboard-agent-grpc-port={p.dashboard_agent_grpc_port}",
            f"--dashboard-agent-listen-port={p.dashboard_agent_listen_port}",
            f"--metrics-export-port={p.metrics_export_port}",
        ]
        logger.info(f"Starting worker: {' '.join(cmd)}")
        print(f"[ray_bootstrap] Starting worker: {' '.join(cmd)}", flush=True)
        try:
            result = subprocess.run(cmd, check=False, timeout=RAY_START_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning(f"Ray worker start timed out after {RAY_START_TIMEOUT_SECONDS}s")
            print(f"[ray_bootstrap] Ray worker start timed out after {RAY_START_TIMEOUT_SECONDS}s", flush=True)
            return None
        if result.returncode != 0:
            print(f"[ray_bootstrap] Worker start failed with code {result.returncode}", flush=True)
            return None
        print("[ray_bootstrap] Worker node started successfully", flush=True)
        return result

    def _wait_for_workers(self) -> None:
        """Poll until all workers have connected to the cluster.

        Raises:
            TimeoutError: If workers don't connect within max_wait_seconds
        """
        poll_interval = 2
        elapsed = 0

        while elapsed < self.max_wait_seconds:
            if self.ended_file.exists():
                raise RuntimeError("ENDED file detected during worker wait")

            worker_units = self._get_worker_units()
            logger.info(f"[INFO] Number of actors online: {worker_units}/{self.expected_worker_units}")
            print(
                f"[ray_bootstrap] Workers online: {worker_units}/{self.expected_worker_units}",
                flush=True,
            )

            if worker_units >= self.expected_worker_units:
                return

            _pause(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(
            f"Timed out waiting for all workers to connect after {self.max_wait_seconds}s. "
            f"Expected {self.expected_worker_units} worker_units."
        )

    def _get_worker_units(self) -> int:
        """Extract worker_units from ray status output.

        Returns:
            Total number of worker_units available in the cluster
        """
        try:
            result = subprocess.run(
                [self.ray_executable, "status"],
                capture_output=True,
                text=True,
                check=False,
                timeout=RAY_STATUS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"ray status timed out after {RAY_STATUS_TIMEOUT_SECONDS}s")
            print(f"[ray_bootstrap] ray status timed out after {RAY_STATUS_TIMEOUT_SECONDS}s", flush=True)
            return 0

        if result.returncode != 0:
            logger.warning(f"ray status failed: {result.stderr}")
            print(f"[ray_bootstrap] ray status failed: {result.stderr}", flush=True)
            return 0

        return self._parse_worker_units(result.stdout)

    @staticmethod
    def _parse_worker_units(status_output: str) -> int:
        """Parse worker_units from ray status output.

        The ray status output contains lines like:
            0.0/1.0 worker_units

        Where the format is: usage/total resource_name
        We want the TOTAL (second number) as that's how many worker_units are available.

        Args:
            status_output: Output from `ray status` command

        Returns:
            Total number of worker_units available in the cluster
        """
        # Match pattern: " 0.0/1.0 worker_units" - extract the total (second number)
        match = re.search(r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\s+worker_units", status_output)
        if match:
            total = int(float(match.group(2)))
            print(
                f"[ray_bootstrap] Parsed worker_units: {match.group(1)}/{match.group(2)} -> total={total}", flush=True
            )
            return total

        # Fallback: look for lines containing worker_units
        for line in status_output.splitlines():
            if "worker_units" in line:
                # Try to extract numbers from format "X/Y worker_units"
                parts = line.strip().split()
                if len(parts) >= 2 and "/" in parts[0]:
                    usage_total = parts[0].split("/")
                    if len(usage_total) == 2:
                        try:
                            total = int(float(usage_total[1]))
                            print(
                                f"[ray_bootstrap] Fallback parsed worker_units: {parts[0]} -> total={total}", flush=True
                            )
                            return total
                        except ValueError:
                            pass

        print("[ray_bootstrap] Could not parse worker_units from ray status", flush=True)
        return 0

    def _run_driver(self, driver_script: str, driver_args: list[str]) -> int:
        """Execute the Python driver script with output capture.

        Runs the driver as a subprocess, streaming output to console in real-time
        while capturing recent lines in a rolling buffer for error extraction.
        The captured output is available via the ``driver_output`` property.

        Args:
            driver_script: Path to the driver script
            driver_args: Arguments for the driver

        Returns:
            Exit code from the driver
        """
        cmd = [self.driver_python, driver_script] + driver_args
        logger.info(f"Running driver with python={self.driver_python}: {' '.join(cmd)}")

        # Build environment with extended PYTHONPATH if configured
        env = os.environ.copy()
        if self.driver_extra_pythonpath:
            existing_pythonpath = env.get("PYTHONPATH", "")
            if existing_pythonpath:
                env["PYTHONPATH"] = f"{existing_pythonpath}{os.pathsep}{self.driver_extra_pythonpath}"
            else:
                env["PYTHONPATH"] = self.driver_extra_pythonpath
            logger.info(f"Driver PYTHONPATH: {env['PYTHONPATH']}")

        # Reset the output buffer for this driver run
        self._driver_output.clear()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self._driver_process = process

        reader_thread = threading.Thread(
            target=read_subprocess_output,
            args=(process, self._driver_output),
            daemon=True,
        )
        reader_thread.start()

        try:
            process.wait()
        except BaseException:
            # If interrupted (e.g. SystemExit from signal handler), terminate the
            # driver process so it doesn't become orphaned.
            self.terminate_driver()
            raise
        finally:
            self._driver_process = None

        # Wait for reader thread to finish capturing remaining output
        if reader_thread.is_alive():
            reader_thread.join(timeout=5)

        return process.returncode

    def _monitor_for_termination(self) -> int:
        """Monitor for ENDED file and handle worker termination.

        Returns:
            Exit code (0 for graceful termination)
        """
        logger.info("Worker monitoring for termination signal")

        while not self._stop_event.is_set():
            if self.ended_file.exists():
                logger.info("Detected ENDED file, terminating worker...")
                self._stop_ray()
                return 0

            _pause(1)

        return 0

    def _signal_termination(self) -> None:
        """Signal termination by creating the ENDED file."""
        logger.info(f"Creating termination signal: {self.ended_file}")
        self.ended_file.touch()

    def _stop_ray(self, grace_period: int = 60, timeout: int | None = None) -> None:
        """Stop Ray with grace period.

        Args:
            grace_period: Seconds to wait for graceful shutdown.
            timeout: Hard wall-clock bound for the ``ray stop`` subprocess. Defaults
                to ``grace_period + 30``; callers under a tight cleanup budget pass a
                smaller value so the (non-daemon) cleanup thread can't keep the
                process alive past that budget.
        """
        effective_timeout = timeout if timeout is not None else grace_period + 30
        logger.info(f"Stopping Ray with {grace_period}s grace period")
        # Bound the call so a wedged `ray stop` can't keep the (non-daemon)
        # cleanup thread — and therefore the whole process — alive indefinitely.
        try:
            subprocess.run(
                [self.ray_executable, "stop", "--force", f"--grace-period={grace_period}"],
                check=False,
                capture_output=True,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"ray stop did not complete within {effective_timeout}s")

    def _cleanup_with_timeout(self, timeout: int = 30) -> None:
        """Cleanup with timeout, force kill if necessary.

        Args:
            timeout: Maximum seconds to wait for cleanup
        """
        logger.info(f"[INFO] Cleaning up Ray cluster from RANK {self.rank}")
        self._stop_event.set()
        self._signal_termination()

        def cleanup() -> None:
            # Keep stop + sleep within the outer `timeout` budget: bound `ray stop`
            # to timeout-10 and sleep the remaining 10, so the non-daemon cleanup
            # thread can't keep the process alive well past `timeout`.
            self._stop_ray(grace_period=20, timeout=max(1, timeout - 10))
            _pause(10)  # Wait for ray to stop
            logger.info("[INFO] Cleanup complete.")

        cleanup_thread = threading.Thread(target=cleanup)
        cleanup_thread.start()
        cleanup_thread.join(timeout=timeout)

        if cleanup_thread.is_alive():
            logger.warning("[WARN] Cleanup timed out. Forcing termination.")

    def _log_ray_status(self) -> None:
        """Log current Ray cluster status."""
        try:
            status_result = subprocess.run(
                [self.ray_executable, "status"],
                capture_output=True,
                text=True,
                check=False,
                timeout=RAY_STATUS_TIMEOUT_SECONDS,
            )
            logger.info(f"Ray status:\n{status_result.stdout}")

            memory_result = subprocess.run(
                [self.ray_executable, "memory"],
                capture_output=True,
                text=True,
                check=False,
                timeout=RAY_STATUS_TIMEOUT_SECONDS,
            )
            logger.info(f"Ray memory:\n{memory_result.stdout}")
        except Exception as e:
            logger.warning(f"Failed to log Ray status: {e}")

    def cleanup(self) -> None:
        """Public cleanup method."""
        self._cleanup_with_timeout()


def create_bootstrap_from_env() -> RayClusterBootstrap:
    """Create RayClusterBootstrap from environment variables.

    Expected environment variables:
        RANK: Node rank (0 for head, >0 for workers)
        WORLD_SIZE: Total number of nodes
        MASTER_ADDR: IP address of the head node
        GPUS_PER_NODE: Number of GPUs per node (optional, default 1)
        BASE_LOG_DIR: Base directory for logs (optional, default /tmp)

    Returns:
        Configured RayClusterBootstrap instance
    """
    return RayClusterBootstrap(
        rank=int(os.getenv("RANK", "0")),
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        master_addr=os.getenv("MASTER_ADDR", "127.0.0.1"),
        driver_python=os.getenv("DRIVER_PYTHON", sys.executable),
        driver_extra_pythonpath=os.getenv("DRIVER_EXTRA_PYTHONPATH", ""),
    )


def main() -> int:
    """Main entry point for Ray cluster bootstrap.

    This can be invoked directly to start a Ray cluster node, or
    with driver arguments to start the cluster and run a training script.

    Usage:
        # Start cluster node (head or worker based on RANK)
        python -m nhx.rl.tasks.training.backends.nemo_rl.ray_bootstrap

        # Start cluster and run driver
        python -m nhx.rl.tasks.training.backends.nemo_rl.ray_bootstrap \
            driver_script.py --config config.yaml --id job-123

    Returns:
        Exit code (0 for success)
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Ray cluster bootstrap")
    parser.add_argument(
        "driver_script",
        nargs="?",
        help="Optional driver script to run after cluster is ready",
    )
    parser.add_argument(
        "driver_args",
        nargs="*",
        help="Arguments to pass to the driver script",
    )

    args = parser.parse_args()

    bootstrap = create_bootstrap_from_env()

    # Setup signal handlers
    def signal_handler(signum: int, frame: FrameType | None) -> None:
        logger.warning(f"Received signal {signum}, initiating cleanup")
        bootstrap.cleanup()
        sys.exit(signum)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.driver_script:
        return bootstrap.run_with_driver(args.driver_script, args.driver_args)
    else:
        bootstrap.start()
        return 0


if __name__ == "__main__":
    sys.exit(main())
