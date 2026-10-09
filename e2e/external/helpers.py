# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for running e2e tests against external clusters."""

import logging
import time

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.secrets.client import SecretsClient

logger = logging.getLogger(__name__)


def _wait_for_workspace_ready(
    client: NemoClient,
    workspace_name: str,
    timeout: float = 60.0,
    min_successes: int = 3,
) -> None:
    """Poll until the workspace role binding has propagated to all OPA instances.

    Auth is embedded in each service pod via OPA WASM that refreshes every ~30s.
    workspaces.create() already waits for one replica to be ready, but other
    replicas refresh independently.

    We probe files and secrets directly and require min_successes consecutive
    successful responses from each.  With round-robin load balancing across N
    replicas, min_successes responses will have hit every replica at least once
    when min_successes >= N.  This is adaptive: fast on single-replica clusters,
    patient on multi-replica ones, and never wastes time sleeping unnecessarily.
    """
    files_client = FilesClient.from_client(client)
    secrets_client = SecretsClient.from_client(client)
    start = time.time()
    deadline = start + timeout
    files_streak = 0
    secrets_streak = 0

    logger.info(
        "waiting for auth to propagate to workspace '%s' (need %d consecutive successes each)",
        workspace_name,
        min_successes,
    )

    while time.time() < deadline:
        elapsed = time.time() - start

        if files_streak < min_successes:
            try:
                files_client.list_filesets(workspace=workspace_name).page()
                files_streak += 1
                logger.info("files probe %d/%d OK (%.1fs elapsed)", files_streak, min_successes, elapsed)
            except Exception as e:
                logger.debug("files probe failed (%.1fs elapsed): %s", elapsed, e)
                files_streak = 0

        if secrets_streak < min_successes:
            try:
                secrets_client.list_secrets(workspace=workspace_name).page()
                secrets_streak += 1
                logger.info("secrets probe %d/%d OK (%.1fs elapsed)", secrets_streak, min_successes, elapsed)
            except Exception as e:
                logger.debug("secrets probe failed (%.1fs elapsed): %s", elapsed, e)
                secrets_streak = 0

        if files_streak >= min_successes and secrets_streak >= min_successes:
            logger.info("workspace '%s' ready (%.1fs)", workspace_name, time.time() - start)
            return

        time.sleep(1)

    logger.warning(
        "timed out after %.1fs waiting for workspace '%s' (files_streak=%d, secrets_streak=%d)",
        time.time() - start,
        workspace_name,
        files_streak,
        secrets_streak,
    )


def wait_for_deployment_status(
    client: NemoClient,
    deployment_name: str,
    workspace: str,
    expected_status: str,
    timeout: float = 60.0,
    poll_interval: float = 2.0,
) -> None:
    """Poll until a model deployment reaches the expected status.

    Args:
        client: Typed platform client.
        deployment_name: Name of the deployment to poll.
        workspace: Workspace containing the deployment.
        expected_status: Target status to wait for (e.g. "READY").
        timeout: Maximum seconds to wait before failing.
        poll_interval: Seconds between polls.

    Raises:
        pytest.fail: If the deployment reaches ERROR state or the timeout expires.
    """
    import pytest

    models_client = ModelsClient.from_client(client)
    start = time.time()
    deadline = start + timeout
    last_status: str | None = None

    while time.time() < deadline:
        deployment = models_client.get_deployment(name=deployment_name, workspace=workspace).data()
        last_status = deployment.status.value
        if last_status == expected_status:
            logger.info("deployment '%s' reached %s (%.1fs)", deployment_name, expected_status, time.time() - start)
            return
        if last_status == "ERROR":
            pytest.fail(f"Deployment '{deployment_name}' reached ERROR state")
        logger.debug("deployment '%s' status=%s, waiting for %s", deployment_name, last_status, expected_status)
        time.sleep(poll_interval)

    pytest.fail(
        f"Timeout: deployment '{deployment_name}' did not reach {expected_status} within {timeout}s "
        f"(last status: {last_status})"
    )
