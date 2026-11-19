# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Upgrade legacy snapshot references at mutation and cleanup boundaries."""

import logging

from nemo_evals.jobs.agent_files_snapshot import discard_registered_agent_files, discard_registered_agent_files_sync
from nemo_evals.jobs.agent_spec import Target, registered_agent_files_receipt, registered_agent_source
from nemo_evals.jobs.utils import run_with_isolated_async_client
from nemo_evals.schema_migrations import migrate_schema_agent_target, migrate_schema_agent_target_sync
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient

logger = logging.getLogger(__name__)


async def discard_snapshot(target: Target | None, files: AsyncFilesClient) -> None:
    """Upgrade and discard a legacy snapshot without masking the operation's outcome."""
    try:
        target = await migrate_schema_agent_target(target, files_client=files)
        receipt = registered_agent_files_receipt(target)
        if receipt is not None:
            await discard_registered_agent_files(files, receipt)
    except Exception:
        logger.warning("Failed to resolve or clean up registered-agent snapshot", exc_info=True)


def discard_snapshot_sync(target: Target | None, client: NemoClient | AsyncNemoClient) -> None:
    """Clean up a worker's local snapshot metadata, leaving its immutable attempt alone."""
    source = registered_agent_source(target)
    if source is None or source.files is None:
        return
    if isinstance(client, AsyncNemoClient):
        run_with_isolated_async_client(
            client, lambda async_client: discard_snapshot(target, AsyncFilesClient.from_client(async_client))
        )
        return
    try:
        target = migrate_schema_agent_target_sync(target, files_client=FilesClient.from_client(client))
        receipt = registered_agent_files_receipt(target)
        if receipt is not None:
            discard_registered_agent_files_sync(client, receipt)
    except Exception:
        logger.warning("Failed to resolve or clean up registered-agent snapshot", exc_info=True)
