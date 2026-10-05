# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator service dependency wiring."""

from unittest.mock import AsyncMock

from nemo_evaluator.api.dependencies import get_taskset_service


async def test_taskset_service_adapts_request_authorizer(entity_store) -> None:
    """Task membership checks use the caller-scoped HTTP authorizer."""
    authorize = AsyncMock()
    service = get_taskset_service(entity_store, AsyncMock(), authorize)

    await service.authorize_task_read("source workspace", "task/name")

    authorize.assert_awaited_once_with("GET", "/apis/evaluator/v2/workspaces/source%20workspace/tasks/task%2Fname")
