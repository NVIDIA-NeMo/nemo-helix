# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest
from nemo_helix_plugin.auth.workload_delegations import (
    SYSTEM_WORKSPACE,
    WorkloadDelegationEntity,
    WorkloadDelegationLookupScope,
    WorkloadDelegationScope,
    WorkloadDelegationStore,
)
from nemo_helix_plugin.entities import ListResponse, PaginationInfo
from nemo_helix_plugin.filter_ops import ComparisonOperation, LogicalOperation
from pydantic import ValidationError


def _empty_page() -> ListResponse[WorkloadDelegationEntity]:
    return ListResponse(
        data=[],
        pagination=PaginationInfo(
            page=1,
            page_size=100,
            current_page_size=0,
            total_pages=1,
            total_results=0,
        ),
    )


def test_lookup_scope_allows_missing_kind_but_rejects_empty_kind() -> None:
    scope = WorkloadDelegationLookupScope(
        workload_workspace="default",
        workload_kind=None,
        workload_instance_id="deployment-123",
    )

    assert scope.workload_kind is None
    with pytest.raises(ValidationError, match="workload_kind"):
        WorkloadDelegationLookupScope(
            workload_workspace="default",
            workload_kind="",
            workload_instance_id="deployment-123",
        )


def test_create_scope_requires_kind() -> None:
    with pytest.raises(ValidationError, match="workload_kind"):
        WorkloadDelegationScope.model_validate(
            {
                "workload_workspace": "default",
                "workload_instance_id": "deployment-123",
            }
        )


@pytest.mark.asyncio
async def test_list_by_workload_omits_kind_for_instance_only_scope() -> None:
    entity_client = AsyncMock()
    entity_client.list.return_value = _empty_page()

    result = await WorkloadDelegationStore(entity_client).list_by_workload(
        WorkloadDelegationLookupScope(
            workload_workspace="default",
            workload_kind=None,
            workload_instance_id="deployment-123",
        ),
    )

    assert result == []
    list_kwargs = entity_client.list.await_args.kwargs
    assert list_kwargs["workspace"] == SYSTEM_WORKSPACE
    filter_operation = list_kwargs["filter_operation"]
    assert isinstance(filter_operation, LogicalOperation)
    comparison_operations: list[ComparisonOperation] = []
    for operation in filter_operation.operations:
        assert isinstance(operation, ComparisonOperation)
        comparison_operations.append(operation)
    assert {operation.field: operation.value for operation in comparison_operations} == {
        "data.workload_workspace": "default",
        "data.workload_id": "deployment-123",
    }
