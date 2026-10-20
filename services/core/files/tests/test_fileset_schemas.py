# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the fileset entity -> response DTO converter and the fileset filter."""

import pytest
from nhx.common.api.filter import ComparisonOperation, FilterOperator
from nhx.core.files.api.v2.filesets.schemas import FilesetFilter, fileset_output_from_entity
from nhx.core.files.entities import Fileset


def _make_fileset(created_by: str | None, updated_by: str | None) -> Fileset:
    fileset = Fileset(
        name="test-fileset",
        workspace="default",
        storage={"type": "local", "path": "/tmp/test-fileset"},
        purpose="generic",
    )
    fileset._created_by = created_by
    fileset._updated_by = updated_by
    return fileset


@pytest.mark.parametrize(
    ("principal", "expected"),
    [("user@example.com", "user@example.com"), (None, None), ("", None)],
)
def test_fileset_output_carries_creator_and_updater(principal: str | None, expected: str | None):
    output = fileset_output_from_entity(_make_fileset(created_by=principal, updated_by=principal))

    assert output.created_by == expected
    assert output.updated_by == expected


@pytest.mark.parametrize("field", ["created_by", "updated_by"])
def test_fileset_filter_targets_principal_columns_not_data(field: str):
    operation = ComparisonOperation(operator=FilterOperator.EQ, field=field, value="user@example.com")

    translated = FilesetFilter.translate_operation(operation)

    assert isinstance(translated, ComparisonOperation)
    assert translated.field == field
