# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the fileset entity -> response DTO converter and the fileset filter."""

import pytest
from nhx.common.api.filter import ComparisonOperation, FilterOperator
from nhx.core.files.api.v2.filesets.schemas import FilesetFilter, FilesetOutput, fileset_output_from_entity
from nhx.core.files.entities import DEFAULT_REVISION, Fileset


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
def test_fileset_output_principal_fields_are_nullable_in_schema(field: str):
    schema = FilesetOutput.model_json_schema(mode="serialization")

    assert schema["properties"][field]["nullable"] is True


@pytest.mark.parametrize("field", ["created_by", "updated_by"])
def test_fileset_filter_targets_principal_columns_not_data(field: str):
    operation = ComparisonOperation(operator=FilterOperator.EQ, field=field, value="user@example.com")

    translated = FilesetFilter.translate_operation(operation)

    assert isinstance(translated, ComparisonOperation)
    assert translated.field == field


@pytest.mark.parametrize("blank", [False, True])
def test_unset_revisions_use_a_fixed_default(blank: bool):
    entity = _make_fileset(None, None)
    if blank:
        entity.revision = ""
        entity.commit_revision = ""
    entity._id = "first-fileset"
    output = fileset_output_from_entity(entity)
    assert output.revision == DEFAULT_REVISION
    assert output.commit_revision == DEFAULT_REVISION
    entity._id = "second-fileset"
    assert fileset_output_from_entity(entity).revision == DEFAULT_REVISION
    assert fileset_output_from_entity(entity).commit_revision == DEFAULT_REVISION
    entity.commit_revision = "committed-token"
    assert fileset_output_from_entity(entity).commit_revision == "committed-token"
    entity.revision = "assigned-token"
    assert fileset_output_from_entity(entity).revision == "assigned-token"
