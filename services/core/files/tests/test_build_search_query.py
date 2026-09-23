# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FilesetFilter entity field mapping."""

from nhx.core.files.api.v2.filesets.schemas import FilesetFilter
from nhx.core.files.entities import FilesetPurpose


def test_entity_field_map_contains_expected_mappings():
    field_map = FilesetFilter._get_entity_field_map()
    assert field_map["purpose"] == "data.purpose"
    assert field_map["storage_type"] == "data.storage.type"
    assert field_map["description"] == "data.description"
    # Base fields should not be in the map
    assert "name" not in field_map
    assert "created_at" not in field_map


def test_filter_has_description_field():
    f = FilesetFilter(description="llama")
    assert f.description == "llama"


def test_filter_has_purpose_field():
    f = FilesetFilter(purpose=FilesetPurpose.DATASET)
    assert f.purpose == FilesetPurpose.DATASET


def test_environment_fileset_purpose_is_supported():
    f = FilesetFilter(purpose=FilesetPurpose.ENVIRONMENT)
    assert f.purpose == FilesetPurpose.ENVIRONMENT
    assert f.purpose.value == "environment"


def test_filter_defaults_to_none():
    f = FilesetFilter()
    assert f.name is None
    assert f.description is None
    assert f.purpose is None
    assert f.storage_type is None
    assert f.created_at is None
    assert f.updated_at is None
