# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fileset names must satisfy the entity store's NAME_PATTERN, not a looser one."""

from __future__ import annotations

import pytest
from nemo_helix_plugin.files.types import NAME_MAX_LENGTH, NAME_PATTERN, CreateFilesetRequest
from pydantic import ValidationError
from pydantic_core import ErrorDetails


def name_errors(name: str) -> list[ErrorDetails]:
    try:
        CreateFilesetRequest(name=name)
    except ValidationError as exc:
        return [err for err in exc.errors() if err["loc"] == ("name",)]
    return []


@pytest.mark.parametrize(
    "name",
    ["Training-Data", "1dataset", "my--dataset", "mydataset-", "a", "x" * 64, "has space", "with/slash"],
)
def test_rejects_names_the_entity_store_would_reject(name):
    assert name_errors(name), f"accepted {name!r}"


@pytest.mark.parametrize("name", ["training-data-v1", "llama-checkpoint", "ab", "ngc_cli"])
def test_accepts_valid_names(name):
    assert not name_errors(name)


def test_advertises_the_entity_store_pattern():
    schema = CreateFilesetRequest.model_json_schema()["properties"]["name"]
    assert schema["pattern"] == NAME_PATTERN
    assert schema["maxLength"] == NAME_MAX_LENGTH


def test_older_fileset_response_has_no_commit_revision():
    from nemo_helix_plugin.files.types import FilesetOutput

    result = FilesetOutput.model_validate(
        {
            "id": "id",
            "name": "files",
            "workspace": "default",
            "description": "",
            "purpose": "generic",
            "storage": {"type": "local", "path": "/tmp/files"},
            "metadata": {},
            "custom_fields": {},
            "project": "",
            "created_at": "",
            "updated_at": "",
        }
    )
    assert result.commit_revision is None
