# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation of the ``name`` job-request field, on both the plugin request and the jobs-service create body."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.api_factory import BaseJobRequest
from nemo_helix_plugin.jobs.types import (
    JOB_FILESET_NAME_PREFIX,
    JOB_NAME_MAX_LENGTH_WITH_DERIVED_FILESET,
    CreateHelixJobRequest,
)
from pydantic import BaseModel, ValidationError


class _Spec(BaseModel):
    pass


def _plugin_request(name: str | None, output_location: str | None) -> BaseModel:
    return BaseJobRequest[_Spec](spec=_Spec(), name=name, output_location=output_location)


def _create_request(name: str | None, output_location: str | None) -> BaseModel:
    return CreateHelixJobRequest.model_validate(
        {
            "name": name,
            "output_location": output_location,
            "source": "test-source",
            "spec": {},
            "platform_spec": {
                "steps": [
                    {"name": "run", "executor": {"provider": "subprocess", "profile": "default", "command": ["true"]}}
                ]
            },
        }
    )


RequestFactory = Callable[[str | None, str | None], BaseModel]
REQUEST_FACTORIES = pytest.mark.parametrize(
    "make_request", [_plugin_request, _create_request], ids=["plugin", "create"]
)


@REQUEST_FACTORIES
@pytest.mark.parametrize(
    "name", [None, "my-job", "nemo-evals.retrieve-bfbbh5ui", "a" * JOB_NAME_MAX_LENGTH_WITH_DERIVED_FILESET]
)
def test_valid_name_is_accepted(make_request: RequestFactory, name: str | None) -> None:
    assert make_request(name, None).name == name


@REQUEST_FACTORIES
@pytest.mark.parametrize(
    "name", ["bad--name", "trailing-hyphen-", "Upper", "1starts-with-digit", "a", "my job", "a" * 64]
)
def test_invalid_name_is_rejected(make_request: RequestFactory, name: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        make_request(name, None)

    assert [error["loc"] for error in exc_info.value.errors()] == [("name",)]


@REQUEST_FACTORIES
def test_name_too_long_for_derived_fileset_is_rejected(make_request: RequestFactory) -> None:
    """Without ``output_location`` the job's fileset is ``job-fileset-<name>``, which caps the name at 51."""
    with pytest.raises(ValidationError, match="at most 51 characters when output_location is not set"):
        make_request("a" * (JOB_NAME_MAX_LENGTH_WITH_DERIVED_FILESET + 1), None)


@REQUEST_FACTORIES
def test_long_name_is_accepted_with_output_location(make_request: RequestFactory) -> None:
    """A caller-supplied fileset is used as-is, so the derived-fileset length cap does not apply."""
    name = "a" * 63
    assert make_request(name, "shared-fs").name == name


def test_longest_accepted_name_forms_a_valid_fileset_name() -> None:
    CreateFilesetRequest(name=f"{JOB_FILESET_NAME_PREFIX}{'a' * JOB_NAME_MAX_LENGTH_WITH_DERIVED_FILESET}")
