# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Syntactic validation of the ``output_location`` job-request field."""

from __future__ import annotations

import pytest
from nemo_helix_plugin.jobs.api_factory import BaseJobRequest
from nemo_helix_plugin.jobs.types import CreatePlatformJobRequest, job_artifact_base_path
from pydantic import BaseModel, ValidationError


class _Spec(BaseModel):
    pass


def _request(output_location: str | None) -> BaseJobRequest[_Spec]:
    return BaseJobRequest[_Spec](spec=_Spec(), output_location=output_location)


def test_bare_fileset_name_is_accepted() -> None:
    assert _request("my-eval-fileset").output_location == "my-eval-fileset"


def test_value_is_stripped() -> None:
    assert _request("  my-fileset  ").output_location == "my-fileset"


def test_none_is_allowed() -> None:
    assert _request(None).output_location is None


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_is_rejected(value: str) -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        _request(value)


def test_subpath_is_rejected() -> None:
    with pytest.raises(ValidationError, match="subpath in output_location is not yet supported"):
        _request("my-fileset#runs/2026")


def test_workspace_qualified_name_is_rejected() -> None:
    with pytest.raises(ValidationError, match="bare fileset name"):
        _request("default/my-fileset")


@pytest.mark.parametrize("value", ["my fileset", "my:fileset", "my*fileset", "a" * 256])
def test_invalid_fileset_name_is_rejected(value: str) -> None:
    """Names a fileset could never have are rejected before any lookup."""
    with pytest.raises(ValidationError):
        _request(value)


def test_create_request_applies_the_same_validation() -> None:
    """The internal jobs-service create body validates identically to the plugin request."""
    # Other fields are left out on purpose; only the output_location error matters here.
    with pytest.raises(ValidationError, match="must not be empty"):
        CreatePlatformJobRequest.model_validate({"output_location": "   "})


def test_artifact_base_path_nests_under_jobs_when_output_location_set() -> None:
    assert job_artifact_base_path("my-job", "shared-fs") == "jobs/my-job"


def test_artifact_base_path_is_none_when_job_owns_the_fileset() -> None:
    assert job_artifact_base_path("my-job", None) is None
