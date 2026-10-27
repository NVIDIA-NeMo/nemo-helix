# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation of the ``name`` job-request field, on both the plugin request and the jobs-service create body."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from nemo_helix_plugin.entity_naming import NAME_MAX_LENGTH
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.api_factory import BaseJobRequest
from nemo_helix_plugin.jobs.types import (
    JOB_FILESET_NAME_PREFIX,
    JOB_NAME_MAX_LENGTH_WITH_DERIVED_FILESET,
    CreateHelixJobRequest,
    derive_job_fileset_name,
)
from pydantic import BaseModel, ValidationError


class _Spec(BaseModel):
    pass


def _plugin_request(name: str | None, output_location: str | None) -> BaseJobRequest[_Spec]:
    return BaseJobRequest[_Spec](spec=_Spec(), name=name, output_location=output_location)


def _create_request(name: str | None, output_location: str | None) -> CreateHelixJobRequest:
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


RequestFactory = Callable[[str | None, str | None], BaseJobRequest[_Spec] | CreateHelixJobRequest]
REQUEST_FACTORIES = pytest.mark.parametrize(
    "make_request", [_plugin_request, _create_request], ids=["plugin", "create"]
)


@REQUEST_FACTORIES
@pytest.mark.parametrize(
    "name",
    [
        None,
        "my-job",
        "nemo-evals.retrieve-bfbbh5ui",
        "a" * JOB_NAME_MAX_LENGTH_WITH_DERIVED_FILESET,
        # The full entity-name limit is usable without output_location: the service derives a
        # length-safe fileset name, so no caller-side reservation of the prefix budget is needed.
        "a" * NAME_MAX_LENGTH,
    ],
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
def test_full_length_name_is_accepted_without_output_location(make_request: RequestFactory) -> None:
    """A job name at the entity-name limit submits even without output_location.

    The service derives a length-safe ``job-fileset-`` name from it, so the name no longer has to
    reserve room for the prefix.
    """
    name = "a" * NAME_MAX_LENGTH
    assert make_request(name, None).name == name


@REQUEST_FACTORIES
def test_long_name_is_accepted_with_output_location(make_request: RequestFactory) -> None:
    """A caller-supplied fileset is used as-is, so no fileset name is derived from the job name."""
    name = "a" * NAME_MAX_LENGTH
    assert make_request(name, "shared-fs").name == name


_WS = "default"


def test_short_name_nests_verbatim_and_is_a_valid_fileset_name() -> None:
    """A job name short enough to nest verbatim keeps the historical ``job-fileset-<name>`` form."""
    name = "a" * JOB_NAME_MAX_LENGTH_WITH_DERIVED_FILESET
    derived = derive_job_fileset_name(name, _WS)
    assert derived == f"{JOB_FILESET_NAME_PREFIX}{name}"
    CreateFilesetRequest(name=derived)


def test_boundary_at_51_chars_is_verbatim_and_52_chars_is_hashed() -> None:
    """The two code paths meet exactly at the prefix budget: 51 verbatim, 52 hashed.

    A 51-char name nests verbatim into a 63-char fileset name; one more char forces the
    normalize-truncate-hash path, so the result is no longer the plain ``job-fileset-<name>``.
    """
    at_limit = "a" * JOB_NAME_MAX_LENGTH_WITH_DERIVED_FILESET
    over_limit = "a" * (JOB_NAME_MAX_LENGTH_WITH_DERIVED_FILESET + 1)

    verbatim = derive_job_fileset_name(at_limit, _WS)
    assert verbatim == f"{JOB_FILESET_NAME_PREFIX}{at_limit}"
    assert len(verbatim) == NAME_MAX_LENGTH

    hashed = derive_job_fileset_name(over_limit, _WS)
    assert hashed != f"{JOB_FILESET_NAME_PREFIX}{over_limit}"
    assert hashed.startswith(JOB_FILESET_NAME_PREFIX)
    assert len(hashed) <= NAME_MAX_LENGTH
    CreateFilesetRequest(name=hashed)


def test_short_name_with_dots_and_underscores_is_preserved_verbatim() -> None:
    """The short path preserves the full job-name charset (dots, underscores) byte-for-byte.

    The shared k8s label normalization would collapse these to hyphens; the short path must not
    invoke it, so a name a caller reconstructs by hand stays exactly ``job-fileset-<name>``.
    """
    name = "nemo-evals.retrieve_eval+v1"
    assert derive_job_fileset_name(name, _WS) == f"{JOB_FILESET_NAME_PREFIX}{name}"


@pytest.mark.parametrize("length", list(range(2, NAME_MAX_LENGTH + 1)))
def test_derived_fileset_name_is_valid_for_every_job_name_length(length: int) -> None:
    """Every valid job-name length yields a valid, within-limit fileset name."""
    name = "a" * length
    derived = derive_job_fileset_name(name, _WS)
    assert len(derived) <= NAME_MAX_LENGTH
    # Constructing the request applies NAME_PATTERN + the length limit, so this asserts validity.
    CreateFilesetRequest(name=derived)


def test_full_length_job_name_derives_within_limit_fileset_name() -> None:
    """The DoD: a job name at the 63-char limit yields a valid fileset name under the limit."""
    derived = derive_job_fileset_name("a" * NAME_MAX_LENGTH, _WS)
    assert len(derived) <= NAME_MAX_LENGTH
    assert derived.startswith(JOB_FILESET_NAME_PREFIX)
    CreateFilesetRequest(name=derived)


def test_long_name_with_dots_still_derives_a_valid_fileset_name() -> None:
    """An over-length name containing dots/underscores still normalizes to a valid fileset name."""
    name = "nemo-evals.retrieve_eval." + "x" * NAME_MAX_LENGTH
    name = name[:NAME_MAX_LENGTH].rstrip("-.")
    derived = derive_job_fileset_name(name, _WS)
    assert len(derived) <= NAME_MAX_LENGTH
    assert derived.startswith(JOB_FILESET_NAME_PREFIX)
    CreateFilesetRequest(name=derived)


def test_derivation_is_deterministic() -> None:
    """Same (workspace, job name) always derives the same fileset name, so retries resolve one fileset."""
    name = "b" * NAME_MAX_LENGTH
    assert derive_job_fileset_name(name, _WS) == derive_job_fileset_name(name, _WS)


def test_distinct_long_job_names_derive_distinct_fileset_names() -> None:
    """Two long job names that truncate to the same visible prefix stay distinct via the hash suffix."""
    base = "a" * (NAME_MAX_LENGTH - 2)
    assert derive_job_fileset_name(base + "bc", _WS) != derive_job_fileset_name(base + "de", _WS)


def test_same_long_name_in_different_workspaces_derives_distinct_names() -> None:
    """The identity hash keys on workspace too, so one long name in two workspaces stays distinct."""
    name = "a" * NAME_MAX_LENGTH
    assert derive_job_fileset_name(name, "ws-a") != derive_job_fileset_name(name, "ws-b")
