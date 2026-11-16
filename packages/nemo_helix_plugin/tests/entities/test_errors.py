# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for entity client errors and the workspace_not_found code."""

import httpx
import pytest
from nemo_helix_plugin.client.errors import (
    NemoHTTPError,
    NemoTransportError,
    NotFoundError,
    UnprocessableEntityError,
    raise_for_status,
)
from nemo_helix_plugin.entities import (
    CLIENT_ERRORS,
    EntityConflictError,
    EntityNotFoundError,
    EntityStoreError,
    EntityValidationError,
    is_workspace_not_found,
)
from nemo_helix_plugin.entities.types import WORKSPACE_NOT_FOUND_CODE
from pydantic import BaseModel, ValidationError


def _response(status: int, body: dict[str, str]) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("POST", "http://test"))


def test_validation_error_keeps_detail_and_code_from_response():
    error = UnprocessableEntityError(
        _response(422, {"detail": "Workspace 'ws' does not exist.", "code": WORKSPACE_NOT_FOUND_CODE})
    )

    converted = EntityValidationError.from_response(error)

    assert str(converted) == "Workspace 'ws' does not exist."
    assert converted.code == WORKSPACE_NOT_FOUND_CODE


def test_validation_error_without_code_in_response():
    converted = EntityValidationError.from_response(UnprocessableEntityError(_response(422, {"detail": "Invalid."})))

    assert str(converted) == "Invalid."
    assert converted.code is None


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(EntityValidationError("gone", code=WORKSPACE_NOT_FOUND_CODE), True, id="entity-client-error"),
        pytest.param(
            UnprocessableEntityError(_response(422, {"detail": "gone", "code": WORKSPACE_NOT_FOUND_CODE})),
            True,
            id="relayed-422",
        ),
        pytest.param(EntityValidationError("Project 'p' does not exist."), False, id="other-validation-error"),
        pytest.param(UnprocessableEntityError(_response(422, {"detail": "Invalid."})), False, id="422-without-code"),
        pytest.param(
            NotFoundError(_response(404, {"detail": "gone", "code": WORKSPACE_NOT_FOUND_CODE})), False, id="404"
        ),
        pytest.param(RuntimeError("boom"), False, id="unrelated"),
    ],
)
def test_is_workspace_not_found(error: BaseException, expected: bool):
    assert is_workspace_not_found(error) is expected


class _Strict(BaseModel):
    count: int


def _validation_error() -> ValidationError:
    try:
        _Strict.model_validate({"count": "x"})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


def _raised(status: int) -> NemoHTTPError:
    """The error the typed clients raise for an upstream response with this status."""
    try:
        raise_for_status(_response(status, {"detail": "upstream"}))
    except NemoHTTPError as exc:
        return exc
    raise AssertionError(f"expected HTTP {status} to raise")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        *(
            pytest.param(_raised(status), True, id=f"http-{status}")
            for status in (400, 401, 403, 404, 405, 409, 410, 422, 429)
        ),
        pytest.param(EntityNotFoundError("gone"), True, id="entity-not-found"),
        pytest.param(EntityConflictError("exists"), True, id="entity-conflict"),
        pytest.param(EntityValidationError("invalid"), True, id="entity-validation"),
        *(pytest.param(_raised(status), False, id=f"http-{status}") for status in (500, 502, 503)),
        pytest.param(
            NemoTransportError(httpx.ConnectError("refused", request=httpx.Request("GET", "http://test"))),
            False,
            id="transport",
        ),
        pytest.param(EntityStoreError("store failure"), False, id="bare-entity-store-error"),
        pytest.param(_validation_error(), False, id="pydantic-validation-error"),
        pytest.param(ValueError("internal"), False, id="value-error"),
    ],
)
def test_client_errors(error: BaseException, expected: bool):
    """Only caller mistakes bypass a route's catch-all 500; server failures do not."""
    assert isinstance(error, CLIENT_ERRORS) is expected
