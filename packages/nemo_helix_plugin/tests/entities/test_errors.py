# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for entity client errors and the workspace_not_found code."""

import httpx
import pytest
from nemo_helix_plugin.client.errors import NotFoundError, UnprocessableEntityError
from nemo_helix_plugin.entities import EntityValidationError, is_workspace_not_found
from nemo_helix_plugin.entities.types import WORKSPACE_NOT_FOUND_CODE


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
