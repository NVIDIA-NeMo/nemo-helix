# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared request-validation handler."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.common.errors.validation_exception_handler import register_validation_exception_handler
from pydantic import BaseModel, Field, JsonValue


class _Body(BaseModel):
    count: int = Field(ge=0)
    ratio: float = Field(le=1)
    attributes: dict[str, str] = Field(default_factory=dict)


def _app(*, with_handler: bool) -> FastAPI:
    app = FastAPI()
    if with_handler:
        register_validation_exception_handler(app)

    @app.post("/")
    def create(body: _Body) -> dict[str, JsonValue]:
        return body.model_dump()

    return app


def _post(app: FastAPI, content: str | bytes, content_type: str = "application/json"):
    return TestClient(app, raise_server_exceptions=False).post(
        "/", content=content, headers={"Content-Type": content_type}
    )


@pytest.mark.parametrize(
    "content",
    [
        pytest.param('{"count": -1, "ratio": 0.5}', id="value out of range"),
        pytest.param('{"ratio": 0.5}', id="missing field"),
        pytest.param('{"count": 1, "ratio": 0.5, "attributes": {"x": 1}}', id="wrong type"),
        pytest.param("{not json", id="malformed JSON"),
    ],
)
def test_an_encodable_error_keeps_fastapis_default_body(content):
    """Every body FastAPI could already send stays exactly as it was, input and ctx included."""
    response = _post(_app(with_handler=True), content)
    default = _post(_app(with_handler=False), content)

    assert response.status_code == 422
    assert response.json() == default.json()


@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        pytest.param('{"count": 1, "ratio": NaN}', "application/json", id="NaN"),
        pytest.param('{"count": 1, "ratio": Infinity}', "application/json", id="infinity"),
        pytest.param(
            '{"count": -Infinity, "ratio": 2}', "application/json", id="negative infinity beside an encodable error"
        ),
        pytest.param(r'{"count": "\ud800", "ratio": 0.5}', "application/json", id="lone surrogate"),
        pytest.param(
            r'{"count": 1, "ratio": 0.5, "attributes": ["\udfff"]}',
            "application/json",
            id="lone surrogate in a nested input",
        ),
        pytest.param(b"\xff", "application/octet-stream", id="non-JSON body that isn't UTF-8"),
    ],
)
def test_an_error_echoing_an_unencodable_value_is_422_without_the_echo(content, content_type):
    """A response can't carry these values, so FastAPI's default answered 500."""
    response = _post(_app(with_handler=True), content, content_type)

    assert _post(_app(with_handler=False), content, content_type).status_code == 500
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert errors
    for error in errors:
        assert set(error) == {"type", "loc", "msg"}
