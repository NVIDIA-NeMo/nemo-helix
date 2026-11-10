# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import httpx
import pytest
from nemo_helix_plugin.client.errors import NemoResponseValidationError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_core import PydanticCustomError

VERSION_HINT = "The server and this client may be on different versions"


class Progress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    completed: int


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    progress: Progress


class Wide(BaseModel):
    a: int
    b: int
    c: int
    d: int
    e: int
    f: int
    g: int


def _response(body: object) -> httpx.Response:
    return httpx.Response(200, request=httpx.Request("GET", "http://test/apis/evals/v2/runs/r1?x=1"), json=body)


def _validation_error(model: type[BaseModel], body: object) -> ValidationError:
    with pytest.raises(ValidationError) as exc_info:
        model.model_validate(body)
    return exc_info.value


def test_extra_field_names_model_location_and_version_hint() -> None:
    body = {"id": "r1", "progress": {"completed": 1, "trials": 3}}
    error = _validation_error(Run, body)

    exc = NemoResponseValidationError(_response(body), error)

    assert str(exc) == (
        "Response from GET /apis/evals/v2/runs/r1 did not match Run (1 validation error(s)):\n"
        "  progress.trials: Extra inputs are not permitted (extra_forbidden)\n"
        f"{VERSION_HINT}; upgrade the CLI or check the deployed version."
    )
    assert exc.error is error
    assert exc.body == body
    assert exc.status_code == 200


def test_missing_field_adds_version_hint() -> None:
    body = {"progress": {"completed": 1}}

    message = str(NemoResponseValidationError(_response(body), _validation_error(Run, body)))

    assert "  id: Field required (missing)" in message
    assert VERSION_HINT in message


def test_type_error_has_no_version_hint() -> None:
    body = {"id": "r1", "progress": {"completed": "many"}}

    message = str(NemoResponseValidationError(_response(body), _validation_error(Run, body)))

    assert "  progress.completed: Input should be a valid integer" in message
    assert "(int_parsing)" in message
    assert VERSION_HINT not in message


class Referenced(BaseModel):
    fileset: str
    schema_ref: str

    @field_validator("fileset")
    @classmethod
    def _known_fileset(cls, value: str) -> str:
        raise ValueError(f"Fileset '{value}' not found.")

    @field_validator("schema_ref")
    @classmethod
    def _known_schema(cls, value: str) -> str:
        raise PydanticCustomError("unknown_schema", "Schema {ref} is unknown", {"ref": value})


def test_custom_validator_messages_are_not_printed() -> None:
    body = {"fileset": "secret-ws/private-files", "schema_ref": "secret-ws/private-schema"}

    message = str(NemoResponseValidationError(_response(body), _validation_error(Referenced, body)))

    assert "secret-ws" not in message
    assert "  fileset: Invalid value (message withheld: it can quote response data) (value_error)" in message
    assert "  schema_ref: Invalid value (message withheld: it can quote response data) (unknown_schema)" in message


class Cat(BaseModel):
    kind: Literal["cat"]


class Dog(BaseModel):
    kind: Literal["dog"]


class Pet(BaseModel):
    pet: Cat | Dog = Field(discriminator="kind")


def test_union_tag_from_the_response_is_not_printed() -> None:
    body = {"pet": {"kind": "secret-tag"}}

    message = str(NemoResponseValidationError(_response(body), _validation_error(Pet, body)))

    assert "secret-tag" not in message
    assert "(union_tag_invalid)" in message


def test_more_than_five_errors_are_capped() -> None:
    error = _validation_error(Wide, {})

    lines = str(NemoResponseValidationError(_response({}), error)).splitlines()

    assert lines[0] == "Response from GET /apis/evals/v2/runs/r1 did not match Wide (7 validation error(s)):"
    assert lines[1:6] == [f"  {name}: Field required (missing)" for name in "abcde"]
    assert lines[6] == "  +2 more"
    assert lines[7].startswith(VERSION_HINT)


def test_response_without_request_omits_source() -> None:
    error = _validation_error(Run, {})

    message = str(NemoResponseValidationError(httpx.Response(200, json={}), error))

    assert message.startswith("Response did not match Run (2 validation error(s)):")


def test_non_pydantic_error_keeps_generic_message() -> None:
    error = ValueError("Paginated responses must be JSON objects")

    exc = NemoResponseValidationError(_response([]), error)

    assert str(exc) == "Data returned by API is invalid for the expected schema"
    assert exc.error is error
