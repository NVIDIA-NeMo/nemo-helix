# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar
from unittest.mock import patch

from nemo_platform_ext.cli.app import app
from nemo_platform_ext.quickstart.config import QuickstartConfig
from nemo_platform_plugin.client.response import PageResult
from nemo_platform_plugin.client.types import OffsetPaginationMetadata
from nemo_platform_plugin.secrets.types import (
    ListSecretsQueryParams,
    PlatformSecretAccessResponse,
    PlatformSecretAdminRotationResponse,
    PlatformSecretCreateRequest,
    PlatformSecretResponse,
    PlatformSecretUpdateRequest,
)
from typer.testing import CliRunner, Result

T = TypeVar("T")

runner = CliRunner()
_qs_no_auth = QuickstartConfig(auth_enabled=False)


@dataclass(frozen=True)
class _Response(Generic[T]):
    body: T

    def data(self) -> T:
        return self.body


class _PaginatedSecretsResponse:
    def __init__(self, items: list[PlatformSecretResponse], metadata: OffsetPaginationMetadata) -> None:
        self._items = items
        self._metadata = metadata
        self.page_called = False
        self.items_called = False

    def page(self) -> PageResult[PlatformSecretResponse, OffsetPaginationMetadata]:
        self.page_called = True
        return PageResult(items=self._items, metadata=self._metadata)

    def items(self) -> Iterator[PlatformSecretResponse]:
        self.items_called = True
        yield from self._items


@dataclass(frozen=True)
class _CreateSecretCall:
    workspace: str | None
    body: PlatformSecretCreateRequest


@dataclass(frozen=True)
class _UpdateSecretCall:
    workspace: str | None
    name: str
    body: PlatformSecretUpdateRequest


@dataclass(frozen=True)
class _ListSecretsCall:
    workspace: str | None
    query_params: ListSecretsQueryParams | None


class _FakeSecretsClient:
    def __init__(self) -> None:
        self.create_calls: list[_CreateSecretCall] = []
        self.update_calls: list[_UpdateSecretCall] = []
        self.list_calls: list[_ListSecretsCall] = []
        self.get_calls: list[tuple[str, str | None]] = []
        self.access_calls: list[tuple[str, str | None]] = []
        self.delete_calls: list[tuple[str, str | None]] = []
        self.rotate_called = False
        self.list_response = _PaginatedSecretsResponse(
            [
                PlatformSecretResponse(name="hf-token", workspace="default"),
                PlatformSecretResponse(name="wandb-key", workspace="default"),
            ],
            {
                "page": 1,
                "page_size": 20,
                "current_page_size": 2,
                "total_pages": 1,
                "total_results": 2,
            },
        )

    def create_secret(
        self, *, workspace: str | None, body: PlatformSecretCreateRequest
    ) -> _Response[PlatformSecretResponse]:
        self.create_calls.append(_CreateSecretCall(workspace=workspace, body=body))
        return _Response(PlatformSecretResponse(name=body.name, workspace=workspace or "default"))

    def update_secret(
        self, *, workspace: str | None, name: str, body: PlatformSecretUpdateRequest
    ) -> _Response[PlatformSecretResponse]:
        self.update_calls.append(_UpdateSecretCall(workspace=workspace, name=name, body=body))
        return _Response(PlatformSecretResponse(name=name, workspace=workspace or "default"))

    def list_secrets(
        self, *, workspace: str | None, query_params: ListSecretsQueryParams | None = None
    ) -> _PaginatedSecretsResponse:
        self.list_calls.append(_ListSecretsCall(workspace=workspace, query_params=query_params))
        return self.list_response

    def get_secret(self, *, workspace: str | None, name: str) -> _Response[PlatformSecretResponse]:
        self.get_calls.append((name, workspace))
        return _Response(PlatformSecretResponse(name=name, workspace=workspace or "default"))

    def access_secret(self, *, workspace: str | None, name: str) -> _Response[PlatformSecretAccessResponse]:
        self.access_calls.append((name, workspace))
        return _Response(PlatformSecretAccessResponse(name=name, workspace=workspace or "default", value="secret"))

    def delete_secret(self, *, workspace: str | None, name: str) -> _Response[None]:
        self.delete_calls.append((name, workspace))
        return _Response(None)

    def rotate_encryption_keys(self) -> _Response[PlatformSecretAdminRotationResponse]:
        self.rotate_called = True
        return _Response(PlatformSecretAdminRotationResponse(rotated_secrets=0, success=True))


class _FakeSDK:
    pass


def _invoke(secrets: _FakeSecretsClient, args: list[str]) -> Result:
    with (
        patch("nemo_platform_ext.quickstart.QuickstartConfig.load", return_value=_qs_no_auth),
        patch("nemo_platform_ext.cli.core.context.CLIContext.get_client", return_value=_FakeSDK()),
        patch("nemo_platform_ext.cli.commands.secrets.client_from_platform", return_value=secrets),
    ):
        return runner.invoke(app, args)


def test_create_builds_typed_request_with_real_secret_value() -> None:
    secrets = _FakeSecretsClient()

    result = _invoke(
        secrets,
        [
            "secrets",
            "create",
            "hf-token",
            "--workspace",
            "default",
            "--value",
            "nvapi-real",
            "--description",
            "NVIDIA key",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(secrets.create_calls) == 1
    call = secrets.create_calls[0]
    assert call.workspace == "default"
    assert call.body.name == "hf-token"
    assert call.body.description == "NVIDIA key"
    assert call.body.value.get_secret_value() == "nvapi-real"
    assert "nvapi-real" not in result.stdout


def test_create_accepts_legacy_data_alias() -> None:
    secrets = _FakeSecretsClient()

    result = _invoke(
        secrets,
        [
            "secrets",
            "create",
            "hf-token",
            "--workspace",
            "default",
            "--data",
            "nvapi-real",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(secrets.create_calls) == 1
    call = secrets.create_calls[0]
    assert call.body.value.get_secret_value() == "nvapi-real"


def test_update_omits_secret_value_when_no_value_source_is_provided() -> None:
    secrets = _FakeSecretsClient()

    result = _invoke(
        secrets,
        [
            "secrets",
            "update",
            "hf-token",
            "--workspace",
            "default",
            "--description",
            "metadata only",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(secrets.update_calls) == 1
    call = secrets.update_calls[0]
    assert call.workspace == "default"
    assert call.name == "hf-token"
    assert call.body.description == "metadata only"
    assert call.body.value is None
    assert call.body.model_dump_json(exclude_unset=True) == '{"description":"metadata only"}'


def test_update_omits_description_when_only_value_source_is_provided() -> None:
    secrets = _FakeSecretsClient()

    result = _invoke(
        secrets,
        [
            "secrets",
            "update",
            "hf-token",
            "--workspace",
            "default",
            "--value",
            "new-secret",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(secrets.update_calls) == 1
    call = secrets.update_calls[0]
    assert call.body.value is not None
    assert call.body.value.get_secret_value() == "new-secret"
    assert call.body.description is None
    assert call.body.model_dump_json(exclude_unset=True) == '{"value":"new-secret"}'


def test_update_accepts_legacy_data_alias() -> None:
    secrets = _FakeSecretsClient()

    result = _invoke(
        secrets,
        [
            "secrets",
            "update",
            "hf-token",
            "--workspace",
            "default",
            "--data",
            "new-secret",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(secrets.update_calls) == 1
    call = secrets.update_calls[0]
    assert call.body.value is not None
    assert call.body.value.get_secret_value() == "new-secret"


def test_list_all_pages_consumes_typed_paginated_items() -> None:
    secrets = _FakeSecretsClient()

    result = _invoke(
        secrets,
        [
            "secrets",
            "list",
            "--workspace",
            "default",
            "--page-size",
            "20",
            "--all-pages",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert secrets.list_calls == [_ListSecretsCall(workspace="default", query_params={"page_size": 20})]
    assert secrets.list_response.items_called is True
    assert secrets.list_response.page_called is False
    assert "hf-token" in result.stdout
    assert "wandb-key" in result.stdout


def test_get_access_delete_and_admin_route_to_typed_client_methods() -> None:
    secrets = _FakeSecretsClient()

    get_result = _invoke(
        secrets,
        ["secrets", "get", "hf-token", "--workspace", "default", "--output-format", "json"],
    )
    access_result = _invoke(
        secrets,
        ["secrets", "access", "hf-token", "--workspace", "default", "--output-format", "json"],
    )
    delete_result = _invoke(secrets, ["secrets", "delete", "hf-token", "--workspace", "default"])
    rotate_result = _invoke(secrets, ["secrets", "admin", "rotate-encryption-keys", "--output-format", "json"])

    assert get_result.exit_code == 0, get_result.output
    assert access_result.exit_code == 0, access_result.output
    assert delete_result.exit_code == 0, delete_result.output
    assert rotate_result.exit_code == 0, rotate_result.output
    assert secrets.get_calls == [("hf-token", "default")]
    assert secrets.access_calls == [("hf-token", "default")]
    assert secrets.delete_calls == [("hf-token", "default")]
    assert secrets.rotate_called is True


def test_code_output_uses_typed_secrets_client() -> None:
    secrets = _FakeSecretsClient()

    result = _invoke(
        secrets,
        [
            "secrets",
            "create",
            "hf-token",
            "--workspace",
            "default",
            "--value",
            "nvapi-real",
            "--output-format",
            "code",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "client_from_platform(platform_client, SecretsClient)" in result.stdout
    assert "PlatformSecretCreateRequest" in result.stdout
    assert "SecretStr" in result.stdout
    assert "<secret-value>" in result.stdout
    assert "nvapi-real" not in result.stdout
    assert secrets.create_calls == []


def test_delete_code_output_uses_typed_secrets_client_without_deleting() -> None:
    secrets = _FakeSecretsClient()

    result = _invoke(
        secrets,
        [
            "secrets",
            "delete",
            "hf-token",
            "--workspace",
            "default",
            "--output-format",
            "code",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "client_from_platform(platform_client, SecretsClient)" in result.stdout
    assert "secrets.delete_secret(" in result.stdout
    assert 'name=str(args["name"])' in result.stdout
    assert secrets.delete_calls == []


def test_list_code_output_omits_none_query_params() -> None:
    secrets = _FakeSecretsClient()

    result = _invoke(
        secrets,
        [
            "secrets",
            "list",
            "--workspace",
            "default",
            "--page-size",
            "20",
            "--output-format",
            "code",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "query_params: ListSecretsQueryParams = {}" in result.stdout
    assert 'query_params["page_size"] = int(page_size)' in result.stdout
    assert "query_params=query_params or None" in result.stdout
    assert "page_result = response.page()" in result.stdout
    assert "response.items()" not in result.stdout
    assert "else None" not in result.stdout
    assert secrets.list_calls == []


def test_list_code_output_consumes_all_pages_only_when_requested() -> None:
    secrets = _FakeSecretsClient()

    result = _invoke(
        secrets,
        [
            "secrets",
            "list",
            "--workspace",
            "default",
            "--page-size",
            "20",
            "--all-pages",
            "--output-format",
            "code",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "for secret in response.items():" in result.stdout
    assert "page_result = response.page()" not in result.stdout
    assert secrets.list_calls == []


def test_update_code_output_preserves_unset_value_for_description_only_update() -> None:
    secrets = _FakeSecretsClient()

    result = _invoke(
        secrets,
        [
            "secrets",
            "update",
            "hf-token",
            "--workspace",
            "default",
            "--description",
            "metadata only",
            "--output-format",
            "code",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "body = PlatformSecretUpdateRequest(description=str(description))" in result.stdout
    assert "value=SecretStr(str(secret_value)) if secret_value is not None else None" not in result.stdout
    assert "description=args.get" not in result.stdout
    assert secrets.update_calls == []


def test_update_code_output_preserves_unset_description_for_value_only_update() -> None:
    secrets = _FakeSecretsClient()

    result = _invoke(
        secrets,
        [
            "secrets",
            "update",
            "hf-token",
            "--workspace",
            "default",
            "--value",
            "new-secret",
            "--output-format",
            "code",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "body = PlatformSecretUpdateRequest(value=SecretStr(str(secret_value)))" in result.stdout
    assert "value=SecretStr(str(secret_value)) if secret_value is not None else None" not in result.stdout
    assert "new-secret" not in result.stdout
    assert "<secret-value>" in result.stdout
    assert secrets.update_calls == []
