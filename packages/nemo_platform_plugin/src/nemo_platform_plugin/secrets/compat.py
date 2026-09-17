# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility facade for the legacy ``NeMoPlatform.secrets`` SDK resource."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import NoReturn, TypeVar

from nemo_platform_plugin.client.errors import NemoHTTPError
from nemo_platform_plugin.schema import PaginationData
from nemo_platform_plugin.secrets.client import AsyncSecretsClient, SecretsClient
from nemo_platform_plugin.secrets.types import (
    ListSecretsQueryParams,
    PlatformSecretAccessResponse,
    PlatformSecretAdminRotationResponse,
    PlatformSecretCreateRequest,
    PlatformSecretResponse,
    PlatformSecretResponsesPage,
    PlatformSecretUpdateRequest,
)
from pydantic import SecretStr

ResponseT = TypeVar("ResponseT")


def _raise_sdk_error(error: NemoHTTPError) -> NoReturn:
    """Map source-owned client HTTP errors back to the public SDK hierarchy."""
    from nemo_platform import _exceptions as sdk_exceptions

    error_cls = {
        400: sdk_exceptions.BadRequestError,
        401: sdk_exceptions.AuthenticationError,
        403: sdk_exceptions.PermissionDeniedError,
        404: sdk_exceptions.NotFoundError,
        409: sdk_exceptions.ConflictError,
        422: sdk_exceptions.UnprocessableEntityError,
        429: sdk_exceptions.RateLimitError,
    }.get(error.status_code)
    if error_cls is None:
        error_cls = sdk_exceptions.InternalServerError if error.status_code >= 500 else sdk_exceptions.APIStatusError
    raise error_cls(error.detail, response=error.http_response, body=error.body) from error


def _translate_client_errors(call: Callable[[], ResponseT]) -> ResponseT:
    try:
        return call()
    except NemoHTTPError as error:
        _raise_sdk_error(error)


async def _translate_async_client_errors(awaitable: Awaitable[ResponseT]) -> ResponseT:
    try:
        return await awaitable
    except NemoHTTPError as error:
        _raise_sdk_error(error)


def _list_query_params(*, page: int | None, page_size: int | None) -> ListSecretsQueryParams | None:
    query_params: ListSecretsQueryParams = {}
    if page is not None:
        query_params["page"] = page
    if page_size is not None:
        query_params["page_size"] = page_size
    return query_params or None


def _page_response(data: list[PlatformSecretResponse], metadata: object) -> PlatformSecretResponsesPage:
    return PlatformSecretResponsesPage(
        data=data,
        pagination=PaginationData.model_validate(metadata),
    )


def _create_body(*, name: str, value: str, description: str | None) -> PlatformSecretCreateRequest:
    if description is not None:
        return PlatformSecretCreateRequest(name=name, value=SecretStr(value), description=description)
    return PlatformSecretCreateRequest(name=name, value=SecretStr(value))


def _create_value(*, value: str | None, data: str | None) -> str:
    if value is not None and data is not None:
        raise TypeError("Pass either `value` or legacy `data`, not both")
    if value is not None:
        return value
    if data is not None:
        return data
    raise TypeError("Missing required argument: `value`")


def _update_value(*, value: str | None, data: str | None) -> str | None:
    if value is not None and data is not None:
        raise TypeError("Pass either `value` or legacy `data`, not both")
    return value if value is not None else data


def _update_body(*, description: str | None, value: str | None) -> PlatformSecretUpdateRequest:
    if description is not None and value is not None:
        return PlatformSecretUpdateRequest(description=description, value=SecretStr(value))
    if description is not None:
        return PlatformSecretUpdateRequest(description=description)
    if value is not None:
        return PlatformSecretUpdateRequest(value=SecretStr(value))
    return PlatformSecretUpdateRequest()


def _sync_client_from_platform(platform: object) -> SecretsClient:
    from nemo_platform import NeMoPlatform
    from nemo_platform_plugin.client.adapter import client_from_platform

    if not isinstance(platform, NeMoPlatform):
        raise TypeError("SecretsResource requires a NeMoPlatform client")
    return client_from_platform(platform, SecretsClient)


def _async_client_from_platform(platform: object) -> AsyncSecretsClient:
    from nemo_platform import AsyncNeMoPlatform
    from nemo_platform_plugin.client.adapter import client_from_platform

    if not isinstance(platform, AsyncNeMoPlatform):
        raise TypeError("AsyncSecretsResource requires an AsyncNeMoPlatform client")
    return client_from_platform(platform, AsyncSecretsClient)


class SecretsAdminResource:
    def __init__(self, client: SecretsClient) -> None:
        self._client = client

    def rotate_encryption_keys(self) -> PlatformSecretAdminRotationResponse:
        return _translate_client_errors(lambda: self._client.rotate_encryption_keys().data())


class AsyncSecretsAdminResource:
    def __init__(self, client: AsyncSecretsClient) -> None:
        self._client = client

    async def rotate_encryption_keys(self) -> PlatformSecretAdminRotationResponse:
        response = await _translate_async_client_errors(self._client.rotate_encryption_keys())
        return response.data()


class SecretsResource:
    """Legacy SDK-shaped secrets resource backed by the source-owned typed client."""

    def __init__(self, platform: object) -> None:
        self._client = _sync_client_from_platform(platform)
        self.admin = SecretsAdminResource(self._client)

    def create(
        self,
        *,
        workspace: str | None = None,
        name: str,
        value: str | None = None,
        data: str | None = None,
        description: str | None = None,
    ) -> PlatformSecretResponse:
        secret_value = _create_value(value=value, data=data)
        return _translate_client_errors(
            lambda: self._client.create_secret(
                workspace=workspace,
                body=_create_body(name=name, value=secret_value, description=description),
            ).data()
        )

    def retrieve(self, name: str, *, workspace: str | None = None) -> PlatformSecretResponse:
        return _translate_client_errors(lambda: self._client.get_secret(name=name, workspace=workspace).data())

    def update(
        self,
        name: str,
        *,
        workspace: str | None = None,
        description: str | None = None,
        value: str | None = None,
        data: str | None = None,
    ) -> PlatformSecretResponse:
        secret_value = _update_value(value=value, data=data)
        return _translate_client_errors(
            lambda: self._client.update_secret(
                name=name,
                workspace=workspace,
                body=_update_body(description=description, value=secret_value),
            ).data()
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
    ) -> PlatformSecretResponsesPage:
        def fetch_page() -> PlatformSecretResponsesPage:
            response = self._client.list_secrets(
                workspace=workspace,
                query_params=_list_query_params(page=page, page_size=page_size),
            )
            first_page = response.page()
            return _page_response(list(first_page.items), first_page.metadata)

        return _translate_client_errors(fetch_page)

    def delete(self, name: str, *, workspace: str | None = None) -> None:
        _translate_client_errors(lambda: self._client.delete_secret(name=name, workspace=workspace).data())

    def access(self, name: str, *, workspace: str | None = None) -> PlatformSecretAccessResponse:
        return _translate_client_errors(lambda: self._client.access_secret(name=name, workspace=workspace).data())


class AsyncSecretsResource:
    """Async legacy SDK-shaped secrets resource backed by the typed client."""

    def __init__(self, platform: object) -> None:
        self._client = _async_client_from_platform(platform)
        self.admin = AsyncSecretsAdminResource(self._client)

    async def create(
        self,
        *,
        workspace: str | None = None,
        name: str,
        value: str | None = None,
        data: str | None = None,
        description: str | None = None,
    ) -> PlatformSecretResponse:
        secret_value = _create_value(value=value, data=data)
        response = await _translate_async_client_errors(
            self._client.create_secret(
                workspace=workspace,
                body=_create_body(name=name, value=secret_value, description=description),
            )
        )
        return response.data()

    async def retrieve(self, name: str, *, workspace: str | None = None) -> PlatformSecretResponse:
        response = await _translate_async_client_errors(self._client.get_secret(name=name, workspace=workspace))
        return response.data()

    async def update(
        self,
        name: str,
        *,
        workspace: str | None = None,
        description: str | None = None,
        value: str | None = None,
        data: str | None = None,
    ) -> PlatformSecretResponse:
        secret_value = _update_value(value=value, data=data)
        response = await _translate_async_client_errors(
            self._client.update_secret(
                name=name,
                workspace=workspace,
                body=_update_body(description=description, value=secret_value),
            )
        )
        return response.data()

    async def list(
        self,
        *,
        workspace: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
    ) -> PlatformSecretResponsesPage:
        response = await _translate_async_client_errors(
            self._client.list_secrets(
                workspace=workspace,
                query_params=_list_query_params(page=page, page_size=page_size),
            )
        )
        first_page = response.page()
        return _page_response(list(first_page.items), first_page.metadata)

    async def delete(self, name: str, *, workspace: str | None = None) -> None:
        response = await _translate_async_client_errors(self._client.delete_secret(name=name, workspace=workspace))
        response.data()

    async def access(self, name: str, *, workspace: str | None = None) -> PlatformSecretAccessResponse:
        response = await _translate_async_client_errors(self._client.access_secret(name=name, workspace=workspace))
        return response.data()
