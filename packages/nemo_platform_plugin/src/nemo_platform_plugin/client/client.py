# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP client for NeMo Platform.

Sends :class:`~.endpoint.PreparedRequest` objects and returns typed
responses.  The return type of :meth:`send` is determined by the endpoint's
``ResponseT``:

- ``BaseModel`` → :class:`~.response.NemoResponse[T]`
- ``None`` → :class:`~.response.NemoResponse[None]`
- ``BinaryContent`` → :class:`~.response.NemoBinaryResponse`
- ``Stream[T]`` → :class:`~.response.NemoStreamResponse[T]`
- ``Paginated[T]`` → :class:`~.response.NemoPaginatedResponse[T]`
"""

from __future__ import annotations

import asyncio
import copy
import email.utils
import json
import logging
import os
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from datetime import timezone
from functools import cache
from pathlib import Path
from typing import Any, Generic, Self, TypeVar, cast, get_args, get_origin, overload
from urllib.parse import quote, urlsplit

import httpx
from nemo_platform_plugin.client.auth import (
    AsyncTokenProvider,
    StaticToken,
    TokenProvider,
    TokenProviderAuth,
    resolve_token_async,
)
from nemo_platform_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_platform_plugin.client.errors import (
    NemoResponseValidationError,
    NemoTransportError,
    raise_for_status,
)
from nemo_platform_plugin.client.response import (
    AsyncNemoBinaryResponse,
    AsyncNemoPaginatedResponse,
    AsyncNemoStreamResponse,
    AsyncPageFetcher,
    NemoBinaryResponse,
    NemoPaginatedResponse,
    NemoResponse,
    NemoStreamResponse,
    SyncPageFetcher,
)
from nemo_platform_plugin.client.types import (
    BinaryContent,
    OffsetPagination,
    Paginated,
    PaginationStrategy,
    PreparedRequest,
    ResponseT,
    RetryPolicy,
    StrategyT,
    Stream,
)
from pydantic import BaseModel, TypeAdapter, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)
HttpClientT = TypeVar("HttpClientT", httpx.Client, httpx.AsyncClient)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60.0
_AUTHORIZATION_HEADER = "Authorization"
_PRINCIPAL_ID_HEADER = "X-NMP-Principal-Id"
_DEFAULT_ORIGIN_PORTS = {"http": 80, "https": 443}


def _has_header(headers: Mapping[str, str] | None, name: str) -> bool:
    if not headers:
        return False
    normalized = name.lower()
    return any(header.lower() == normalized for header in headers)


def _url_origin(url: str) -> tuple[str, str, int | None] | None:
    parsed = urlsplit(url)
    if not parsed.scheme or parsed.hostname is None:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    return scheme, parsed.hostname, port if port is not None else _DEFAULT_ORIGIN_PORTS.get(scheme)


@overload
def _resolve_implicit_workload_auth(
    *,
    base_url: str,
    auth: TokenProvider | str | None,
    default_headers: Mapping[str, str] | None,
    allow_env_bootstrap: bool,
) -> TokenProvider | str | None: ...


@overload
def _resolve_implicit_workload_auth(
    *,
    base_url: str,
    auth: TokenProvider | AsyncTokenProvider | str | None,
    default_headers: Mapping[str, str] | None,
    allow_env_bootstrap: bool,
) -> TokenProvider | AsyncTokenProvider | str | None: ...


def _resolve_implicit_workload_auth(
    *,
    base_url: str,
    auth: TokenProvider | AsyncTokenProvider | str | None,
    default_headers: Mapping[str, str] | None,
    allow_env_bootstrap: bool,
) -> TokenProvider | AsyncTokenProvider | str | None:
    if auth is not None or not allow_env_bootstrap:
        return auth
    if _has_header(default_headers, _AUTHORIZATION_HEADER) or _has_header(default_headers, _PRINCIPAL_ID_HEADER):
        return None

    subject_token_file = os.environ.get(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR)
    if not subject_token_file:
        return None

    from nemo_platform_plugin.client.oidc_factory import resolve_workload_exchange_provider

    return resolve_workload_exchange_provider(
        base_url=base_url,
        subject_token_file=Path(subject_token_file),
    )


@cache
def _type_adapter(response_type: type[ResponseT]) -> TypeAdapter[ResponseT]:
    """Build each response annotation's validation schema once."""
    return TypeAdapter(response_type)


def _parse_json_body(response_type: type[ResponseT], data: object) -> ResponseT:
    """Parse a decoded JSON body against an endpoint's return annotation.

    ``TypeAdapter`` handles both model classes and arbitrary annotations such as
    ``list[Profile]`` while preserving the annotation's type for callers.
    """
    return _type_adapter(response_type).validate_python(data)


def _parse_response_body(response_type: type[ResponseT], response: httpx.Response) -> ResponseT:
    """Decode and validate a response, normalizing contract failures."""
    try:
        return _parse_json_body(response_type, response.json())
    except (ValueError, ValidationError) as exc:
        raise NemoResponseValidationError(response, exc) from exc


def _get_stream_model_type(response_type: type[Stream[ModelT]]) -> type[ModelT]:
    """Extract the ModelT from a Stream[ModelT] generic alias."""
    args = get_args(response_type)
    if not args:
        raise TypeError(f"Stream response type must be parameterized, got {response_type}")
    return cast(type[ModelT], args[0])


def _get_paginated_types(
    response_type: type[Paginated[ModelT, StrategyT]],
) -> tuple[type[ModelT], type[StrategyT]]:
    """Extract (ModelT, StrategyT) from a Paginated[ModelT, StrategyT] generic alias."""
    args = get_args(response_type)
    if not args:
        raise TypeError(f"Paginated response type must be parameterized, got {response_type}")
    model_type = args[0]
    strategy_type = args[1] if len(args) > 1 else OffsetPagination
    return cast(type[ModelT], model_type), cast(type[StrategyT], strategy_type)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


def _retry_after(response: httpx.Response) -> float | None:
    """Parse a reasonable server-requested retry delay in seconds."""
    retry_after_ms = response.headers.get("retry-after-ms")
    try:
        delay = float(retry_after_ms) / 1000
    except (TypeError, ValueError):
        retry_after = response.headers.get("retry-after")
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            # Retry-After may instead be an RFC 5322 / HTTP-date, e.g.
            # "Fri, 31 Dec 2027 23:59:59 GMT"; convert that to a delta.
            try:
                retry_date = email.utils.parsedate_to_datetime(retry_after)
            except (TypeError, ValueError, OverflowError):
                return None
            if retry_date.tzinfo is None:
                retry_date = retry_date.replace(tzinfo=timezone.utc)
            try:
                delay = retry_date.timestamp() - time.time()
            except (OverflowError, OSError, ValueError):
                return None
    return delay if 0 < delay <= 60 else None


# Transport failures httpx raises before it starts reading the request body: the
# connection has to exist before any body byte can be written. A one-shot body is
# therefore still untouched when one of these surfaces, so the request can be sent
# again as-is. Everything else — a read/write timeout, a dropped connection, a
# protocol error — can land mid-body, where a replay would send a truncated body
# under the original ``Content-Length``.
_PRE_BODY_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ProxyError,
    httpx.UnsupportedProtocol,
)


def _is_replayable(content: bytes | Iterable[bytes] | AsyncIterable[bytes] | None) -> bool:
    """Whether a request body can be handed to httpx more than once.

    ``bytes``, ``str``, no body at all, and in-memory sequences replay fine —
    httpx re-reads them from the start on every attempt.

    A generator, file object or other one-shot iterable does not: httpx drains it
    on the first attempt, so a replay sends a short body while the original
    ``Content-Length`` still stands on the request, and h11 aborts with ``Too
    little data for declared Content-Length`` — masking whatever actually failed
    the first time. Such a body may only be re-sent while it is still untouched;
    once it has been read, the retry has to happen a level up, where the caller
    can build a fresh iterator over the source.

    ``bytearray`` and ``memoryview`` are deliberately absent: httpx treats
    anything that is not ``bytes``/``str`` as an iterable of chunks, and
    iterating either of those yields ``int``, so it rejects them on the first
    attempt regardless of what this returns.
    """
    return content is None or isinstance(content, (bytes, str, list, tuple))


def _should_retry(
    request: PreparedRequest,
    response: httpx.Response | None,
    exc: httpx.TransportError | None,
    attempt: int,
    policy: RetryPolicy,
) -> float | None:
    """Decide whether to retry and return the backoff duration, or None to stop.

    Shared decision logic used by both sync and async retry paths.
    Returns the sleep duration if a retry should happen, or ``None`` if
    the response should be returned / the exception re-raised.

    The policy decides first, without regard for the body; only then does the
    body get a veto. Keeping that order means a one-shot body (see
    :func:`_is_replayable`) is reported as the reason a retry stopped exactly
    when it is the reason, rather than on every attempt that was never going to
    be retried anyway.
    """
    if attempt >= policy.max_retries:
        return None

    backoff = policy.backoff_base * (2**attempt)
    if exc is not None:
        # A connection that never opened leaves the body untouched; anything
        # later can land mid-body. See :data:`_PRE_BODY_TRANSPORT_ERRORS`.
        body_is_spent = not isinstance(exc, _PRE_BODY_TRANSPORT_ERRORS)
    elif response is None:
        return None
    else:
        # A response only arrives once the body has gone out on the wire.
        body_is_spent = True
        decision = response.headers.get("x-should-retry") if policy.respect_retry_decision_headers else None
        if policy.respect_retry_decision_headers and response.status_code < 400:
            return None
        if decision == "false":
            return None
        if decision != "true":
            # No explicit verdict from the server, so fall back to the status code.
            retryable_status = response.status_code in policy.retryable_status_codes
            if policy.retry_all_server_errors and response.status_code >= 500:
                retryable_status = True
            if not retryable_status:
                return None
        if policy.respect_retry_after_headers:
            backoff = _retry_after(response) or backoff

    if body_is_spent and not _is_replayable(request.content):
        logger.info(
            "Not retrying %s %s: the request body is a one-shot stream that has already been read, "
            "so it cannot be sent again. Retry at a level that can rebuild it.",
            request.method,
            request.path_template,
        )
        return None
    return backoff


def _should_resolve_conflict(response: httpx.Response, request: PreparedRequest) -> bool:
    """Whether a 409 should be resolved by replaying the linked retrieve request.

    True when the create was sent with ``exist_ok=True``, the server responded
    409 Conflict, and the endpoint declared a ``get_on_conflict`` resolver (whose
    prebuilt GET is on ``request.on_conflict_get``). In that case ``send()``
    replays that GET and returns the existing entity instead of raising.
    """
    if response.status_code != 409:
        return False
    if not (request.client_options or {}).get("exist_ok"):
        return False
    if request.on_conflict_get is None:
        raise ValueError(
            "exist_ok=True was set on a create request whose endpoint declares no "
            "get_on_conflict resolver, so the existing entity cannot be retrieved "
            "on conflict. Add get_on_conflict=<resolver> to the @post endpoint."
        )
    return True


class _InferenceNamespace:
    """Compat shim for the legacy ``sdk.inference.*`` resource tree.

    The old Stainless SDK exposed ``sdk.inference.providers``, ``.deployments``,
    ``.deployment_configs``, and ``.virtual_models`` as sub-resources.  In the
    typed-client world providers/deployments/deployment_configs live on
    :class:`ModelsClient` and virtual_models on :class:`VirtualModelsClient`.
    This namespace dispatches accordingly so existing ``sdk.inference.<x>``
    call sites keep working while callers migrate to ``sdk.models.<method>``.

    Each accessor picks the sync or async resource client to match the owning
    client, so an :class:`AsyncNemoClient` never routes its ``httpx.AsyncClient``
    through a synchronous ``send()``.
    """

    def __init__(self, client: NemoClient | AsyncNemoClient) -> None:
        self._client = client

    def _models_client(self) -> NemoClient | AsyncNemoClient:
        from nemo_platform_plugin.models.client import AsyncModelsClient, ModelsClient

        if isinstance(self._client, AsyncNemoClient):
            return AsyncModelsClient.from_client(self._client)
        return ModelsClient.from_client(self._client)

    @property
    def providers(self) -> NemoClient | AsyncNemoClient:
        return self._models_client()

    @property
    def deployments(self) -> NemoClient | AsyncNemoClient:
        return self._models_client()

    @property
    def deployment_configs(self) -> NemoClient | AsyncNemoClient:
        return self._models_client()

    @property
    def virtual_models(self) -> NemoClient | AsyncNemoClient:
        from nemo_platform_plugin.virtual_models.client import AsyncVirtualModelsClient, VirtualModelsClient

        if isinstance(self._client, AsyncNemoClient):
            return AsyncVirtualModelsClient.from_client(self._client)
        return VirtualModelsClient.from_client(self._client)


class BaseNemoClient(Generic[HttpClientT]):
    """Shared logic for sync and async NeMo clients.

    Handles URL construction and request serialisation.
    Subclasses provide the actual HTTP transport (sync or async).
    """

    _http: HttpClientT
    _owns_http: bool

    def __init__(
        self,
        *,
        base_url: str,
        workspace: str | None = None,
        auth: TokenProvider | AsyncTokenProvider | str | None = None,
        retry: RetryPolicy | None = None,
        default_headers: Mapping[str, str] | None = None,
        timeout: float | httpx.Timeout | None = None,
        url_resolver: Callable[[str], str | httpx.URL] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._workspace = workspace
        self._auth: TokenProvider | AsyncTokenProvider | None = StaticToken(auth) if isinstance(auth, str) else auth
        self._retry = retry
        self._default_headers = dict(default_headers) if default_headers else {}
        self._url_resolver = url_resolver
        self._timeout: float | httpx.Timeout | None = timeout

    @property
    def base_url(self) -> str:
        return self._base_url

    def is_platform_url(self, url: str) -> bool:
        """Return whether *url* has the same origin as this client's platform base URL."""
        origin = _url_origin(url)
        return origin is not None and origin == _url_origin(self._base_url)

    @property
    def _client(self) -> httpx.Client | httpx.AsyncClient:
        """Underlying httpx transport.

        Legacy plugin SDK resources access ``NeMoPlatform._client`` to make raw
        HTTP calls. Exposing the same transport property here lets those
        resources work with ``NemoClient`` while they migrate.
        """
        return self._http

    @property
    def default_headers(self) -> dict[str, str]:
        """Default headers sent with every request."""
        return self._default_headers or {}

    @property
    def workspace(self) -> str | None:
        return self._workspace

    def resolve_workspace(self, workspace: str | None = None) -> str:
        """Return the explicit workspace, client default, or platform default."""
        return workspace or self._workspace or "default"

    def require_workspace(self, workspace: str | None = None) -> str:
        """Return the explicit workspace or client default, raising if neither is set."""
        resolved_workspace = workspace or self._workspace
        if resolved_workspace is None:
            raise ValueError("workspace must be provided when the client has no default workspace")
        return resolved_workspace

    @property
    def retry(self) -> RetryPolicy | None:
        return self._retry

    def _resolve_retry(self, retry: RetryPolicy | None) -> RetryPolicy | None:
        """Resolve retry policy: per-call override > client default."""
        if retry is not None:
            return retry
        return self._retry

    def _resolve_path(self, request: PreparedRequest) -> str:
        """Resolve path template with client defaults and explicit params.

        Client-level defaults (e.g. workspace) are merged under explicit
        params — explicit always wins.  Raises ``ValueError`` if any
        placeholders remain unresolved.
        """
        params: dict[str, str] = {}
        if self._workspace:
            params["workspace"] = self._workspace
        params.update(request.path_params)
        encoded_params = {name: quote(str(value), safe="") for name, value in params.items()}
        try:
            path = request.path_template.format_map(encoded_params)
        except KeyError as exc:
            raise ValueError(f"Missing path parameter {exc} for {request.method} {request.path_template}") from exc
        url = self._base_url + path
        if self._url_resolver is not None:
            return str(self._url_resolver(url))
        return url

    def _request_headers(self, request: PreparedRequest) -> dict[str, str] | None:
        headers: dict[str, str] = {}
        if self._default_headers:
            headers.update(self._default_headers)
        if request.content_type is not None:
            headers["Content-Type"] = request.content_type
        if request.extra_headers:
            headers.update(request.extra_headers)
        return headers or None

    def _is_binary(self, request: PreparedRequest) -> bool:
        return request.response_type is BinaryContent

    def _is_stream(self, request: PreparedRequest) -> bool:
        return get_origin(request.response_type) is Stream

    def _is_paginated(self, request: PreparedRequest) -> bool:
        return get_origin(request.response_type) is Paginated

    def with_options(
        self,
        *,
        headers: Mapping[str, str] | None = None,
        retry: RetryPolicy | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> Self:
        """Return a copy of this client with the given options merged in.

        The returned client shares the underlying HTTP transport, so it is
        cheap to create.  Useful for one-off header, retry, or timeout
        overrides when calling ``method()``-bound endpoints::

            client.with_headers({"Range": "bytes=0-99"}).download_file(...)
            client.with_options(timeout=300).update_fileset(...)
        """
        clone = copy.copy(self)
        clone._owns_http = False
        if headers:
            clone._default_headers = {**self._default_headers, **headers}
        if retry is not None:
            clone._retry = retry
        if timeout is not None:
            clone._timeout = timeout
        return clone

    def with_headers(self, headers: Mapping[str, str]) -> Self:
        """Shorthand for ``with_options(headers=...)``."""
        return self.with_options(headers=headers)

    def with_workspace(self, workspace: str) -> Self:
        """Return a copy of this client with *workspace* as the default workspace."""
        clone = copy.copy(self)
        clone._owns_http = False
        clone._workspace = workspace
        return clone

    def with_retry(self, retry: RetryPolicy) -> Self:
        """Shorthand for ``with_options(retry=...)``."""
        return self.with_options(retry=retry)

    def _resource_client(
        self,
        sync_client_type: type[NemoClient],
        async_client_type: type[AsyncNemoClient],
    ) -> NemoClient | AsyncNemoClient:
        if isinstance(self, AsyncNemoClient):
            return async_client_type.from_client(self)
        if isinstance(self, NemoClient):
            return sync_client_type.from_client(self)
        raise TypeError(f"{type(self).__name__} is not a concrete NeMo client")

    # ---------------------------------------------------------------------------
    # Convenience properties — return typed resource clients sharing this
    # client's transport.  Lazy imports avoid circular dependencies (typed
    # clients subclass NemoClient / AsyncNemoClient).
    # ---------------------------------------------------------------------------

    @property
    def files(self) -> NemoClient | AsyncNemoClient:
        from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient

        return self._resource_client(FilesClient, AsyncFilesClient)

    @property
    def models(self) -> NemoClient | AsyncNemoClient:
        from nemo_platform_plugin.models.client import AsyncModelsClient, ModelsClient

        return self._resource_client(ModelsClient, AsyncModelsClient)

    @property
    def workspaces(self) -> NemoClient | AsyncNemoClient:
        from nemo_platform_plugin.workspaces.client import AsyncWorkspacesClient, WorkspacesClient

        return self._resource_client(WorkspacesClient, AsyncWorkspacesClient)

    @property
    def secrets(self) -> NemoClient | AsyncNemoClient:
        from nemo_platform_plugin.secrets.client import AsyncSecretsClient, SecretsClient

        return self._resource_client(SecretsClient, AsyncSecretsClient)

    @property
    def jobs(self) -> NemoClient | AsyncNemoClient:
        from nemo_platform_plugin.jobs.client import AsyncJobsClient, JobsClient

        return self._resource_client(JobsClient, AsyncJobsClient)

    @property
    def auth(self) -> NemoClient | AsyncNemoClient:
        from nemo_platform_plugin.auth.client import AsyncAuthenticationClient, AuthenticationClient

        return self._resource_client(AuthenticationClient, AsyncAuthenticationClient)

    @property
    def access_keys(self) -> NemoClient | AsyncNemoClient:
        from nemo_platform_plugin.auth.access_keys.client import AccessKeysClient, AsyncAccessKeysClient

        return self._resource_client(AccessKeysClient, AsyncAccessKeysClient)

    @property
    def iam(self) -> NemoClient | AsyncNemoClient:
        from nemo_platform_plugin.iam.client import AsyncIAMClient, IAMClient

        return self._resource_client(IAMClient, AsyncIAMClient)

    @property
    def agents(self) -> NemoClient | AsyncNemoClient:
        from nemo_platform_plugin.agents.client import AgentsClient, AsyncAgentsClient

        return self._resource_client(AgentsClient, AsyncAgentsClient)

    @property
    def auditor(self) -> NemoClient | AsyncNemoClient:
        from nemo_platform_plugin.auditor.client import AsyncAuditorClient, AuditorClient

        return self._resource_client(AuditorClient, AsyncAuditorClient)

    @property
    def guardrail(self) -> NemoClient | AsyncNemoClient:
        from nemo_platform_plugin.guardrail.client import AsyncGuardrailClient, GuardrailClient

        return self._resource_client(GuardrailClient, AsyncGuardrailClient)

    @property
    def evaluator(self) -> NemoClient | AsyncNemoClient:
        from nemo_platform_plugin.evaluator.client import AsyncEvaluatorClient, EvaluatorClient

        return self._resource_client(EvaluatorClient, AsyncEvaluatorClient)

    @property
    def projects(self) -> NemoClient | AsyncNemoClient:
        from nemo_platform_plugin.projects.client import AsyncProjectsClient, ProjectsClient

        return self._resource_client(ProjectsClient, AsyncProjectsClient)

    @property
    def intake(self) -> NemoClient | AsyncNemoClient:
        from nemo_platform_plugin.intake.client import AsyncIntakeClient, IntakeClient

        return self._resource_client(IntakeClient, AsyncIntakeClient)

    @property
    def data_designer(self) -> NemoClient | AsyncNemoClient:
        from nemo_platform_plugin.data_designer.client import AsyncDataDesignerClient, DataDesignerClient

        return self._resource_client(DataDesignerClient, AsyncDataDesignerClient)

    @property
    def agent_hardener(self) -> NemoClient | AsyncNemoClient:
        from nemo_platform_plugin.agent_hardener.client import AgentHardenerClient, AsyncAgentHardenerClient

        return self._resource_client(AgentHardenerClient, AsyncAgentHardenerClient)

    @property
    def inference(self: NemoClient | AsyncNemoClient) -> _InferenceNamespace:
        return _InferenceNamespace(self)

    def _resolve_query_params(self, request: PreparedRequest) -> dict[str, str | int | bool] | None:
        """Filter out None values and JSON-serialize dicts/lists in query params."""
        if request.query_params is None:
            return None
        filtered = {}
        for k, v in request.query_params.items():
            if v is None:
                continue
            if isinstance(v, (dict, list)):
                filtered[k] = json.dumps(v)
            else:
                filtered[k] = v
        return filtered or None


class NemoClient(BaseNemoClient[httpx.Client]):
    """Sync HTTP client for NeMo Platform APIs."""

    _auth: TokenProvider | None

    def __init__(
        self,
        *,
        base_url: str,
        workspace: str | None = None,
        auth: TokenProvider | str | None = None,
        default_headers: Mapping[str, str] | None = None,
        timeout: float | httpx.Timeout | None = None,
        retry: RetryPolicy | None = None,
        http_client: httpx.Client | None = None,
        owns_http_client: bool | None = None,
        url_resolver: Callable[[str], str | httpx.URL] | None = None,
    ) -> None:
        """Create a client.

        *timeout* is applied per request, so it holds even when *http_client* is
        supplied and shared with another caller — an httpx client's own timeout
        is fixed when it is built and cannot be changed afterwards. ``None``
        defers to the transport's timeout, giving one we build ourselves
        :data:`DEFAULT_TIMEOUT`; ``httpx.Timeout(None)`` waits indefinitely.
        """
        auth = _resolve_implicit_workload_auth(
            base_url=base_url,
            auth=auth,
            default_headers=default_headers,
            allow_env_bootstrap=http_client is None,
        )
        super().__init__(
            base_url=base_url,
            workspace=workspace,
            auth=auth,
            retry=retry,
            default_headers=default_headers,
            timeout=timeout,
            url_resolver=url_resolver,
        )
        self._owns_http = http_client is None if owns_http_client is None else owns_http_client
        self._http = http_client or httpx.Client(
            headers=dict(default_headers) if default_headers else None,
            timeout=timeout if timeout is not None else DEFAULT_TIMEOUT,
            auth=TokenProviderAuth(self._auth) if self._auth else None,
        )

    @classmethod
    def from_client(cls, client: NemoClient) -> Self:
        """Create an instance of this subclass sharing the transport of *client*."""
        return cls(
            base_url=client.base_url,
            workspace=client.workspace,
            auth=client._auth,
            default_headers=client._default_headers or None,
            timeout=client._timeout,
            retry=client._retry,
            http_client=client._http,
            owns_http_client=False,
            url_resolver=client._url_resolver,
        )

    def close(self) -> None:
        """Close the underlying sync HTTP transport."""
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @overload
    def send(
        self,
        request: PreparedRequest[BinaryContent],
        *,
        headers: dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
    ) -> NemoBinaryResponse: ...
    @overload
    def send(
        self,
        request: PreparedRequest[Stream[ModelT]],
        *,
        headers: dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
    ) -> NemoStreamResponse[ModelT]: ...
    @overload
    def send(
        self,
        request: PreparedRequest[Paginated[ModelT, StrategyT]],
        *,
        headers: dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
    ) -> NemoPaginatedResponse[ModelT, StrategyT]: ...
    @overload
    def send(
        self,
        request: PreparedRequest[None],
        *,
        headers: dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
    ) -> NemoResponse[None]: ...
    @overload
    def send(
        self,
        request: PreparedRequest[ResponseT],
        *,
        headers: dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
    ) -> NemoResponse[ResponseT]: ...

    @classmethod
    def from_config(
        cls,
        context: str | None = None,
        config_path: Path | str | None = None,
    ) -> NemoClient:
        """Create a NemoClient from the user's nmp config file.

        Args:
            context: Context name to use (default: active context).
            config_path: Path to config file (default: ``~/.config/nmp/config.yaml``).
        """
        return _client_from_config(cls, context=context, config_path=config_path)

    def send(
        self,
        request: PreparedRequest,
        *,
        headers: dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
    ) -> NemoResponse | NemoBinaryResponse | NemoStreamResponse | NemoPaginatedResponse:
        """Send a prepared request and return a typed response.

        Args:
            request: The prepared request to send.
            headers: Optional per-request headers merged on top of client
                defaults and content-type headers.
            retry: Optional per-request retry policy override. Takes
                precedence over client-level defaults.

        For binary and streaming endpoints, the caller should use the
        response as a context manager to ensure the connection is closed::

            with client.send(endpoints.download(name="file.csv")) as resp:
                for chunk in resp:
                    f.write(chunk)
        """
        if headers:
            request = request.with_headers(headers)

        # Inject auth header if a TokenProvider is configured.
        # NOTE: If a 401 occurs despite this, a future enhancement could
        # call provider.force_refresh() and retry once. The proactive
        # refresh margin (60s) makes this unlikely in practice.
        if self._auth:
            token = self._auth.get_access_token()
            request = request.with_headers({"Authorization": f"Bearer {token}"})

        url = self._resolve_path(request)
        req_headers = self._request_headers(request)
        params = self._resolve_query_params(request)
        resolved_retry = self._resolve_retry(retry)

        if self._is_binary(request):
            stream_ctx = self._stream_with_retry(request, url, req_headers, params, resolved_retry)
            return NemoBinaryResponse(stream_ctx, request)

        if self._is_stream(request):
            assert request.response_type is not None
            stream_ctx = self._stream_with_retry(request, url, req_headers, params, resolved_retry)
            model_type = _get_stream_model_type(request.response_type)
            return NemoStreamResponse(stream_ctx, model_type, request)

        if self._is_paginated(request):
            assert request.response_type is not None
            raw = self._request_with_retry(request, url, req_headers, params, resolved_retry)
            model_type, strategy = _get_paginated_types(request.response_type)
            return NemoPaginatedResponse(
                raw, model_type, request, self._make_page_fetcher(strategy, resolved_retry), strategy
            )

        raw = self._request_with_retry(request, url, req_headers, params, resolved_retry)
        if _should_resolve_conflict(raw, request):
            assert request.on_conflict_get is not None
            return self.send(request.on_conflict_get, headers=headers, retry=retry)
        raise_for_status(raw)
        body = None
        if request.response_type is not None:
            body = _parse_response_body(request.response_type, raw)
        return NemoResponse(http_response=raw, body=body, request=request)

    def _request_with_retry(
        self,
        request: PreparedRequest,
        url: str,
        headers: dict[str, str] | None,
        params: dict | None,
        retry: RetryPolicy | None,
    ) -> httpx.Response:
        """Execute a single HTTP request with optional retry."""
        last_response: httpx.Response | None = None
        for attempt in range(retry.max_retries + 1 if retry else 1):
            try:
                kwargs: dict = {"content": request.content, "headers": headers, "params": params}
                if self._timeout is not None:
                    kwargs["timeout"] = self._timeout
                raw = self._http.request(request.method, url, **kwargs)
            except httpx.TransportError as exc:
                backoff = _should_retry(request, None, exc, attempt, retry) if retry else None
                if backoff is not None:
                    time.sleep(backoff)
                    continue
                raise NemoTransportError(exc) from exc
            if retry:
                backoff = _should_retry(request, raw, None, attempt, retry)
                if backoff is not None:
                    last_response = raw
                    time.sleep(backoff)
                    continue
            return raw

        assert last_response is not None
        return last_response

    @contextmanager
    def _stream_with_retry(
        self,
        request: PreparedRequest,
        url: str,
        headers: dict[str, str] | None,
        params: dict | None,
        retry: RetryPolicy | None,
    ) -> Iterator[httpx.Response]:
        """Open a stream, retrying failures before handing it to the caller."""
        for attempt in range(retry.max_retries + 1 if retry else 1):
            yielded = False
            try:
                kwargs: dict = {"content": request.content, "headers": headers, "params": params}
                if self._timeout is not None:
                    kwargs["timeout"] = self._timeout
                with self._http.stream(request.method, url, **kwargs) as raw:
                    backoff = _should_retry(request, raw, None, attempt, retry) if retry else None
                    if backoff is not None:
                        time.sleep(backoff)
                        continue
                    yielded = True
                    yield raw
                    return
            except httpx.TransportError as exc:
                if yielded:
                    raise NemoTransportError(exc) from exc
                backoff = _should_retry(request, None, exc, attempt, retry) if retry else None
                if backoff is not None:
                    time.sleep(backoff)
                    continue
                raise NemoTransportError(exc) from exc

    def _make_page_fetcher(
        self, strategy: type[PaginationStrategy[Any, Any]], retry: RetryPolicy | None = None
    ) -> SyncPageFetcher:
        """Create a page-fetching callback bound to this client and strategy."""

        def fetch(request: PreparedRequest, page: Any) -> httpx.Response:
            url = self._resolve_path(request)
            req_headers = self._request_headers(request)
            existing_params = self._resolve_query_params(request) or {}
            page_params = strategy.page_query_params(page)
            params = {**existing_params, **page_params}
            return self._request_with_retry(request, url, req_headers, params, retry)

        return fetch


class AsyncNemoClient(BaseNemoClient[httpx.AsyncClient]):
    """Async HTTP client for NeMo Platform APIs.

    Async twin of :class:`NemoClient`.
    """

    def __init__(
        self,
        *,
        base_url: str,
        workspace: str | None = None,
        auth: TokenProvider | AsyncTokenProvider | str | None = None,
        default_headers: Mapping[str, str] | None = None,
        timeout: float | httpx.Timeout | None = None,
        retry: RetryPolicy | None = None,
        http_client: httpx.AsyncClient | None = None,
        owns_http_client: bool | None = None,
        url_resolver: Callable[[str], str | httpx.URL] | None = None,
    ) -> None:
        """Create a client. See :meth:`NemoClient.__init__` for *timeout*."""
        auth = _resolve_implicit_workload_auth(
            base_url=base_url,
            auth=auth,
            default_headers=default_headers,
            allow_env_bootstrap=http_client is None,
        )
        super().__init__(
            base_url=base_url,
            workspace=workspace,
            auth=auth,
            retry=retry,
            default_headers=default_headers,
            timeout=timeout,
            url_resolver=url_resolver,
        )
        self._owns_http = http_client is None if owns_http_client is None else owns_http_client
        self._http = http_client or httpx.AsyncClient(
            headers=dict(default_headers) if default_headers else None,
            timeout=timeout if timeout is not None else DEFAULT_TIMEOUT,
            auth=TokenProviderAuth(self._auth) if self._auth else None,
        )

    @classmethod
    def from_client(cls, client: AsyncNemoClient) -> Self:
        """Create an instance of this subclass sharing the transport of *client*."""
        return cls(
            base_url=client.base_url,
            workspace=client.workspace,
            auth=client._auth,
            default_headers=client._default_headers or None,
            timeout=client._timeout,
            retry=client._retry,
            http_client=client._http,
            owns_http_client=False,
            url_resolver=client._url_resolver,
        )

    async def close(self) -> None:
        """Close the underlying async HTTP transport."""
        if self._owns_http:
            await self._http.aclose()

    async def aclose(self) -> None:
        """Alias for compatibility with httpx-style async resources."""
        await self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    def with_http_client(self, http_client: httpx.AsyncClient) -> Self:
        """Return a copy of this client using a different async transport."""
        transport_owner = AsyncNemoClient(
            base_url=self.base_url,
            workspace=self.workspace,
            auth=self._auth,
            default_headers=self._default_headers or None,
            timeout=self._timeout,
            retry=self._retry,
            http_client=http_client,
            owns_http_client=False,
            url_resolver=self._url_resolver,
        )
        return type(self).from_client(transport_owner)

    @overload
    async def send(
        self,
        request: PreparedRequest[BinaryContent],
        *,
        headers: dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
    ) -> AsyncNemoBinaryResponse: ...
    @overload
    async def send(
        self,
        request: PreparedRequest[Stream[ModelT]],
        *,
        headers: dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
    ) -> AsyncNemoStreamResponse[ModelT]: ...
    @overload
    async def send(
        self,
        request: PreparedRequest[Paginated[ModelT, StrategyT]],
        *,
        headers: dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
    ) -> AsyncNemoPaginatedResponse[ModelT, StrategyT]: ...
    @overload
    async def send(
        self,
        request: PreparedRequest[None],
        *,
        headers: dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
    ) -> NemoResponse[None]: ...
    @overload
    async def send(
        self,
        request: PreparedRequest[ResponseT],
        *,
        headers: dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
    ) -> NemoResponse[ResponseT]: ...

    @classmethod
    def from_config(
        cls,
        context: str | None = None,
        config_path: Path | str | None = None,
    ) -> AsyncNemoClient:
        """Create an AsyncNemoClient from the user's nmp config file.

        Args:
            context: Context name to use (default: active context).
            config_path: Path to config file (default: ``~/.config/nmp/config.yaml``).
        """
        return _client_from_config(cls, context=context, config_path=config_path)

    async def send(
        self,
        request: PreparedRequest,
        *,
        headers: dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
    ) -> NemoResponse | AsyncNemoBinaryResponse | AsyncNemoStreamResponse | AsyncNemoPaginatedResponse:
        """Send a prepared request and return a typed response."""
        if headers:
            request = request.with_headers(headers)

        if self._auth:
            request = request.with_headers({"Authorization": f"Bearer {await resolve_token_async(self._auth)}"})

        url = self._resolve_path(request)
        req_headers = self._request_headers(request)
        params = self._resolve_query_params(request)
        resolved_retry = self._resolve_retry(retry)

        if self._is_binary(request):
            stream_ctx = self._stream_with_retry(request, url, req_headers, params, resolved_retry)
            return AsyncNemoBinaryResponse(stream_ctx, request)

        if self._is_stream(request):
            assert request.response_type is not None
            stream_ctx = self._stream_with_retry(request, url, req_headers, params, resolved_retry)
            model_type = _get_stream_model_type(request.response_type)
            return AsyncNemoStreamResponse(stream_ctx, model_type, request)

        if self._is_paginated(request):
            assert request.response_type is not None
            raw = await self._request_with_retry(request, url, req_headers, params, resolved_retry)
            model_type, strategy = _get_paginated_types(request.response_type)
            return AsyncNemoPaginatedResponse(
                raw, model_type, request, self._make_page_fetcher(strategy, resolved_retry), strategy
            )

        raw = await self._request_with_retry(request, url, req_headers, params, resolved_retry)
        if _should_resolve_conflict(raw, request):
            assert request.on_conflict_get is not None
            return await self.send(request.on_conflict_get, headers=headers, retry=retry)
        raise_for_status(raw)
        body = None
        if request.response_type is not None:
            body = _parse_response_body(request.response_type, raw)
        return NemoResponse(http_response=raw, body=body, request=request)

    async def _request_with_retry(
        self,
        request: PreparedRequest,
        url: str,
        headers: dict[str, str] | None,
        params: dict | None,
        retry: RetryPolicy | None,
    ) -> httpx.Response:
        """Execute a single async HTTP request with optional retry."""
        last_response: httpx.Response | None = None
        for attempt in range(retry.max_retries + 1 if retry else 1):
            try:
                kwargs: dict = {"content": request.content, "headers": headers, "params": params}
                if self._timeout is not None:
                    kwargs["timeout"] = self._timeout
                raw = await self._http.request(request.method, url, **kwargs)
            except httpx.TransportError as exc:
                backoff = _should_retry(request, None, exc, attempt, retry) if retry else None
                if backoff is not None:
                    await asyncio.sleep(backoff)
                    continue
                raise NemoTransportError(exc) from exc
            if retry:
                backoff = _should_retry(request, raw, None, attempt, retry)
                if backoff is not None:
                    last_response = raw
                    await asyncio.sleep(backoff)
                    continue
            return raw

        assert last_response is not None
        return last_response

    @asynccontextmanager
    async def _stream_with_retry(
        self,
        request: PreparedRequest,
        url: str,
        headers: dict[str, str] | None,
        params: dict | None,
        retry: RetryPolicy | None,
    ) -> AsyncIterator[httpx.Response]:
        """Open an async stream, retrying failures before handing it to the caller."""
        for attempt in range(retry.max_retries + 1 if retry else 1):
            yielded = False
            try:
                kwargs: dict = {"content": request.content, "headers": headers, "params": params}
                if self._timeout is not None:
                    kwargs["timeout"] = self._timeout
                async with self._http.stream(request.method, url, **kwargs) as raw:
                    backoff = _should_retry(request, raw, None, attempt, retry) if retry else None
                    if backoff is not None:
                        await asyncio.sleep(backoff)
                        continue
                    yielded = True
                    yield raw
                    return
            except httpx.TransportError as exc:
                if yielded:
                    raise NemoTransportError(exc) from exc
                backoff = _should_retry(request, None, exc, attempt, retry) if retry else None
                if backoff is not None:
                    await asyncio.sleep(backoff)
                    continue
                raise NemoTransportError(exc) from exc

    def _make_page_fetcher(
        self, strategy: type[PaginationStrategy[Any, Any]], retry: RetryPolicy | None = None
    ) -> AsyncPageFetcher:
        """Create an async page-fetching callback bound to this client and strategy."""

        async def fetch(request: PreparedRequest, page: Any) -> httpx.Response:
            url = self._resolve_path(request)
            req_headers = self._request_headers(request)
            existing_params = self._resolve_query_params(request) or {}
            page_params = strategy.page_query_params(page)
            params = {**existing_params, **page_params}
            return await self._request_with_retry(request, url, req_headers, params, retry)

        return fetch


# ---------------------------------------------------------------------------
# from_config helper (shared by NemoClient and AsyncNemoClient)
# ---------------------------------------------------------------------------

_ClientT = TypeVar("_ClientT", NemoClient, AsyncNemoClient)


def _client_from_config(
    cls: type[_ClientT],
    *,
    context: str | None = None,
    config_path: Path | str | None = None,
) -> _ClientT:
    """Shared implementation for NemoClient.from_config / AsyncNemoClient.from_config."""
    from nemo_platform_plugin.client.config.config import Config
    from nemo_platform_plugin.client.config.models import ConfigParams, OAuthUser
    from nemo_platform_plugin.client.oidc_factory import resolve_oidc_provider

    resolved_path = Path(config_path) if isinstance(config_path, str) else config_path
    overrides: ConfigParams | None = None
    if context is not None:
        overrides = {"current_context": context}
    config = Config.load(config_path=resolved_path, overrides=overrides)
    actual_config_path = config.get_config_path() or Config.get_default_config_path()
    config_exists = actual_config_path.exists()
    # If the token came from NMP_ACCESS_TOKEN (env override), it's not from
    # the config file — don't cache or persist provider state for it.
    explicit_access_token = config.access_token is not None
    ctx = config.resolve()

    auth: TokenProvider | str | None = None
    workload_identity_token_file = os.environ.get(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR)
    use_implicit_workload_auth = bool(workload_identity_token_file and not explicit_access_token)

    if not use_implicit_workload_auth and isinstance(ctx.user, OAuthUser):
        auth = resolve_oidc_provider(
            base_url=str(ctx.cluster.base_url),
            context_name=ctx.context_name,
            access_token=ctx.user.token.get_secret_value(),
            refresh_token=ctx.user.refresh_token.get_secret_value() if ctx.user.refresh_token else None,
            config_exists=config_exists,
            config_path=actual_config_path,
            explicit_access_token=explicit_access_token,
        )
    elif not use_implicit_workload_auth and ctx.user:
        client_config = ctx.user.get_client_config()
        raw_headers = client_config.get("default_headers")
        if isinstance(raw_headers, dict):
            raw_auth = dict(raw_headers).get("Authorization")
            if isinstance(raw_auth, str) and raw_auth.startswith("Bearer "):
                auth = raw_auth.removeprefix("Bearer ")

    return cls(base_url=str(ctx.cluster.base_url), workspace=ctx.workspace, auth=auth)
