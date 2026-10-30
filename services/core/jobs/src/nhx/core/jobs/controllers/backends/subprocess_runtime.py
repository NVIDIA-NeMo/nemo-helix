# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import time_ns
from typing import IO, Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import httpx
import requests
from nemo_helix_plugin.client.constants import (
    ACCESS_TOKEN_ENVVAR,
    ACCESS_TOKEN_FILE_ENVVAR,
    WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
)
from nemo_helix_plugin.client.oidc_factory import resolve_workload_exchange_provider
from nhx.common.auth.models import NHX_PRINCIPAL_ENVVAR, Principal
from nhx.common.auth.workload_tokens import ServiceWorkloadAccessTokenProvider
from nhx.common.config import get_auth_config
from nhx.common.jobs.constants import NEMO_JOB_SECRETS_ENVVAR
from opentelemetry._logs import Logger
from opentelemetry._logs.severity import SeverityNumber
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)

NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT_ENVVAR = "NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT"
NHX_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH_ENVVAR = "NHX_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH"
NHX_JOB_LAUNCHER_OTLP_LOGS_TRANSPORT_ENVVAR = "NHX_JOB_LAUNCHER_OTLP_LOGS_TRANSPORT"
SERVICE_JOBS_BEARER_HEADERS = {"Authorization": "Bearer service:jobs"}


@dataclass(frozen=True)
class SecretReference:
    env_var_name: str
    workspace: str
    secret_name: str


@dataclass
class SubprocessOtelLogger:
    logger: Logger
    provider: LoggerProvider

    def emit(self, message: str, stream_name: str) -> None:
        severity_number = SeverityNumber.INFO if stream_name == "stdout" else SeverityNumber.ERROR
        severity_text = "INFO" if stream_name == "stdout" else "ERROR"
        self.logger.emit(
            body=message,
            timestamp=time_ns(),
            severity_number=severity_number,
            severity_text=severity_text,
            attributes={"stream": stream_name},
        )

    def close(self) -> None:
        self.provider.force_flush()
        self.provider.shutdown()


class _UnixSocketOTLPSession(requests.Session):
    def __init__(self, socket_path: str) -> None:
        super().__init__()
        self._client = httpx.Client(
            transport=httpx.HTTPTransport(uds=socket_path),
            follow_redirects=True,
        )

    def request(
        self,
        method: Any,
        url: Any,
        params: Any = None,
        data: Any = None,
        headers: Any = None,
        cookies: Any = None,
        files: Any = None,
        auth: Any = None,
        timeout: Any = None,
        allow_redirects: bool = True,
        proxies: Any = None,
        hooks: Any = None,
        stream: Any = None,
        verify: Any = None,
        cert: Any = None,
        json: Any = None,
    ) -> requests.Response:
        try:
            response = self._client.request(
                str(method),
                str(url),
                params=params,
                content=data,
                headers=_merge_headers(self.headers, headers),
                json=json,
                timeout=timeout,
                follow_redirects=allow_redirects,
            )
        except httpx.TimeoutException as error:
            raise requests.exceptions.Timeout(str(error)) from error
        except httpx.TransportError as error:
            raise requests.exceptions.ConnectionError(str(error)) from error
        except httpx.HTTPError as error:
            raise requests.exceptions.RequestException(str(error)) from error
        return _to_requests_response(response)

    def close(self) -> None:
        self._client.close()
        super().close()


def _merge_headers(base_headers: Mapping[Any, Any], extra_headers: Any) -> dict[str, str]:
    headers = {str(key): str(value) for key, value in base_headers.items()}
    if extra_headers is None:
        return headers
    if isinstance(extra_headers, Mapping):
        headers.update({str(key): str(value) for key, value in extra_headers.items()})
        return headers
    headers.update({str(key): str(value) for key, value in extra_headers})
    return headers


def _to_requests_response(response: httpx.Response) -> requests.Response:
    requests_response = requests.Response()
    requests_response.status_code = response.status_code
    requests_response.reason = response.reason_phrase
    requests_response.url = str(response.url)
    requests_response.headers.update(response.headers)
    requests_response._content = response.content
    requests_response.encoding = response.encoding
    return requests_response


def parse_secret_references(secrets_env: str) -> list[SecretReference]:
    if not secrets_env:
        return []

    references: list[SecretReference] = []
    for pair in secrets_env.split(","):
        pair = pair.strip()
        if not pair:
            continue
        env_var_name, sep, secret_ref = pair.partition("=")
        if sep != "=" or not env_var_name.strip():
            raise ValueError(f"invalid secret reference format: {pair} (expected ENV_VAR=workspace/secret_name)")
        workspace, slash, secret_name = secret_ref.strip().partition("/")
        workspace = workspace.strip()
        secret_name = secret_name.strip()
        if slash != "/" or not workspace or not secret_name:
            raise ValueError(f"invalid secret reference format: {secret_ref.strip()} (expected workspace/secret_name)")
        references.append(
            SecretReference(
                env_var_name=env_var_name.strip(),
                workspace=workspace,
                secret_name=secret_name,
            )
        )
    return references


def _mint_task_access_token(principal: Principal | None) -> str | None:
    """A ``jobs`` service token delegated to *principal*, or ``None`` without workload token exchange."""
    auth_config = get_auth_config()
    if not (auth_config.enabled and auth_config.oidc.workload is not None):
        return None
    on_behalf_of = principal.effective_principal if principal is not None and principal.id else None
    return ServiceWorkloadAccessTokenProvider(auth_config, "jobs", on_behalf_of).get_access_token()


def _delegable_principal(env: dict[str, str]) -> Principal | None:
    principal = _principal_from_env(env)
    return principal if principal is not None and principal.id else None


def write_task_access_token(token_file: Path, token: str) -> None:
    """Atomically replace *token_file* with *token*, readable only by this user."""
    token_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = token_file.with_name(f".{token_file.name}.tmp")
    # The child shares this uid and can pre-plant the temp path: refuse to follow or reuse anything there.
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(token)
    os.replace(tmp, token_file)


def inject_subprocess_task_auth(env: dict[str, str], token_file: Path) -> dict[str, str]:
    """Give the child a short-lived bearer token delegated from its job principal, via a rotatable file.

    Without workload token exchange, or without a principal to delegate from, the child gets no
    token: it must never hold an undelegated service identity.
    """
    env.pop(ACCESS_TOKEN_ENVVAR, None)
    env.pop(ACCESS_TOKEN_FILE_ENVVAR, None)
    principal = _delegable_principal(env)
    if principal is None or not (token := _mint_task_access_token(principal)):
        return env
    write_task_access_token(token_file, token)
    env[ACCESS_TOKEN_FILE_ENVVAR] = str(token_file)
    return env


def task_access_token_refresh_interval() -> float:
    """Seconds between re-minting a child's token: half the token lifetime."""
    workload = get_auth_config().oidc.workload
    return (workload.token_ttl_seconds if workload is not None else 300) / 2


def refresh_subprocess_task_auth(env: dict[str, str], token_file: Path) -> bool:
    """Re-mint the child's delegated token into *token_file*; the child re-reads it on each request.

    ``False`` when there was nothing to mint, so the caller keeps retrying instead of treating it as rotated.
    """
    principal = _delegable_principal(env)
    if principal is None or not (token := _mint_task_access_token(principal)):
        return False
    write_task_access_token(token_file, token)
    return True


def _access_token_from_env(env: dict[str, str]) -> str | None:
    token_file = env.get(ACCESS_TOKEN_FILE_ENVVAR)
    if not token_file:
        return None
    try:
        return Path(token_file).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def inject_secret_env_vars(env: dict[str, str], *, secrets_url: str | None = None) -> dict[str, str]:
    secrets_env = env.get(NEMO_JOB_SECRETS_ENVVAR, "")
    references = parse_secret_references(secrets_env)
    if not references:
        return env

    api_base_url = secrets_url or env.get("NHX_SECRETS_URL", "")
    if not api_base_url:
        raise ValueError("NHX_SECRETS_URL is required when NEMO_JOB_SECRETS is set")
    _validate_http_url(api_base_url, "NHX_SECRETS_URL")

    principal = _principal_from_env(env)
    for reference in references:
        env[reference.env_var_name] = _fetch_secret(
            api_base_url=api_base_url,
            principal=principal,
            access_token=_access_token_from_env(env),
            workspace=reference.workspace,
            secret_name=reference.secret_name,
        )
    return env


def start_log_capture(
    pipe: IO[str] | None,
    *,
    log_path: Path,
    log_lock: threading.Lock,
    otel_logger: SubprocessOtelLogger | None,
    stream_name: str,
    job: str,
    step: str,
    task_id: str,
) -> threading.Thread | None:
    if pipe is None:
        return None

    def capture() -> None:
        try:
            for line in pipe:
                message = line.rstrip("\n")
                if not message:
                    continue
                if stream_name != "stdout":
                    message = f"[{stream_name}] {message}"
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "job": job,
                    "job_step": step,
                    "job_task": task_id,
                    "message": message,
                }
                with log_lock:
                    with log_path.open("a", encoding="utf-8") as log_file:
                        log_file.write(json.dumps(record) + "\n")
                if otel_logger is not None:
                    otel_logger.emit(message, stream_name)
        finally:
            pipe.close()

    thread = threading.Thread(target=capture, name=f"jobs-log-capture-{task_id}-{stream_name}", daemon=True)
    thread.start()
    return thread


def create_otel_logger(
    *,
    env: dict[str, str],
    workspace: str,
    job: str,
    attempt_id: str,
    step: str,
    task_id: str,
) -> SubprocessOtelLogger | None:
    endpoint = env.get(NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT_ENVVAR)
    if not endpoint:
        return None

    resource = Resource.create(
        {
            "workspace": workspace,
            "job": job,
            "job_attempt": attempt_id,
            "job_step": step,
            "job_task": task_id,
        }
    )
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(_build_otlp_log_exporter(env, endpoint)))
    logger.info("Created local OTEL logger", extra={"endpoint": endpoint, "job": job, "step": step})
    return SubprocessOtelLogger(logger_provider.get_logger("nhx.jobs.subprocess"), logger_provider)


def _build_otlp_log_exporter(env: dict[str, str], endpoint: str) -> OTLPLogExporter:
    headers = _otlp_log_auth_headers(env)
    transport = env.get(NHX_JOB_LAUNCHER_OTLP_LOGS_TRANSPORT_ENVVAR, "tcp")
    if transport == "tcp":
        return OTLPLogExporter(endpoint=endpoint, headers=headers)
    if transport != "uds":
        raise ValueError(f"unsupported OTLP logs transport: {transport!r}")

    socket_path = env.get(NHX_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH_ENVVAR)
    if not socket_path:
        raise ValueError(f"{NHX_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH_ENVVAR} is required for UDS OTLP logs")
    return OTLPLogExporter(
        endpoint=endpoint,
        headers=headers,
        session=_UnixSocketOTLPSession(socket_path),
    )


def _otlp_log_auth_headers(env: dict[str, str]) -> dict[str, str]:
    subject_token_file = env.get(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR)
    if not subject_token_file:
        return dict(SERVICE_JOBS_BEARER_HEADERS)

    base_url = env.get("NHX_AUTH_URL") or env.get("NHX_BASE_URL")
    if not base_url:
        raise ValueError(f"NHX_AUTH_URL or NHX_BASE_URL is required when {WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR} is set")

    provider = resolve_workload_exchange_provider(
        base_url=base_url,
        subject_token_file=Path(subject_token_file),
    )
    return {"Authorization": f"Bearer {provider.get_access_token()}"}


def _principal_from_env(env: dict[str, str]) -> Principal | None:
    principal_json = env.get(NHX_PRINCIPAL_ENVVAR, "")
    if not principal_json:
        return None
    try:
        return Principal.model_validate_json(principal_json)
    except Exception:
        return None


def _fetch_secret(
    *,
    api_base_url: str,
    principal: Principal | None,
    access_token: str | None,
    workspace: str,
    secret_name: str,
) -> str:
    request = Request(
        url=f"{api_base_url}/apis/secrets/v2/workspaces/{quote(workspace, safe='')}/secrets/{quote(secret_name, safe='')}/access",
        method="GET",
    )
    for name, value in _secret_request_headers(principal, access_token=access_token).items():
        request.add_header(name, value)

    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"failed to fetch secret {workspace}/{secret_name}: status code {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"failed to fetch secret {workspace}/{secret_name}: {exc.reason}") from exc

    value = payload.get("value")
    if not isinstance(value, str):
        raise RuntimeError(f"failed to fetch secret {workspace}/{secret_name}: missing string value field")
    return value


def _validate_http_url(url: str, field_name: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute http or https URL")


def _secret_request_headers(principal: Principal | None, *, access_token: str | None = None) -> dict[str, str]:
    if access_token:
        return {"Authorization": f"Bearer {access_token}"}

    if token := _mint_task_access_token(principal):
        return {"Authorization": f"Bearer {token}"}

    if principal is None or not principal.id:
        return Principal(id="service:jobs", authz_aliases=["service:jobs"]).get_headers()
    effective_principal = principal.effective_principal
    service_principal = Principal(
        id="service:jobs",
        authz_aliases=["service:jobs"],
        on_behalf_of=effective_principal.id,
        on_behalf_of_email=effective_principal.email,
        on_behalf_of_groups=effective_principal.groups,
        on_behalf_of_account_id=effective_principal.account_id,
        on_behalf_of_authz_aliases=effective_principal.authz_aliases,
    )
    return service_principal.get_headers()
