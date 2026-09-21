# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticate Fabric model requests with the executing job's identity."""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from nemo_agents_plugin.agent_config import AgentConfig, ModelConfig
from nemo_platform_plugin.auth import platform_auth_enabled
from nemo_platform_plugin.client.auth import TokenProviderAuth
from nemo_platform_plugin.client.auth_proxy import build_auth_proxy_app
from nemo_platform_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_platform_plugin.client.oidc_factory import resolve_workload_exchange_provider
from nemo_platform_plugin.sdk_provider import get_forwarding_headers, get_platform_sdk

logger = logging.getLogger(__name__)

_STARTUP_TIMEOUT_SECONDS = 10.0
_SHUTDOWN_TIMEOUT_SECONDS = 5


def _gateway_models(config: AgentConfig) -> Iterator[tuple[ModelConfig, str]]:
    models = [*config.models.values(), *(h.model for h in config.harnesses.values() if h.model is not None)]
    for model in models:
        base_url = model.base_url or model.settings.get("base_url")
        if isinstance(base_url, str) and urlsplit(base_url).path.startswith("/apis/inference-gateway/"):
            yield model, base_url


@contextmanager
def authenticated_gateway_config(config: AgentConfig) -> Iterator[AgentConfig]:
    """Route gateway models through a loopback proxy for this job invocation.

    The job's token provider stays outside Fabric's serializable config and
    refreshes on demand. No identity or credential is taken from the agent spec.
    Other execution modes and explicitly external providers keep their routing.
    """
    token_file = os.environ.get(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR)
    if not (token_file or os.environ.get("NMP_PRINCIPAL")) or not any(_gateway_models(config)):
        yield config
        return

    base_url = os.environ.get("NMP_BASE_URL", "")
    parsed_base = urlsplit(base_url)
    if parsed_base.scheme not in {"http", "https"} or not parsed_base.hostname:
        raise ValueError("NMP_BASE_URL must be set to the Platform URL for authenticated agent jobs")
    if token_file:
        provider = resolve_workload_exchange_provider(base_url=base_url, subject_token_file=Path(token_file))
        # Fail before starting Fabric if the job cannot authenticate. Subsequent
        # requests use the same provider, which exchanges again as tokens expire.
        provider.get_access_token()
        app = build_auth_proxy_app(base_url=base_url, auth=TokenProviderAuth(provider))
    else:
        with get_platform_sdk() as sdk:
            headers = get_forwarding_headers(sdk)
        if not any(name.lower() == "x-nmp-principal-id" and value.strip() for name, value in headers.items()):
            # Auth-disabled platforms serialize an anonymous Principal into
            # jobs too. Preserve that existing unauthenticated execution mode.
            if not platform_auth_enabled():
                yield config
                return
            raise ValueError("NMP_PRINCIPAL must provide a principal ID for authenticated agent jobs")
        app = build_auth_proxy_app(base_url=base_url, headers=headers)

    runtime_config = config.model_copy(deep=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        proxy_origin = f"127.0.0.1:{listener.getsockname()[1]}"
        for model, original_url in _gateway_models(runtime_config):
            model.base_url = urlsplit(original_url)._replace(scheme="http", netloc=proxy_origin).geturl()
            model.settings.pop("base_url", None)

        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=listener.getsockname()[1],
                access_log=False,
                log_config=None,
                timeout_graceful_shutdown=_SHUTDOWN_TIMEOUT_SECONDS,
            )
        )
        errors: list[BaseException] = []

        def serve() -> None:
            try:
                server.run(sockets=[listener])
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=serve, name="agent-job-gateway", daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
            while not server.started:
                if not thread.is_alive():
                    raise RuntimeError("Agent job gateway proxy failed to start") from (errors[0] if errors else None)
                if time.monotonic() >= deadline:
                    raise TimeoutError("Agent job gateway proxy startup timed out")
                thread.join(timeout=0.01)
            yield runtime_config
        finally:
            server.should_exit = True
            thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS + 1)
            if thread.is_alive():
                server.force_exit = True
                thread.join(timeout=1)
            if thread.is_alive():
                logger.error("Agent job gateway proxy did not stop within its shutdown deadline")
