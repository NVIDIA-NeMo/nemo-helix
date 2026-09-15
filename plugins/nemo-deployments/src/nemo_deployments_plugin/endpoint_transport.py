# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How platform components dial a deployment ``Endpoint``.

An openshell endpoint is reached through its gateway, which routes exposed sandbox
services by Host header: dial the gateway, send the advertised host as ``Host``, present
the executor's TLS material. Docker and k8s endpoints are dialed as-is.
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urlunparse

from nemo_deployments_plugin.config import DeploymentsConfig


@dataclass(frozen=True)
class EndpointTransport:
    """Connection settings for reaching endpoints minted by one executor."""

    connect_base: str | None = None
    """``scheme://host:port`` to dial in place of the URL's own. None dials the URL itself."""
    ca_cert: str | None = None
    client_cert: tuple[str, str] | None = field(default=None)

    def target(self, url: str) -> tuple[str, dict[str, str]]:
        """The URL to dial for *url*, plus the headers that keep it routing."""
        if self.connect_base is None:
            return url, {}
        parsed = urlparse(url)
        base = urlparse(self.connect_base)
        dial = parsed._replace(scheme=base.scheme, netloc=base.netloc)
        return urlunparse(dial), {"Host": parsed.netloc}

    def client_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``httpx.AsyncClient`` carrying the TLS material."""
        if self.ca_cert is None and self.client_cert is None:
            return {}
        context = ssl.create_default_context(cafile=self.ca_cert)
        if self.client_cert is not None:
            context.load_cert_chain(*self.client_cert)
        return {"verify": context}


DIRECT = EndpointTransport()
"""Dial the endpoint URL as advertised, with default TLS verification."""


def endpoint_transport(executor: str | None) -> EndpointTransport:
    """Transport for endpoints minted by the named executor; ``None`` means ``default_executor``."""
    config = DeploymentsConfig.get()
    name = executor or config.default_executor
    entry = next((e for e in config.executors if e.name == name), None)
    if entry is None or entry.backend != "openshell":
        return DIRECT
    from nemo_deployments_plugin.backends.openshell.config import OpenShellExecutorConfig

    return OpenShellExecutorConfig.model_validate(entry.config).service_transport()
