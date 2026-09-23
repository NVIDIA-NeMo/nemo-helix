# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TLS configuration shared by NeMo Helix SDK and CLI clients."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

NHX_CLIENT_SSL_CERT_FILE_ENVVAR = "NHX_CLIENT_SSL_CERT_FILE"


class HttpxTLSConfig(TypedDict, total=False):
    """TLS kwargs passed to HTTPX client and request calls."""

    verify: str


def client_certificate_authority_from_env(
    certificate_authority: str | None = None,
    env: Mapping[str, str] | None = None,
    *,
    cert_file_envvars: Sequence[str] = (NHX_CLIENT_SSL_CERT_FILE_ENVVAR,),
) -> str | None:
    """Return the configured CA bundle path, if one is configured."""
    for envvar in cert_file_envvars:
        cert_file = (env[envvar] if env is not None and envvar in env else os.environ.get(envvar, "")).strip()
        if cert_file:
            return cert_file
    return certificate_authority or None


def httpx_tls_config_from_env(
    certificate_authority: str | None = None,
    env: Mapping[str, str] | None = None,
    *,
    cert_file_envvars: Sequence[str] = (NHX_CLIENT_SSL_CERT_FILE_ENVVAR,),
) -> HttpxTLSConfig:
    """Return HTTPX TLS kwargs for NeMo Helix client requests."""
    ca_bundle = client_certificate_authority_from_env(
        certificate_authority,
        env=env,
        cert_file_envvars=cert_file_envvars,
    )
    return {"verify": ca_bundle} if ca_bundle is not None else {}


def client_verify_from_env(certificate_authority: str | None = None) -> str | Literal[True]:
    """Return the httpx verify setting for NeMo Helix client requests.

    The environment variable remains an explicit runtime override. A saved
    cluster ``certificate_authority`` is used when no override is set.
    """
    return client_certificate_authority_from_env(certificate_authority) or True
