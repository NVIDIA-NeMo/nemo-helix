# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TLS configuration shared by NeMo Helix plugin clients."""

from __future__ import annotations

import os

NHX_CLIENT_SSL_CERT_FILE_ENVVAR = "NHX_CLIENT_SSL_CERT_FILE"


def client_verify_from_env() -> str | bool:
    """Return the httpx verify setting for NeMo Helix client requests."""
    cert_file = os.environ.get(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, "").strip()
    return cert_file or True
