# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OIDC client authentication and token-at-rest encryption."""

from __future__ import annotations

import base64
import hashlib
import os
from urllib.parse import quote_plus

from cryptography.fernet import Fernet


def pkce_challenge(verifier: str) -> str:
    """Return the S256 code challenge for a PKCE verifier."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def client_secret_basic_header(client_id: str, client_secret: str) -> str:
    """Build an RFC 6749 section 2.3.1 HTTP Basic client credential."""
    encoded_id = quote_plus(client_id, safe="")
    encoded_secret = quote_plus(client_secret, safe="")
    raw = f"{encoded_id}:{encoded_secret}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def _fernet(key_material: str) -> Fernet:
    digest = hashlib.sha256(key_material.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_secret(value: str, key_material: str) -> str:
    """Encrypt a provider token for storage."""
    return _fernet(key_material).encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_secret(value: str, key_material: str) -> str:
    """Decrypt a provider token loaded from storage."""
    return _fernet(key_material).decrypt(value.encode("ascii")).decode("utf-8")


def read_env(name: str) -> str:
    """Return a required environment variable or raise ValueError."""
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"Environment variable {name} is not set")
    return value
