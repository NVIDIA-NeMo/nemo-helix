# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared client constants and env checks."""

import os
from pathlib import Path

WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR = "NHX_WORKLOAD_IDENTITY_TOKEN_FILE"
ACCESS_TOKEN_ENVVAR = "NHX_ACCESS_TOKEN"
ACCESS_TOKEN_FILE_ENVVAR = "NHX_ACCESS_TOKEN_FILE"
JWT_WORKLOAD_SUBJECT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"
DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE = "urn:nvidia:nemo:params:oauth:token-type:docker-opaque-workload-proof"
OPAQUE_DOCKER_PROOF_PREFIX = "nhx_obo_v1"


def is_workload_identity_token_file_set() -> bool:
    return bool(os.environ.get(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR))


def get_env_access_token() -> str | None:
    """Bearer token from ``NHX_ACCESS_TOKEN_FILE`` (re-read on every call) or ``NHX_ACCESS_TOKEN``.

    The file form lets a supervisor rotate a short-lived token under a running process.
    ``None`` when neither source yields a non-blank token.
    """
    if token_file := os.environ.get(ACCESS_TOKEN_FILE_ENVVAR):
        try:
            token = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError:
            token = ""
        if token:
            return token
    return os.environ.get(ACCESS_TOKEN_ENVVAR, "").strip() or None


def subject_token_type_for_exchange(subject_token: str) -> str:
    """Return the RFC 8693 subject_token_type for a workload identity subject token."""
    if subject_token.startswith(f"{OPAQUE_DOCKER_PROOF_PREFIX}."):
        return DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE
    return JWT_WORKLOAD_SUBJECT_TOKEN_TYPE
