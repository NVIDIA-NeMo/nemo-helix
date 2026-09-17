# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a retrieval spec's ``hf_token_secret`` reference.

Job steps never resolve the reference themselves: ``retrieval_common`` compiles it
into a ``from_secret`` environment variable so the plaintext stays out of the job
spec. Retrieval preview runs in-process instead of as a job step, so it reads the
value through the Secrets service here.
"""

from __future__ import annotations

import logging

from data_designer_nemo.errors import NDDInternalError, NDDInvalidConfigError
from data_designer_nemo.secret_resolver import parse_secret_reference
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_platform_plugin.secrets.client import AsyncSecretsClient

logger = logging.getLogger(__name__)


async def resolve_hf_token(
    async_sdk: AsyncNeMoPlatform,
    hf_token_secret: str | None,
    workspace: str,
) -> str | None:
    """Return the plaintext token for ``workspace/name`` (or ``name``) secret reference."""
    if not hf_token_secret:
        return None
    secret_workspace, name = parse_secret_reference(hf_token_secret, workspace)
    try:
        secrets = client_from_platform(async_sdk, AsyncSecretsClient)
        response = (await secrets.access_secret(name=name, workspace=secret_workspace)).data()
    except NotFoundError as exc:
        raise NDDInvalidConfigError(f"Could not find secret {name!r} in workspace {secret_workspace!r}") from exc
    except PermissionDeniedError as exc:
        raise NDDInvalidConfigError(f"Access denied to workspace {secret_workspace!r}") from exc
    except Exception as exc:
        logger.exception("Unexpected error accessing configured HF token secret")
        raise NDDInternalError("An unexpected error occurred while accessing the configured HF token secret.") from exc
    return response.value or None
