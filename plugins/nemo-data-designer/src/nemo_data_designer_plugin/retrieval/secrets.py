# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a retrieval spec's ``hf_token_secret`` reference.

Job steps never resolve the reference themselves: ``retrieval_common`` compiles it
into a ``from_secret`` environment variable so the plaintext stays out of the job
spec. Retrieval preview runs in-process instead of as a job step, so it reads the
value through the Secrets service here.
"""

from __future__ import annotations

from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.secrets.client import AsyncSecretsClient


async def resolve_hf_token(
    async_sdk: AsyncNeMoPlatform,
    hf_token_secret: str | None,
    workspace: str,
) -> str | None:
    """Return the plaintext token for ``workspace/name`` (or ``name``) secret reference."""
    if not hf_token_secret:
        return None
    secret_workspace, _, name = hf_token_secret.rpartition("/")
    secrets = client_from_platform(async_sdk, AsyncSecretsClient)
    response = (await secrets.access_secret(name=name, workspace=secret_workspace or workspace)).data()
    return response.value or None
