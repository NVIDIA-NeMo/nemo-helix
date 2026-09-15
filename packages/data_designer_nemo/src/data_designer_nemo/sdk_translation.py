# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for bridging sync and async NeMo Platform SDK entry points.

``sync_to_async_sdk`` exists because Data Designer validation and provider
resolution are async-first: they call platform services to validate filesets,
secrets, personas, and model providers before handing work to the upstream
Data Designer engine, and in the case of preview this work happens within the
FastAPI process with an injected ``AsyncNeMoPlatform``. However, some legitimate
callers still start with a sync ``NeMoPlatform`` instance, notably sync SDK/CLI
validation and job-container runtime code, so those paths need an async sibling
that preserves the same base URL, workspace, headers, query defaults, timeout,
and retry settings.

The inverse boundary is deliberately absent. DuckDB and upstream Data Designer
seed/person readers call fsspec synchronously, so callers must pass a separate
sync ``NeMoPlatform`` into execution paths instead of rebuilding one from an
``AsyncNeMoPlatform``.
"""

from nemo_platform import AsyncNeMoPlatform, NeMoPlatform


def sync_to_async_sdk(sdk: NeMoPlatform) -> AsyncNeMoPlatform:
    """Build an async :class:`AsyncNeMoPlatform` mirroring the sync SDK's config."""
    async_sdk = AsyncNeMoPlatform(
        base_url=sdk.base_url,
        default_headers=dict(sdk._custom_headers) if sdk._custom_headers else None,
        default_query=dict(sdk._custom_query) if sdk._custom_query else None,
        timeout=sdk.timeout,
        max_retries=sdk.max_retries,
        workspace=sdk.workspace,
        inference_base_url=sdk.inference_base_url,
    )
    return async_sdk
