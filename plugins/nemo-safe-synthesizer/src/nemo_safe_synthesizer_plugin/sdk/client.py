# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Safe Synthesizer plugin API."""

from __future__ import annotations

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method
from nemo_safe_synthesizer_plugin.sdk import endpoints


class _SafeSynthesizerMethods:
    create_job = method(endpoints.create_job)
    list_jobs = method(endpoints.list_jobs)
    get_job = method(endpoints.get_job)


class SafeSynthesizerClient(_SafeSynthesizerMethods, NemoClient):
    """Sync client for the Safe Synthesizer plugin API."""


class AsyncSafeSynthesizerClient(_SafeSynthesizerMethods, AsyncNemoClient):
    """Async client for the Safe Synthesizer plugin API."""
