# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Guardrails service.

Wraps the endpoint functions from ``guardrail.endpoints`` as direct methods
using the ``method()`` descriptor, following the files/models pattern.
"""

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method
from nemo_helix_plugin.guardrail import endpoints


class _GuardrailMethods:
    get_guardrail_config = method(endpoints.get_guardrail_config)
    list_guardrail_configs = method(endpoints.list_guardrail_configs)
    create_guardrail_config = method(endpoints.create_guardrail_config)
    update_guardrail_config = method(endpoints.update_guardrail_config)
    delete_guardrail_config = method(endpoints.delete_guardrail_config)
    check_guardrail = method(endpoints.check_guardrail)


class GuardrailClient(_GuardrailMethods, NemoClient):
    """Sync client for the Guardrails service API."""


class AsyncGuardrailClient(_GuardrailMethods, AsyncNemoClient):
    """Async client for the Guardrails service API."""
