# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Header utilities for internal NeMo Helix service-to-service HTTP calls."""

from nhx.common.observability.otel import get_otel_headers
from nhx.common.platform_client_context import service_principal_auth_headers


def build_downstream_service_headers(service_name: str) -> dict[str, str]:
    """Build the full set of headers for HTTP requests to downstream NeMo Helix services.

    This is used when constructing service-to-service clients and raw HTTP
    requests that need to carry the same identity as platform clients.

    This should also be used when a service needs to manually forward headers to a NeMo Helix
    service via a raw HTTP client (i.e. not via the pre-configured NeMo Helix SDK). For example,
    Guardrails invokes IGW via Langchain, which internally makes the call via its own HTTP
    client. This function returns the headers that should be forwarded to IGW via that
    internal client.

    Args:
        service_name: The calling service's name (e.g. ``"guardrails"``).

    Returns:
        Header dictionary ready to merge into an outbound request.

    Example::

        headers = build_downstream_service_headers("guardrails")
        # headers == {
        #   "traceparent": "...",
        #   "X-NHX-Principal-Id": "service:guardrails",
        #   "X-NHX-Principal-On-Behalf-Of": "<user-id>",   # if user in context
        # }
    """
    return {**get_otel_headers(), **service_principal_auth_headers(service_name)}
