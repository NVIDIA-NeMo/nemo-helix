# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Auth service IAM API."""

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method
from nemo_helix_plugin.iam import endpoints


class _IAMMethods:
    list_role_bindings = method(endpoints.list_role_bindings)
    create_role_binding = method(endpoints.create_role_binding)
    get_role_binding = method(endpoints.get_role_binding)
    revoke_role_binding = method(endpoints.revoke_role_binding)
    evaluate_authorization = method(endpoints.evaluate_authorization)
    get_opa_bundle = method(endpoints.get_opa_bundle)


class IAMClient(_IAMMethods, NemoClient):
    """Sync client for IAM role bindings, authorization, and OPA bundles."""


class AsyncIAMClient(_IAMMethods, AsyncNemoClient):
    """Async client for IAM role bindings, authorization, and OPA bundles."""
