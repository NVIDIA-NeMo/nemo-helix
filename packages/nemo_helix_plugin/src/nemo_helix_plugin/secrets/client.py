# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Secrets service.

Wraps the endpoint functions from ``secrets.endpoints`` as direct methods
using the ``method()`` descriptor, following the Files client pattern.

Usage::

    client = SecretsClient(base_url="...", workspace="default")
    secret = client.create_secret(body=HelixSecretCreateRequest(name="hf-token", value="...")).data()
    value = client.access_secret(name="hf-token").data().value
"""

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method
from nemo_helix_plugin.secrets import endpoints


class _SecretsMethods:
    create_secret = method(endpoints.create_secret)
    list_secrets = method(endpoints.list_secrets)
    get_secret = method(endpoints.get_secret)
    update_secret = method(endpoints.update_secret)
    delete_secret = method(endpoints.delete_secret)
    access_secret = method(endpoints.access_secret)
    rotate_encryption_keys = method(endpoints.rotate_encryption_keys)


class SecretsClient(_SecretsMethods, NemoClient):
    """Sync client for the Secrets service API."""


class AsyncSecretsClient(_SecretsMethods, AsyncNemoClient):
    """Async client for the Secrets service API."""
