# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the deployments-plugin API.

Wraps the ``deployments.endpoints`` functions as methods via the ``method()``
descriptor (the Files/Secrets pattern). Methods return the standard response
envelope: unwrap single responses with ``.data()`` and iterate list responses
with ``.items()``. ``delete_*`` returns ``NemoResponse[None]`` (volume/deployment
deletes are soft and reconciler-driven; deployment-config delete is immediate —
see ``endpoints``).

Usage::

    from nemo_helix_plugin.deployments.client import DeploymentsClient
    from nemo_helix_plugin.deployments.types import CreateVolumeRequest

    client = DeploymentsClient(base_url="...", workspace="default")
    vol = client.create_volume(body=CreateVolumeRequest(name="weights")).data()
    for v in client.list_volumes().items():
        print(v.name, v.status)
    client.delete_volume(name="weights")
"""

from __future__ import annotations

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method
from nemo_helix_plugin.deployments import endpoints


class _DeploymentsMethods:
    # Volumes
    create_volume = method(endpoints.create_volume)
    list_volumes = method(endpoints.list_volumes)
    get_volume = method(endpoints.get_volume)
    delete_volume = method(endpoints.delete_volume)

    # Deployment configs
    create_deployment_config = method(endpoints.create_deployment_config)
    list_deployment_configs = method(endpoints.list_deployment_configs)
    get_deployment_config = method(endpoints.get_deployment_config)
    delete_deployment_config = method(endpoints.delete_deployment_config)

    # Deployments
    create_deployment = method(endpoints.create_deployment)
    list_deployments = method(endpoints.list_deployments)
    get_deployment = method(endpoints.get_deployment)
    delete_deployment = method(endpoints.delete_deployment)


class DeploymentsClient(_DeploymentsMethods, NemoClient):
    """Sync client for the deployments-plugin API.

    Methods return the standard response envelope: unwrap single responses with
    ``.data()`` and iterate list responses with ``.items()`` (the Files/Secrets
    pattern).
    """


class AsyncDeploymentsClient(_DeploymentsMethods, AsyncNemoClient):
    """Async client for the deployments-plugin API.

    Methods return the standard response envelope: unwrap single responses with
    ``.data()`` and iterate list responses with ``.items()`` (the Files/Secrets
    pattern).
    """
