# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for Projects (Entity Store).

Wraps the endpoint functions from ``projects.endpoints`` as direct methods
using the ``method()`` descriptor, following the files/models pattern.
"""

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method
from nemo_helix_plugin.projects import endpoints


class _ProjectsMethods:
    get_project = method(endpoints.get_project)
    list_projects = method(endpoints.list_projects)
    create_project = method(endpoints.create_project)
    update_project = method(endpoints.update_project)
    delete_project = method(endpoints.delete_project)


class ProjectsClient(_ProjectsMethods, NemoClient):
    """Sync client for the Projects API."""


class AsyncProjectsClient(_ProjectsMethods, AsyncNemoClient):
    """Async client for the Projects API."""
