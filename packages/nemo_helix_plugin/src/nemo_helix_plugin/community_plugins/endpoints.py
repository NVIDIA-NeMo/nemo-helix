# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the external plugin install lifecycle.

Paths include the ``/apis/plugins`` gateway prefix. These routes are
HelixAdmin-gated server side; ``install`` re-fetches the manifest and rejects the
request if its digest no longer matches the one the caller previewed.
"""

from __future__ import annotations

from abc import abstractmethod

from nemo_helix_plugin.client.endpoint import delete, get, patch, post
from nemo_helix_plugin.community_plugins.registry import ExternalPluginRegistry
from nemo_helix_plugin.community_plugins.types import (
    CatalogResponse,
    EnableRequest,
    InstallRequest,
    InstallResponse,
    ManifestSource,
    PreviewResponse,
)


@get("/apis/plugins/-/catalog")
@abstractmethod
def get_plugin_catalog() -> CatalogResponse: ...


@post("/apis/plugins/-/preview")
@abstractmethod
def preview_plugin(*, body: ManifestSource) -> PreviewResponse: ...


@post("/apis/plugins/-/installed")
@abstractmethod
def install_plugin(*, body: InstallRequest) -> InstallResponse: ...


@get("/apis/plugins/-/installed")
@abstractmethod
def list_installed_plugins() -> ExternalPluginRegistry: ...


@patch("/apis/plugins/-/installed/{name}")
@abstractmethod
def set_plugin_enabled(*, name: str, body: EnableRequest) -> None: ...


@delete("/apis/plugins/-/installed/{name}")
@abstractmethod
def uninstall_plugin(*, name: str) -> None: ...
