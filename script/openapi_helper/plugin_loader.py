# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convention-based loader that yields a FastAPI app for any nemo plugin.

Plugins register their service class under the ``nemo.services`` entry-point
group. ``discover_services()`` enumerates that group; we instantiate the class
(no constructor args by contract — per-request deps inject at runtime via
FastAPI), wrap it in ``NemoServiceAdapter`` to obtain the per-service FastAPI
app, and mount that app's router under ``/apis/<plugin-name>`` on a parent
FastAPI so the generated OpenAPI paths match production routing (the platform
runner mounts plugins at the same prefix at startup — see
``nhx.platform_runner.server`` ``include_router(prefix=f"/apis/{service_instance.name}")``).

A plugin that needs special construction (e.g. constructor args, env-var
preconditions) sets ``factory_override`` on its ``PluginConfig`` instead of
relying on this loader.
"""

from fastapi import FastAPI
from nemo_helix_plugin.discovery import discover_services
from nhx.common.api.utils import install_query_param_schema_openapi_hook
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter


def build_plugin_app(plugin_name: str) -> FastAPI:
    services = discover_services()
    if plugin_name not in services:
        available = ", ".join(sorted(services)) or "<none>"
        raise KeyError(f"plugin '{plugin_name}' not found in nemo.services entry-points; available: {available}")
    cls = services[plugin_name]
    sub_app = NemoServiceAdapter(cls()).create_app()

    parent = FastAPI(title=f"{plugin_name} (plugin)")
    parent.include_router(sub_app.router, prefix=f"/apis/{plugin_name}")
    install_query_param_schema_openapi_hook(parent)
    return parent
