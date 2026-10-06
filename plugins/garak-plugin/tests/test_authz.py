# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorization derivation for the garak_plugin plugin (every route ruled, no problems)."""

from __future__ import annotations

from garak_plugin.service import GarakPluginService
from nemo_helix_plugin.authz_discovery import _derive_service_contribution


def test_garak_plugin_authz_derivation_has_no_problems() -> None:
    contrib, problems, _warnings = _derive_service_contribution(GarakPluginService())
    assert problems == []

    # healthz: authenticated, no permission required.
    healthz = contrib.endpoints["/apis/garak-plugin/v1/healthz"]["get"]
    assert healthz.callers == ["principal"]
    assert healthz.permissions == []

    # configs CRUD.
    configs = "/apis/garak-plugin/v2/workspaces/{workspace}/configs"
    assert contrib.endpoints[configs]["post"].permissions == ["garak-plugin.configs.create"]
    assert contrib.endpoints[configs]["get"].permissions == ["garak-plugin.configs.list"]
    assert contrib.endpoints[f"{configs}/{{name}}"]["get"].permissions == ["garak-plugin.configs.read"]
    assert contrib.endpoints[f"{configs}/{{name}}"]["put"].permissions == ["garak-plugin.configs.update"]
    assert contrib.endpoints[f"{configs}/{{name}}"]["delete"].permissions == ["garak-plugin.configs.delete"]

    # targets CRUD: spot-check + every referenced permission is declared.
    targets = "/apis/garak-plugin/v2/workspaces/{workspace}/targets"
    assert contrib.endpoints[targets]["post"].permissions == ["garak-plugin.targets.create"]
    assert {"garak-plugin.targets.read", "garak-plugin.targets.delete"} <= set(contrib.permissions)

    # No service-only routes in this plugin: every route stays reachable by a human.
    # Not an equality check: the factory-generated job routes also carry
    # ``service_principal`` (see authz.GENERATED_ROUTE_CALLERS).
    for methods in contrib.endpoints.values():
        for binding in methods.values():
            assert binding.callers is not None and "principal" in binding.callers
            assert binding.deny is False


def test_legacy_scope_area_is_accepted_alongside_the_new_one() -> None:
    contrib, problems, _warnings = _derive_service_contribution(GarakPluginService())
    assert problems == []

    configs = "/apis/garak-plugin/v2/workspaces/{workspace}/configs"
    assert contrib.endpoints[configs]["get"].scopes == [
        "garak-plugin:read",
        "platform:read",
        "auditor:read",
    ]
    assert contrib.endpoints[configs]["post"].scopes == [
        "garak-plugin:write",
        "platform:write",
        "auditor:write",
    ]
    # job routes are built from scope.child("scan"); the legacy area must survive child().
    job_scopes = {
        tuple(methods["get"].scopes)
        for path, methods in contrib.endpoints.items()
        if "/jobs/scan" in path and "get" in methods
    }
    assert job_scopes
    assert all("auditor:read" in scopes for scopes in job_scopes)
