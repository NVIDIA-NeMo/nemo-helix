# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorization derivation for the example plugin (every route ruled, no problems)."""

from __future__ import annotations

from nemo_example_plugin.service import ExampleService
from nemo_helix_plugin.authz_discovery import _derive_service_contribution


def test_example_authz_derivation_has_no_problems() -> None:
    contrib, problems, _warnings = _derive_service_contribution(ExampleService())
    assert problems == []

    # Minimal hello endpoint (non-workspace-scoped).
    assert contrib.endpoints["/apis/example/hello/{name}"]["get"].permissions == ["example.hello.read"]

    # Items CRUD.
    items = "/apis/example/v2/workspaces/{workspace}/items"
    assert contrib.endpoints[items]["post"].permissions == ["example.items.create"]
    assert contrib.endpoints[items]["get"].permissions == ["example.items.list"]
    assert contrib.endpoints[f"{items}/{{name}}"]["get"].permissions == ["example.items.read"]
    assert contrib.endpoints[f"{items}/{{name}}"]["patch"].permissions == ["example.items.update"]
    assert contrib.endpoints[f"{items}/{{name}}"]["delete"].permissions == ["example.items.delete"]

    # Middleware-config CRUD (hyphenated namespace segment).
    mw = "/apis/example/v2/workspaces/{workspace}/middleware-configs"
    assert contrib.endpoints[mw]["post"].permissions == ["example.middleware-configs.create"]
    assert contrib.endpoints[f"{mw}/{{name}}"]["delete"].permissions == ["example.middleware-configs.delete"]

    # Factory-stamped function routes: the permissions they reference must be declared.
    assert {"example.greet", "example.count"} <= set(contrib.permissions)

    # Every route is PRINCIPAL and none is denied.
    for methods in contrib.endpoints.values():
        for binding in methods.values():
            assert binding.callers == ["principal"]
            assert binding.deny is False
