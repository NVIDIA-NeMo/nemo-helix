# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorization derivation for the auditor plugin (every route ruled, no problems)."""

from __future__ import annotations

from nemo_auditor.service import AuditorPluginService
from nemo_helix_plugin.authz_discovery import _derive_service_contribution


def test_auditor_authz_derivation_has_no_problems() -> None:
    contrib, problems, _warnings = _derive_service_contribution(AuditorPluginService())
    assert problems == []

    # healthz: authenticated, no permission required.
    healthz = contrib.endpoints["/apis/auditor/v1/healthz"]["get"]
    assert healthz.callers == ["principal"]
    assert healthz.permissions == []

    # configs CRUD.
    configs = "/apis/auditor/v2/workspaces/{workspace}/configs"
    assert contrib.endpoints[configs]["post"].permissions == ["auditor.configs.create"]
    assert contrib.endpoints[configs]["get"].permissions == ["auditor.configs.list"]
    assert contrib.endpoints[f"{configs}/{{name}}"]["get"].permissions == ["auditor.configs.read"]
    assert contrib.endpoints[f"{configs}/{{name}}"]["put"].permissions == ["auditor.configs.update"]
    assert contrib.endpoints[f"{configs}/{{name}}"]["delete"].permissions == ["auditor.configs.delete"]

    # targets CRUD: spot-check + every referenced permission is declared.
    targets = "/apis/auditor/v2/workspaces/{workspace}/targets"
    assert contrib.endpoints[targets]["post"].permissions == ["auditor.targets.create"]
    assert {"auditor.targets.read", "auditor.targets.delete"} <= set(contrib.permissions)

    # All routes are PRINCIPAL (no service-only routes in this plugin).
    for methods in contrib.endpoints.values():
        for binding in methods.values():
            assert binding.callers == ["principal"]
            assert binding.deny is False
