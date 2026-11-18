# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorization derivation for the notif plugin (every route ruled, no problems)."""

from __future__ import annotations

from nemo_helix_plugin.authz_discovery import _derive_service_contribution
from nemo_notif_plugin.service import NotifService


def test_notif_authz_derivation_has_no_problems() -> None:
    contrib, problems, _warnings = _derive_service_contribution(NotifService())
    assert problems == []

    events = "/apis/notif/v2/workspaces/{workspace}/events"
    assert contrib.endpoints[events]["post"].permissions == ["notif.events.create"]
    assert contrib.endpoints[f"{events}/stream"]["get"].permissions == ["notif.events.read"]
    for methods in contrib.endpoints.values():
        for binding in methods.values():
            assert binding.callers is not None and {"principal", "service_principal"} <= set(binding.callers)
