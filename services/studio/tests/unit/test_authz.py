# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorization derivation for the Studio Assistant bridge."""

from nemo_helix_plugin.authz_discovery import _derive_service_contribution
from nhx.studio.service import StudioService


def test_studio_assistant_routes_are_ruled_for_authenticated_principals() -> None:
    contribution, problems, _warnings = _derive_service_contribution(StudioService())

    assert problems == []
    skills = contribution.endpoints["/apis/studio/v2/assistant/skills"]["get"]
    messages = contribution.endpoints["/apis/studio/v2/assistant/sessions/{session_id}/messages"]["post"]
    assert skills.callers == ["principal"]
    assert skills.permissions == []
    assert skills.scopes == ["studio:read", "platform:read"]
    assert not skills.deny
    assert messages.callers == ["principal"]
    assert messages.permissions == []
    assert messages.scopes == ["studio:write", "platform:write"]
    assert not messages.deny
