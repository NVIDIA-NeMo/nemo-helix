# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

try:
    from scaled_evals.api.auth import CurrentPrincipal
    from scaled_evals.api.settings import settings
    from scaled_evals.api.tenancy import is_admin, require_admin
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def _clear_admin_settings(monkeypatch) -> None:
    monkeypatch.setattr(settings, "control_plane_admin_subjects", "")
    monkeypatch.setattr(settings, "control_plane_admin_emails", "")
    monkeypatch.setattr(settings, "control_plane_admin_groups", "")
    monkeypatch.setattr(settings, "control_plane_admin_roles", "")


def test_admin_group_uses_verified_starfleet_group_id(monkeypatch) -> None:
    _clear_admin_settings(monkeypatch)
    group_id = "sFZ-AvS0ZRSpyhC63regahzQ0Gp8GyO2ZGNYzq2NUL0"
    monkeypatch.setattr(settings, "control_plane_admin_groups", group_id)

    assert is_admin(CurrentPrincipal(owner_type="USER", owner_id="user-1", groups=(group_id,)))
    assert not is_admin(
        CurrentPrincipal(
            owner_type="USER",
            owner_id="user-2",
            groups=("scaled-evaluations-ssa-admin",),
        )
    )


def test_admin_subject_email_and_role_fallbacks(monkeypatch) -> None:
    _clear_admin_settings(monkeypatch)
    monkeypatch.setattr(settings, "control_plane_admin_subjects", "subject-1")
    monkeypatch.setattr(settings, "control_plane_admin_emails", "mstaats@nvidia.com")
    monkeypatch.setattr(settings, "control_plane_admin_roles", "scaled-evals-admin")

    assert is_admin(CurrentPrincipal(owner_type="USER", owner_id="subject-1"))
    assert is_admin(CurrentPrincipal(owner_type="USER", owner_id="subject-2", email="MSTAATS@nvidia.com"))
    assert is_admin(CurrentPrincipal(owner_type="USER", owner_id="subject-3", roles=("scaled-evals-admin",)))
    assert not is_admin(CurrentPrincipal(owner_type="USER", owner_id="subject-4"))


def test_require_admin_allows_only_disabled_or_configured_principals(monkeypatch) -> None:
    _clear_admin_settings(monkeypatch)
    db = MagicMock()
    local = CurrentPrincipal(owner_type="DEV", owner_id="dev", source="disabled")

    assert require_admin(local, db) == local
    db.users.upsert.assert_called_once_with("dev", email=None, username=None, display_name=None)

    hosted = CurrentPrincipal(owner_type="USER", owner_id="hosted-user", source="starfleet_jwt")
    with pytest.raises(HTTPException) as exc_info:
        require_admin(hosted, db)
    assert exc_info.value.status_code == 404

    monkeypatch.setattr(settings, "control_plane_admin_subjects", "hosted-user")
    assert require_admin(hosted, db) == hosted
