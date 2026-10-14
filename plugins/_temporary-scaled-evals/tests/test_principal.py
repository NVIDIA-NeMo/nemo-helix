# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ownership resolution after the in-plugin identity provider was removed."""

from __future__ import annotations

import pytest

# This plugin is absent from `enabled-plugins`, so a default sync leaves it and its database
# driver uninstalled and the repo-wide test run still sweeps this directory. Skip rather than
# error there; the job that owns these tests installs the `scaled-evals` group first.
try:
    from fastapi import HTTPException
    from nemo_helix_plugin.auth import AuthContext
    from scaled_evals.api import auth
    from scaled_evals.api.auth import current_principal
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def test_principal_follows_platform_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ownership is the platform principal, or the shared owner when auth is off.

    Auth off, every caller must land on owner ``dev`` with source ``disabled``:
    that exact string is what several call sites read to skip owner filtering.
    Auth on without a principal must be 401, never the shared owner. A service
    acting on behalf of a user owns rows as that user.
    """
    monkeypatch.setattr(auth, "platform_auth_enabled", lambda: False)
    principal = current_principal()
    assert (principal.owner_id, principal.owner_type, principal.source) == ("dev", "DEV", "disabled")

    monkeypatch.setattr(auth, "platform_auth_enabled", lambda: True)
    monkeypatch.setattr(auth, "current_auth_context", lambda: None)
    with pytest.raises(HTTPException) as raised:
        current_principal()
    assert raised.value.status_code == 401

    user = AuthContext(principal_id="usr_1", principal_email="a@example.com", principal_groups=["g1"])
    monkeypatch.setattr(auth, "current_auth_context", lambda: user)
    principal = current_principal()
    assert (principal.owner_id, principal.owner_type, principal.source) == ("usr_1", "USER", "platform")
    assert (principal.email, principal.groups) == ("a@example.com", ("g1",))

    delegated = AuthContext(
        principal_id="service:scaled-evals",
        principal_on_behalf_of="usr_2",
        principal_on_behalf_of_email="b@example.com",
    )
    monkeypatch.setattr(auth, "current_auth_context", lambda: delegated)
    principal = current_principal()
    assert (principal.owner_id, principal.owner_type, principal.email) == ("usr_2", "USER", "b@example.com")
