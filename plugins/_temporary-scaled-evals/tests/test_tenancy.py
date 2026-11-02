# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path

import pytest
import yaml
from fastapi import HTTPException

try:
    from nemo_scaled_evals_plugin.service import ScaledEvalsService
    from scaled_evals.api.tenancy import _ADMIN_PROBE, caller_is_admin
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def test_caller_is_admin_probes_the_platform_authorizer() -> None:
    probed: list[tuple[str, str]] = []

    def authorizer(status: int | None):
        async def authorize(method: str, path: str) -> None:
            probed.append((method, path))
            if status is not None:
                raise HTTPException(status_code=status)

        return authorize

    assert asyncio.run(caller_is_admin(authorizer(None))) is True
    assert asyncio.run(caller_is_admin(authorizer(403))) is False
    with pytest.raises(HTTPException) as raised:
        asyncio.run(caller_is_admin(authorizer(503)))
    assert raised.value.status_code == 503
    assert probed == [_ADMIN_PROBE] * 3


@pytest.mark.parametrize(
    ("route_pattern", "method", "matching_scope", "other_scope"),
    [
        ("/apis/scaled-evals/v1/admin/capacity", "GET", "platform:read", "platform:write"),
        ("/apis/scaled-evals/v1/benchmarks/{benchmark_id}/promote", "POST", "platform:write", "platform:read"),
    ],
)
@pytest.mark.parametrize(
    ("roles", "workspace", "matching_scopes", "expected_allowed"),
    [
        ([], "system", True, False),
        (["Viewer"], "system", True, False),
        (["Editor"], "system", True, False),
        (["Admin"], "system", True, False),
        (["Editor"], "ordinary", True, False),
        (["Admin"], "ordinary", True, False),
        (["HelixAdmin"], "system", True, True),
        (["ScaledEvalsOperator"], "system", True, True),
        (["ScaledEvalsOperator"], "ordinary", True, False),
        (["ScaledEvalsOperator"], "system", False, False),
        (["HelixAdmin"], "system", False, False),
    ],
)
def test_admin_policy_preserves_explicit_operator_grants(
    route_pattern: str,
    method: str,
    matching_scope: str,
    other_scope: str,
    roles: list[str],
    workspace: str,
    matching_scopes: bool,
    expected_allowed: bool,
) -> None:
    from nemo_helix_plugin.authz_discovery import _derive_service_contribution
    from nemo_helix_plugin.authz_merge import merge_authz_contributions
    from nemo_scaled_evals_plugin.authz import ADMIN
    from nhx.core.auth.app.embedded_pdp import OPAPolicy
    from nhx.core.auth.app.embedded_pdp.policy_wasm import ensure_embedded_policy_wasm
    from nhx.core.auth.app.scope_families import with_scope_families

    contribution, errors, _warnings = _derive_service_contribution(ScaledEvalsService())
    assert not errors
    assert contribution.endpoints[route_pattern][method.lower()].permissions == [ADMIN.id]

    repo_root = Path(__file__).resolve().parents[3]
    static_policy = repo_root / "services/core/auth/src/nhx/core/auth/assets/static-authz.yaml"
    data = merge_authz_contributions(yaml.safe_load(static_policy.read_text()), [contribution.to_dict()])
    assert data["authz"]["permissions"]["scaled-evals"]["admin"]["has_role"] is False
    for role_name in ("Viewer", "Editor", "Admin"):
        assert ADMIN.id not in data["authz"]["roles"][role_name]["permissions"]

    data["authz"]["roles"]["ScaledEvalsOperator"] = {"permissions": [ADMIN.id]}
    data["authz"]["principals"] = {"operator@example.com": {"workspaces": {workspace: roles}}}
    policy = OPAPolicy(str(ensure_embedded_policy_wasm()))
    policy.set_data(with_scope_families(data))

    result = policy.evaluate(
        {
            "principal_id": "operator@example.com",
            "caller_kind": "principal",
            "path": route_pattern.replace("{benchmark_id}", "benchmark"),
            "method": method,
            "scopes": [matching_scope if matching_scopes else other_scope],
        }
    )
    assert result[0]["result"]["allowed"] is expected_allowed
