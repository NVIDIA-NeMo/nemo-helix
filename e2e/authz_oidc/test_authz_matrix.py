# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parametrized authz verification matrix (see matrix.py for the case list)."""

from __future__ import annotations

import pytest
from harness import IDENTITIES, WS_A, WS_B, Platform
from idp import KID_ROGUE, MintSpec
from matrix import MATRIX, Case
from report import ReportCollector, Row

# subprocess_only: the harness spawns its own OIDC-configured ``nemo services`` platforms
# (see conftest.py); it never targets a shared cluster, so it is skipped when NHX_BASE_URL
# is set (the Kubernetes/kind e2e job) rather than spawning subprocess platforms there and
# timing out under that job's resource pressure.
pytestmark = [pytest.mark.e2e, pytest.mark.subprocess_only]

_DEFECT_OVERRIDES: dict[str, dict] = {
    "expired": {"expires_in": -3600},
    "wrong-issuer": {"issuer": "http://127.0.0.1:1/evil-issuer"},
    "wrong-audience": {"audience": "some-other-audience"},
    "unknown-key": {"kid": KID_ROGUE},
    "unsigned": {"unsigned": True},
}


def _mint_token(platform: Platform, case: Case) -> tuple[str | None, str]:
    """Return (token, claims-description) for the case's identity + defect."""
    if case.identity == "anonymous":
        return None, "(no Authorization header)"
    if case.token_defect == "garbage":
        return "not-a-jwt-at-all", "(garbage bearer string)"

    info = IDENTITIES[case.identity]
    overrides = dict(_DEFECT_OVERRIDES.get(case.token_defect or "", {}))
    spec = MintSpec(sub=info["sub"], email=info["email"], scopes=case.scopes, **overrides)
    return platform.issuer.mint(spec), platform.issuer.claims_summary(spec)


def _run_case(platform: Platform, case: Case, report: ReportCollector) -> None:
    token, claims = _mint_token(platform, case)
    path = case.path.format(wsA=WS_A, wsB=WS_B)
    body = case.body
    if body is not None:
        body = {k: (v.format(case=case.id.lower()) if isinstance(v, str) else v) for k, v in body.items()}

    response = platform.request(case.method, path, token=token, body=body)
    observed = response.status_code

    if isinstance(case.expected, str):
        passed = observed != 403
        expected_str = "not 403"
    else:
        passed = observed in case.expected
        expected_str = "/".join(str(c) for c in sorted(case.expected))

    report.record(
        Row(
            case_id=case.id,
            group=case.group,
            description=case.description,
            method=case.method,
            path=path,
            identity=case.identity,
            claims=claims,
            expected=expected_str,
            observed=observed,
            passed=passed,
            phase=case.phase,
            notes=case.notes,
        )
    )
    assert passed, (
        f"[{case.id}] {case.description}: expected {expected_str}, observed {observed} "
        f"for {case.method} {path} as {case.identity} ({claims}); body: {response.text[:300]}"
    )


@pytest.mark.parametrize("case", [c for c in MATRIX if c.phase == "default"], ids=lambda c: c.id)
def test_authz_default_knobs(platform: Platform, report: ReportCollector, case: Case) -> None:
    _run_case(platform, case, report)


@pytest.mark.parametrize("case", [c for c in MATRIX if c.phase == "knobs"], ids=lambda c: c.id)
def test_authz_quarantine_and_admin_exempt_knobs(platform_knobs: Platform, report: ReportCollector, case: Case) -> None:
    _run_case(platform_knobs, case, report)
