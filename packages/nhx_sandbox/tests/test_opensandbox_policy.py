# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Applied-policy verification shared by every OpenSandbox caller."""

import logging
from dataclasses import dataclass, field

import pytest
from nhx_sandbox.egress import EgressPolicy, EgressRule
from nhx_sandbox.opensandbox_policy import EgressVerificationError, EgressVerificationMode, verify_applied_egress

EXPECTED = EgressPolicy(
    rules=(
        EgressRule(target="api.example.com", action="allow"),
        EgressRule(target="::ffff:0:0/96", action="deny"),
    )
)


@dataclass
class _Rule:
    action: str
    target: str


@dataclass
class _Policy:
    default_action: str | None
    egress: list[_Rule] | None = field(default_factory=list)


class _Sandbox:
    def __init__(self, applied: _Policy) -> None:
        self._applied = applied

    async def get_egress_policy(self) -> _Policy:
        return self._applied


def _matching() -> _Policy:
    # The sidecar re-serializes: Go spells the IPv4-mapped prefix differently and adds its own rules.
    return _Policy(
        default_action="deny",
        egress=[
            _Rule("allow", "API.example.com."),
            _Rule("deny", "::ffff:0.0.0.0/96"),
            _Rule("allow", "10.96.0.10"),
        ],
    )


@pytest.mark.parametrize("mode", ["default_action", "strict"])
async def test_matching_policy_passes(mode: EgressVerificationMode) -> None:
    await verify_applied_egress(_Sandbox(_matching()), EXPECTED, mode=mode, label="sandbox s1")


@pytest.mark.parametrize("applied_default", ["allow", None])
@pytest.mark.parametrize("mode", ["default_action", "strict"])
async def test_default_action_mismatch_fails_in_every_checking_mode(
    applied_default: str | None, mode: EgressVerificationMode
) -> None:
    sandbox = _Sandbox(_Policy(default_action=applied_default, egress=_matching().egress))

    with pytest.raises(EgressVerificationError, match="sandbox s1 applied egress default_action"):
        await verify_applied_egress(sandbox, EXPECTED, mode=mode, label="sandbox s1")


async def test_missing_rule_fails_only_under_strict(caplog: pytest.LogCaptureFixture) -> None:
    sandbox = _Sandbox(_Policy(default_action="deny", egress=[_Rule("allow", "api.example.com")]))

    with pytest.raises(EgressVerificationError, match="did not apply 1 requested egress rule"):
        await verify_applied_egress(sandbox, EXPECTED, mode="strict", label="sandbox s1")

    with caplog.at_level(logging.WARNING):
        await verify_applied_egress(sandbox, EXPECTED, mode="default_action", label="sandbox s1")
    assert "did not report 1 requested egress rule" in caplog.text


async def test_off_skips_the_readback() -> None:
    sandbox = _Sandbox(_Policy(default_action="allow"))

    await verify_applied_egress(sandbox, EXPECTED, mode="off", label="sandbox s1")


async def test_unreadable_policy_warns_by_default_and_fails_when_readback_is_required(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        await verify_applied_egress(object(), EXPECTED, mode="default_action", label="sandbox s1")
    assert "cannot report its egress policy" in caplog.text

    with pytest.raises(EgressVerificationError, match="cannot report its applied egress policy"):
        await verify_applied_egress(
            object(), EXPECTED, mode="default_action", label="sandbox s1", require_readback=True
        )
