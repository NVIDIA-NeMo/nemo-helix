# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Applied-policy verification shared by every OpenSandbox caller."""

import logging
from dataclasses import dataclass, field

import pytest
from nhx_sandbox.egress import EgressPolicy, EgressRule
from nhx_sandbox.opensandbox_policy import (
    EgressVerificationError,
    EgressVerificationMode,
    egress_target_covers,
    verify_applied_egress,
)


@pytest.mark.parametrize(
    ("trusted", "target", "covered"),
    [
        # Same target, any spelling.
        ("pypi.org", "PyPI.org.", True),
        ("::ffff:0:0/96", "::ffff:0.0.0.0/96", True),
        # A wildcard covers its subdomains at any depth, and narrower wildcards, but not the bare domain.
        ("*.wikipedia.org", "en.wikipedia.org", True),
        ("*.wikipedia.org", "upload.en.wikipedia.org", True),
        ("*.wikipedia.org", "*.en.wikipedia.org", True),
        ("*.wikipedia.org", "wikipedia.org", False),
        ("*.wikipedia.org", "notwikipedia.org", False),
        ("*.en.wikipedia.org", "*.wikipedia.org", False),
        # A CIDR covers the addresses and ranges inside it, in its own family only.
        ("10.0.0.0/8", "10.1.2.3", True),
        ("10.0.0.0/8", "10.1.0.0/16", True),
        ("10.0.0.0/8", "11.0.0.1", False),
        ("10.1.0.0/16", "10.0.0.0/8", False),
        ("2001:db8::/48", "2001:db8::1", True),
        ("10.0.0.0/8", "::ffff:10.1.2.3", False),
        # An exact host covers only itself; hostnames and addresses never cover each other.
        ("en.wikipedia.org", "*.en.wikipedia.org", False),
        ("10.1.2.3", "10.0.0.0/8", False),
        ("10.0.0.0/8", "internal.example", False),
        ("*.example", "10.1.2.3", False),
    ],
)
def test_egress_target_covers(trusted: str, target: str, covered: bool) -> None:
    assert egress_target_covers(trusted, target) is covered


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
    # The sidecar re-serializes: Go spells the IPv4-mapped prefix differently, and an operator rule of
    # its own may appear. A deny it adds only narrows what the sandbox can reach.
    return _Policy(
        default_action="deny",
        egress=[
            _Rule("allow", "API.example.com."),
            _Rule("deny", "::ffff:0.0.0.0/96"),
            _Rule("deny", "169.254.0.0/16"),
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
    assert "did not apply 1 requested egress rule(s)" in caplog.text
    assert "Strict verification would fail it" in caplog.text


async def test_an_allow_rule_nobody_requested_fails_only_under_strict(caplog: pytest.LogCaptureFixture) -> None:
    """It lets the sandbox reach something it shouldn't, which a check of the requested rules alone misses."""
    sandbox = _Sandbox(_Policy(default_action="deny", egress=[*(_matching().egress or []), _Rule("allow", "*.com")]))

    with pytest.raises(EgressVerificationError, match=r"allows 1 egress target\(s\) that were not requested.*\*\.com"):
        await verify_applied_egress(sandbox, EXPECTED, mode="strict", label="sandbox s1")

    with caplog.at_level(logging.WARNING):
        await verify_applied_egress(sandbox, EXPECTED, mode="default_action", label="sandbox s1")
    assert "allows 1 egress target(s) that were not requested" in caplog.text


async def test_strict_reports_missing_and_extra_rules_together() -> None:
    sandbox = _Sandbox(
        _Policy(default_action="deny", egress=[_Rule("deny", "::ffff:0:0/96"), _Rule("allow", "evil.example")])
    )

    with pytest.raises(EgressVerificationError) as raised:
        await verify_applied_egress(sandbox, EXPECTED, mode="strict", label="sandbox s1")

    assert "did not apply 1 requested egress rule(s)" in str(raised.value)
    assert "allows 1 egress target(s) that were not requested" in str(raised.value)


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
