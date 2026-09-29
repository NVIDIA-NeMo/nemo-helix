# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenSandbox wire format for the shared egress policy.

Every component that creates OpenSandbox sandboxes must render the same :class:`EgressPolicy`
into the same create block and check it the same way, so the rendering and the check live beside
the policy type rather than in any one caller.
"""

import ipaddress
import logging
from collections.abc import Mapping
from typing import Any, Literal

from nhx_sandbox.egress import EgressPolicy

LOGGER = logging.getLogger(__name__)

NETWORK_POLICY_KEY = "network_policy"

EgressVerificationMode = Literal["off", "default_action", "strict"]


class EgressVerificationError(RuntimeError):
    """A sandbox's applied egress policy does not match the policy it was created with."""


def to_opensandbox_policy(policy: EgressPolicy) -> dict[str, Any]:
    """Render an egress policy in the shape the OpenSandbox sidecar expects."""
    return {
        "defaultAction": policy.default_action,
        "egress": [{"action": rule.action, "target": rule.target} for rule in policy.rules],
    }


def create_options_with_policy(create: Mapping[str, Any] | None, policy: EgressPolicy) -> dict[str, Any]:
    """Overlay our egress policy onto a deployment-supplied create block.

    The policy is applied last and unconditionally: whatever a deployment puts under
    ``network_policy``, ours is what reaches the SDK. Every caller depends on that, so it is
    stated once here rather than reimplemented per caller.

    Args:
        create: Deployment-supplied create options, or ``None``.
        policy: The policy that must win.

    Returns:
        A new mapping; ``create`` is not mutated.
    """
    return {
        **(dict(create) if create else {}),
        NETWORK_POLICY_KEY: to_opensandbox_policy(policy),
    }


def canonical_egress_target(target: str) -> str:
    """Canonicalize an egress target so two spellings of the same range compare equal.

    The sidecar does not echo the policy it was given -- it re-serializes a merged one, with
    operator-managed always-rules and nameserver addresses folded in. Go renders an IPv4-mapped
    prefix as ``::ffff:0.0.0.0/96`` where we send ``::ffff:0:0/96``, and casing of IPv6 literals is
    not guaranteed either. Parsing both sides removes that class of false mismatch; anything that
    is not an address or network is treated as a domain.
    """
    try:
        return str(ipaddress.ip_network(target, strict=False))
    except ValueError:
        return target.strip().rstrip(".").lower()


async def verify_applied_egress(
    sandbox: Any,
    expected: EgressPolicy,
    *,
    mode: EgressVerificationMode,
    label: str,
    require_readback: bool = False,
) -> None:
    """Confirm the egress policy an OpenSandbox sandbox reports is the one it was created with.

    Setting the policy is enforcement by construction; this is enforcement by verification, so a
    server that ignored the policy, or a deployment whose sidecar is not wired up, surfaces as a
    failure rather than as a sandbox with quietly unrestricted network access.

    ``default_action`` is always fatal when it disagrees: it is unambiguous, it decides what happens
    to everything no rule matches, and create is the only moment it can be set. Missing *rules* are
    only fatal under ``strict``, because the sidecar returns a merged, re-serialized policy and a
    textual difference there is more likely to mean reformatting than a real gap.

    Args:
        sandbox: A live OpenSandbox SDK ``Sandbox``.
        expected: The policy the sandbox was created with.
        mode: ``off`` skips the check; ``default_action`` fails only on a default-action mismatch;
            ``strict`` also fails on any requested rule the sandbox does not report.
        label: Names the sandbox in errors and logs, e.g. ``"episode abc123"``.
        require_readback: Fail, rather than warn and skip, when the SDK object cannot report its
            applied policy.

    Raises:
        EgressVerificationError: The applied policy does not satisfy ``mode``, or it could not be
            read and ``require_readback`` is set.
    """
    if mode == "off":
        return
    get_policy = getattr(sandbox, "get_egress_policy", None)
    if get_policy is None:
        if require_readback:
            raise EgressVerificationError(f"{label} cannot report its applied egress policy")
        LOGGER.warning("%s cannot report its egress policy; skipping verification", label)
        return

    applied = await get_policy()
    applied_default = getattr(applied, "default_action", None)
    if applied_default != expected.default_action:
        raise EgressVerificationError(
            f"{label} applied egress default_action={applied_default!r}, expected {expected.default_action!r}"
        )

    applied_targets = {
        (getattr(rule, "action", None), canonical_egress_target(str(getattr(rule, "target", ""))))
        for rule in (getattr(applied, "egress", None) or ())
    }
    missing = sorted(
        (rule.action, rule.target)
        for rule in expected.rules
        if (rule.action, canonical_egress_target(rule.target)) not in applied_targets
    )
    if not missing:
        return

    if mode == "strict":
        raise EgressVerificationError(
            f"{label} did not apply {len(missing)} requested egress rule(s); first few: {missing[:5]}"
        )
    LOGGER.warning(
        "%s did not report %d requested egress rule(s) (first few: %s). The default action matched, "
        "so this may be the sidecar re-serializing a merged policy rather than a real gap. Use strict "
        "verification once a deployment is known to round-trip rules cleanly.",
        label,
        len(missing),
        missing[:5],
    )
