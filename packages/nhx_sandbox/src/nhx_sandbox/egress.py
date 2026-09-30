# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Egress policies for sandboxes.

Provider-neutral and shared by every component that creates sandboxes, so nothing here is named
for any one of them.

Callers supply allow targets only; deny CIDRs for non-public space are computed here with those
addresses subtracted. Explicit denies are required because OpenSandbox ``dns+nft`` learns A/AAAA
from allowed domains into ``@dyn_allow`` *after* ``@deny``, so a wildcard like ``*.com`` can
otherwise rebind to a private IP that was only blocked by omission.
"""

import ipaddress
import logging
import socket
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict

LOGGER = logging.getLogger(__name__)

# A parsed allow target or deny range, in either address family.
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network

# Where local_resolver_addresses reads nameservers from by default.
RESOLV_CONF_PATH = "/etc/resolv.conf"


class EgressRule(BaseModel):
    """One egress rule. ``target`` is an FQDN, IP, or CIDR."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str
    action: Literal["allow", "deny"] = "allow"


class EgressPolicy(BaseModel):
    """Egress policy applied to a sandbox.

    Always sent explicitly, even when it has no rules. A backend that treats an absent policy as
    allow-all would otherwise grant unrestricted network access by default, and on OpenSandbox
    create is the only moment the default action can be established at all -- the post-create
    egress calls merge rules but preserve ``defaultAction``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_action: Literal["deny"] = "deny"
    rules: tuple[EgressRule, ...] = ()


class EgressAllowlist(BaseModel):
    """The allow-only inputs a trusted caller passes to :func:`build_egress_policy`.

    Attributes:
        targets: Allowed FQDNs, addresses, or CIDRs. Wildcard suffixes such as ``*.com`` are
            permitted and contribute no deny-range carve-out, since they name no fixed address.
        allow_internet: Whether to append the public DNS suffixes. Never adds a public CIDR:
            public space is reached by resolving an allowed name, which is what keeps the
            whitelist meaningful.
        public_dns_allow: Extra suffixes consulted only when ``allow_internet`` is set.
        resolver_addresses: Nameservers to keep out of the deny ranges. ``None`` means use this
            process's own resolvers; an empty tuple deliberately carves nothing out.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    targets: tuple[str, ...] = ()
    allow_internet: bool = False
    public_dns_allow: tuple[str, ...] = ()
    resolver_addresses: tuple[str, ...] | None = None


# Domain suffixes an allowlist gets when it sets ``allow_internet``; public space is reached only by name.
DEFAULT_PUBLIC_DNS_SUFFIXES: tuple[str, ...] = ("*.com", "*.org")

# Everything a sandbox has no business reaching by address. Public space is deliberately absent:
# it is reached by resolving an allowed domain, which is what makes the whitelist meaningful --
# an allowed public CIDR would let a sandbox dial any host in it without ever asking the resolver.
#
# Loopback is absent too, and must stay that way. It is the sandbox's own network namespace, so it
# crosses no boundary, and denying it breaks the sandbox outright: the egress sidecar redirects all
# port-53 traffic to a DNS proxy on 127.0.0.1, and Gym reaches its sub-servers the same way.
_DENIED_IPV4_CIDRS: tuple[str, ...] = (
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.0.0.0/24",
    "192.0.2.0/24",
    "192.168.0.0/16",
    "198.18.0.0/15",
    "198.51.100.0/24",
    "203.0.113.0/24",
    "224.0.0.0/4",
    "240.0.0.0/4",
)

# The same reachability applies over AAAA, so the same ranges are denied in v6 terms. ``::ffff:0:0/96``
# covers IPv4-mapped answers, which would otherwise describe a private v4 destination in v6 form.
_DENIED_IPV6_CIDRS: tuple[str, ...] = (
    "::/128",
    "::ffff:0:0/96",
    "64:ff9b:1::/48",
    "100::/64",
    "2001:db8::/32",
    "fc00::/7",
    "fe80::/10",
    "ff00::/8",
)


def _unique_targets(targets: Iterable[str]) -> tuple[str, ...]:
    """Strip whitespace and trailing dots, then drop empty and repeated targets, keeping first-seen order."""
    return tuple(
        dict.fromkeys(target.strip().rstrip(".") for target in targets if target and target.strip().rstrip("."))
    )


def _as_network(target: str) -> IPNetwork | None:
    """Parse an allow target as an address or CIDR, or return ``None`` for a domain."""
    try:
        return ipaddress.ip_network(target, strict=False)
    except ValueError:
        return None


def _resolve_host(target: str) -> tuple[IPNetwork, ...]:
    """Resolve a hostname to the addresses that must be carved out of the deny ranges.

    Resolution happens on the trusted side, which shares a resolver with the sandbox in a
    deployed cluster. A name that does not resolve here is not fatal -- the sandbox may still
    resolve it, and the DNS-learned allow covers public answers -- but a name that resolves to
    private space and could not be looked up would be denied. Callers that hit that case pass the
    literal address alongside the name.
    """
    try:
        info = socket.getaddrinfo(target, None, proto=socket.IPPROTO_TCP)
    except OSError:
        LOGGER.warning(
            "Egress target %r did not resolve while building the deny ranges. If it points at a "
            "private address, add that address as an allow target or the sandbox will be denied.",
            target,
        )
        return ()
    return tuple(dict.fromkeys(ipaddress.ip_network(entry[4][0], strict=False) for entry in info))


def local_resolver_addresses(path: str | None = None) -> tuple[str, ...]:
    """Return the nameservers this process resolves through.

    A sandbox reaches its resolver by address, so a resolver left inside a denied range denies
    every lookup and nothing resolves at all -- including the allowed names. In a deployed cluster
    the trusted side and the sandbox share a resolver, which is what makes this a usable default;
    callers whose sandboxes resolve elsewhere pass the addresses instead.
    """
    resolv_conf = RESOLV_CONF_PATH if path is None else path
    try:
        content = Path(resolv_conf).read_text()
    except OSError:
        LOGGER.warning("Could not read %s; no resolver will be carved out.", resolv_conf)
        return ()
    addresses: list[str] = []
    for line in content.splitlines():
        fields = line.split("#", 1)[0].split(";", 1)[0].split()
        if len(fields) < 2 or fields[0] != "nameserver":
            continue
        address = fields[1].split("%", 1)[0]
        if _as_network(address) is not None:
            addresses.append(address)
    return tuple(dict.fromkeys(addresses))


def allowed_networks(allow_targets: Iterable[str]) -> tuple[IPNetwork, ...]:
    """Return the addresses an allow whitelist grants, resolving hostnames where possible."""
    networks: list[IPNetwork] = []
    for target in allow_targets:
        if target.startswith("*"):
            continue
        network = _as_network(target)
        networks.extend((network,) if network is not None else _resolve_host(target))
    return tuple(dict.fromkeys(networks))


def denied_cidrs(
    allow_targets: Iterable[str] = (),
    resolver_addresses: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return the non-public ranges to deny, with every allowed address subtracted out.

    Subtraction is what keeps a deny range from swallowing a whitelisted host that lives inside
    it: denying ``10.0.0.0/8`` around an allowed ``10.0.0.51`` yields the ``/8`` split into the
    prefixes above and below that address, and a second allowed address splits whichever prefix
    now contains it. The resolver is subtracted on the same footing, defaulting to this process's
    nameservers; pass ``()`` to keep it in the deny ranges.
    """
    resolvers = local_resolver_addresses() if resolver_addresses is None else tuple(resolver_addresses)
    holes = allowed_networks((*allow_targets, *resolvers))
    denied: list[IPNetwork] = []
    for family in (_DENIED_IPV4_CIDRS, _DENIED_IPV6_CIDRS):
        for raw in family:
            fragments: list[IPNetwork] = [ipaddress.ip_network(raw)]
            for hole in holes:
                if hole.version != fragments[0].version:
                    continue
                remaining: list[IPNetwork] = []
                for fragment in fragments:
                    if fragment.subnet_of(hole):  # type: ignore[arg-type]
                        continue
                    if hole.subnet_of(fragment):  # type: ignore[arg-type]
                        remaining.extend(fragment.address_exclude(hole))  # type: ignore[arg-type]
                    else:
                        remaining.append(fragment)
                fragments = remaining
                if not fragments:
                    break
            # Resolvers may return a host's addresses in any order; sorting keeps the policy, and
            # its hash, identical across builds.
            denied.extend(sorted(fragments))  # type: ignore[type-var]
    return tuple(str(network) for network in denied)


def _allow_targets(allowlist: EgressAllowlist) -> tuple[str, ...]:
    """Expand an allowlist into the whitelist its rules are drawn from.

    Public DNS suffixes are added only when ``allow_internet`` is set. No public CIDR is ever
    added: public space is reachable by resolving an allowed name, which is the mechanism the
    whitelist is expressed in.
    """
    targets = list(allowlist.targets)
    if allowlist.allow_internet:
        targets.extend(DEFAULT_PUBLIC_DNS_SUFFIXES)
        targets.extend(allowlist.public_dns_allow)
    return _unique_targets(targets)


def build_egress_policy(allowlist: EgressAllowlist) -> EgressPolicy:
    """Build a deny-by-default policy from an allow-only whitelist.

    The deny rules are emitted whether or not ``allow_internet`` is set. They close the
    DNS-learned path into private space, and which names a sandbox may resolve is not the only way
    it can come to hold one.

    Args:
        allowlist: The caller's allow-only inputs.

    Returns:
        A policy whose ``default_action`` is ``deny``, carrying one allow rule per whitelisted
        target and one deny rule per non-public range that survives the carve-outs.
    """
    allow_targets = _allow_targets(allowlist)
    resolvers = (
        local_resolver_addresses()
        if allowlist.resolver_addresses is None
        else _unique_targets(allowlist.resolver_addresses)
    )
    policy_allow_targets = _unique_targets((*allow_targets, *resolvers))
    rules = [EgressRule(target=target, action="allow") for target in policy_allow_targets]
    rules.extend(EgressRule(target=target, action="deny") for target in denied_cidrs(allow_targets, resolvers))
    return EgressPolicy(default_action="deny", rules=tuple(rules))
