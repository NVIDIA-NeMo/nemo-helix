# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ipaddress

import pytest
from nhx_sandbox import egress
from nhx_sandbox.egress import EgressAllowlist, _as_network, build_egress_policy, denied_cidrs


def _denies(policy):
    return {rule.target for rule in policy.rules if rule.action == "deny"}


def _allows(policy):
    return {rule.target for rule in policy.rules if rule.action == "allow"}


def _is_denied(address, denied_targets):
    addr = ipaddress.ip_address(address)
    return any(
        addr in ipaddress.ip_network(target)
        for target in denied_targets
        if ipaddress.ip_network(target).version == addr.version
    )


def test_allow_internet_is_still_default_deny():
    policy = build_egress_policy(EgressAllowlist(targets=("vllm.svc.cluster.local",), allow_internet=True))
    assert policy.default_action == "deny"
    assert {"vllm.svc.cluster.local", "*.com", "*.org"} <= _allows(policy)
    # Public space is reached by resolving an allowed name, never by allowing the address range:
    # an allowed public CIDR would let the sandbox dial any host in it without asking the resolver.
    networks = (_as_network(target) for target in _allows(policy))
    assert not any(network is not None and not network.is_private for network in networks)


def test_without_internet_there_are_no_dns_suffixes():
    policy = build_egress_policy(
        EgressAllowlist(
            targets=("broker.svc.cluster.local",),
            allow_internet=False,
            public_dns_allow=("*.io",),
            resolver_addresses=(),
        )
    )
    assert policy.default_action == "deny"
    assert _allows(policy) == {"broker.svc.cluster.local"}
    # Deny rules are not conditional on allow_internet: a sandbox can come to hold a private
    # address through a name it was allowed for other reasons.
    assert _is_denied("10.0.0.1", _denies(policy))


def test_custom_public_dns_suffixes_are_added_with_internet():
    policy = build_egress_policy(EgressAllowlist(allow_internet=True, public_dns_allow=("*.io",)))
    assert {"*.com", "*.org", "*.io"} <= _allows(policy)


@pytest.mark.parametrize(
    "address",
    [
        "10.0.0.1",
        "100.64.0.1",
        "169.254.169.254",
        "172.16.0.1",
        "192.168.0.1",
        "198.18.0.1",
        "224.0.0.1",
        "240.0.0.1",
        "fd00::1",
        "fe80::1",
    ],
)
def test_denied_cidrs_cover_non_public_space(address):
    assert _is_denied(address, denied_cidrs(resolver_addresses=()))


@pytest.mark.parametrize("address", ["127.0.0.1", "127.0.0.53", "::1"])
def test_denied_cidrs_leave_loopback_alone(address):
    """Loopback is the sandbox's own namespace, and the egress sidecar's DNS proxy lives there.

    All port-53 traffic is redirected to that proxy on ``127.0.0.1``, and deny beats allow, so a
    denied loopback drops every lookup before it can reach the resolver.
    """
    assert not _is_denied(address, denied_cidrs(resolver_addresses=()))


def test_denied_cidrs_leave_public_space_reachable():
    """A deny that swallowed public space would make the DNS whitelist useless."""
    denied = denied_cidrs(resolver_addresses=())
    assert not _is_denied("8.8.8.8", denied)
    assert not _is_denied("2606:4700::1111", denied)
    assert "0.0.0.0/0" not in denied


def test_denied_cidrs_carve_out_an_allowed_private_address():
    """The whole point: an allowed host inside a denied range must survive the deny.

    OpenSandbox evaluates ``@deny`` before both the DNS-learned set and ``@allow``, so a denied
    ``10.0.0.0/8`` would beat an allowed ``10.0.0.51`` rather than lose to it.
    """
    denied = denied_cidrs(("10.0.0.51",), resolver_addresses=())
    assert not _is_denied("10.0.0.51", denied)
    assert _is_denied("10.0.0.50", denied)
    assert _is_denied("10.0.0.52", denied)
    assert _is_denied("10.255.255.255", denied)


def test_denied_cidrs_split_repeatedly_for_several_allowed_addresses():
    """Each further allowed address splits whichever fragment now contains it."""
    allowed = ("10.0.0.51", "10.0.0.83", "10.0.0.193")
    denied = denied_cidrs(allowed, resolver_addresses=())
    for address in allowed:
        assert not _is_denied(address, denied)
    for address in ("10.0.0.50", "10.0.0.84", "10.0.0.192", "10.0.0.194", "10.1.0.1"):
        assert _is_denied(address, denied)
    # Fragments must stay disjoint; nftables interval sets reject overlapping elements.
    networks = sorted(ipaddress.ip_network(target) for target in denied if ":" not in target)
    for earlier, later in zip(networks, networks[1:]):
        assert earlier.broadcast_address < later.network_address


def test_denied_cidrs_carve_out_a_resolvable_hostname(monkeypatch):
    """Hostnames are resolved so a private endpoint named by DNS is not denied by address."""
    monkeypatch.setattr(egress, "_resolve_host", lambda target: (ipaddress.ip_network("10.0.0.51"),))
    denied = denied_cidrs(("private.svc.cluster.local",), resolver_addresses=())
    assert not _is_denied("10.0.0.51", denied)
    assert _is_denied("10.0.0.52", denied)


def test_denied_cidrs_do_not_depend_on_resolution_order(monkeypatch):
    """DNS round-robin reorders a host's addresses; the deny list must come out the same."""
    answers = [
        (ipaddress.ip_network("10.48.202.65"), ipaddress.ip_network("10.48.202.204")),
        (ipaddress.ip_network("10.48.202.204"), ipaddress.ip_network("10.48.202.65")),
    ]
    monkeypatch.setattr(egress, "_resolve_host", lambda target: answers.pop(0))
    first = denied_cidrs(("inference.example.com",), resolver_addresses=())
    second = denied_cidrs(("inference.example.com",), resolver_addresses=())
    assert first == second


def test_denied_cidrs_ignore_wildcard_dns_targets():
    assert denied_cidrs(("*.com",), resolver_addresses=()) == denied_cidrs(resolver_addresses=())


def test_denied_cidrs_carve_out_the_resolver():
    """A denied resolver drops every lookup, so nothing resolves and no allow rule can fire."""
    denied = denied_cidrs(resolver_addresses=("10.96.5.5",))
    assert not _is_denied("10.96.5.5", denied)
    assert _is_denied("10.96.5.6", denied)


def test_policy_explicitly_allows_the_resolver():
    policy = build_egress_policy(EgressAllowlist(resolver_addresses=("10.96.5.5",)))
    assert "10.96.5.5" in _allows(policy)
    assert not _is_denied("10.96.5.5", _denies(policy))


def test_denied_cidrs_default_the_resolver_to_the_local_nameservers(tmp_path, monkeypatch):
    resolv_conf = tmp_path / "resolv.conf"
    resolv_conf.write_text("search svc.cluster.local\nnameserver 10.96.5.5  # cluster\n")
    monkeypatch.setattr(egress, "RESOLV_CONF_PATH", str(resolv_conf))

    assert egress.local_resolver_addresses() == ("10.96.5.5",)
    assert not _is_denied("10.96.5.5", denied_cidrs())


def test_local_resolver_addresses_tolerate_a_missing_resolv_conf(tmp_path, monkeypatch):
    monkeypatch.setattr(egress, "RESOLV_CONF_PATH", str(tmp_path / "absent"))
    assert egress.local_resolver_addresses() == ()
