# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import hashlib
import hmac

import pytest
from nemo_helix_plugin.files.known_hosts import (
    KnownHostEntries,
    LocalVerdict,
    check_against_known_hosts,
    key_with_fingerprint,
    known_host_entries,
    known_hosts_name,
    normalize_fingerprint,
)
from nemo_helix_plugin.files.storage_config import parse_ssh_remote
from nemo_helix_plugin.files.types import SshHostKey

ED25519 = ("ssh-ed25519", "AAAAC3NzaC1lZDI1NTE5AAAAIEd25519KeyBlob")
RSA = ("ssh-rsa", "AAAAB3NzaC1yc2EAAAADAQABAAABAQRsaKeyBlob")


def _scanned(host: str, key_type: str, key: str, fingerprint: str) -> SshHostKey:
    return SshHostKey(key_type=key_type, fingerprint=fingerprint, known_hosts_line=f"{host} {key_type} {key}")


def _hashed(name: str) -> str:
    salt = b"0123456789abcdefghij"
    digest = hmac.new(salt, name.encode(), hashlib.sha1).digest()
    return f"|1|{base64.b64encode(salt).decode()}|{base64.b64encode(digest).decode()}"


@pytest.mark.parametrize(
    ("url", "name"),
    [
        ("git@gitlab.example.com:org/repo.git", "gitlab.example.com"),
        ("ssh://git@gitlab.example.com/org/repo.git", "gitlab.example.com"),
        ("ssh://git@gitlab.example.com:2222/org/repo.git", "[gitlab.example.com]:2222"),
    ],
)
def test_known_hosts_name_brackets_only_a_non_default_port(url: str, name: str) -> None:
    assert known_hosts_name(parse_ssh_remote(url)) == name


def test_known_host_entries_matches_plain_comma_separated_and_hashed_hosts() -> None:
    known_hosts = "\n".join(
        [
            "# comment",
            f"other.example.com,gitlab.example.com {ED25519[0]} {ED25519[1]}",
            f"{_hashed('gitlab.example.com')} {RSA[0]} {RSA[1]}",
            f"{_hashed('elsewhere.example.com')} ssh-ed25519 AAAAunrelated",
        ]
    )

    entries = known_host_entries(known_hosts, "gitlab.example.com")

    assert entries.trusted == {ED25519, RSA}
    assert entries.revoked == frozenset()


def test_known_host_entries_honors_wildcards_and_negation() -> None:
    known_hosts = f"*.example.com,!gitlab.example.com {ED25519[0]} {ED25519[1]}\n*.example.com {RSA[0]} {RSA[1]}"

    assert known_host_entries(known_hosts, "gitlab.example.com").trusted == {RSA}
    assert known_host_entries(known_hosts, "git.example.com").trusted == {ED25519, RSA}


def test_known_host_entries_separates_revoked_and_skips_cert_authorities() -> None:
    known_hosts = "\n".join(
        [
            f"@revoked gitlab.example.com {ED25519[0]} {ED25519[1]}",
            f"@cert-authority gitlab.example.com {RSA[0]} {RSA[1]}",
        ]
    )

    entries = known_host_entries(known_hosts, "gitlab.example.com")

    assert entries.trusted == frozenset()
    assert entries.revoked == {ED25519}


def test_known_host_entries_matches_bracketed_port_form() -> None:
    known_hosts = f"[gitlab.example.com]:2222 {ED25519[0]} {ED25519[1]}"

    assert known_host_entries(known_hosts, "[gitlab.example.com]:2222").trusted == {ED25519}
    assert known_host_entries(known_hosts, "gitlab.example.com").trusted == frozenset()


def test_check_returns_the_scanned_key_listed_locally() -> None:
    scanned = [
        _scanned("gitlab.example.com", *ED25519, fingerprint="SHA256:ed"),
        _scanned("gitlab.example.com", *RSA, fingerprint="SHA256:rsa"),
    ]

    check = check_against_known_hosts(scanned, KnownHostEntries(trusted=frozenset({RSA}), revoked=frozenset()))

    assert check.verdict is LocalVerdict.MATCH
    assert check.key == scanned[1]


def test_check_is_a_mismatch_when_the_host_is_listed_under_other_keys() -> None:
    scanned = [_scanned("gitlab.example.com", *ED25519, fingerprint="SHA256:ed")]

    check = check_against_known_hosts(scanned, KnownHostEntries(trusted=frozenset({RSA}), revoked=frozenset()))

    assert check.verdict is LocalVerdict.MISMATCH
    assert check.key is None


def test_check_never_matches_a_revoked_key() -> None:
    scanned = [_scanned("gitlab.example.com", *ED25519, fingerprint="SHA256:ed")]

    check = check_against_known_hosts(
        scanned, KnownHostEntries(trusted=frozenset({ED25519}), revoked=frozenset({ED25519}))
    )

    assert check.verdict is LocalVerdict.MISMATCH


def test_check_is_unknown_for_an_unlisted_host() -> None:
    scanned = [_scanned("gitlab.example.com", *ED25519, fingerprint="SHA256:ed")]

    check = check_against_known_hosts(scanned, KnownHostEntries(trusted=frozenset(), revoked=frozenset()))

    assert check.verdict is LocalVerdict.UNKNOWN


@pytest.mark.parametrize("typed", ["SHA256:abc", "abc", "abc=", " SHA256:abc= "])
def test_fingerprints_compare_with_or_without_prefix_and_padding(typed: str) -> None:
    scanned = [_scanned("gitlab.example.com", *ED25519, fingerprint="SHA256:abc")]

    assert normalize_fingerprint(typed) == "SHA256:abc"
    assert key_with_fingerprint(scanned, typed) == scanned[0]
