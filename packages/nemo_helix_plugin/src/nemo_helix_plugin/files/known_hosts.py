# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pick which of a git host's scanned SSH keys to trust, the way ssh decides on a connection."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import re
from dataclasses import dataclass
from enum import Enum

from nemo_helix_plugin.files.storage_config import SshRemote
from nemo_helix_plugin.files.types import SshHostKey

_DEFAULT_SSH_PORT = 22
_HASHED_HOST_PREFIX = "|1|"


@dataclass(frozen=True)
class KnownHostEntries:
    """The ``(key_type, key)`` pairs a known_hosts file lists for one host."""

    trusted: frozenset[tuple[str, str]]
    revoked: frozenset[tuple[str, str]]


class LocalVerdict(Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LocalHostKeyCheck:
    verdict: LocalVerdict
    key: SshHostKey | None = None


def known_hosts_name(remote: SshRemote) -> str:
    """The host as known_hosts names it: bare on port 22, ``[host]:port`` otherwise."""
    if remote.port in (None, _DEFAULT_SSH_PORT):
        return remote.host
    return f"[{remote.host}]:{remote.port}"


def _hashed_host_matches(pattern: str, name: str) -> bool:
    parts = pattern[len(_HASHED_HOST_PREFIX) :].split("|")
    if len(parts) != 2:
        return False
    try:
        salt = base64.b64decode(parts[0], validate=True)
        expected = base64.b64decode(parts[1], validate=True)
    except (binascii.Error, ValueError):
        return False
    actual = hmac.new(salt, name.encode(), hashlib.sha1).digest()
    return hmac.compare_digest(actual, expected)


def _pattern_matches(pattern: str, name: str) -> bool:
    # ssh patterns know only * and ?, so the brackets of [host]:port stay literal.
    regex = "".join(".*" if char == "*" else "." if char == "?" else re.escape(char) for char in pattern)
    return re.fullmatch(regex, name, flags=re.DOTALL) is not None


def _host_matches(patterns: str, name: str) -> bool:
    if patterns.startswith(_HASHED_HOST_PREFIX):
        return _hashed_host_matches(patterns, name)
    matched = False
    for pattern in patterns.lower().split(","):
        negated = pattern.startswith("!")
        if not _pattern_matches(pattern.removeprefix("!"), name):
            continue
        if negated:
            return False
        matched = True
    return matched


def known_host_entries(known_hosts: str, name: str) -> KnownHostEntries:
    """Collect the keys *known_hosts* lists for *name*. ``@cert-authority`` lines are skipped."""
    trusted: set[tuple[str, str]] = set()
    revoked: set[tuple[str, str]] = set()
    lowered_name = name.lower()
    for raw_line in known_hosts.splitlines():
        fields = raw_line.split()
        if not fields or fields[0].startswith("#"):
            continue
        marker = None
        if fields[0].startswith("@"):
            marker, fields = fields[0], fields[1:]
        if len(fields) < 3 or marker == "@cert-authority":
            continue
        patterns, key_type, key = fields[:3]
        if not _host_matches(patterns, lowered_name):
            continue
        (revoked if marker == "@revoked" else trusted).add((key_type, key))
    return KnownHostEntries(trusted=frozenset(trusted), revoked=frozenset(revoked))


def _key_pair(host_key: SshHostKey) -> tuple[str, str]:
    fields = host_key.known_hosts_line.split()
    return fields[1], fields[2]


def is_revoked(host_key: SshHostKey, entries: KnownHostEntries) -> bool:
    return _key_pair(host_key) in entries.revoked


def check_against_known_hosts(host_keys: list[SshHostKey], entries: KnownHostEntries) -> LocalHostKeyCheck:
    """Match scanned keys against local entries, as ssh would.

    A host listed locally whose scanned keys all differ is a ``MISMATCH``: ssh refuses that
    connection, since either the host was re-keyed or someone is in the middle.
    """
    for host_key in host_keys:
        pair = _key_pair(host_key)
        if pair in entries.trusted and pair not in entries.revoked:
            return LocalHostKeyCheck(LocalVerdict.MATCH, host_key)
    if entries.trusted:
        return LocalHostKeyCheck(LocalVerdict.MISMATCH)
    return LocalHostKeyCheck(LocalVerdict.UNKNOWN)


def normalize_fingerprint(fingerprint: str) -> str:
    """``SHA256:<base64>`` without padding, whether or not the caller typed the prefix."""
    value = fingerprint.strip().removeprefix("SHA256:").rstrip("=")
    return f"SHA256:{value}"


def key_with_fingerprint(host_keys: list[SshHostKey], fingerprint: str) -> SshHostKey | None:
    wanted = normalize_fingerprint(fingerprint)
    return next((host_key for host_key in host_keys if host_key.fingerprint == wanted), None)
