# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decide which SSH host key a git storage trusts: ``--host-key``, then local known_hosts, then a prompt."""

from __future__ import annotations

from pathlib import Path

import click
import typer
from nemo_helix_plugin.cli_input import is_stdin_available
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.known_hosts import (
    LocalHostKeyCheck,
    LocalVerdict,
    check_against_known_hosts,
    key_with_fingerprint,
    known_host_entries,
    known_hosts_name,
    normalize_fingerprint,
)
from nemo_helix_plugin.files.storage_config import GitStorageConfig, SshRemote
from nemo_helix_plugin.files.types import FindGitRepositoryFilesRequest, SshHostKey
from typer import confirm

DEFAULT_KNOWN_HOSTS_FILE = Path("~/.ssh/known_hosts")

_KEY_TYPE_LABELS = {"ssh-ed25519": "ED25519", "ssh-rsa": "RSA"}


def key_type_label(key_type: str) -> str:
    """The algorithm name ssh prints beside a fingerprint, e.g. ED25519 or ECDSA."""
    if key_type.startswith("ecdsa-"):
        return "ECDSA"
    return _KEY_TYPE_LABELS.get(key_type, key_type)


def _describe(host_key: SshHostKey) -> str:
    return f"{key_type_label(host_key.key_type)} {host_key.fingerprint}"


def _key_list(host_keys: list[SshHostKey]) -> str:
    return "\n".join(f"  {_describe(host_key)}" for host_key in host_keys)


def _check_local(host_keys: list[SshHostKey], remote: SshRemote, known_hosts_file: Path) -> LocalHostKeyCheck:
    try:
        text = known_hosts_file.expanduser().read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return LocalHostKeyCheck(LocalVerdict.UNKNOWN)
    return check_against_known_hosts(host_keys, known_host_entries(text, known_hosts_name(remote)))


def choose_host_key(
    host_keys: list[SshHostKey],
    remote: SshRemote,
    *,
    host_key: str | None,
    known_hosts_file: Path,
) -> SshHostKey:
    """Pick the scanned key to trust, or raise ``click.ClickException`` / ``click.Abort``."""
    authority = remote.authority
    if not host_keys:
        raise click.ClickException(f"{authority} returned no SSH host keys.")

    if host_key is not None:
        chosen = key_with_fingerprint(host_keys, host_key)
        if chosen is None:
            raise click.ClickException(
                f"{authority} offered no key with fingerprint {normalize_fingerprint(host_key)}. "
                f"It offered:\n{_key_list(host_keys)}"
            )
        return chosen

    check = _check_local(host_keys, remote, known_hosts_file)
    if check.verdict is LocalVerdict.MATCH and check.key is not None:
        typer.echo(f"Host key for {authority} matches {known_hosts_file}: {_describe(check.key)}", err=True)
        return check.key
    if check.verdict is LocalVerdict.MISMATCH:
        raise click.ClickException(
            f"The host key of {authority} does not match {known_hosts_file}. The host may have been re-keyed, "
            "or someone may be intercepting the connection. Nothing was read. If you know the host changed "
            f"its key, update {known_hosts_file} or pass --host-key with the new fingerprint. It offered:\n"
            f"{_key_list(host_keys)}"
        )

    if is_stdin_available():
        raise click.UsageError(
            f"{authority} is not in {known_hosts_file}. Check one of its fingerprints against one from a "
            f"trusted source, then pass it with --host-key. It offered:\n{_key_list(host_keys)}"
        )
    preferred = host_keys[0]
    typer.echo(
        f"The authenticity of host '{authority}' can't be established.\n"
        f"{key_type_label(preferred.key_type)} key fingerprint is {preferred.fingerprint}.",
        err=True,
    )
    confirm("Trust this key and continue?", abort=True, err=True)
    return preferred


def trust_git_host(
    files_client: FilesClient,
    *,
    workspace: str | None,
    storage: GitStorageConfig,
    file_name: str,
    host_key: str | None,
    known_hosts_file: Path = DEFAULT_KNOWN_HOSTS_FILE,
) -> GitStorageConfig:
    """Return *storage* with ``known_hosts`` set to the one scanned key the caller trusts."""
    scan = files_client.find_git_repository_files(
        workspace=workspace,
        body=FindGitRepositoryFilesRequest(storage=storage.model_copy(update={"known_hosts": ""}), file_name=file_name),
    ).data()
    chosen = choose_host_key(scan.host_keys, storage.remote, host_key=host_key, known_hosts_file=known_hosts_file)
    return storage.model_copy(update={"known_hosts": chosen.known_hosts_line})
