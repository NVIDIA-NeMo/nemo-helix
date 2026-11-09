# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import click
import pytest
from nemo_helix_plugin import cli_git_host_keys
from nemo_helix_plugin.cli_git_host_keys import choose_host_key, key_type_label
from nemo_helix_plugin.files.storage_config import parse_ssh_remote
from nemo_helix_plugin.files.types import SshHostKey

REMOTE = parse_ssh_remote("git@gitlab.example.com:org/repo.git")
ED25519 = SshHostKey(
    key_type="ssh-ed25519",
    fingerprint="SHA256:edfingerprint",
    known_hosts_line="gitlab.example.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEd25519KeyBlob",
)
RSA = SshHostKey(
    key_type="ssh-rsa",
    fingerprint="SHA256:rsafingerprint",
    known_hosts_line="gitlab.example.com ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQRsaKeyBlob",
)
SCANNED = [ED25519, RSA]


@pytest.fixture
def no_known_hosts(tmp_path: Path) -> Path:
    return tmp_path / "missing_known_hosts"


@pytest.fixture
def interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_git_host_keys, "is_stdin_available", lambda: False)


@pytest.fixture
def piped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_git_host_keys, "is_stdin_available", lambda: True)


def _known_hosts(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "known_hosts"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize(
    ("key_type", "label"),
    [("ssh-ed25519", "ED25519"), ("ssh-rsa", "RSA"), ("ecdsa-sha2-nistp256", "ECDSA"), ("sk-ssh-x", "sk-ssh-x")],
)
def test_key_type_label(key_type: str, label: str) -> None:
    assert key_type_label(key_type) == label


def test_host_key_option_picks_the_matching_key(piped: None, no_known_hosts: Path) -> None:
    chosen = choose_host_key(SCANNED, REMOTE, host_key="rsafingerprint", known_hosts_file=no_known_hosts)

    assert chosen == RSA


def test_host_key_option_without_a_match_lists_what_the_host_offered(piped: None, no_known_hosts: Path) -> None:
    with pytest.raises(click.ClickException) as raised:
        choose_host_key(SCANNED, REMOTE, host_key="SHA256:other", known_hosts_file=no_known_hosts)

    assert "offered no key with fingerprint SHA256:other" in raised.value.message
    assert "ED25519 SHA256:edfingerprint" in raised.value.message


def test_key_already_in_known_hosts_is_trusted_without_asking(piped: None, tmp_path: Path) -> None:
    known_hosts = _known_hosts(tmp_path, "gitlab.example.com ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQRsaKeyBlob")

    chosen = choose_host_key(SCANNED, REMOTE, host_key=None, known_hosts_file=known_hosts)

    assert chosen == RSA


def test_host_listed_under_another_key_is_refused(interactive: None, tmp_path: Path) -> None:
    known_hosts = _known_hosts(tmp_path, "gitlab.example.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDifferentKey")

    with pytest.raises(click.ClickException) as raised:
        choose_host_key(SCANNED, REMOTE, host_key=None, known_hosts_file=known_hosts)

    assert "does not match" in raised.value.message


def test_unknown_host_without_a_terminal_names_host_key(piped: None, no_known_hosts: Path) -> None:
    with pytest.raises(click.UsageError) as raised:
        choose_host_key(SCANNED, REMOTE, host_key=None, known_hosts_file=no_known_hosts)

    assert "--host-key" in raised.value.message
    assert "RSA SHA256:rsafingerprint" in raised.value.message


def test_unknown_host_on_a_terminal_confirms_the_preferred_key(
    interactive: None, no_known_hosts: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli_git_host_keys.typer, "confirm", lambda *args, **kwargs: True)

    chosen = choose_host_key(SCANNED, REMOTE, host_key=None, known_hosts_file=no_known_hosts)

    assert chosen == ED25519
    assert "ED25519 key fingerprint is SHA256:edfingerprint" in capsys.readouterr().err


def test_declining_the_prompt_aborts(interactive: None, no_known_hosts: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _decline(*args: object, **kwargs: object) -> bool:
        raise click.Abort

    monkeypatch.setattr(cli_git_host_keys.typer, "confirm", _decline)

    with pytest.raises(click.Abort):
        choose_host_key(SCANNED, REMOTE, host_key=None, known_hosts_file=no_known_hosts)


def test_no_scanned_keys_is_an_error(piped: None, no_known_hosts: Path) -> None:
    with pytest.raises(click.ClickException, match="returned no SSH host keys"):
        choose_host_key([], REMOTE, host_key=None, known_hosts_file=no_known_hosts)
