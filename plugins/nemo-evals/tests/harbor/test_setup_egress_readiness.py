# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The lookup that checks an opened setup egress window, run under /bin/sh with fake resolver tools."""

import subprocess
from pathlib import Path

import pytest
from nemo_evals.harbor.setup_egress_readiness import RESOLVE_HOSTS_SCRIPT, resolvable_hosts


def _tool(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)


def _resolve(bin_dir: Path, hosts: str, attempts: int = 3) -> subprocess.CompletedProcess[str]:
    # bin_dir comes first so its fakes win over any real tool; /bin keeps sh and sleep.
    env = {"PATH": f"{bin_dir}:/bin", "HOSTS": hosts, "ATTEMPTS": str(attempts), "DELAY": "0"}
    return subprocess.run(["/bin/sh", "-c", RESOLVE_HOSTS_SCRIPT], env=env, capture_output=True, text=True, timeout=30)


@pytest.fixture
def bin_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "bin"
    directory.mkdir()
    return directory


def test_hosts_that_resolve_pass(bin_dir):
    _tool(bin_dir, "getent", "exit 0")
    result = _resolve(bin_dir, "astral.sh github.com")
    assert result.returncode == 0 and result.stdout == ""


def test_a_lookup_that_fails_at_first_is_retried(bin_dir, tmp_path):
    """The sidecar's DNS proxy may not answer yet right after the rules are applied."""
    count = tmp_path / "count"
    count.write_text("0")
    _tool(bin_dir, "getent", f"n=$(($(cat {count}) + 1)); echo $n > {count}; [ $n -ge 3 ]")
    assert _resolve(bin_dir, "astral.sh", attempts=3).returncode == 0
    assert count.read_text().strip() == "3"


def test_hosts_that_never_resolve_are_named(bin_dir):
    _tool(bin_dir, "getent", '[ "$2" != astral.sh ] && [ "$2" != nodejs.org ]')
    result = _resolve(bin_dir, "astral.sh github.com nodejs.org")
    assert result.returncode == 1
    assert result.stdout.split() == ["astral.sh", "nodejs.org"]


def test_nslookup_is_used_when_there_is_no_getent_or_python(bin_dir):
    _tool(bin_dir, "nslookup", '[ "$1" != astral.sh ]')
    result = _resolve(bin_dir, "astral.sh github.com", attempts=1)
    assert result.returncode == 1 and result.stdout.split() == ["astral.sh"]


def test_an_image_without_a_resolver_tool_is_not_checked(bin_dir):
    assert _resolve(bin_dir, "astral.sh").returncode == 0


def test_only_hostnames_are_looked_up():
    hosts = ("astral.sh", "*.githubusercontent.com", "10.77.0.9", "10.0.0.0/8", "::1", "nodejs.org")
    assert resolvable_hosts(hosts) == ("astral.sh", "nodejs.org")
