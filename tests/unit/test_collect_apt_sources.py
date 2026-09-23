# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import stat
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "docker" / "scripts" / "collect-apt-sources.sh"


def write_executable(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).lstrip())
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_installed_sources_use_installed_source_version_and_explicit_sources_stay_unversioned(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    apt_get_log = tmp_path / "apt-get.log"
    dpkg_query_log = tmp_path / "dpkg-query.log"

    write_executable(
        fake_bin / "dpkg-query",
        f"""
        #!/usr/bin/env bash
        printf '%s\\n' "$*" >> "{dpkg_query_log}"
        printf 'libssl3:amd64\\t3.0.2-0ubuntu1.20\\topenssl\\t3.0.2-0ubuntu1.20\\n'
        """,
    )
    write_executable(
        fake_bin / "apt-cache",
        """
        #!/usr/bin/env bash
        package="${@: -1}"
        case "${package}" in
            libssl3:amd64)
                printf 'Package: libssl3\\nSource: openssl (3.0.2-0ubuntu1.20)\\n'
                ;;
            explicit-bin)
                printf 'Package: explicit-bin\\nSource: explicit-src (9.9-1)\\n'
                ;;
        esac
        """,
    )
    write_executable(
        fake_bin / "apt-get",
        f"""
        #!/usr/bin/env bash
        printf '%s\\n' "$*" >> "{apt_get_log}"
        case "$1" in
            update|source|clean)
                exit 0
                ;;
        esac
        exit 1
        """,
    )

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["NHX_COLLECT_SOURCES"] = "1"
    output_dir = tmp_path / "out"

    result = subprocess.run(
        ["bash", str(SCRIPT), str(output_dir), "--installed", "explicit-bin"],
        check=False,
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    dpkg_query_args = dpkg_query_log.read_text()
    assert "${source:Package}" in dpkg_query_args
    assert "${source:Version}" in dpkg_query_args

    apt_get_calls = apt_get_log.read_text().splitlines()
    assert "source --download-only --only-source openssl=3.0.2-0ubuntu1.20" in apt_get_calls
    assert "source --download-only --only-source explicit-src" in apt_get_calls
    assert not any("explicit-src=" in call for call in apt_get_calls)

    downloaded_sources = (output_dir / "manifests" / "downloaded-apt-sources.txt").read_text().splitlines()
    assert "openssl=3.0.2-0ubuntu1.20" in downloaded_sources
    assert "explicit-src" in downloaded_sources
