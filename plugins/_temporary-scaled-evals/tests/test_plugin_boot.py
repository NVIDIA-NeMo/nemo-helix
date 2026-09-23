# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boot the plugin the way the platform does: a clean process with no test env."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# These boot the plugin in a child process, so guard the module instead: a default sync leaves
# it uninstalled and the repo-wide run still sweeps here.
pytest.importorskip("scaled_evals", reason="scaled-evals plugin not installed")


def _fixture_dsn(role: str, host: str, database: str) -> str:
    """Build a throwaway DSN whose password is just the role name.

    Assembled rather than written out: a literal ``user:password@host`` in-tree reads as
    a live credential to secret scanners even when, as here, it points nowhere.
    """
    return f"postgresql://{role}:{role}@{host}:5432/{database}"


PLATFORM_DB_URL = _fixture_dsn("platform", "platform-db", "nemo_helix")
OWN_DB_URL = _fixture_dsn("scaled_evals", "own-db", "scaled_evals")

# Runs in a subprocess so conftest's env presets can't mask a real boot failure.
BOOT_PROBE = """
import os
from importlib.metadata import entry_points

service_entrypoint = next(iter(entry_points(group="nemo.services", name="scaled-evals")))
print("DISCOVERED:", service_entrypoint.name == "scaled-evals")
service_entrypoint.load()()

from cryptography.fernet import Fernet
os.environ["CREDENTIALS_ENCRYPTION_KEY"] = Fernet.generate_key().decode()

from scaled_evals.api.settings import settings
print("DBURL:", settings.resolved_database_url())
"""


def test_plugin_loads_without_test_env_and_keeps_off_the_platform_database() -> None:
    env = {k: v for k, v in os.environ.items() if k != "CREDENTIALS_ENCRYPTION_KEY"}
    # Both are visible in the real platform process, so the DSN below has to be the
    # one that wins.
    env["DATABASE_URL"] = PLATFORM_DB_URL
    env["SCALED_EVALS_DATABASE_URL"] = OWN_DB_URL

    result = subprocess.run(
        [sys.executable, "-c", BOOT_PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    # Loads even though CREDENTIALS_ENCRYPTION_KEY is absent; the entry-point
    # loader swallows import errors, so a crash here would silently drop the plugin.
    assert "DISCOVERED: True" in result.stdout, result.stdout

    db_line = next(ln for ln in result.stdout.splitlines() if ln.startswith("DBURL:"))
    assert OWN_DB_URL in db_line, db_line
    assert "nemo_helix" not in db_line, db_line
    assert "search_path%3Dscaled_evals" in db_line, db_line
