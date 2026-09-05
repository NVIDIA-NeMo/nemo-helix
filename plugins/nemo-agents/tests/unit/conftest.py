# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for ``plugins/nemo-agents/tests/unit/``."""

from __future__ import annotations

from pathlib import Path

import pytest
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults


@pytest.fixture(autouse=True)
def _isolate_nmp_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory):
    """Pin ``NMP_DATA_DIR`` to a per-test tempdir for every test in this tree.

    Belt-and-braces against accidental writes to ``~/.local/share/nemo``: any
    test that resolves :func:`nmp_user_data_dir` (directly or via
    ``AgentsConfig`` defaults) will land under a tempdir that pytest
    cleans up, never the developer's real homedir.  Tests that need to
    exercise the env-var resolution itself (XDG, explicit path) override
    this in their own fixtures via ``monkeypatch``.
    """
    isolated = tmp_path_factory.mktemp("nmp-data")
    monkeypatch.setenv("NMP_DATA_DIR", str(isolated))
    # Configuration is cached; reset so the override takes effect for tests
    # that read `AgentsConfig.get()`.
    from nemo_platform_plugin.config import Configuration

    Configuration.clear_cache()
    yield
    Configuration.clear_cache()


@pytest.fixture
def ctx(tmp_path: Path) -> JobContext:
    """:class:`JobContext` with platform-style storage rooted in ``tmp_path``.

    Mirrors the shape ``run_task`` builds via
    :func:`nemo_platform_plugin.tasks.dispatcher.build_ctx_from_env` and the
    scheduler builds via ``_build_local_context``: a per-job tempdir
    containing ``persistent/`` and ``ephemeral/`` subdirs plus a
    :class:`LocalJobResults` sink rooted at ``persistent/results/``.
    """
    persistent = tmp_path / "persistent"
    ephemeral = tmp_path / "ephemeral"
    persistent.mkdir(exist_ok=True)
    ephemeral.mkdir(exist_ok=True)
    return JobContext(
        workspace="default",
        storage=StoragePaths(ephemeral=ephemeral, persistent=persistent),
        results=LocalJobResults(root=persistent / "results"),
    )


@pytest.fixture(autouse=True)
def _released_contract_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """A source checkout reports a version the packaging guard rejects."""
    import nemo_agents_plugin.container.template as template

    monkeypatch.setattr(template, "get_contract_version", lambda: "1.0.0")
