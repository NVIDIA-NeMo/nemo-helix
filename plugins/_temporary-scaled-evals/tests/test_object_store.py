# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cover the startup storage-readiness hooks after the Files migration.

``ensure_bucket`` used to create the object-store bucket on a fresh RustFS/MinIO volume; with
artifacts on the Files service there is no scaled-evals-owned bucket, so it is now a no-op and
``check_bucket`` (used by ``/v1/readyz``) delegates to the Files readiness probe. These tests
pin that behavior and that a storage outage degrades the plugin without aborting platform
startup.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

# This plugin is absent from `enabled-plugins`, so a default sync leaves it and its database
# driver uninstalled and the repo-wide test run still sweeps this directory. Skip rather than
# error there; the job that owns these tests installs the `scaled-evals` group first.
try:
    from nemo_scaled_evals_plugin.service import ScaledEvalsService
    from scaled_evals.api import s3
    from scaled_evals.api.settings import settings
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def test_ensure_bucket_is_a_noop_returning_the_workspace() -> None:
    # No object-store bucket to create anymore; filesets are created on demand at write time.
    # ensure_bucket just reports the Files workspace and never raises.
    assert s3.ensure_bucket() == settings.files_workspace


def test_check_bucket_delegates_to_files_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(s3._files_backend, "readiness_probe", lambda: calls.append("probe"))  # noqa: SLF001
    s3.check_bucket()
    assert calls == ["probe"]


def test_check_bucket_propagates_a_files_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode() -> None:
        raise RuntimeError("files unreachable")

    monkeypatch.setattr(s3._files_backend, "readiness_probe", explode)  # noqa: SLF001
    with pytest.raises(RuntimeError, match="files unreachable"):
        s3.check_bucket()


def test_startup_survives_unreachable_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Storage being down must degrade the plugin, not abort platform startup."""

    def explode(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("files unreachable")

    monkeypatch.setattr(s3, "ensure_bucket", explode)
    # Keep the database side out of it; this test is about storage readiness.
    monkeypatch.setattr(ScaledEvalsService, "_apply_migrations", lambda _self: None)

    asyncio.run(ScaledEvalsService().on_startup())
