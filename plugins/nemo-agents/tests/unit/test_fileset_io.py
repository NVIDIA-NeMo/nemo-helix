# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared fileset staging helpers used by agents jobs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nemo_agents_plugin.jobs.fileset_io import resolve_output, resolve_staged_config, split_fileset_ref
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.refs import FilesetRef


def test_resolve_staged_config_local_pass_through(tmp_path: Path, ctx: JobContext) -> None:
    target = str(tmp_path / "optimize.yml")
    with resolve_staged_config(target, None, workspace="default", ctx=ctx, sdk=None, kind="optimize-config") as p:
        assert p == Path(target)


@pytest.mark.parametrize("ref", ["", "/name", "workspace/", "workspace/name/extra"])
def test_split_fileset_ref_rejects_invalid_refs(ref: str) -> None:
    with pytest.raises(ValueError, match="invalid entity reference"):
        split_fileset_ref(ref, "default")


def test_resolve_staged_config_fileset_downloads_via_sdk(tmp_path: Path, ctx: JobContext) -> None:
    sdk = MagicMock()
    manager = MagicMock()

    def _fake_download(_ref: str, *, local_dir: Path) -> None:
        Path(local_dir, "optimize.yml").write_text("optimizer: {}")

    manager.download_from_url.side_effect = _fake_download

    with (
        patch("nemo_agents_plugin.jobs.fileset_io.client_from_platform", return_value=MagicMock()),
        patch("nemo_agents_plugin.jobs.fileset_io._fileset_manager", return_value=manager) as manager_factory,
    ):
        with resolve_staged_config(
            "optimize.yml",
            FilesetRef("nemo-agent-optimize-calc"),
            workspace="default",
            ctx=ctx,
            sdk=sdk,
            kind="optimize-config",
        ) as resolved:
            assert resolved.is_file()
            assert resolved.name == "optimize.yml"
            assert resolved.read_text() == "optimizer: {}"

    manager_factory.assert_called_once()
    assert manager_factory.call_args.kwargs["fileset"] == "nemo-agent-optimize-calc"
    assert manager_factory.call_args.kwargs["workspace"] == "default"
    manager.download_from_url.assert_called_once()
    assert manager.download_from_url.call_args.args == ("default/nemo-agent-optimize-calc",)


def test_resolve_staged_config_fileset_without_sdk_raises(ctx: JobContext) -> None:
    with pytest.raises(Exception) as exc:
        with resolve_staged_config(
            "optimize.yml",
            FilesetRef("fs"),
            workspace="default",
            ctx=ctx,
            sdk=None,
            kind="optimize-config",
        ):
            pass
    assert "sdk" in str(exc.value).lower()


def test_resolve_staged_config_empty_fileset_ref_is_invalid(ctx: JobContext) -> None:
    with pytest.raises(ValueError, match="invalid entity reference"):
        with resolve_staged_config(
            "optimize.yml",
            FilesetRef(""),
            workspace="default",
            ctx=ctx,
            sdk=None,
            kind="optimize-config",
        ):
            pass


def test_resolve_staged_config_rejects_path_escape(ctx: JobContext) -> None:
    sdk = MagicMock()
    manager = MagicMock()
    with (
        patch("nemo_agents_plugin.jobs.fileset_io.client_from_platform", return_value=MagicMock()),
        patch("nemo_agents_plugin.jobs.fileset_io._fileset_manager", return_value=manager),
        pytest.raises(ValueError, match="outside the downloaded fileset"),
    ):
        with resolve_staged_config(
            "../evil.yml",
            FilesetRef("fs"),
            workspace="default",
            ctx=ctx,
            sdk=sdk,
            kind="optimize-config",
        ):
            pass


def test_resolve_output_none_uses_persistent_results(ctx: JobContext) -> None:
    with resolve_output(None, workspace="default", ctx=ctx, sdk=None, kind="optimize") as base:
        assert base == ctx.storage.persistent / "results"
        assert base.is_dir()


def test_resolve_output_fileset_uploads_on_clean_exit(ctx: JobContext) -> None:
    sdk = MagicMock()
    manager = MagicMock()

    with (
        patch("nemo_agents_plugin.jobs.fileset_io.client_from_platform", return_value=MagicMock()),
        patch("nemo_agents_plugin.jobs.fileset_io._fileset_manager", return_value=manager) as manager_factory,
    ):
        with resolve_output(FilesetRef("optimize-out"), workspace="default", ctx=ctx, sdk=sdk, kind="optimize"):
            pass

    manager_factory.assert_called_once()
    assert manager_factory.call_args.kwargs["fileset"] == "optimize-out"
    assert manager_factory.call_args.kwargs["workspace"] == "default"
    assert manager_factory.call_args.kwargs["ensure_fileset_exists"] is True
    manager.validate_storage.assert_called_once_with()
    manager.upload.assert_called_once()
    assert manager.upload.call_args.kwargs["remote_path"] == ""


def test_resolve_output_fileset_skips_upload_when_body_raises(ctx: JobContext) -> None:
    sdk = MagicMock()
    manager = MagicMock()
    with pytest.raises(RuntimeError, match="boom"):
        with (
            patch("nemo_agents_plugin.jobs.fileset_io.client_from_platform", return_value=MagicMock()),
            patch("nemo_agents_plugin.jobs.fileset_io._fileset_manager", return_value=manager),
        ):
            with resolve_output(FilesetRef("optimize-out"), workspace="default", ctx=ctx, sdk=sdk, kind="optimize"):
                raise RuntimeError("boom")
    manager.upload.assert_not_called()


def test_resolve_output_empty_fileset_ref_is_invalid(ctx: JobContext) -> None:
    sdk = MagicMock()
    with pytest.raises(ValueError, match="invalid entity reference"):
        with resolve_output(FilesetRef(""), workspace="default", ctx=ctx, sdk=sdk, kind="optimize"):
            pass
