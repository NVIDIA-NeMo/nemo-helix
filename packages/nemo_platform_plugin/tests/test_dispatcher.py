# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`nemo_platform_plugin.tasks.dispatcher`.

Pin the typed task-entrypoint contract:

- Step config comes from :data:`NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR`.
- ``JobContext`` is built from platform-provided ``NEMO_JOB_*`` env vars.
- Sync and async task entrypoints accept concrete typed clients.
- Exit-code mapping is centralized in :func:`exit_code_for`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.errors import LocalRunError
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.tasks import dispatcher as dispatcher_module
from nemo_platform_plugin.tasks.dispatcher import (
    build_ctx_from_env,
    exit_code_for,
    read_step_config,
    run_task_with_async_client,
    run_task_with_client,
)


def _setup_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    step_config: Mapping[str, object] | None = None,
    workspace: str = "ws",
    job_id: str | None = "submitted-job-name",
) -> Path:
    """Wire platform task env and return the step-config path."""
    config_path = tmp_path / "step_config.json"
    if step_config is not None:
        config_path.write_text(json.dumps(step_config), encoding="utf-8")
    monkeypatch.setenv("NEMO_JOB_STEP_CONFIG_FILE_PATH", str(config_path))
    monkeypatch.setenv("NEMO_JOB_WORKSPACE", workspace)
    monkeypatch.setenv("NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH", str(tmp_path / "p"))
    monkeypatch.setenv("NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH", str(tmp_path / "e"))
    if job_id is None:
        monkeypatch.delenv("NEMO_JOB_ID", raising=False)
    else:
        monkeypatch.setenv("NEMO_JOB_ID", job_id)

    def _fake_platform_job_results(**_kwargs: object) -> MagicMock:
        return MagicMock(name="PlatformJobResults")

    monkeypatch.setattr(
        dispatcher_module,
        "PlatformJobResults",
        _fake_platform_job_results,
    )
    return config_path


def _sdk() -> NeMoPlatform:
    return NeMoPlatform(base_url="http://platform.test", workspace="ws")


def _sync_client() -> NemoClient:
    return NemoClient(
        base_url="http://platform.test",
        workspace="ws",
        http_client=httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(404))),
        owns_http_client=True,
    )


def _async_client() -> AsyncNemoClient:
    return AsyncNemoClient(
        base_url="http://platform.test",
        workspace="ws",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(404))),
        owns_http_client=True,
    )


class TestExitCodeFor:
    def test_returns_0_when_result_is_success_shape(self) -> None:
        assert exit_code_for({"status": "completed"}) == 0

    def test_returns_0_for_arbitrary_dict_shapes(self) -> None:
        assert exit_code_for({"result": "greeting", "artifact": "file://..."}) == 0

    def test_returns_1_for_status_failed(self) -> None:
        assert exit_code_for({"status": "failed", "returncode": 124}) == 1

    def test_returns_1_for_non_zero_exit_code(self) -> None:
        assert exit_code_for({"exit_code": 1}) == 1

    def test_returns_0_for_zero_exit_code(self) -> None:
        assert exit_code_for({"exit_code": 0}) == 0

    def test_returns_0_for_non_dict_result(self) -> None:
        assert exit_code_for("hello") == 0

    def test_returns_1_for_none_return(self) -> None:
        assert exit_code_for(None) == 1


class TestReadStepConfig:
    def test_reads_dict_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        payload = {"hello": "world"}
        _setup_env(monkeypatch, tmp_path, step_config=payload)

        assert read_step_config() == payload

    def test_missing_envvar_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _setup_env(monkeypatch, tmp_path, step_config={})
        monkeypatch.delenv("NEMO_JOB_STEP_CONFIG_FILE_PATH", raising=False)

        with pytest.raises(RuntimeError, match="NEMO_JOB_STEP_CONFIG_FILE_PATH"):
            read_step_config()

    def test_non_object_config_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_path = _setup_env(monkeypatch, tmp_path)
        config_path.write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(RuntimeError, match="must be a JSON object"):
            read_step_config()

    def test_missing_file_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_path = _setup_env(monkeypatch, tmp_path)
        assert not config_path.exists()

        with pytest.raises(FileNotFoundError, match=str(config_path)):
            read_step_config()

    def test_invalid_json_reports_path_and_size(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_path = _setup_env(monkeypatch, tmp_path)
        config_path.write_text('{"broken":', encoding="utf-8")

        with pytest.raises(RuntimeError) as exc_info:
            read_step_config()

        message = str(exc_info.value)
        assert "Invalid JSON in step config" in message
        assert str(config_path) in message
        assert f"{config_path.stat().st_size} bytes" in message


class TestTypedTaskDispatch:
    def test_sync_entrypoint_passes_config_ctx_and_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        payload = {"a": 1}
        _setup_env(monkeypatch, tmp_path, step_config=payload, workspace="injected-ws")
        client = _sync_client()
        captured: dict[str, object] = {}

        class _Job(NemoJob):
            name = "sync-runtime"

            def run(self, config: dict, *, ctx: JobContext, client: NemoClient) -> dict[str, object]:
                captured["config"] = config
                captured["ctx"] = ctx
                captured["client"] = client
                return {"status": "completed"}

        try:
            rc = run_task_with_client(_Job, client=client, ctx=build_ctx_from_env(_sdk()))
        finally:
            client.close()

        assert rc == 0
        assert captured["config"] == payload
        assert captured["client"] is client
        ctx = captured["ctx"]
        assert isinstance(ctx, JobContext)
        assert ctx.workspace == "injected-ws"
        assert ctx.storage.persistent == tmp_path / "p"

    def test_async_entrypoint_passes_config_ctx_and_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        payload = {"a": 1}
        _setup_env(monkeypatch, tmp_path, step_config=payload, workspace="injected-ws")
        client = _async_client()
        captured: dict[str, object] = {}

        class _Job(NemoJob):
            name = "async-runtime"

            def run(self, config: dict, *, ctx: JobContext, async_client: AsyncNemoClient) -> dict[str, object]:
                captured["config"] = config
                captured["ctx"] = ctx
                captured["async_client"] = async_client
                return {"status": "completed"}

        try:
            rc = run_task_with_async_client(_Job, async_client=client, ctx=build_ctx_from_env(_sdk()))
        finally:
            import asyncio

            asyncio.run(client.close())

        assert rc == 0
        assert captured["config"] == payload
        assert captured["async_client"] is client
        ctx = captured["ctx"]
        assert isinstance(ctx, JobContext)
        assert ctx.workspace == "injected-ws"

    def test_constructor_failure_maps_to_setup_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _setup_env(monkeypatch, tmp_path, step_config={})
        client = _sync_client()

        class _Job(NemoJob):
            name = "bad-ctor"

            def __init__(self) -> None:
                raise RuntimeError("ctor blew up")

            def run(self, config: dict, *, ctx: JobContext, client: NemoClient) -> dict[str, object]:
                return {"status": "completed"}

        try:
            rc = run_task_with_client(_Job, client=client, ctx=build_ctx_from_env(_sdk()))
        finally:
            client.close()

        assert rc == 2

    def test_run_exception_maps_to_run_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _setup_env(monkeypatch, tmp_path, step_config={})
        client = _sync_client()

        class _Job(NemoJob):
            name = "raises"

            def run(self, config: dict, *, ctx: JobContext, client: NemoClient) -> dict[str, object]:
                raise RuntimeError("kaboom")

        try:
            rc = run_task_with_client(_Job, client=client, ctx=build_ctx_from_env(_sdk()))
        finally:
            client.close()

        assert rc == 1

    def test_local_run_error_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _setup_env(monkeypatch, tmp_path, step_config={})
        client = _sync_client()

        class _Job(NemoJob):
            name = "raises-local-run-error"

            def run(self, config: dict, *, ctx: JobContext, client: NemoClient) -> dict[str, object]:
                raise LocalRunError("missing fileset upload")

        try:
            with pytest.raises(LocalRunError, match="fileset upload"):
                run_task_with_client(_Job, client=client, ctx=build_ctx_from_env(_sdk()))
        finally:
            client.close()

    def test_config_read_failure_maps_to_setup_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _setup_env(monkeypatch, tmp_path)
        client = _sync_client()

        class _Job(NemoJob):
            name = "x"

            def run(self, config: dict, *, ctx: JobContext, client: NemoClient) -> dict[str, object]:
                return {"status": "completed"}

        try:
            rc = run_task_with_client(_Job, client=client, ctx=build_ctx_from_env(_sdk()))
        finally:
            client.close()

        assert rc == 2


class TestBuildCtxFromEnv:
    """``build_ctx_from_env`` reads the platform-injected env vars."""

    @staticmethod
    def _patch_results(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        sentinel = MagicMock(name="PlatformJobResults")

        def _fake_platform_job_results(**_kwargs: object) -> MagicMock:
            return sentinel

        monkeypatch.setattr(
            dispatcher_module,
            "PlatformJobResults",
            _fake_platform_job_results,
        )
        return sentinel

    def test_reads_workspace_and_paths_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_results(monkeypatch)
        persistent = tmp_path / "persistent"
        ephemeral = tmp_path / "ephemeral"
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", "platform-ws")
        monkeypatch.setenv("NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH", str(persistent))
        monkeypatch.setenv("NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH", str(ephemeral))
        monkeypatch.setenv("NEMO_JOB_ID", "submitted-job-name")

        ctx = build_ctx_from_env(_sdk())

        assert ctx.workspace == "platform-ws"
        assert ctx.storage.persistent == persistent
        assert ctx.storage.ephemeral == ephemeral
        assert ctx.job_id == "submitted-job-name"

    def test_results_use_submitted_job_id_not_class_name(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def _capture_platform_job_results(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            return MagicMock(name="PlatformJobResults")

        monkeypatch.setattr(
            dispatcher_module,
            "PlatformJobResults",
            _capture_platform_job_results,
        )
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", "ws")
        monkeypatch.setenv("NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH", str(tmp_path / "p"))
        monkeypatch.setenv("NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH", str(tmp_path / "e"))
        monkeypatch.setenv("NEMO_JOB_ID", "evaluate-agent-abc123")

        build_ctx_from_env(_sdk())

        assert captured["job_name"] == "evaluate-agent-abc123"
        assert captured["workspace"] == "ws"

    def test_results_default_is_platform_job_results(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentinel = self._patch_results(monkeypatch)
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", "ws")
        monkeypatch.setenv("NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH", str(tmp_path / "p"))
        monkeypatch.setenv("NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH", str(tmp_path / "e"))
        monkeypatch.setenv("NEMO_JOB_ID", "submitted-job-name")

        ctx = build_ctx_from_env(_sdk())

        assert ctx.results is sentinel

    def test_missing_workspace_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEMO_JOB_WORKSPACE", raising=False)

        with pytest.raises(RuntimeError, match="NEMO_JOB_WORKSPACE"):
            build_ctx_from_env(_sdk())

    def test_empty_workspace_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", "")

        with pytest.raises(RuntimeError, match="NEMO_JOB_WORKSPACE"):
            build_ctx_from_env(_sdk())

    def test_whitespace_workspace_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", "   ")

        with pytest.raises(RuntimeError, match="NEMO_JOB_WORKSPACE"):
            build_ctx_from_env(_sdk())

    def test_missing_persistent_storage_builds_ctx_but_access_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._patch_results(monkeypatch)
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", "ws")
        monkeypatch.delenv("NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH", raising=False)
        monkeypatch.setenv("NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH", str(tmp_path / "e"))
        monkeypatch.setenv("NEMO_JOB_ID", "test-job")

        ctx = build_ctx_from_env(_sdk())
        assert ctx.storage.ephemeral == tmp_path / "e"

        with pytest.raises(RuntimeError, match="did not request persistent storage"):
            _ = ctx.storage.persistent

    def test_missing_ephemeral_storage_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", "ws")
        monkeypatch.setenv("NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH", str(tmp_path / "p"))
        monkeypatch.delenv("NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH", raising=False)

        with pytest.raises(RuntimeError, match="NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH"):
            build_ctx_from_env(_sdk())

    def test_missing_job_id_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", "ws")
        monkeypatch.setenv("NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH", str(tmp_path / "p"))
        monkeypatch.setenv("NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH", str(tmp_path / "e"))
        monkeypatch.delenv("NEMO_JOB_ID", raising=False)

        with pytest.raises(RuntimeError, match="NEMO_JOB_ID"):
            build_ctx_from_env(_sdk())
