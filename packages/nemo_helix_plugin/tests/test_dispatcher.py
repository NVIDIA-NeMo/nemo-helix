# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`nemo_helix_plugin.tasks.dispatcher`.

Pin the contract:

- Reads the step config from :data:`NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR`.
- Builds the :class:`JobContext` from the ``NEMO_JOB_*`` env vars when
  no override is supplied; honours an explicit ``ctx=`` kwarg verbatim.
- Invokes ``job.run`` with signature-based DI of ``ctx`` / ``sdk`` /
  ``async_sdk`` (mirrors :func:`nemo_helix_plugin.run_dependencies.resolve_run_kwargs`).
- Exit codes follow two recognised in-tree failure conventions:
  ``{"status": "failed", ...}`` → ``1``;
  ``{"exit_code": <non-zero>, ...}`` → ``1``;
  any other return → ``0``; ``run`` raising → ``1``;
  pre-run setup failures → ``2``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults
from nemo_helix_plugin.tasks import dispatcher as dispatcher_module
from nemo_helix_plugin.tasks.dispatcher import build_ctx_from_env, run_task


def _setup_env(
    monkeypatch,
    tmp_path: Path,
    *,
    step_config: dict | None = None,
    workspace: str = "ws",
    job_id: str | None = "submitted-job-name",
) -> Path:
    """Wire the env so :func:`run_task` can read step config + ctx.

    Patches the task-SDK-to-results boundary to a no-op so the env-default ctx
    can be built without a real Files / Jobs SDK round-trip; tests that care
    about the results sink override ``ctx`` explicitly.

    Returns the step-config path so tests can re-point or delete it.
    """
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


_DEFAULT_SDK = NeMoHelix(base_url="http://platform.test", workspace="ws")


class TestExitCodes:
    def test_returns_0_when_run_returns_dict(self, monkeypatch, tmp_path: Path) -> None:
        _setup_env(monkeypatch, tmp_path, step_config={"hello": "world"})

        class _Job(NemoJob):
            name = "ok"

            def run(self, config: dict) -> dict:
                return {"status": "completed", "got": config}

        rc = run_task(_Job, sdk=_DEFAULT_SDK)

        assert rc == 0

    def test_returns_0_for_arbitrary_dict_shapes(self, monkeypatch, tmp_path: Path) -> None:
        # The NemoJob contract doesn't require a ``status`` or
        # ``exit_code`` field — the in-tree say-hello job returns
        # ``{"result": ..., "artifact": ...}``, for example. Dicts
        # without either field map to success.
        _setup_env(monkeypatch, tmp_path, step_config={})

        class _Job(NemoJob):
            name = "no-status"

            def run(self, config: dict) -> dict:
                return {"result": "greeting", "artifact": "file://..."}

        rc = run_task(_Job, sdk=_DEFAULT_SDK)

        assert rc == 0

    def test_returns_1_for_status_failed(self, monkeypatch, tmp_path: Path) -> None:
        # Pin the agent-style job convention: a
        # ``{"status": "failed", "returncode": ...}`` return signals
        # task failure that propagates as a non-zero process exit.
        _setup_env(monkeypatch, tmp_path, step_config={})

        class _Job(NemoJob):
            name = "agents-style"

            def run(self, config: dict) -> dict:
                return {"status": "failed", "returncode": 124}

        rc = run_task(_Job, sdk=_DEFAULT_SDK)

        assert rc == 1

    def test_returns_1_for_non_zero_exit_code(self, monkeypatch, tmp_path: Path) -> None:
        # Pin the data-designer CreateJob convention: ``{"exit_code": N}``
        # where ``N != 0`` signals failure.
        _setup_env(monkeypatch, tmp_path, step_config={})

        class _Job(NemoJob):
            name = "data-designer-style"

            def run(self, config: dict) -> dict:
                return {"exit_code": 1}

        rc = run_task(_Job, sdk=_DEFAULT_SDK)

        assert rc == 1

    def test_returns_0_for_zero_exit_code(self, monkeypatch, tmp_path: Path) -> None:
        # ``{"exit_code": 0}`` is a successful CreateJob return.
        _setup_env(monkeypatch, tmp_path, step_config={})

        class _Job(NemoJob):
            name = "dd-success"

            def run(self, config: dict) -> dict:
                return {"exit_code": 0}

        rc = run_task(_Job, sdk=_DEFAULT_SDK)

        assert rc == 0

    def test_returns_0_for_non_dict_result(self, monkeypatch, tmp_path: Path) -> None:
        # The dispatcher doesn't police the return type — non-dict,
        # non-None values are treated as success. Jobs that want to
        # signal failure raise instead.
        _setup_env(monkeypatch, tmp_path, step_config={})

        class _Job(NemoJob):
            name = "non-dict"

            def run(self, config: dict) -> dict:
                # Intentionally violate the typed contract to pin the dispatcher
                # behavior for bad runtime returns without hiding it behind a cast.
                return "hello"  # ty: ignore[invalid-return-type]

        rc = run_task(_Job, sdk=_DEFAULT_SDK)

        assert rc == 0

    def test_returns_1_for_none_return(self, monkeypatch, tmp_path: Path) -> None:
        # ``None`` (likely a refactor that dropped a ``return``)
        # collapses to exit 1 instead of silently reporting success.
        _setup_env(monkeypatch, tmp_path, step_config={})

        class _Job(NemoJob):
            name = "forgot-return"

            def run(self, config: dict) -> dict:  # ty: ignore[empty-body]
                # Intentionally no return — most plausible failure mode.
                pass

        rc = run_task(_Job, sdk=_DEFAULT_SDK)

        assert rc == 1

    def test_returns_1_when_run_raises(self, monkeypatch, tmp_path: Path) -> None:
        _setup_env(monkeypatch, tmp_path, step_config={})

        class _Job(NemoJob):
            name = "raises"

            def run(self, config: dict) -> dict:
                raise RuntimeError("kaboom")

        rc = run_task(_Job, sdk=_DEFAULT_SDK)

        assert rc == 1

    def test_returns_2_when_step_config_envvar_missing(self, monkeypatch, tmp_path: Path) -> None:
        # No step-config envvar at all → setup failure, not a run failure.
        _setup_env(monkeypatch, tmp_path, step_config={})
        monkeypatch.delenv("NEMO_JOB_STEP_CONFIG_FILE_PATH", raising=False)

        class _Job(NemoJob):
            name = "x"

            def run(self, config: dict) -> dict:
                return {"status": "completed"}

        rc = run_task(_Job, sdk=_DEFAULT_SDK)

        assert rc == 2

    def test_returns_2_when_step_config_is_not_a_dict(self, monkeypatch, tmp_path: Path) -> None:
        # JSON list / scalar at the top level would otherwise propagate to
        # ``job.run`` and surface as a Pydantic ``ValidationError`` (run
        # failure).  The dispatcher rejects it as a setup failure (exit 2)
        # since the malformed config never makes sense for any job.
        config_path = _setup_env(monkeypatch, tmp_path)
        config_path.write_text("[1, 2, 3]", encoding="utf-8")

        class _Job(NemoJob):
            name = "x"

            def run(self, config: dict) -> dict:
                return {"status": "completed"}

        rc = run_task(_Job, sdk=_DEFAULT_SDK)

        assert rc == 2

    def test_returns_2_when_step_config_file_missing(self, monkeypatch, tmp_path: Path) -> None:
        # Envvar points at a path that doesn't exist on disk.
        _setup_env(monkeypatch, tmp_path)  # step_config=None → file not written
        assert not (tmp_path / "step_config.json").exists()

        class _Job(NemoJob):
            name = "x"

            def run(self, config: dict) -> dict:
                return {"status": "completed"}

        rc = run_task(_Job, sdk=_DEFAULT_SDK)

        assert rc == 2

    def test_returns_2_when_step_config_json_invalid_logs_path_and_size(
        self,
        monkeypatch,
        tmp_path: Path,
        caplog,
    ) -> None:
        config_path = _setup_env(monkeypatch, tmp_path)
        config_path.write_text('{"broken":', encoding="utf-8")
        caplog.set_level(logging.ERROR, logger="nemo_helix_plugin.tasks.dispatcher")

        class _Job(NemoJob):
            name = "x"

            def run(self, config: dict) -> dict:
                return {"status": "completed"}

        rc = run_task(_Job, sdk=_DEFAULT_SDK)

        assert rc == 2
        assert "Invalid JSON in step config" in caplog.text
        assert str(config_path) in caplog.text
        assert f"{config_path.stat().st_size} bytes" in caplog.text

    def test_returns_2_when_step_config_json_empty_logs_zero_bytes(
        self,
        monkeypatch,
        tmp_path: Path,
        caplog,
    ) -> None:
        config_path = _setup_env(monkeypatch, tmp_path)
        config_path.write_text("", encoding="utf-8")
        caplog.set_level(logging.ERROR, logger="nemo_helix_plugin.tasks.dispatcher")

        class _Job(NemoJob):
            name = "x"

            def run(self, config: dict) -> dict:
                return {"status": "completed"}

        rc = run_task(_Job, sdk=_DEFAULT_SDK)

        assert rc == 2
        assert "Invalid JSON in step config" in caplog.text
        assert str(config_path) in caplog.text
        assert "0 bytes" in caplog.text

    def test_returns_2_when_workspace_envvar_missing(self, monkeypatch, tmp_path: Path) -> None:
        _setup_env(monkeypatch, tmp_path, step_config={})
        monkeypatch.delenv("NEMO_JOB_WORKSPACE", raising=False)

        class _Job(NemoJob):
            name = "x"

            def run(self, config: dict) -> dict:
                return {"status": "completed"}

        rc = run_task(_Job, sdk=_DEFAULT_SDK)

        assert rc == 2

    def test_returns_2_when_constructor_raises(self, monkeypatch, tmp_path: Path) -> None:
        # A job whose ``__init__`` raises (e.g. bad class-level config
        # validation) is a setup error, not a run error — surface 2.
        _setup_env(monkeypatch, tmp_path, step_config={})

        class _Job(NemoJob):
            name = "bad-ctor"

            def __init__(self) -> None:
                raise RuntimeError("ctor blew up")

            def run(self, config: dict) -> dict:
                return {"status": "completed"}

        rc = run_task(_Job, sdk=_DEFAULT_SDK)

        assert rc == 2

    def test_returns_2_when_sdk_missing_and_no_explicit_ctx(self, monkeypatch, tmp_path: Path, caplog) -> None:
        # The dispatcher's auto-built ctx wires PlatformJobResults, which
        # needs the sdk. Surface the misconfig as exit 2 instead of crashing
        # while constructing the default JobContext.
        _setup_env(monkeypatch, tmp_path, step_config={})
        caplog.set_level(logging.ERROR, logger="nemo_helix_plugin.tasks.dispatcher")

        class _Job(NemoJob):
            name = "needs-sdk-or-ctx"

            def run(self, config: dict) -> dict:
                return {"status": "completed"}

        rc = run_task(_Job)

        assert rc == 2
        assert "requires sdk=" in caplog.text

    def test_local_run_error_propagates(self, monkeypatch, tmp_path: Path) -> None:
        # ``LocalRunError`` from ``resolve_run_kwargs`` indicates a
        # plugin-author bug (run declares ``sdk`` as required but the
        # caller didn't pass one). Propagating beats collapsing it into
        # the same exit-2 bucket as a missing env var.
        from nemo_helix_plugin.run_dependencies import LocalRunError

        _setup_env(monkeypatch, tmp_path, step_config={})
        # Pass ``ctx`` explicitly so the dispatcher's own sdk-requirement
        # is bypassed; we want ``resolve_run_kwargs`` to be the layer that
        # raises when *the job's* required ``sdk`` param can't be bound.
        ctx = JobContext(
            workspace="ws",
            storage=StoragePaths(ephemeral=tmp_path / "e", persistent=tmp_path / "p"),
            results=LocalJobResults(root=tmp_path / "r"),
        )

        class _Job(NemoJob):
            name = "needs-sdk"

            def run(self, config: dict, *, sdk) -> dict:
                return {"status": "completed"}

        with pytest.raises(LocalRunError, match="sdk"):
            run_task(_Job, ctx=ctx)  # no sdk passed → LocalRunError

    def test_local_run_error_from_job_run_propagates(self, monkeypatch, tmp_path: Path) -> None:
        # LocalRunError from job.run must propagate, not collapse to exit 1.
        from nemo_helix_plugin.run_dependencies import LocalRunError

        _setup_env(monkeypatch, tmp_path, step_config={})

        class _Job(NemoJob):
            name = "raises-local-run-error"

            def run(self, config: dict) -> dict:
                raise LocalRunError("missing sdk for fileset upload")

        with pytest.raises(LocalRunError, match="fileset upload"):
            run_task(_Job, sdk=_DEFAULT_SDK)

    def test_unsupported_required_run_param_raises_local_run_error(self, monkeypatch, tmp_path: Path) -> None:
        # Unknown required run() param surfaces as LocalRunError, not TypeError.
        from nemo_helix_plugin.run_dependencies import LocalRunError

        _setup_env(monkeypatch, tmp_path, step_config={})

        class _Job(NemoJob):
            name = "weird-required-param"

            def run(self, config: dict, *, foo) -> dict:
                return {"status": "completed", "foo": foo}

        with pytest.raises(LocalRunError, match="foo"):
            run_task(_Job, sdk=_DEFAULT_SDK)


class TestSignatureDI:
    def test_ctx_bound_when_declared(self, monkeypatch, tmp_path: Path) -> None:
        _setup_env(monkeypatch, tmp_path, step_config={}, workspace="injected-ws")
        captured: dict = {}

        class _Job(NemoJob):
            name = "ctx-job"

            def run(self, config: dict, *, ctx: JobContext) -> dict:
                captured["ctx"] = ctx
                return {"status": "completed"}

        rc = run_task(_Job, sdk=_DEFAULT_SDK)

        assert rc == 0
        assert isinstance(captured["ctx"], JobContext)
        assert captured["ctx"].workspace == "injected-ws"
        assert captured["ctx"].storage.persistent == tmp_path / "p"

    def test_sdk_bound_when_declared(self, monkeypatch, tmp_path: Path) -> None:
        _setup_env(monkeypatch, tmp_path, step_config={})
        fake_sdk = MagicMock()
        captured: dict = {}

        class _Job(NemoJob):
            name = "sdk-job"

            def run(self, config: dict, *, sdk) -> dict:
                captured["sdk"] = sdk
                return {"status": "completed"}

        rc = run_task(_Job, sdk=fake_sdk)

        assert rc == 0
        assert captured["sdk"] is fake_sdk

    def test_unrecognised_param_left_unbound(self, monkeypatch, tmp_path: Path) -> None:
        _setup_env(monkeypatch, tmp_path, step_config={})

        class _Job(NemoJob):
            name = "extra-param"

            # An extra keyword-only param the dispatcher doesn't know
            # about: signature DI leaves it alone, the default applies.
            def run(self, config: dict, *, custom: str = "default-value") -> dict:
                return {"status": "completed", "custom": custom}

        rc = run_task(_Job, sdk=_DEFAULT_SDK)

        assert rc == 0

    def test_run_with_only_config_param(self, monkeypatch, tmp_path: Path) -> None:
        # Job's run takes only the config dict; dispatcher must not try
        # to bind extra kwargs even when ctx/sdk are available.
        _setup_env(monkeypatch, tmp_path, step_config={"a": 1})
        captured: dict = {}

        class _Job(NemoJob):
            name = "minimal"

            def run(self, config: dict) -> dict:
                captured["config"] = config
                return {"status": "completed"}

        rc = run_task(_Job, sdk=MagicMock())

        assert rc == 0
        assert captured["config"] == {"a": 1}


class TestStepConfigPlumbing:
    def test_step_config_passed_as_first_positional(self, monkeypatch, tmp_path: Path) -> None:
        payload = {"agent": "calc", "eval_config": "config.yml", "workspace": "ws"}
        _setup_env(monkeypatch, tmp_path, step_config=payload)
        captured: dict = {}

        class _Job(NemoJob):
            name = "echo"

            def run(self, config: dict, *, ctx: JobContext) -> dict:
                captured["config"] = config
                return {"status": "completed"}

        rc = run_task(_Job, sdk=_DEFAULT_SDK)

        assert rc == 0
        assert captured["config"] == payload


class TestCtxOverride:
    """``ctx=`` kwarg lets callers swap the auto-built context.

    Most plugins want the env-derived default. Plugins that need a
    platform-backed :class:`~nemo_helix_plugin.job_results.JobResults` sink
    (``PlatformJobResults`` for fileset upload + jobs-results
    registration) build their own :class:`JobContext` and pass it in.
    """

    def test_explicit_ctx_replaces_from_env_default(self, monkeypatch, tmp_path: Path) -> None:
        _setup_env(monkeypatch, tmp_path, step_config={}, workspace="env-ws")
        # Construct a custom ctx whose workspace differs from env so we
        # can prove ``from_env`` was bypassed.
        custom_ctx = JobContext(
            workspace="override-ws",
            storage=StoragePaths(ephemeral=tmp_path / "ce", persistent=tmp_path / "cp"),
            results=LocalJobResults(root=tmp_path / "cr"),
            job_id="explicit-job-id",
        )
        captured: dict = {}

        class _Job(NemoJob):
            name = "ctx-override"

            def run(self, config: dict, *, ctx: JobContext) -> dict:
                captured["ctx"] = ctx
                return {"status": "completed"}

        rc = run_task(_Job, ctx=custom_ctx)

        assert rc == 0
        assert captured["ctx"] is custom_ctx
        assert captured["ctx"].workspace == "override-ws"
        assert captured["ctx"].job_id == "explicit-job-id"

    def test_explicit_ctx_skips_from_env_when_workspace_unset(self, monkeypatch, tmp_path: Path) -> None:
        # Without an override, missing NEMO_JOB_WORKSPACE returns 2.
        # With an explicit ctx, the dispatcher must never call from_env(),
        # so the missing envvar is irrelevant.
        _setup_env(monkeypatch, tmp_path, step_config={})
        monkeypatch.delenv("NEMO_JOB_WORKSPACE", raising=False)

        custom_ctx = JobContext(
            workspace="ws",
            storage=StoragePaths(ephemeral=tmp_path / "e", persistent=tmp_path / "p"),
            results=LocalJobResults(root=tmp_path / "r"),
        )

        class _Job(NemoJob):
            name = "ctx-override-nm"

            def run(self, config: dict) -> dict:
                return {"status": "completed"}

        rc = run_task(_Job, ctx=custom_ctx)

        assert rc == 0

    def test_explicit_ctx_carries_custom_results_sink(self, monkeypatch, tmp_path: Path) -> None:
        # Pin that the override path is the seam for plugins to inject a
        # platform-backed results sink (PlatformJobResults in real
        # deployments; a mock here).
        _setup_env(monkeypatch, tmp_path, step_config={})
        custom_results = MagicMock(spec=LocalJobResults)
        custom_ctx = JobContext(
            workspace="ws",
            storage=StoragePaths(ephemeral=tmp_path / "e", persistent=tmp_path / "p"),
            results=custom_results,
        )
        captured: dict = {}

        class _Job(NemoJob):
            name = "results-sink"

            def run(self, config: dict, *, ctx: JobContext) -> dict:
                captured["results"] = ctx.results
                return {"status": "completed"}

        rc = run_task(_Job, ctx=custom_ctx)

        assert rc == 0
        assert captured["results"] is custom_results


class TestBuildCtxFromEnv:
    """``build_ctx_from_env`` reads the platform-injected env vars.

    Tested in isolation here so the env-vars-to-JobContext mapping has
    direct coverage — :func:`run_task` is the only public call site,
    but the helper has enough invariants (required envvars, results-sink
    wiring) to deserve its own pinning.
    """

    @staticmethod
    def _patch_results(monkeypatch) -> MagicMock:
        sentinel = MagicMock(name="PlatformJobResults")

        def _fake_platform_job_results(**_kwargs: object) -> MagicMock:
            return sentinel

        monkeypatch.setattr(
            dispatcher_module,
            "PlatformJobResults",
            _fake_platform_job_results,
        )
        return sentinel

    class _Job(NemoJob):
        name = "test-job"

        def run(self, config: dict) -> dict:
            return {"status": "completed"}

    def test_reads_workspace_and_paths_from_env(self, tmp_path: Path, monkeypatch) -> None:
        self._patch_results(monkeypatch)
        persistent = tmp_path / "persistent"
        ephemeral = tmp_path / "ephemeral"
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", "platform-ws")
        monkeypatch.setenv("NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH", str(persistent))
        monkeypatch.setenv("NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH", str(ephemeral))
        monkeypatch.setenv("NEMO_JOB_ID", "submitted-job-name")

        ctx = build_ctx_from_env(_DEFAULT_SDK)

        assert ctx.workspace == "platform-ws"
        assert ctx.storage.persistent == persistent
        assert ctx.storage.ephemeral == ephemeral
        assert ctx.job_id == "submitted-job-name"

    def test_results_use_submitted_job_id_not_class_name(self, tmp_path: Path, monkeypatch) -> None:
        # Regression: ``PlatformJobResults`` must be keyed off the *submitted*
        # platform job name (``NEMO_JOB_ID``) so ``ResultManager.create_result``
        # can retrieve it via the typed Jobs client. Earlier versions passed
        # ``job_cls.name`` (the NemoJob class identifier like ``"evaluate"``)
        # which points at a non-existent job record.
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

        build_ctx_from_env(_DEFAULT_SDK)

        assert captured["job_name"] == "evaluate-agent-abc123"
        assert captured["workspace"] == "ws"

    def test_results_default_is_platform_job_results(self, tmp_path: Path, monkeypatch) -> None:
        # Pin Mike's intent (PR #205): the dispatcher always wires
        # ``PlatformJobResults`` so results upload through the Files
        # service regardless of executor (docker container, subprocess
        # on local host). Plugin authors don't choose the sink.
        sentinel = self._patch_results(monkeypatch)
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", "ws")
        monkeypatch.setenv("NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH", str(tmp_path / "p"))
        monkeypatch.setenv("NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH", str(tmp_path / "e"))
        monkeypatch.setenv("NEMO_JOB_ID", "submitted-job-name")

        ctx = build_ctx_from_env(_DEFAULT_SDK)

        assert ctx.results is sentinel

    def test_missing_workspace_raises(self, monkeypatch) -> None:
        monkeypatch.delenv("NEMO_JOB_WORKSPACE", raising=False)

        with pytest.raises(RuntimeError, match="NEMO_JOB_WORKSPACE"):
            build_ctx_from_env(_DEFAULT_SDK)

    def test_empty_workspace_raises(self, monkeypatch) -> None:
        # Empty string is treated as missing; the platform never sets a
        # blank workspace, so this catches misconfigured runtimes.
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", "")

        with pytest.raises(RuntimeError, match="NEMO_JOB_WORKSPACE"):
            build_ctx_from_env(_DEFAULT_SDK)

    def test_whitespace_workspace_raises(self, monkeypatch) -> None:
        # Whitespace-only is treated as missing too — same realistic
        # failure mode (e.g. a deployment template that emitted ``" "``
        # instead of the real value).
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", "   ")

        with pytest.raises(RuntimeError, match="NEMO_JOB_WORKSPACE"):
            build_ctx_from_env(_DEFAULT_SDK)

    def test_missing_persistent_storage_builds_ctx_but_access_raises(self, tmp_path: Path, monkeypatch) -> None:
        # Persistent storage is optional — the ctx builds successfully
        # without it, but accessing ctx.storage.persistent raises a clear
        # RuntimeError so jobs that need it fail fast with guidance.
        self._patch_results(monkeypatch)
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", "ws")
        monkeypatch.delenv("NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH", raising=False)
        monkeypatch.setenv("NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH", str(tmp_path / "e"))
        monkeypatch.setenv("NEMO_JOB_ID", "test-job")

        ctx = build_ctx_from_env(_DEFAULT_SDK)
        assert ctx.storage.ephemeral == tmp_path / "e"

        with pytest.raises(RuntimeError, match="did not request persistent storage"):
            _ = ctx.storage.persistent

    def test_missing_ephemeral_storage_raises(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", "ws")
        monkeypatch.setenv("NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH", str(tmp_path / "p"))
        monkeypatch.delenv("NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH", raising=False)

        with pytest.raises(RuntimeError, match="NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH"):
            build_ctx_from_env(_DEFAULT_SDK)

    def test_missing_job_id_raises(self, tmp_path: Path, monkeypatch) -> None:
        # ``NEMO_JOB_ID`` is required for the default ctx because
        # ``PlatformJobResults`` keys all result-registration calls on the
        # submitted platform job name. Without it, every ``ctx.results.save()``
        # would target the wrong (or nonexistent) job record.
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", "ws")
        monkeypatch.setenv("NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH", str(tmp_path / "p"))
        monkeypatch.setenv("NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH", str(tmp_path / "e"))
        monkeypatch.delenv("NEMO_JOB_ID", raising=False)

        with pytest.raises(RuntimeError, match="NEMO_JOB_ID"):
            build_ctx_from_env(_DEFAULT_SDK)
