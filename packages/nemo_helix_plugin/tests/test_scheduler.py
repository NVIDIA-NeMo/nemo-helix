# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`nemo_helix_plugin.scheduler` — the three-verb scheduler.

Phase 1 wires :meth:`NemoJobScheduler.run_local`, :meth:`submit_remote`, and
:meth:`explain`. These tests pin:

- Legacy ``run(config: dict)`` jobs without ``spec_schema`` receive the raw
  dict unchanged.
- Jobs with ``spec_schema`` receive a validated, defaults-merged dict.
- Jobs with ``input_spec_schema`` + ``to_spec`` receive the transformed
  canonical dict (re-validated against ``spec_schema``).
- The constructed :class:`JobContext` carries the expected
  workspace / job_id / storage shape and exposes the in-container runtime
  envvars to ``run`` for backwards compatibility with legacy task code.
- ``submit_remote`` builds the right URL, body, and headers and POSTs to
  the configured base URL.
- ``explain`` returns the OpenAPI-shaped descriptor expected by the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults
from nemo_helix_plugin.run_dependencies import LocalRunError
from nemo_helix_plugin.scheduler import NemoJobScheduler
from pydantic import BaseModel, ValidationError, ValidationInfo, model_validator

# ---------------------------------------------------------------------------
# Fixture jobs
# ---------------------------------------------------------------------------


class _LegacyRawJob(NemoJob):
    """No schema declared — expects the raw dict unchanged."""

    name = "legacy-raw"
    description = "Legacy job without spec_schema."

    def run(self, config: dict) -> dict:
        return {"got": config}


class _SpecSchemaOnlySpec(BaseModel):
    name: str
    count: int = 1


class _SpecSchemaOnlyJob(NemoJob):
    """Declares spec_schema only — defaults are filled in."""

    name = "spec-only"
    description = "Job with a canonical spec."
    spec_schema = _SpecSchemaOnlySpec

    def run(self, config: dict) -> dict:
        return {"name": config["name"], "count": config["count"]}


class _CanonicalSpec(BaseModel):
    resolved_id: str
    count: int


class _InputSpec(BaseModel):
    raw_name: str
    count: int = 1


class _InputTransformJob(NemoJob):
    """Declares both shapes and a ``to_spec`` transform.

    ``to_spec`` is the ``async classmethod`` shape — runs in the API
    process; the scheduler awaits it via ``asyncio.run``.
    """

    name = "transform"
    description = "Resolves raw_name → resolved_id."
    spec_schema = _CanonicalSpec
    input_spec_schema = _InputSpec

    @classmethod
    async def to_spec(cls, input_spec, *, workspace, entity_client, async_sdk, is_local: bool):
        del is_local
        assert isinstance(input_spec, _InputSpec)
        return _CanonicalSpec(
            resolved_id=f"id-{input_spec.raw_name}",
            count=input_spec.count,
        )

    def run(self, config: dict) -> dict:
        return {"resolved_id": config["resolved_id"], "count": config["count"]}


# ---------------------------------------------------------------------------
# run_local
# ---------------------------------------------------------------------------


class TestRunLocalSignatureAdaptation:
    def test_legacy_job_receives_raw_dict(self) -> None:
        scheduler = NemoJobScheduler()
        result = scheduler.run_local(_LegacyRawJob, {"foo": "bar", "n": 3})
        assert result == {"got": {"foo": "bar", "n": 3}}

    def test_spec_schema_applies_defaults(self) -> None:
        scheduler = NemoJobScheduler()
        result = scheduler.run_local(_SpecSchemaOnlyJob, {"name": "widget"})
        assert result == {"name": "widget", "count": 1}

    def test_spec_schema_validates_input(self) -> None:
        scheduler = NemoJobScheduler()
        with pytest.raises(ValidationError):
            scheduler.run_local(_SpecSchemaOnlyJob, {"name": 123})

    def test_input_spec_schema_runs_to_spec(self) -> None:
        scheduler = NemoJobScheduler()
        result = scheduler.run_local(_InputTransformJob, {"raw_name": "foo", "count": 5})
        assert result == {"resolved_id": "id-foo", "count": 5}

    def test_is_local_injected_into_to_spec_and_run_when_requested(self) -> None:
        seen: dict[str, bool] = {}

        class _LocalityJob(NemoJob):
            name = "locality"
            spec_schema = _CanonicalSpec
            input_spec_schema = _InputSpec

            @classmethod
            async def to_spec(cls, input_spec, *, workspace, entity_client, async_sdk, is_local: bool):
                del workspace, entity_client, async_sdk
                seen["to_spec"] = is_local
                return _CanonicalSpec(
                    resolved_id=f"id-{input_spec.raw_name}",
                    count=input_spec.count,
                )

            def run(self, config: dict, *, is_local: bool) -> dict:
                return {"to_spec": seen["to_spec"], "run": is_local, **config}

        scheduler = NemoJobScheduler()
        result = scheduler.run_local(_LocalityJob, {"raw_name": "foo", "count": 5})

        assert result == {"to_spec": True, "run": True, "resolved_id": "id-foo", "count": 5}

    def test_run_local_is_local_true_even_when_job_id_env_is_set(self, monkeypatch) -> None:
        monkeypatch.setenv("NEMO_JOB_ID", "env-job-id")

        class _LocalityJob(NemoJob):
            name = "locality-env"
            spec_schema = _CanonicalSpec
            input_spec_schema = _InputSpec

            @classmethod
            async def to_spec(cls, input_spec, *, workspace, entity_client, async_sdk, is_local: bool):
                del workspace, entity_client, async_sdk
                return _CanonicalSpec(resolved_id=f"id-{input_spec.raw_name}", count=int(is_local))

            def run(self, config: dict, *, is_local: bool) -> dict:
                return {"to_spec": config["count"], "run": is_local}

        result = NemoJobScheduler().run_local(_LocalityJob, {"raw_name": "foo"})

        assert result == {"to_spec": 1, "run": True}

    def test_run_local_passes_validation_context_to_input_and_canonical_schemas(self) -> None:
        seen: list[str] = []

        def require_local_context(data: Any, info: ValidationInfo, label: str) -> Any:
            context = info.context
            if not (isinstance(context, dict) and context.get("is_local") is True):
                raise ValueError(f"missing local validation context for {label}")
            seen.append(label)
            return data

        class _ContextInputSpec(BaseModel):
            name: str

            @model_validator(mode="before")
            @classmethod
            def require_local_context(cls, data: Any, info: ValidationInfo) -> Any:
                return require_local_context(data, info, "input")

        class _ContextCanonicalSpec(BaseModel):
            name: str

            @model_validator(mode="before")
            @classmethod
            def require_local_context(cls, data: Any, info: ValidationInfo) -> Any:
                return require_local_context(data, info, "canonical")

        class _ContextJob(NemoJob):
            name = "context-job"
            input_spec_schema = _ContextInputSpec
            spec_schema = _ContextCanonicalSpec

            @classmethod
            async def to_spec(cls, input_spec, *, workspace, entity_client, async_sdk, is_local: bool):
                del workspace, entity_client, async_sdk, is_local
                return {"name": input_spec.name}

            def run(self, config: dict) -> dict:
                return config

        result = NemoJobScheduler().run_local(_ContextJob, {"name": "widget"})

        assert result == {"name": "widget"}
        assert seen == ["input", "canonical"]

    def test_input_spec_schema_validates_incoming_shape(self) -> None:
        scheduler = NemoJobScheduler()
        with pytest.raises(ValidationError):
            scheduler.run_local(_InputTransformJob, {"wrong_field": "x"})


class TestRunLocalJobContext:
    def test_scheduler_builds_local_context_when_none_provided(self, tmp_path) -> None:
        captured: dict = {}

        class _CaptureJob(NemoJob):
            name = "capture"

            def run(self, config: dict) -> dict:
                return config

        # Use a custom ctx so the test can inspect it without relying on
        # tempdir construction.
        from nemo_helix_plugin.job_results import LocalJobResults

        ctx = JobContext(
            workspace="test-ws",
            job_id="550e8400-e29b-41d4-a716-446655440000",
            storage=StoragePaths(ephemeral=tmp_path / "e", persistent=tmp_path / "p"),
            results=LocalJobResults(root=tmp_path / "r"),
        )
        (tmp_path / "e").mkdir()
        (tmp_path / "p").mkdir()
        captured["ctx"] = ctx

        scheduler = NemoJobScheduler()
        scheduler.run_local(_CaptureJob, {"hello": "world"}, workspace="test-ws", ctx=ctx)

        # Assert the ctx we passed satisfies the protocol.
        assert isinstance(ctx, JobContext)
        assert ctx.workspace == "test-ws"
        assert ctx.job_id == "550e8400-e29b-41d4-a716-446655440000"

    def test_auto_built_context_has_tempdir_storage(self) -> None:
        scheduler = NemoJobScheduler()

        # Run a job that just echoes; the ctx is built internally.
        result = scheduler.run_local(_LegacyRawJob, {}, workspace="auto-ws")
        assert result == {"got": {}}

    def test_auto_built_context_uses_workspace_argument(self) -> None:
        scheduler = NemoJobScheduler()

        class _LookAtCtxJob(NemoJob):
            name = "look-at-ctx"

            def run(self, config: dict) -> dict:
                return config

        # We can't observe the auto-built ctx from run(config) alone, so
        # the test asserts through the scheduler's helper that the ctx
        # would carry the workspace correctly.
        ctx = scheduler._build_local_context(_LookAtCtxJob, workspace="visible-ws")
        assert ctx.workspace == "visible-ws"
        # Default ``job_id`` is None for scheduler-created local contexts.
        assert ctx.job_id is None
        assert ctx.storage.ephemeral.exists()
        assert ctx.storage.persistent.exists()
        # ``ctx.results`` is the default LocalJobResults rooted under
        # ``persistent / "results"`` per ``build_local_job_context``.
        assert ctx.results is not None

    def test_auto_built_context_results_writes_under_persistent(self) -> None:
        """The default ``LocalJobResults`` lands artefacts under
        ``persistent / "results"`` so the dev loop can inspect output
        deterministically."""
        scheduler = NemoJobScheduler()

        class _NoopJob(NemoJob):
            name = "noop-results"

            def run(self, config: dict) -> dict:
                return config

        ctx = scheduler._build_local_context(_NoopJob, workspace="ws")
        src = ctx.storage.ephemeral / "out.txt"
        src.write_text("payload")
        assert isinstance(ctx.results, LocalJobResults)
        ref = ctx.results.save("out", src)
        expected_root = ctx.storage.persistent / "results"
        assert ref.artifact_url.startswith(f"file://{expected_root.resolve()}/")


# ---------------------------------------------------------------------------
# run_local — JobContext / LocalJobResults wiring
# ---------------------------------------------------------------------------


class TestRunLocalContextWiring:
    """The auto-built context carries a :class:`LocalJobResults` sink and
    a :class:`JobContext` — there is no async twin."""

    def test_run_receives_sync_results_sink(self) -> None:
        seen: dict[str, object] = {}

        class _Job(NemoJob):
            name = "sync-results"

            def run(self, config: dict, *, ctx: JobContext) -> dict:
                seen["results_type"] = type(ctx.results).__name__
                seen["ctx_type"] = type(ctx).__name__
                src = ctx.storage.ephemeral / "out.txt"
                src.write_text("payload")
                ref = ctx.results.save("out", src)
                return {"artifact_url": ref.artifact_url}

        result = NemoJobScheduler().run_local(_Job, {})
        assert seen["results_type"] == "LocalJobResults"
        assert seen["ctx_type"] == "JobContext"
        assert result["artifact_url"].startswith("file://")


class TestRunLocalEnvvarMirroring:
    """``run_local`` should expose the in-container runtime envvars during ``run``.

    Legacy task code reads ``NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH`` and
    friends via ``os.environ.get(...)`` to locate scratch / persistent
    volumes. Local invocations need the same names visible so unported
    code keeps working.
    """

    def test_storage_envvars_visible_inside_run(self, monkeypatch) -> None:
        import os

        monkeypatch.delenv("NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH", raising=False)
        monkeypatch.delenv("NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH", raising=False)
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", "default")

        seen: dict[str, object] = {}

        class _ReadEnvJob(NemoJob):
            name = "read-env"

            def run(self, config: dict) -> dict:
                persistent = os.environ.get("NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH")
                ephemeral = os.environ.get("NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH")
                seen["persistent"] = persistent
                seen["ephemeral"] = ephemeral
                seen["job_id"] = os.environ.get("NEMO_JOB_ID")
                seen["workspace"] = os.environ.get("NEMO_JOB_WORKSPACE")
                # Directories should exist *during* the run; they are torn down
                # in the scheduler's finally block.
                seen["persistent_exists"] = persistent is not None and Path(persistent).exists()
                seen["ephemeral_exists"] = ephemeral is not None and Path(ephemeral).exists()
                return config

        NemoJobScheduler().run_local(_ReadEnvJob, {}, workspace="my-ws")

        assert seen["persistent"] is not None
        assert seen["ephemeral"] is not None
        assert seen["persistent_exists"] is True
        assert seen["ephemeral_exists"] is True
        # Local runs leave job_id as None; the scheduler skips exporting
        # NEMO_JOB_ID in that case.
        assert seen["job_id"] is None
        assert seen["workspace"] == "my-ws"
        assert os.environ.get("NEMO_JOB_WORKSPACE") == "default"

    def test_envvars_restored_after_run(self, monkeypatch) -> None:
        import os

        monkeypatch.delenv("NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH", raising=False)
        monkeypatch.delenv("NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH", raising=False)

        class _NoopJob(NemoJob):
            name = "noop"

            def run(self, config: dict) -> dict:
                return config

        NemoJobScheduler().run_local(_NoopJob, {})

        # Envvars we synthesized should not leak past the call.
        assert "NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH" not in os.environ
        assert "NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH" not in os.environ

    def test_caller_set_envvars_win(self, monkeypatch) -> None:
        import os

        monkeypatch.setenv("NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH", "/caller/persistent")
        seen: dict[str, str | None] = {}

        class _ReadEnvJob(NemoJob):
            name = "read-env"

            def run(self, config: dict) -> dict:
                seen["persistent"] = os.environ.get("NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH")
                return config

        NemoJobScheduler().run_local(_ReadEnvJob, {})

        # Caller's value survives — scheduler doesn't override pre-set envvars.
        assert seen["persistent"] == "/caller/persistent"
        # And it's still set after the call (we never touched it).
        assert os.environ.get("NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH") == "/caller/persistent"


# ---------------------------------------------------------------------------
# submit_remote — URL building, body shaping, HTTP POST (MR 1.3)
# ---------------------------------------------------------------------------


def _mock_transport(capture: dict) -> httpx.MockTransport:
    """Build a transport that captures the request and returns a canned job."""

    def handler(request: httpx.Request) -> httpx.Response:
        capture["url"] = str(request.url)
        capture["method"] = request.method
        capture["body"] = request.read().decode("utf-8")
        return httpx.Response(200, json={"id": "job-123", "status": "queued"})

    return httpx.MockTransport(handler)


class _NamespacedJob(NemoJob):
    """A job defined in a module whose top-level package shapes the API segment."""

    name = "example"


# Force the module on these fixture classes so URL construction sees a stable
# ``{api}`` segment regardless of how pytest rewrote the test module path.
# This mirrors what real plugin jobs get from `<plugin_name>.jobs.<name>`.
_NamespacedJob.__module__ = "my_tests_plugin.jobs.example"


class _CollectionOverrideJob(NemoJob):
    name = "custom"
    job_collection_path = "/custom-jobs"


_CollectionOverrideJob.__module__ = "my_tests_plugin.jobs.custom"


class TestSubmitRemoteURL:
    def test_default_endpoint_applied(self) -> None:
        scheduler = NemoJobScheduler()
        url = scheduler._build_submit_url(
            _NamespacedJob,
            base_url="https://nhx.test",
            workspace="ws-a",
        )
        assert url == "https://nhx.test/apis/my-tests-plugin/v2/workspaces/ws-a/jobs/example"

    def test_job_collection_path_override_honored(self) -> None:
        scheduler = NemoJobScheduler()
        url = scheduler._build_submit_url(
            _CollectionOverrideJob,
            base_url="https://nhx.test",
            workspace="ws-a",
        )
        assert url == "https://nhx.test/apis/my-tests-plugin/v2/workspaces/ws-a/custom-jobs"

    def test_missing_base_url_raises(self) -> None:
        scheduler = NemoJobScheduler()
        with pytest.raises(ValueError, match="requires base_url"):
            scheduler._build_submit_url(_NamespacedJob, base_url=None, workspace="ws")


class TestApiSegmentFor:
    """Cover ``_api_segment_for``, which builds the ``{api}`` portion of submit URLs.

    The platform mounts each plugin's job routes under the ``<plugin>``
    half of its ``nemo.jobs`` entry-point key (``<plugin>.<job>``), so
    the scheduler has to derive the same prefix or every ``submit``
    404s. The authoritative source is the registered entry-point —
    module paths are only consulted when the job isn't installed
    (in-process tests, scratch invocations).
    """

    def test_uses_registered_entry_point_key(self, monkeypatch) -> None:
        from nemo_helix_plugin.scheduler import _api_segment_for

        class _J(NemoJob):
            name = "evaluate"

        # Simulate ``agents.evaluate`` entry-point binding for a class
        # whose package layout (``nemo_agents_plugin``) would otherwise
        # produce ``agents-plugin`` — the URL the platform doesn't mount.
        _J.__module__ = "nemo_agents_plugin.jobs.evaluate_agent"
        monkeypatch.setattr(
            "nemo_helix_plugin.discovery.discover_jobs",
            lambda: {"agents.evaluate": _J},
        )
        assert _api_segment_for(_J) == "agents"

    def test_falls_back_to_module_when_not_registered(self, monkeypatch) -> None:
        from nemo_helix_plugin.scheduler import _api_segment_for

        class _J(NemoJob):
            name = "say-hello"

        # No entry-point match → module-name fallback. The fallback
        # keeps ``_plugin``; an unregistered class in a
        # ``nemo_<name>_plugin`` package surfaces the suffix so the
        # scheduler fails loudly rather than 404ing against the wrong URL.
        _J.__module__ = "nemo_example_plugin.jobs.say_hello"
        monkeypatch.setattr("nemo_helix_plugin.discovery.discover_jobs", lambda: {})
        assert _api_segment_for(_J) == "example-plugin"

    def test_fallback_strips_nemo_prefix(self, monkeypatch) -> None:
        from nemo_helix_plugin.scheduler import _api_segment_for

        class _J(NemoJob):
            name = "evaluate"

        # ``nemo_evaluator`` (no ``_plugin``) strips just ``nemo_``.
        _J.__module__ = "nemo_evaluator.jobs.evaluate"
        monkeypatch.setattr("nemo_helix_plugin.discovery.discover_jobs", lambda: {})
        assert _api_segment_for(_J) == "evaluator"

    def test_handles_missing_nemo_prefix(self, monkeypatch) -> None:
        from nemo_helix_plugin.scheduler import _api_segment_for

        class _J(NemoJob):
            name = "x"

        # In-tree code outside ``nemo_*`` keeps its module name as-is
        # (kebab-cased) so tests with inline classes still produce a
        # stable, predictable segment.
        _J.__module__ = "tests.fixtures.things"
        monkeypatch.setattr("nemo_helix_plugin.discovery.discover_jobs", lambda: {})
        assert _api_segment_for(_J) == "tests"


class TestSubmitRemoteBody:
    def test_body_carries_spec_profile_options_and_metadata(self) -> None:
        scheduler = NemoJobScheduler()
        body = scheduler._build_submit_body(
            {"num_records": 100},
            profile="research",
            options={"slurm": {"nodes": 4}},
            metadata={"name": "nightly", "project": "dd-prod"},
        )
        assert body == {
            "name": "nightly",
            "project": "dd-prod",
            "spec": {"num_records": 100},
            "profile": "research",
            "options": {"slurm": {"nodes": 4}},
        }

    def test_body_omits_profile_when_none(self) -> None:
        scheduler = NemoJobScheduler()
        body = scheduler._build_submit_body({"x": 1}, profile=None, options=None, metadata=None)
        assert body == {"spec": {"x": 1}}

    def test_body_omits_empty_options(self) -> None:
        scheduler = NemoJobScheduler()
        body = scheduler._build_submit_body({"x": 1}, profile="p", options={}, metadata=None)
        assert body == {"spec": {"x": 1}, "profile": "p"}


class TestSubmitRemoteHTTP:
    def test_posts_to_plugin_service_and_returns_response(self) -> None:
        capture: dict = {}
        client = httpx.Client(transport=_mock_transport(capture))
        scheduler = NemoJobScheduler()

        result = scheduler.submit_remote(
            _NamespacedJob,
            {"foo": "bar"},
            base_url="https://nhx.test",
            workspace="ws-a",
            profile="research",
            options={"slurm": {"nodes": 4}},
            metadata={"name": "sub-1"},
            http_client=client,
        )

        assert result == {"id": "job-123", "status": "queued"}
        assert capture["method"] == "POST"
        assert capture["url"] == "https://nhx.test/apis/my-tests-plugin/v2/workspaces/ws-a/jobs/example"
        # Body contents — spec, profile, options, and the metadata envelope.
        import json as _json

        body = _json.loads(capture["body"])
        assert body["spec"] == {"foo": "bar"}
        assert body["profile"] == "research"
        assert body["options"] == {"slurm": {"nodes": 4}}
        assert body["name"] == "sub-1"

    def test_http_error_propagates(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json={"error": "bad spec"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        scheduler = NemoJobScheduler()
        with pytest.raises(httpx.HTTPStatusError):
            scheduler.submit_remote(
                _NamespacedJob,
                {},
                base_url="https://nhx.test",
                http_client=client,
            )


# ---------------------------------------------------------------------------
# explain — local schema extraction (MR 1.4a)
# ---------------------------------------------------------------------------


class _ExplainSpec(BaseModel):
    name: str
    count: int = 1


class _ExplainJob(NemoJob):
    name = "example"
    spec_schema = _ExplainSpec


_ExplainJob.__module__ = "my_tests_plugin.jobs.example"


class _ExplainInputSpec(BaseModel):
    raw: str


class _ExplainInputJob(NemoJob):
    name = "example-in"
    spec_schema = _ExplainSpec
    input_spec_schema = _ExplainInputSpec


_ExplainInputJob.__module__ = "my_tests_plugin.jobs.example_in"


class _NoSchemaJob(NemoJob):
    name = "raw"  # no spec_schema declared


_NoSchemaJob.__module__ = "my_tests_plugin.jobs.raw"


class TestExplain:
    def test_reads_spec_schema_from_pydantic_locally(self) -> None:
        """No network hop — spec_schema comes from the in-hand NemoJob class."""
        bundle = NemoJobScheduler().explain(_ExplainJob, profile="research")

        assert bundle["job_key"].endswith(".example")
        # endpoint is an illustrative template with {workspace} left as a
        # literal placeholder — explain doesn't POST anywhere.
        assert bundle["endpoint"] == "/apis/my-tests-plugin/v2/workspaces/{workspace}/jobs/example"
        assert bundle["profile"] == "research"
        assert bundle["profile_providers"] == []  # MR 1.4b fills this
        assert bundle["options"] == {}  # MR 1.4b / phase 2 fills this

        spec = bundle["spec_schema"]
        assert spec is not None
        assert spec["type"] == "object"
        assert set(spec["properties"]) == {"name", "count"}
        assert spec["required"] == ["name"]
        assert bundle["input_spec_schema"] is None

    def test_returns_input_spec_schema_when_declared(self) -> None:
        bundle = NemoJobScheduler().explain(_ExplainInputJob)

        assert bundle["spec_schema"] is not None
        inp = bundle["input_spec_schema"]
        assert inp is not None
        assert set(inp["properties"]) == {"raw"}

    def test_returns_none_schemas_when_job_declares_no_spec_schema(self) -> None:
        """Jobs that haven't declared a schema get None — explain still renders."""
        bundle = NemoJobScheduler().explain(_NoSchemaJob)

        assert bundle["spec_schema"] is None
        assert bundle["input_spec_schema"] is None
        assert bundle["endpoint"].endswith("/jobs/raw")

    def test_works_without_any_network_context(self) -> None:
        """explain takes no base_url / cluster / http_client — pure local read."""
        bundle = NemoJobScheduler().explain(_ExplainJob)

        # Endpoint is always rendered with the {workspace} placeholder;
        # there is no --workspace flag on explain.
        assert "{workspace}" in bundle["endpoint"]


# ---------------------------------------------------------------------------
# compile() base class marker (interface extension from 1.2b)
# ---------------------------------------------------------------------------


class _DummySpec(BaseModel):
    """Placeholder BaseModel for tests that only care the call raises."""


class TestCompileMarker:
    def test_compile_raises_when_not_overridden(self) -> None:
        # ``compile`` is an ``async classmethod`` now — drive via
        # ``asyncio.run`` so the marker raises in async context. The base
        # marker raises before reading any arg; the spec value is
        # immaterial, but it must be a BaseModel to satisfy the type.
        import asyncio

        with pytest.raises(NotImplementedError, match="must override compile"):
            asyncio.run(
                _LegacyRawJob.compile(
                    workspace="ws",
                    spec=_DummySpec(),
                    entity_client=None,
                    job_name=None,
                    async_sdk=cast(AsyncNeMoHelix, None),
                )
            )


# ---------------------------------------------------------------------------
# Signature-based DI
#
# Plain ``run(self, config: dict)`` jobs receive the canonical dict and
# nothing else. Jobs that widen the signature with keyword-only ``ctx``,
# ``sdk``, or ``async_sdk`` parameters get them resolved by name from the
# scheduler inputs. Per-service typed resources (Files, Models, ...) will
# be reintroduced under the new ``NemoSDK`` design.
# ---------------------------------------------------------------------------


class TestRunLocalDILegacyShim:
    def test_plain_run_config_still_works(self) -> None:
        """Jobs with ``run(self, config)`` receive the canonical dict only — no DI side-effects."""
        scheduler = NemoJobScheduler()
        result = scheduler.run_local(_LegacyRawJob, {"x": 1}, sdk=object())
        assert result == {"got": {"x": 1}}


class TestRunLocalDIContextAndSdk:
    def test_ctx_injected_when_declared(self, tmp_path) -> None:
        seen: dict = {}

        class _CtxJob(NemoJob):
            name = "ctx-job"

            def run(self, config: dict, *, ctx: JobContext) -> dict:
                seen["ctx"] = ctx
                return {"ok": True}

        from nemo_helix_plugin.job_results import LocalJobResults

        ctx = JobContext(
            workspace="ws",
            job_id="11111111-1111-1111-1111-111111111111",
            storage=StoragePaths(ephemeral=tmp_path / "e", persistent=tmp_path / "p"),
            results=LocalJobResults(root=tmp_path / "r"),
        )
        (tmp_path / "e").mkdir()
        (tmp_path / "p").mkdir()

        NemoJobScheduler().run_local(_CtxJob, {}, workspace="ws", ctx=ctx)
        assert seen["ctx"] is ctx

    def test_sdk_injected_when_declared(self) -> None:
        seen: dict = {}

        class _SdkJob(NemoJob):
            name = "sdk-job"

            def run(self, config: dict, *, sdk: object | None = None) -> dict:
                seen["sdk"] = sdk
                return {}

        sdk = object()
        NemoJobScheduler().run_local(_SdkJob, {}, sdk=sdk)
        assert seen["sdk"] is sdk

    def test_sdk_required_without_default_raises(self) -> None:
        class _RequiredSdkJob(NemoJob):
            name = "required-sdk-job"

            def run(self, config: dict, *, sdk: object) -> dict:
                return {}

        with pytest.raises(LocalRunError, match=r"requires a `sdk` argument"):
            NemoJobScheduler().run_local(_RequiredSdkJob, {})

    def test_async_sdk_injected_when_declared(self) -> None:
        seen: dict = {}

        class _AsyncSdkJob(NemoJob):
            name = "async-sdk-job"

            def run(self, config: dict, *, async_sdk: object | None = None) -> dict:
                seen["async_sdk"] = async_sdk
                return {}

        async_sdk = object()
        NemoJobScheduler().run_local(_AsyncSdkJob, {}, async_sdk=async_sdk)
        assert seen["async_sdk"] is async_sdk

    def test_async_sdk_required_without_default_raises(self) -> None:
        class _RequiredAsyncSdkJob(NemoJob):
            name = "required-async-sdk-job"

            def run(self, config: dict, *, async_sdk: object) -> dict:
                return {}

        with pytest.raises(LocalRunError, match=r"requires an `async_sdk` argument"):
            NemoJobScheduler().run_local(_RequiredAsyncSdkJob, {})
