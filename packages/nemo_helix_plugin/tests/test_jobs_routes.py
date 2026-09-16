# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`nemo_helix_plugin.jobs.routes` — ``add_job_routes`` helper.

These tests cover the boundary the wrapper owns — service-name /
job-type derivation, the two signature adapters, profile stamping,
and the overall ``APIRouter`` handover from :func:`job_route_factory`.

The full HTTP round trip (TestClient → factory POST → Jobs service
mock) is not exercised here; it's owned by the factory's own tests.
Keeping this module focused on the shim keeps it fast and stable.
"""

from __future__ import annotations

import pytest
from fastapi import APIRouter
from fastapi.routing import APIRoute
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.jobs.api_factory import JobRouteOption
from nemo_helix_plugin.jobs.routes import (
    _adapt_compile,
    _adapt_to_spec,
    _derive_job_type,
    _derive_service_name,
    add_job_routes,
)
from nhx.common.jobs.exceptions import PlatformJobCompilationError
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Fixture jobs
# ---------------------------------------------------------------------------


class _FakeStep:
    """Minimal step stub with an executor — enough for stamp_profile."""

    def __init__(self, profile: str | None = None) -> None:
        self.executor = _FakeExecutor(profile=profile)


class _FakeExecutor:
    def __init__(self, profile: str | None = None) -> None:
        self.profile = profile


class _FakePlatformSpec:
    def __init__(self, steps: list[_FakeStep]) -> None:
        self.steps = steps


class _WidgetSpec(BaseModel):
    name: str
    count: int = 1


class _WidgetJob(NemoJob):
    name = "widget"
    description = "Makes widgets."
    spec_schema = _WidgetSpec

    def run(self, config: dict) -> dict:
        return {"got": config}

    @classmethod
    async def compile(
        cls,
        *,
        workspace: str,
        spec: BaseModel,
        entity_client,
        job_name,
        async_sdk,
        profile=None,
        options=None,
    ):
        # A real compile would produce a PlatformJobSpec; fake shape is
        # fine here — stamp_profile only needs spec.steps[*].executor.
        return _FakePlatformSpec(steps=[_FakeStep(profile=None)])


_WidgetJob.__module__ = "my_plugin.jobs.widget"


class _InputSpec(BaseModel):
    raw_name: str


class _CanonicalSpec(BaseModel):
    resolved_id: str


class _WithInputJob(NemoJob):
    name = "resolve"
    spec_schema = _CanonicalSpec
    input_spec_schema = _InputSpec

    @classmethod
    async def to_spec(cls, input_spec, *, workspace, entity_client, async_sdk, is_local: bool):
        del is_local
        assert isinstance(input_spec, _InputSpec)
        return _CanonicalSpec(resolved_id=f"id-{input_spec.raw_name}")

    def run(self, config: dict) -> dict:
        return config

    @classmethod
    async def compile(
        cls,
        *,
        workspace: str,
        spec: BaseModel,
        entity_client,
        job_name,
        async_sdk,
        profile=None,
        options=None,
    ):
        assert isinstance(spec, _CanonicalSpec)
        return _FakePlatformSpec(steps=[_FakeStep(profile=None), _FakeStep(profile=None)])


_WithInputJob.__module__ = "my_plugin.jobs.resolve"


class _NoCompileJob(NemoJob):
    """Inherits the NotImplementedError-raising compile from NemoJob."""

    name = "no-compile"
    spec_schema = _WidgetSpec

    def run(self, config: dict) -> dict:
        return config


_NoCompileJob.__module__ = "my_plugin.jobs.no_compile"


class _NoSpecSchemaJob(NemoJob):
    name = "raw"  # spec_schema intentionally missing

    def run(self, config: dict) -> dict:
        return config


# ---------------------------------------------------------------------------
# Derivation helpers
# ---------------------------------------------------------------------------


class TestDeriveServiceName:
    def test_single_segment_module(self) -> None:
        class J(NemoJob):
            name = "j"
            spec_schema = _WidgetSpec

            def run(self, config: dict) -> dict:
                return config

        J.__module__ = "my_plugin"
        assert _derive_service_name(J) == "my-plugin"

    def test_nested_module_uses_top_level(self) -> None:
        assert _derive_service_name(_WidgetJob) == "my-plugin"

    def test_underscores_become_hyphens(self) -> None:
        class J(NemoJob):
            name = "j"
            spec_schema = _WidgetSpec

            def run(self, config: dict) -> dict:
                return config

        J.__module__ = "snake_case_plugin.jobs.widget"
        assert _derive_service_name(J) == "snake-case-plugin"


class TestDeriveJobType:
    def test_single_word(self) -> None:
        assert _derive_job_type(_WidgetJob) == "Widget"

    def test_hyphenated_becomes_pascalcase(self) -> None:
        class J(NemoJob):
            name = "metric-eval"
            spec_schema = _WidgetSpec

            def run(self, config: dict) -> dict:
                return config

        assert _derive_job_type(J) == "MetricEval"

    def test_underscore_treated_as_hyphen(self) -> None:
        class J(NemoJob):
            name = "raw_job"
            spec_schema = _WidgetSpec

            def run(self, config: dict) -> dict:
                return config

        assert _derive_job_type(J) == "RawJob"

    def test_dotted_name_folds_into_pascalcase(self) -> None:
        # A dot must not survive into the schema class name: Pydantic renders it
        # as a ``__`` separator in the OpenAPI ref, which schema-name
        # normalization then strips, collapsing per-backend names (e.g.
        # "automodel.jobs" and "rl.jobs") to a single "jobsJobRequest".
        class J(NemoJob):
            name = "automodel.jobs"
            spec_schema = _WidgetSpec

            def run(self, config: dict) -> dict:
                return config

        job_type = _derive_job_type(J)
        assert job_type == "AutomodelJobs"
        assert "." not in job_type


class TestPerJobTypeSchemaNaming:
    """Guard the end-to-end path where dotted job names used to collapse.

    Two backends whose ``name`` differs only before the dot (``alpha.jobs`` /
    ``beta.jobs``) must produce distinct request schemas that survive
    ``tweak_spec``'s schema-name normalization, each referencing its own spec
    schema. Before the ``_derive_job_type`` dot fix, both collapsed to a single
    ``jobsJobRequest`` and every backend's POST body aliased the first one.
    """

    def test_dotted_backends_keep_distinct_request_schemas(self) -> None:
        from fastapi import FastAPI
        from nhx.common.api.utils import tweak_spec

        class AlphaSpec(BaseModel):
            alpha_field: str

        class BetaSpec(BaseModel):
            beta_field: int

        def _make_job(job_name: str, spec_cls: type[BaseModel]) -> type[NemoJob]:
            class _J(NemoJob):
                name = job_name
                spec_schema = spec_cls

                def run(self, config: dict) -> dict:
                    return config

                @classmethod
                async def compile(cls, **kwargs):  # pragma: no cover - not invoked by openapi()
                    raise NotImplementedError

            return _J

        app = FastAPI()
        app.include_router(add_job_routes(_make_job("alpha.jobs", AlphaSpec)))
        app.include_router(add_job_routes(_make_job("beta.jobs", BetaSpec)))

        spec = tweak_spec(app.openapi())
        schemas = spec["components"]["schemas"]

        request_keys = {k for k in schemas if k.endswith("JobRequest")}
        assert request_keys == {"AlphaJobsJobRequest", "BetaJobsJobRequest"}

        def _spec_ref(request_key: str) -> str:
            return schemas[request_key]["properties"]["spec"]["$ref"].split("/")[-1]

        assert _spec_ref("AlphaJobsJobRequest") == "AlphaSpec"
        assert _spec_ref("BetaJobsJobRequest") == "BetaSpec"


# ---------------------------------------------------------------------------
# _adapt_to_spec
# ---------------------------------------------------------------------------


class TestAdaptToSpec:
    @pytest.mark.asyncio
    async def test_adapter_invokes_nemo_to_spec_with_correct_kwargs(self) -> None:
        seen: dict = {}

        class J(NemoJob):
            name = "capture"
            spec_schema = _CanonicalSpec
            input_spec_schema = _InputSpec

            @classmethod
            async def to_spec(cls, input_spec, *, workspace, entity_client, async_sdk, is_local: bool):
                seen.update(
                    workspace=workspace,
                    entity_client=entity_client,
                    async_sdk=async_sdk,
                    is_local=is_local,
                    input_type=type(input_spec),
                )
                return _CanonicalSpec(resolved_id="ok")

            def run(self, config: dict) -> dict:
                return config

        adapter = _adapt_to_spec(J)
        out = await adapter(_InputSpec(raw_name="x"), "ws", "ent", "name", "sdk")

        assert isinstance(out, _CanonicalSpec)
        assert out.resolved_id == "ok"
        assert seen == {
            "workspace": "ws",
            "entity_client": "ent",
            "async_sdk": "sdk",
            "is_local": False,
            "input_type": _InputSpec,
        }

    @pytest.mark.asyncio
    async def test_adapter_injects_is_local_false_when_requested(self) -> None:
        seen: dict = {}

        class J(NemoJob):
            name = "capture-locality"
            spec_schema = _CanonicalSpec
            input_spec_schema = _InputSpec

            @classmethod
            async def to_spec(cls, input_spec, *, workspace, entity_client, async_sdk, is_local: bool):
                del input_spec, workspace, entity_client, async_sdk
                seen["is_local"] = is_local
                return _CanonicalSpec(resolved_id="ok")

            def run(self, config: dict) -> dict:
                return config

        adapter = _adapt_to_spec(J)
        await adapter(_InputSpec(raw_name="x"), "ws", "ent", "name", "sdk")

        assert seen == {"is_local": False}


# ---------------------------------------------------------------------------
# _adapt_compile
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compile_adapter_invokes_nemo_compile_and_stamps_default_profile() -> None:
    adapter = _adapt_compile(_WidgetJob, default_profile="research")
    spec = _WidgetSpec(name="w")
    platform_spec = await adapter("ws", spec, spec, "entity_client", "job-1", "sdk")

    assert isinstance(platform_spec, _FakePlatformSpec)
    # Profile stamped on every step since the compiler didn't set one.
    for step in platform_spec.steps:
        assert step.executor.profile == "research"


@pytest.mark.asyncio
async def test_compile_adapter_forwards_submitted_profile_and_options() -> None:
    seen: dict = {}

    class CaptureSubmitControls(NemoJob):
        name = "capture-submit-controls"
        spec_schema = _WidgetSpec

        def run(self, config: dict) -> dict:
            return config

        @classmethod
        async def compile(
            cls,
            *,
            workspace: str,
            spec: BaseModel,
            entity_client,
            job_name,
            async_sdk,
            profile=None,
            options=None,
        ):
            seen.update(profile=profile, options=options)
            return _FakePlatformSpec(steps=[_FakeStep(profile=None)])

    adapter = _adapt_compile(CaptureSubmitControls, default_profile="default")
    platform_spec = await adapter(
        "ws",
        None,
        _WidgetSpec(name="x"),
        "ec",
        None,
        "sdk",
        profile="research",
        options={"slurm": {"nodes": 4}},
    )

    assert seen == {"profile": "research", "options": {"slurm": {"nodes": 4}}}
    assert platform_spec.steps[0].executor.profile == "research"


@pytest.mark.asyncio
async def test_compile_adapter_omits_empty_controls_for_legacy_compile() -> None:
    seen: dict[str, object] = {}

    class LegacyCompile(NemoJob):
        name = "legacy-compile"
        spec_schema = _WidgetSpec

        def run(self, config: dict) -> dict:
            return config

        @classmethod
        async def compile(  # ty: ignore[invalid-method-override]
            cls,
            *,
            workspace: str,
            spec: BaseModel,
            entity_client,
            job_name,
            async_sdk,
        ):
            seen.update(workspace=workspace, spec=spec, entity_client=entity_client, job_name=job_name, sdk=async_sdk)
            return _FakePlatformSpec(steps=[_FakeStep(profile=None)])

    adapter = _adapt_compile(LegacyCompile, default_profile="default")
    platform_spec = await adapter(
        "ws",
        None,
        _WidgetSpec(name="x"),
        "ec",
        "job-1",
        "sdk",
        profile=None,
        options={},
    )

    assert seen == {
        "workspace": "ws",
        "spec": _WidgetSpec(name="x"),
        "entity_client": "ec",
        "job_name": "job-1",
        "sdk": "sdk",
    }
    assert platform_spec.steps[0].executor.profile == "default"


@pytest.mark.asyncio
async def test_compile_adapter_rejects_unsupported_submitted_controls_for_legacy_compile() -> None:
    class LegacyCompile(NemoJob):
        name = "legacy-compile"
        spec_schema = _WidgetSpec

        def run(self, config: dict) -> dict:
            return config

        @classmethod
        async def compile(  # ty: ignore[invalid-method-override]
            cls,
            *,
            workspace: str,
            spec: BaseModel,
            entity_client,
            job_name,
            async_sdk,
        ):
            del workspace, spec, entity_client, job_name, async_sdk
            return _FakePlatformSpec(steps=[_FakeStep(profile=None)])

    adapter = _adapt_compile(LegacyCompile, default_profile="default")

    with pytest.raises(PlatformJobCompilationError, match="does not support submit field\\(s\\): profile, options"):
        await adapter(
            "ws",
            None,
            _WidgetSpec(name="x"),
            "ec",
            None,
            "sdk",
            profile="research",
            options={"slurm": {"nodes": 4}},
        )


@pytest.mark.asyncio
async def test_compile_adapter_preserves_profile_set_by_plugin_compile() -> None:
    class CompileSetsProfile(NemoJob):
        name = "pre-stamped"
        spec_schema = _WidgetSpec

        def run(self, config: dict) -> dict:
            return config

        @classmethod
        async def compile(cls, **kwargs):
            return _FakePlatformSpec(steps=[_FakeStep(profile="explicit")])

    adapter = _adapt_compile(CompileSetsProfile, default_profile="default")
    platform_spec = await adapter("ws", None, _WidgetSpec(name="x"), "ec", None, "sdk")

    assert platform_spec.steps[0].executor.profile == "explicit"


@pytest.mark.asyncio
async def test_compile_adapter_converts_not_implemented_to_compilation_error() -> None:
    adapter = _adapt_compile(_NoCompileJob, default_profile="default")
    with pytest.raises(PlatformJobCompilationError, match="must override compile"):
        await adapter("ws", None, _WidgetSpec(name="x"), "ec", None, "sdk")


# ---------------------------------------------------------------------------
# add_job_routes (integration-ish)
# ---------------------------------------------------------------------------


class TestAddJobRoutes:
    def test_returns_api_router(self) -> None:
        router = add_job_routes(_WidgetJob)
        assert isinstance(router, APIRouter)

    def test_router_mounts_the_expected_routes(self) -> None:
        router = add_job_routes(_WidgetJob)
        methods_and_paths = {(tuple(sorted(r.methods or [])), r.path) for r in router.routes if isinstance(r, APIRoute)}
        # add_job_routes rebases the factory's generic `/jobs` routes onto
        # the NemoJob collection path (`/{job.name}/jobs` by default).
        assert any(path == "/jobs/widget" and "POST" in methods for (methods, path) in methods_and_paths)
        assert any(path == "/jobs/widget" and "GET" in methods for (methods, path) in methods_and_paths)

    def test_raises_typeerror_when_spec_schema_is_none(self) -> None:
        with pytest.raises(TypeError, match="spec_schema is None"):
            add_job_routes(_NoSpecSchemaJob)

    def test_accepts_route_options_passthrough(self) -> None:
        router = add_job_routes(_WidgetJob, route_options=[JobRouteOption.CORE])
        assert isinstance(router, APIRouter)

    def test_accepts_service_name_override(self) -> None:
        # The override is plumbed to the factory. We can't observe it
        # directly on the router, but the call must not raise.
        router = add_job_routes(_WidgetJob, service_name="custom-name")
        assert isinstance(router, APIRouter)

    def test_handles_job_with_input_spec_schema(self) -> None:
        router = add_job_routes(_WithInputJob)
        assert isinstance(router, APIRouter)

    def test_respects_job_collection_path_override(self) -> None:
        class FlatCollectionJob(_WidgetJob):
            name = "flat"
            job_collection_path = "/flat-jobs"

        router = add_job_routes(FlatCollectionJob)
        paths = {r.path for r in router.routes if isinstance(r, APIRoute)}

        assert "/flat-jobs" in paths
        assert "/flat-jobs/{name}" in paths
        assert "/flat-jobs/jobs" not in paths
