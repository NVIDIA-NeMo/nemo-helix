# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``run-strategy`` routes to the strategy job and otherwise stays out of the way."""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from nemo_agent_optimization_plugin.jobs import run_strategy
from nemo_agent_optimization_plugin.jobs.run_strategy import RunStrategyJob
from nemo_agent_optimization_plugin.schemas.optimize import RunStrategySpec, RunStrategySubmitSpec
from nemo_agent_optimization_plugin.schemas.strategies import OptimizationStrategy
from nemo_helix_plugin.errors import LocalRunError
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import (
    HelixJobSpec,
    HelixJobStep,
    SubprocessExecutionProviderSpec,
)
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError, HelixJobDependencyUnavailableError
from pydantic import BaseModel, model_validator

#: What every test submits: the router's own fields, one strategy deep.
SUBMITTED: dict[str, Any] = {
    "strategy": "fake",
    "optimize_config": "optimize.yml",
    "optimize_config_fileset": "default/opt-bundle",
    "agent": "react-agent",
}


class _FakeStrategySpec(BaseModel):
    """A strategy's own spec — the router's fields minus ``strategy``."""

    optimize_config: str
    optimize_config_fileset: str | None = None
    agent: str | None = None
    output: str | None = None
    workspace: str = "default"

    @model_validator(mode="after")
    def _require_a_staged_bundle(self) -> "_FakeStrategySpec":
        if self.optimize_config_fileset is None:
            raise ValueError("this strategy only runs from a staged bundle")
        return self


class _FakeStrategyJob(NemoJob):
    name: ClassVar[str] = "optimize"
    nemo_agent_optimization_strategy: ClassVar[OptimizationStrategy] = OptimizationStrategy(name="fake")
    spec_schema: ClassVar[type[BaseModel]] = _FakeStrategySpec

    #: Recorded by ``compile`` so tests can assert what the router forwarded.
    compiled_with: ClassVar[dict[str, Any]] = {}

    @classmethod
    async def compile(  # ty: ignore[invalid-method-override]
        cls,
        *,
        workspace: str,
        spec: _FakeStrategySpec,
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict | None = None,
    ) -> HelixJobSpec:
        cls.compiled_with = {"workspace": workspace, "spec": spec, "profile": profile, "options": options}
        return HelixJobSpec(
            steps=[
                HelixJobStep(
                    name="fake-study",
                    executor=SubprocessExecutionProviderSpec(
                        provider="subprocess", profile="default", command=["python", "-m", "fake.task"]
                    ),
                    config=spec.model_dump(mode="json"),
                )
            ]
        )

    def run(self, config: dict, *, ctx: JobContext) -> dict[str, Any]:
        return {"ran": config}


class _NoSchemaStrategyJob(_FakeStrategyJob):
    """A strategy that declares no ``spec_schema`` and takes the raw payload."""

    name: ClassVar[str] = "passthrough"
    nemo_agent_optimization_strategy: ClassVar[OptimizationStrategy] = OptimizationStrategy(name="passthrough")
    spec_schema: ClassVar[type[BaseModel] | None] = None

    @classmethod
    async def compile(  # ty: ignore[invalid-method-override]
        cls,
        *,
        workspace: str,
        spec: dict,
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict | None = None,
    ) -> HelixJobSpec:
        cls.compiled_with = {"workspace": workspace, "spec": spec, "profile": profile, "options": options}
        return HelixJobSpec(
            steps=[
                HelixJobStep(
                    name="passthrough-study",
                    executor=SubprocessExecutionProviderSpec(
                        provider="subprocess", profile="default", command=["python", "-m", "fake.task"]
                    ),
                    config=dict(spec),
                )
            ]
        )


class _FakeSubmitSpec(BaseModel):
    """What a submitter types: the agent by *name*, and a remote-only rule the canonical shape lacks."""

    optimize_config: str
    optimize_config_fileset: str | None = None
    agent: str | None = None
    output: str | None = None
    workspace: str = "default"

    @model_validator(mode="after")
    def _submit_needs_a_staged_bundle(self) -> "_FakeSubmitSpec":
        if self.optimize_config_fileset is None:
            raise ValueError("a submission must name a staged bundle")
        return self


class _FakeResolvedSpec(BaseModel):
    """What ``compile`` needs: the agent resolved to an id.  Accepts a missing bundle."""

    optimize_config: str
    optimize_config_fileset: str | None = None
    agent_id: str | None = None
    output: str | None = None
    workspace: str = "default"


class _ResolvingStrategyJob(_FakeStrategyJob):
    """A strategy whose ``to_spec`` does real work, so skipping it would be visible."""

    name: ClassVar[str] = "resolving"
    nemo_agent_optimization_strategy: ClassVar[OptimizationStrategy] = OptimizationStrategy(name="resolving")
    spec_schema: ClassVar[type[BaseModel]] = _FakeResolvedSpec
    input_spec_schema: ClassVar[type[BaseModel]] = _FakeSubmitSpec

    #: Recorded by ``to_spec`` so tests can assert what the router handed it.
    to_spec_called_with: ClassVar[dict[str, Any]] = {}

    @classmethod
    async def to_spec(
        cls,
        input_spec: BaseModel,
        *,
        workspace: str,
        entity_client: object,
        async_sdk: object,
        is_local: bool,
    ) -> _FakeResolvedSpec:
        cls.to_spec_called_with = {
            "input_spec": input_spec,
            "workspace": workspace,
            "entity_client": entity_client,
            "async_sdk": async_sdk,
            "is_local": is_local,
        }
        payload = input_spec.model_dump()
        agent = payload.pop("agent")
        return _FakeResolvedSpec.model_validate({**payload, "agent_id": f"agent-{agent}", "workspace": workspace})


@pytest.fixture(autouse=True)
def installed_strategies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin discovery to the fakes above so no real plugin install can change a result."""
    monkeypatch.setattr(run_strategy, "discover_strategy_jobs", lambda: {"fake": _FakeStrategyJob})


def submitted_spec(**overrides: Any) -> RunStrategySpec:
    return RunStrategySpec.model_validate({**SUBMITTED, "workspace": "default", **overrides})


async def compile_spec(
    spec: RunStrategySpec,
    *,
    workspace: str = "default",
    profile: str | None = None,
    entity_client: Any = None,
    async_sdk: Any = None,
) -> Any:
    return await RunStrategyJob.compile(
        workspace=workspace,
        spec=spec,
        entity_client=MagicMock() if entity_client is None else entity_client,
        job_name=None,
        async_sdk=MagicMock() if async_sdk is None else async_sdk,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# to_spec
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_to_spec_stamps_the_resolved_workspace() -> None:
    submitted = RunStrategySubmitSpec.model_validate(SUBMITTED)

    spec = await RunStrategyJob.to_spec(
        submitted,
        workspace="staging",
        entity_client=MagicMock(),
        async_sdk=MagicMock(),
        is_local=False,
    )

    assert spec.workspace == "staging"
    assert spec.strategy == "fake"


def test_the_submit_spec_carries_no_workspace() -> None:
    """Workspace comes from the route, not the submitter."""
    assert "workspace" not in RunStrategySubmitSpec.model_fields


# ---------------------------------------------------------------------------
# compile — delegation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compile_returns_the_strategys_own_steps() -> None:
    platform_spec = await compile_spec(submitted_spec(), workspace="staging")

    step = next(iter(platform_spec.steps))
    assert step.name == "fake-study"
    assert step.config["optimize_config_fileset"] == "default/opt-bundle"


@pytest.mark.asyncio
async def test_compile_forwards_everything_but_the_strategy_name() -> None:
    await compile_spec(submitted_spec(), workspace="staging", profile="gpu")

    forwarded = _FakeStrategyJob.compiled_with
    assert forwarded["workspace"] == "staging"
    assert forwarded["profile"] == "gpu"
    assert isinstance(forwarded["spec"], _FakeStrategySpec)
    assert not hasattr(forwarded["spec"], "strategy")
    assert forwarded["spec"].agent == "react-agent"


@pytest.mark.asyncio
async def test_compile_rejects_an_uninstalled_strategy() -> None:
    with pytest.raises(HelixJobCompilationError, match=r"'nope' is not installed.*\['fake'\]"):
        await compile_spec(submitted_spec(strategy="nope"))


@pytest.mark.asyncio
async def test_compile_surfaces_the_strategys_own_spec_rules_as_a_submit_error() -> None:
    """The router validates nothing about a bundle; the strategy that owns it does."""
    spec = submitted_spec(optimize_config_fileset=None)

    with pytest.raises(HelixJobCompilationError, match="not valid for optimization strategy 'fake'"):
        await compile_spec(spec)


@pytest.mark.asyncio
async def test_a_strategy_without_a_spec_schema_gets_the_raw_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_strategy, "discover_strategy_jobs", lambda: {"passthrough": _NoSchemaStrategyJob})

    await compile_spec(submitted_spec(strategy="passthrough"))

    forwarded = _NoSchemaStrategyJob.compiled_with["spec"]
    assert isinstance(forwarded, dict)
    assert "strategy" not in forwarded


# ---------------------------------------------------------------------------
# compile — the strategy's own submit lifecycle
# ---------------------------------------------------------------------------


@pytest.fixture
def resolving_strategy(monkeypatch: pytest.MonkeyPatch) -> type[_ResolvingStrategyJob]:
    monkeypatch.setattr(run_strategy, "discover_strategy_jobs", lambda: {"resolving": _ResolvingStrategyJob})
    _ResolvingStrategyJob.to_spec_called_with = {}
    _ResolvingStrategyJob.compiled_with = {}
    return _ResolvingStrategyJob


@pytest.mark.asyncio
async def test_compile_hands_the_strategy_what_its_own_to_spec_produced(
    resolving_strategy: type[_ResolvingStrategyJob],
) -> None:
    """The strategy's ``compile`` sees its canonical shape, not a re-validation of the router's fields."""
    await compile_spec(submitted_spec(strategy="resolving"), workspace="staging")

    forwarded = resolving_strategy.compiled_with["spec"]
    assert isinstance(forwarded, _FakeResolvedSpec)
    assert forwarded.agent_id == "agent-react-agent"
    assert forwarded.workspace == "staging"
    assert not hasattr(forwarded, "agent")


@pytest.mark.asyncio
async def test_compile_calls_to_spec_the_way_the_route_layer_would(
    resolving_strategy: type[_ResolvingStrategyJob],
) -> None:
    """Same submit context the strategy would get if it were submitted directly, and never local."""
    entity_client = object()
    async_sdk = object()

    await compile_spec(
        submitted_spec(strategy="resolving"), workspace="staging", entity_client=entity_client, async_sdk=async_sdk
    )

    called_with = resolving_strategy.to_spec_called_with
    assert isinstance(called_with["input_spec"], _FakeSubmitSpec)
    assert called_with["input_spec"].agent == "react-agent"
    assert called_with["workspace"] == "staging"
    assert called_with["entity_client"] is entity_client
    assert called_with["async_sdk"] is async_sdk
    assert called_with["is_local"] is False


@pytest.mark.asyncio
async def test_compile_applies_the_strategys_submit_rules_not_only_its_canonical_ones(
    resolving_strategy: type[_ResolvingStrategyJob],
) -> None:
    """A rule that lives only on the submit shape (like nat's remote-only bundle check) still fires."""
    spec = submitted_spec(strategy="resolving", optimize_config_fileset=None)

    with pytest.raises(HelixJobCompilationError, match="not valid for optimization strategy 'resolving'"):
        await compile_spec(spec)

    assert resolving_strategy.to_spec_called_with == {}, "to_spec must not run on input its own schema rejected"


@pytest.mark.asyncio
async def test_a_to_spec_that_cannot_prepare_its_spec_is_a_submit_error(
    resolving_strategy: type[_ResolvingStrategyJob], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever ``to_spec`` raises becomes a 422, as it would at the route layer -- not a 500."""

    async def _unresolvable(_input_spec: BaseModel, **_kwargs: Any) -> BaseModel:
        raise LookupError("no agent named 'react-agent'")

    monkeypatch.setattr(resolving_strategy, "to_spec", _unresolvable)

    with pytest.raises(HelixJobCompilationError, match=r"'resolving' could not prepare its spec.*no agent named"):
        await compile_spec(submitted_spec(strategy="resolving"))


@pytest.mark.asyncio
async def test_a_to_spec_that_reports_a_dependency_outage_keeps_its_own_error(
    resolving_strategy: type[_ResolvingStrategyJob], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The factory maps this one to a 503; wrapping it would downgrade a retryable outage to a 422."""

    async def _platform_down(_input_spec: BaseModel, **_kwargs: Any) -> BaseModel:
        raise HelixJobDependencyUnavailableError("entities service is unavailable")

    monkeypatch.setattr(resolving_strategy, "to_spec", _platform_down)

    with pytest.raises(HelixJobDependencyUnavailableError, match="entities service is unavailable"):
        await compile_spec(submitted_spec(strategy="resolving"))


# ---------------------------------------------------------------------------
# run — local delegation
# ---------------------------------------------------------------------------


def test_run_delegates_to_the_strategy_job(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _run(_self: object, spec: dict, *, ctx: JobContext, sdk: object | None = None) -> dict[str, Any]:
        captured.update(spec=spec, ctx=ctx, sdk=sdk)
        return {"delegated": True}

    monkeypatch.setattr(_FakeStrategyJob, "run", _run)

    ctx = MagicMock()
    sdk = MagicMock()
    result = RunStrategyJob().run(
        submitted_spec().model_dump(mode="json"),
        ctx=ctx,
        sdk=sdk,
        # Not declared by this strategy: forwarding it would be a TypeError.
        client=MagicMock(),
    )

    assert result == {"delegated": True}
    # The caller's context carries through, so the strategy writes its artifacts there.
    assert captured["ctx"] is ctx
    # A dependency the strategy declares is forwarded; one it does not is dropped.
    assert captured["sdk"] is sdk
    assert captured["spec"]["workspace"] == "default"
    assert "strategy" not in captured["spec"]
    assert captured["spec"]["optimize_config"] == "optimize.yml"


def test_run_forwards_every_dependency_to_a_strategy_that_takes_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A catch-all names nothing, so filtering by parameter name would drop everything."""
    captured: dict[str, Any] = {}

    def _run(_self: object, spec: dict, *, ctx: JobContext, **dependencies: Any) -> dict[str, Any]:
        captured.update(dependencies)
        return {"delegated": True}

    monkeypatch.setattr(_FakeStrategyJob, "run", _run)

    sdk = MagicMock()
    client = MagicMock()
    RunStrategyJob().run(submitted_spec().model_dump(mode="json"), ctx=MagicMock(), sdk=sdk, client=client)

    assert captured == {"sdk": sdk, "client": client}


def test_run_rejects_an_uninstalled_strategy() -> None:
    with pytest.raises(LocalRunError, match=r"'nope' is not installed"):
        RunStrategyJob().run(submitted_spec(strategy="nope").model_dump(mode="json"), ctx=MagicMock())


def test_strategy_is_required() -> None:
    assert RunStrategySubmitSpec.model_fields["strategy"].is_required()
