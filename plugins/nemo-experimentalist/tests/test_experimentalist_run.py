# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``run_experimentalist`` wires the CLI's inputs into one :class:`ExperimentRunner`.

What the runner then does with them is covered by the runner's own tests; these check
the hand-off, and that the caller keeps ownership of its platform client.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from doubles import fake_client
from nemo_experimentalist_plugin.entities import DatasetRef
from nemo_experimentalist_plugin.experimentalist import run as experimentalist_run
from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import LocalExperimentalistBackend
from nemo_experimentalist_plugin.experimentalist.result import ExperimentalistResult
from nemo_experimentalist_plugin.experimentalist.strategies.evolutionary import EvolutionaryOptimizerConfig
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.nooa_model_client import ConfiguredModelRefs


class ClosingClient(AsyncNemoClient):
    def __init__(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(500, request=request, json={"detail": "fake client"})
        )
        super().__init__(
            base_url="http://platform.test",
            workspace="default",
            http_client=httpx.AsyncClient(transport=transport),
        )
        self.closed = False

    async def close(self) -> None:
        self.closed = True
        await super().close()


@dataclass
class ClosingModelClients:
    """Stands in for the resolved Model Entity clients (#1159)."""

    default: object = "default"
    fast: object = "fast"
    refs: ConfiguredModelRefs = ConfiguredModelRefs(default="default/quality", fast="default/fast")
    closed: bool = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def model_clients(monkeypatch: pytest.MonkeyPatch) -> ClosingModelClients:
    clients = ClosingModelClients()

    async def resolve(*_: object) -> ClosingModelClients:
        return clients

    monkeypatch.setattr(experimentalist_run, "resolve_model_clients", resolve)
    return clients


@dataclass
class ExperimentRunPaths:
    agent: Path
    train: Path
    validation: Path
    experiment: Path


@dataclass
class BackendFactoryCall:
    client: AsyncNemoClient
    experiments_output: str


@dataclass
class AgentFactoryCall:
    working_dir: Path
    config: EvolutionaryOptimizerConfig


@dataclass
class RecordingRunner:
    """Stands in for the real runner and keeps the kwargs it was constructed with."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, **kwargs: Any) -> "RecordingRunner":
        self.calls.append(kwargs)
        return self

    async def run(self) -> ExperimentalistResult:
        return ExperimentalistResult(summary="optimization complete", run_id="run-1", progress_completed=1)


def _make_run_paths(tmp_path: Path) -> ExperimentRunPaths:
    agent = tmp_path / "agent"
    train = tmp_path / "train"
    validation = tmp_path / "validation"
    for path in (agent, train, validation):
        path.mkdir()
    return ExperimentRunPaths(
        agent=agent,
        train=train,
        validation=validation,
        experiment=tmp_path / "experiment",
    )


@pytest.mark.asyncio
async def test_run_experimentalist_builds_and_runs_complete_local_contract(
    model_clients: ClosingModelClients,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _make_run_paths(tmp_path)
    client = ClosingClient()
    backend = LocalExperimentalistBackend(client=fake_client(), path=tmp_path / "backend")
    optimizer_config = EvolutionaryOptimizerConfig(max_rounds=2)
    strategy = object()
    runner = RecordingRunner()
    backend_calls: list[BackendFactoryCall] = []
    agent_calls: list[AgentFactoryCall] = []

    def make_backend(
        *,
        client: AsyncNemoClient,
        experiments_output: str,
        storage: object = None,
    ) -> LocalExperimentalistBackend:
        backend_calls.append(BackendFactoryCall(client=client, experiments_output=experiments_output))
        return backend

    def build_agent(
        *,
        working_dir: Path,
        config: EvolutionaryOptimizerConfig,
        framework_skills_dirs: list[Path] | None,
    ) -> object:
        assert framework_skills_dirs is None
        agent_calls.append(AgentFactoryCall(working_dir=working_dir, config=config))
        return strategy

    monkeypatch.setattr(experimentalist_run, "make_experimentalist_backend", make_backend)
    monkeypatch.setattr(experimentalist_run, "build_experimentalist_agent", build_agent)
    monkeypatch.setattr(experimentalist_run, "ExperimentRunner", runner)

    train_dataset = DatasetRef(uri=str(paths.train))
    validation_dataset = DatasetRef(uri=str(paths.validation))

    summary = await experimentalist_run.run_experimentalist(
        agent=paths.agent,
        insight=None,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        experiment_dir=paths.experiment,
        workspace="workspace-a",
        client=client,
        config=optimizer_config,
    )

    assert summary == "optimization complete"
    assert paths.experiment.is_dir()
    assert backend_calls == [BackendFactoryCall(client=client, experiments_output=str(paths.experiment.resolve()))]
    assert [(c.working_dir, c.config) for c in agent_calls] == [(paths.experiment.resolve(), optimizer_config)]
    assert not client.closed

    (call,) = runner.calls
    assert call["backend"] is backend
    assert call["strategy"] is strategy
    assert call["config"] is optimizer_config
    assert call["workspace"] == "workspace-a"
    assert call["root"] == paths.experiment.resolve()
    # ``agent`` is forwarded verbatim (it may be a git url@ref); the runner resolves it.
    assert call["agent"] == paths.agent
    assert call["insight"] is None
    assert call["train_dataset"] == train_dataset
    assert call["validation_dataset"] == validation_dataset
    assert call["ethos"] is None


@pytest.mark.asyncio
async def test_run_experimentalist_forwards_platform_insight_id_verbatim(
    model_clients: ClosingModelClients,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _make_run_paths(tmp_path)
    runner = RecordingRunner()

    monkeypatch.setattr(experimentalist_run, "make_experimentalist_backend", lambda **_: object())
    monkeypatch.setattr(experimentalist_run, "build_experimentalist_agent", lambda **_: object())
    monkeypatch.setattr(experimentalist_run, "ExperimentRunner", runner)

    await experimentalist_run.run_experimentalist(
        insight="insight-remote-123",
        train_dataset=DatasetRef(uri=str(paths.train)),
        validation_dataset=DatasetRef(uri=str(paths.validation)),
        task_template=DatasetRef(uri=str(paths.train)),
        experiment_dir=paths.experiment,
        workspace="workspace-a",
        client=ClosingClient(),
        config=EvolutionaryOptimizerConfig(),
    )

    # A str id is not resolved to a Path — it flows through untouched to the backend.
    assert runner.calls[0]["insight"] == "insight-remote-123"


@pytest.mark.asyncio
async def test_run_experimentalist_forwards_ethos_uri(
    model_clients: ClosingModelClients,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _make_run_paths(tmp_path)
    runner = RecordingRunner()

    monkeypatch.setattr(experimentalist_run, "make_experimentalist_backend", lambda **_: object())
    monkeypatch.setattr(experimentalist_run, "build_experimentalist_agent", lambda **_: object())
    monkeypatch.setattr(experimentalist_run, "ExperimentRunner", runner)

    ethos_uri = "/path/to/ETHOS.md"
    await experimentalist_run.run_experimentalist(
        agent=paths.agent,
        ethos=ethos_uri,
        insight=None,
        train_dataset=DatasetRef(uri=str(paths.train)),
        validation_dataset=DatasetRef(uri=str(paths.validation)),
        experiment_dir=paths.experiment,
        workspace="default",
        client=ClosingClient(),
        config=EvolutionaryOptimizerConfig(),
    )

    assert runner.calls[0]["ethos"] == ethos_uri


@pytest.mark.asyncio
async def test_run_experimentalist_does_not_close_caller_client_when_backend_creation_fails(
    model_clients: ClosingModelClients,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _make_run_paths(tmp_path)
    client = ClosingClient()

    def fail_backend_creation(**_: object) -> object:
        raise RuntimeError("backend creation failed")

    monkeypatch.setattr(experimentalist_run, "make_experimentalist_backend", fail_backend_creation)

    with pytest.raises(RuntimeError, match="backend creation failed"):
        await experimentalist_run.run_experimentalist(
            agent=paths.agent,
            insight=None,
            train_dataset=DatasetRef(uri=str(paths.train)),
            validation_dataset=DatasetRef(uri=str(paths.validation)),
            experiment_dir=paths.experiment,
            workspace="default",
            client=client,
            config=EvolutionaryOptimizerConfig(),
        )

    assert not client.closed
