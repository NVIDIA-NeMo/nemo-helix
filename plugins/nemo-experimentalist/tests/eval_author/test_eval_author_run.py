# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The orchestration boundary accepts only authoring inputs."""

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from nemo_experimentalist_plugin.entities import Dataset, DatasetRef, ResourceRef, Task
from nemo_experimentalist_plugin.eval_author import run as eval_author_run
from nemo_experimentalist_plugin.eval_author.agent import EvalAuthor
from nemo_experimentalist_plugin.eval_author.models import EvalAuthorConfig, EvalAuthorResult
from nemo_experimentalist_plugin.experimentalist.components import dataset_staging
from nemo_experimentalist_plugin.experimentalist.components.evaluator.base import EvaluatorType
from nemo_experimentalist_plugin.experimentalist.components.evaluator.harbor import HarborDataset
from nemo_insights_plugin.entities import Insight


@dataclass
class ClosingClient:
    closed: bool = False

    async def close(self) -> None:
        self.closed = True


@dataclass
class ClosingModelClients:
    default: object = None
    fast: object = None
    closed: bool = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def model_clients(monkeypatch: pytest.MonkeyPatch) -> ClosingModelClients:
    clients = ClosingModelClients()
    refs = eval_author_run.ConfiguredModelRefs(
        default="workspace-a/default-model",
        fast="workspace-a/fast-model",
    )

    async def resolve(*_: object) -> ClosingModelClients:
        return clients

    monkeypatch.setattr(eval_author_run, "configured_model_refs", lambda: refs)
    monkeypatch.setattr(eval_author_run, "resolve_model_clients", resolve)
    return clients


class FakeBackend:
    def __init__(self, insight: Insight) -> None:
        self.insight = insight
        self.insight_calls: list[tuple[str, str]] = []
        self.agent_calls: list[tuple[str, str | Path, Path]] = []

    async def get_insight(self, *, workspace: str, insight_id: str) -> Insight:
        self.insight_calls.append((workspace, insight_id))
        return self.insight

    async def get_agent_code(self, *, workspace: str, agent: str | Path, dest: Path) -> None:
        self.agent_calls.append((workspace, agent, dest))


class FakeDatasetFactory:
    def __init__(self) -> None:
        self.template = Task(id="template", uri="file:///template")
        self.template_calls: list[tuple[str, DatasetRef]] = []
        self.dataset_calls: list[tuple[str, DatasetRef]] = []
        self.datasets: list[Dataset] = []

    def build_task_template(self, evaluator_type: str, template_ref: DatasetRef) -> Task:
        self.template_calls.append((evaluator_type, template_ref))
        return self.template

    def build_dataset(self, evaluator_type: str, dataset_ref: DatasetRef) -> Dataset:
        self.dataset_calls.append((evaluator_type, dataset_ref))
        dataset = Dataset(id=Path(dataset_ref.uri).name, source=ResourceRef(uri=Path(dataset_ref.uri).as_uri()))
        self.datasets.append(dataset)
        return dataset


class FakeEvalAuthor:
    def __init__(self) -> None:
        self.call: tuple[Insight, Path, Task, Dataset, Dataset, ClosingClient] | None = None
        self.insight_suite = Dataset(id="insight-suite")

    async def run(
        self,
        insight: Insight,
        agent_path: Path,
        task_template: Task,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        *,
        client: ClosingClient,
    ) -> EvalAuthorResult:
        self.call = (insight, agent_path, task_template, train_dataset, validation_dataset, client)
        return EvalAuthorResult(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            insight_suite=self.insight_suite,
            insight_suite_identity=f"sha256:{'a' * 64}",
            metric_keys=("uses_correct_tool",),
            summary="Eval Author complete.",
        )


def _write_minimal_harbor_task(task_dir: Path, *, name: str, instruction: str) -> None:
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(f'[task]\nname = "{name}"\n', encoding="utf-8")
    (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")


def _dataset_snapshot(dataset: Dataset) -> dict[str, Any]:
    return {
        "type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
        "id": dataset.id,
        "source": dataset.source.model_dump(mode="json") if dataset.source is not None else None,
        "tasks": [task.model_dump(mode="json") for task in dataset.list_tasks()],
        "metadata": dict(dataset.metadata),
    }


@pytest.mark.asyncio
async def test_run_eval_author_fails_before_side_effects_when_model_configuration_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "eval_author"
    make_client = MagicMock()

    def missing_model_refs() -> eval_author_run.ConfiguredModelRefs:
        raise ValueError("No default model is configured. Run `nemo setup` and select agent models.")

    monkeypatch.setattr(eval_author_run, "configured_model_refs", missing_model_refs)
    monkeypatch.setattr(eval_author_run, "make_client", make_client)

    with pytest.raises(ValueError, match="No default model is configured"):
        await eval_author_run.run_eval_author(
            insight="insight-remote-123",
            train_dataset=DatasetRef(uri="train"),
            validation_dataset=DatasetRef(uri="validation"),
            task_template=DatasetRef(uri="template"),
            experiment_dir=experiment_dir,
            workspace="workspace-a",
            base_url="http://platform.test",
            config=EvalAuthorConfig(),
        )

    assert not experiment_dir.exists()
    make_client.assert_not_called()


@pytest.mark.parametrize("evaluator_type", ["harbor-native", "harbor-runner"])
@pytest.mark.asyncio
async def test_run_eval_author_resolves_inputs_and_returns_datasets(
    monkeypatch: pytest.MonkeyPatch,
    model_clients: ClosingModelClients,
    tmp_path: Path,
    evaluator_type: EvaluatorType,
) -> None:
    client = ClosingClient()
    insight = Insight(
        workspace="workspace-a",
        title="failure",
        description="description",
        agent="insight-agent",
        trace_refs=["trace-1"],
    )
    backend = FakeBackend(insight)
    dataset_factory = FakeDatasetFactory()
    eval_author = FakeEvalAuthor()
    monkeypatch.setattr(eval_author_run, "make_client", lambda _: client)
    monkeypatch.setattr(eval_author_run, "make_experimentalist_backend", lambda **_: backend)
    monkeypatch.setattr(eval_author_run, "DatasetFactory", lambda: dataset_factory)
    monkeypatch.setattr(eval_author_run, "build_eval_author_agent", lambda **_: eval_author)
    template = tmp_path / "template"
    template.mkdir()
    (template / "task.toml").write_text("template\n", encoding="utf-8")
    train = tmp_path / "train"
    validation = tmp_path / "validation"
    train.mkdir()
    validation.mkdir()

    result = await eval_author_run.run_eval_author(
        insight="insight-123",
        train_dataset=DatasetRef(uri=str(train)),
        validation_dataset=DatasetRef(uri=str(validation)),
        task_template=DatasetRef(uri=str(template)),
        experiment_dir=tmp_path / "experiment",
        workspace="workspace-a",
        base_url="http://platform.test",
        config=EvalAuthorConfig(),
        evaluator_type=evaluator_type,
    )

    experiment_dir = (tmp_path / "experiment").resolve()
    assert result.train_dataset is dataset_factory.datasets[0]
    assert result.validation_dataset is dataset_factory.datasets[1]
    assert result.insight_suite is eval_author.insight_suite
    assert result.insight_suite_identity == f"sha256:{'a' * 64}"
    assert result.metric_keys == ("uses_correct_tool",)
    assert backend.insight_calls == [("workspace-a", "insight-123")]
    assert backend.agent_calls == [
        ("workspace-a", "insight-agent", experiment_dir / "eval_author" / "source-agent"),
    ]
    assert [call[0] for call in dataset_factory.dataset_calls] == [evaluator_type, evaluator_type]
    assert dataset_factory.template_calls[0][0] == evaluator_type
    assert eval_author.call == (
        insight,
        experiment_dir / "eval_author" / "source-agent",
        dataset_factory.template,
        dataset_factory.datasets[0],
        dataset_factory.datasets[1],
        client,
    )
    assert client.closed
    assert model_clients.closed


def test_public_apis_accept_train_validation_and_generated_task_inputs() -> None:
    orchestration = inspect.signature(eval_author_run.run_eval_author).parameters
    agent_run = inspect.signature(EvalAuthor.run).parameters
    agent_private_run = inspect.signature(EvalAuthor._run).parameters

    assert {"insight", "task_template", "train_dataset", "validation_dataset"} <= set(orchestration)
    assert {"request", "reference_task_sets"}.isdisjoint(orchestration)
    expected = {
        "self",
        "insight",
        "agent_path",
        "task_template",
        "train_dataset",
        "validation_dataset",
        "client",
    }
    assert set(agent_run) == expected
    assert set(agent_private_run) == expected


@pytest.mark.asyncio
async def test_run_eval_author_builds_equivalent_real_harbor_inputs_for_both_evaluator_types(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    template_dir = tmp_path / "task-template"
    _write_minimal_harbor_task(
        train_dir / "train-task",
        name="parity/train-task",
        instruction="Complete the training task.\n",
    )
    _write_minimal_harbor_task(
        validation_dir / "validation-task",
        name="parity/validation-task",
        instruction="Complete the validation task.\n",
    )
    _write_minimal_harbor_task(
        template_dir,
        name="parity/task-template",
        instruction="Complete {{ instruction }}.\n",
    )

    insight = Insight(
        workspace="workspace-a",
        title="failure",
        description="description",
        agent=str(tmp_path / "agent-src"),
        trace_refs=["trace-1"],
    )
    clients: list[ClosingClient] = []
    model_client_sets: list[ClosingModelClients] = []
    eval_authors: list[FakeEvalAuthor] = []
    refs = eval_author_run.ConfiguredModelRefs(
        default="workspace-a/default-model",
        fast="workspace-a/fast-model",
    )

    def make_client(_base_url: str | None) -> ClosingClient:
        client = ClosingClient()
        clients.append(client)
        return client

    def make_backend(**_: object) -> FakeBackend:
        return FakeBackend(insight)

    def build_eval_author_agent(**_: object) -> FakeEvalAuthor:
        eval_author = FakeEvalAuthor()
        eval_authors.append(eval_author)
        return eval_author

    async def resolve_model_clients(*_: object) -> ClosingModelClients:
        model_clients = ClosingModelClients()
        model_client_sets.append(model_clients)
        return model_clients

    monkeypatch.setattr(eval_author_run, "make_client", make_client)
    monkeypatch.setattr(eval_author_run, "make_experimentalist_backend", make_backend)
    monkeypatch.setattr(eval_author_run, "build_eval_author_agent", build_eval_author_agent)
    monkeypatch.setattr(eval_author_run, "configured_model_refs", lambda: refs)
    monkeypatch.setattr(eval_author_run, "resolve_model_clients", resolve_model_clients)

    train_ref = DatasetRef(uri=str(train_dir), metadata={"id": "train"})
    validation_ref = DatasetRef(uri=str(validation_dir), metadata={"id": "validation"})
    template_ref = DatasetRef(uri=str(template_dir), metadata={"id": "task-template"})
    experiment_dir = tmp_path / "eval-author"
    results: list[EvalAuthorResult] = []
    calls: list[tuple[Insight, Path, Task, Dataset, Dataset, ClosingClient]] = []

    for evaluator_type in ("harbor-native", "harbor-runner"):
        results.append(
            await eval_author_run.run_eval_author(
                insight="insight-remote-123",
                train_dataset=train_ref,
                validation_dataset=validation_ref,
                task_template=template_ref,
                experiment_dir=experiment_dir,
                workspace="workspace-a",
                base_url="http://platform.test",
                config=EvalAuthorConfig(),
                evaluator_type=evaluator_type,
            )
        )
        call = eval_authors[-1].call
        assert call is not None
        calls.append(call)

    native_call, sdk_call = calls
    _, _, native_template, native_train, native_validation, _ = native_call
    _, _, sdk_template, sdk_train, sdk_validation, _ = sdk_call
    for call in calls:
        assert isinstance(call[3], HarborDataset)
        assert isinstance(call[4], HarborDataset)

    assert _dataset_snapshot(native_train) == _dataset_snapshot(sdk_train)
    assert _dataset_snapshot(native_validation) == _dataset_snapshot(sdk_validation)
    assert native_template.model_dump(mode="json") == sdk_template.model_dump(mode="json")

    native_result, sdk_result = results
    assert _dataset_snapshot(native_result.train_dataset) == _dataset_snapshot(sdk_result.train_dataset)
    assert _dataset_snapshot(native_result.validation_dataset) == _dataset_snapshot(sdk_result.validation_dataset)
    assert native_result.summary == sdk_result.summary == "Eval Author complete."
    assert len(clients) == 2
    assert all(client.closed for client in clients)
    assert len(model_client_sets) == 2
    assert all(model_clients.closed for model_clients in model_client_sets)


@pytest.mark.parametrize("evaluator_type", ["harbor-native", "harbor-runner"])
@pytest.mark.asyncio
async def test_run_eval_author_hydrates_fileset_task_template(
    monkeypatch: pytest.MonkeyPatch,
    model_clients: ClosingModelClients,
    tmp_path: Path,
    evaluator_type: EvaluatorType,
) -> None:
    listings: list[tuple[str, str, object]] = []
    downloads: list[tuple[str, str, str]] = []

    @dataclass(frozen=True)
    class _FilesetItem:
        path: str

    @dataclass(frozen=True)
    class _FilesetListing:
        data: list[_FilesetItem]

    class _FilesetResponse:
        def data(self) -> _FilesetListing:
            return _FilesetListing([_FilesetItem("task.toml")])

    class _Download:
        async def read(self) -> bytes:
            return b"template\n"

    class FakeFiles:
        async def list_files(self, *, workspace: str, name: str, query_params: object) -> _FilesetResponse:
            listings.append((workspace, name, query_params))
            return _FilesetResponse()

        async def download_file(self, *, workspace: str, name: str, path: str) -> _Download:
            downloads.append((workspace, name, path))
            return _Download()

    client = ClosingClient()
    backend = FakeBackend(
        Insight(workspace="workspace-a", title="failure", description="description", agent="insight-agent")
    )
    dataset_factory = FakeDatasetFactory()
    monkeypatch.setattr(eval_author_run, "make_client", lambda _: client)
    monkeypatch.setattr(dataset_staging.AsyncFilesClient, "from_client", lambda _: FakeFiles())
    monkeypatch.setattr(eval_author_run, "make_experimentalist_backend", lambda **_: backend)
    monkeypatch.setattr(eval_author_run, "DatasetFactory", lambda: dataset_factory)
    monkeypatch.setattr(eval_author_run, "build_eval_author_agent", lambda **_: FakeEvalAuthor())
    template_ref = DatasetRef(uri="fileset://workspace-a/template")
    train = tmp_path / "train"
    validation = tmp_path / "validation"
    train.mkdir()
    validation.mkdir()

    await eval_author_run.run_eval_author(
        insight="insight-123",
        train_dataset=DatasetRef(uri=str(train)),
        validation_dataset=DatasetRef(uri=str(validation)),
        task_template=template_ref,
        experiment_dir=tmp_path / "experiment",
        workspace="workspace-a",
        base_url=None,
        config=EvalAuthorConfig(),
        evaluator_type=evaluator_type,
    )

    staged = (tmp_path / "experiment").resolve() / "dataset" / "task-template"
    assert listings == [("workspace-a", "template", None)]
    assert downloads == [("workspace-a", "template", "task.toml")]
    assert (staged / "task.toml").read_text(encoding="utf-8") == "template\n"
    assert dataset_factory.template_calls == [(evaluator_type, template_ref.model_copy(update={"uri": str(staged)}))]
    assert [Path(ref.uri).name for _, ref in dataset_factory.dataset_calls] == ["train", "validation"]
    assert [call[0] for call in dataset_factory.dataset_calls] == [evaluator_type, evaluator_type]
    assert client.closed
    assert model_clients.closed


@pytest.mark.asyncio
async def test_run_eval_author_uses_agent_override(
    monkeypatch: pytest.MonkeyPatch,
    model_clients: ClosingModelClients,
    tmp_path: Path,
) -> None:
    client = ClosingClient()
    backend = FakeBackend(
        Insight(workspace="workspace-a", title="failure", description="description", agent="insight-agent")
    )
    monkeypatch.setattr(eval_author_run, "make_client", lambda _: client)
    monkeypatch.setattr(eval_author_run, "make_experimentalist_backend", lambda **_: backend)
    monkeypatch.setattr(eval_author_run, "DatasetFactory", FakeDatasetFactory)
    monkeypatch.setattr(eval_author_run, "build_eval_author_agent", lambda **_: FakeEvalAuthor())
    template = tmp_path / "template"
    train = tmp_path / "train"
    validation = tmp_path / "validation"
    template.mkdir()
    train.mkdir()
    validation.mkdir()
    override = tmp_path / "override-agent"

    await eval_author_run.run_eval_author(
        insight="insight-123",
        agent=override,
        train_dataset=DatasetRef(uri=str(train)),
        validation_dataset=DatasetRef(uri=str(validation)),
        task_template=DatasetRef(uri=str(template)),
        experiment_dir=tmp_path / "experiment",
        workspace="workspace-a",
        base_url="http://platform.test",
        config=EvalAuthorConfig(),
    )

    assert backend.agent_calls[0][1] == override
    assert client.closed
    assert model_clients.closed


@pytest.mark.asyncio
async def test_run_eval_author_closes_clients_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    model_clients: ClosingModelClients,
    tmp_path: Path,
) -> None:
    client = ClosingClient()
    monkeypatch.setattr(eval_author_run, "make_client", lambda _: client)
    monkeypatch.setattr(
        eval_author_run,
        "make_experimentalist_backend",
        lambda **_: (_ for _ in ()).throw(RuntimeError("backend failed")),
    )

    with pytest.raises(RuntimeError, match="backend failed"):
        await eval_author_run.run_eval_author(
            insight="insight-123",
            train_dataset=DatasetRef(uri="train"),
            validation_dataset=DatasetRef(uri="validation"),
            task_template=DatasetRef(uri="template"),
            experiment_dir=tmp_path,
            workspace="workspace-a",
            base_url=None,
            config=EvalAuthorConfig(),
        )

    assert client.closed
    assert model_clients.closed
