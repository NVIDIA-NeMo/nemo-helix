# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The runner's own contract: prepare inputs, run one strategy, persist and publish."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, cast

import pytest
from doubles import FakeBackend, FakeEvaluator, fake_client, make_candidate
from nemo_experimentalist_plugin.config import CandidateStorageConfig, EvolutionaryOptimizerConfig
from nemo_experimentalist_plugin.entities import (
    Candidate,
    Dataset,
    DatasetRef,
    ExperimentRun,
    MetricTarget,
    ResourceRef,
    RewardRecord,
    Task,
)
from nemo_experimentalist_plugin.experimentalist import runner as runner_module
from nemo_experimentalist_plugin.experimentalist.context import ExperimentContext
from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import LocalExperimentalistBackend
from nemo_experimentalist_plugin.experimentalist.roles import Strategy
from nemo_experimentalist_plugin.experimentalist.runner import ExperimentRunner
from nemo_experimentalist_plugin.experimentalist.seam import StrategyContext
from nemo_insights_plugin.entities import Insight
from nemo_platform_plugin.config import Configuration


class RecordingStrategy(Strategy):
    """Keeps the context it was handed, then returns (or raises) what it was told to."""

    supports_resume: ClassVar[bool] = True

    def __init__(self, winner: Candidate | None = None, error: Exception | None = None) -> None:
        self.winner = winner
        self.error = error
        self.ctx: ExperimentContext | None = None

    async def run(self, ctx: StrategyContext) -> Candidate | None:
        self.ctx = cast(ExperimentContext, ctx)
        if self.error is not None:
            raise self.error
        if self.winner is not None:
            await ctx.update_candidate(self.winner)
        return self.winner


class NonResumableStrategy(RecordingStrategy):
    supports_resume: ClassVar[bool] = False


def _stub_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace evaluator/dataset construction, which would otherwise need Harbor."""
    monkeypatch.setattr(
        runner_module,
        "EvaluatorFactory",
        lambda: type("F", (), {"build_evaluator": staticmethod(lambda *a, **k: FakeEvaluator())})(),
    )
    monkeypatch.setattr(
        runner_module,
        "DatasetFactory",
        lambda: type(
            "F",
            (),
            {
                "build_dataset": staticmethod(lambda _type, ref, **_options: Dataset(id=ref.uri)),
                "build_task_template": staticmethod(lambda _type, ref: Task(id="template", uri=ref.uri)),
            },
        )(),
    )


def _make_runner(
    tmp_path: Path,
    *,
    strategy: RecordingStrategy,
    monkeypatch: pytest.MonkeyPatch,
    backend: FakeBackend | None = None,
    config: EvolutionaryOptimizerConfig | None = None,
) -> tuple[ExperimentRunner, FakeBackend]:
    backend = backend or FakeBackend()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(exist_ok=True)
    _stub_factories(monkeypatch)
    return (
        ExperimentRunner(
            backend=backend,
            strategy=strategy,
            config=config or EvolutionaryOptimizerConfig(),
            workspace="default",
            root=tmp_path / "experiment",
            agent=agent_dir,
            train_dataset=DatasetRef(uri="train"),
            validation_dataset=DatasetRef(uri="validation"),
        ),
        backend,
    )


@pytest.mark.asyncio
async def test_strategy_failure_marks_the_run_failed(monkeypatch, tmp_path) -> None:
    strategy = RecordingStrategy(error=ValueError("baseline failed"))
    runner, _ = _make_runner(tmp_path, strategy=strategy, monkeypatch=monkeypatch)

    with pytest.raises(ValueError, match="baseline failed"):
        await runner.run()

    assert strategy.ctx is not None
    assert strategy.ctx._run.status == "failed"


@pytest.mark.asyncio
async def test_a_completed_run_persists_its_result_and_winner(monkeypatch, tmp_path) -> None:
    winner = make_candidate(
        run_id="run-1",
        label="agent-1",
        ancestor="agent-0",
        generation=1,
        description="add a tool",
        rewards={"validation": RewardRecord(metrics={"reward": 0.75})},
    )
    (tmp_path / "experiment" / "eval-and-optimize" / "agents" / "agent-1").mkdir(parents=True)
    runner, backend = _make_runner(tmp_path, strategy=RecordingStrategy(winner=winner), monkeypatch=monkeypatch)

    result = await runner.run()

    assert result.winner is winner
    assert "winner=agent-1" in result.summary
    assert backend.results == [result]
    # publish_winner defaults to True upstream, so the winner is published.
    assert backend.published == ["agent-1"]
    # run.json names the winner by label, because every artifact path is built from
    # the label -- 'agents/agent-1', 'results/agent-1-validation'. Writing the id here
    # yields a reference that resolves to nothing, and an empty lookup downstream reads
    # as "the winner has no results" rather than as a broken reference.
    assert backend.runs[-1].winner_agent == "agent-1"
    assert backend.runs[-1].winner_agent != winner.id


@pytest.mark.asyncio
async def test_the_winner_is_published_when_storage_asks_for_it(monkeypatch, tmp_path) -> None:
    winner = make_candidate(run_id="run-1", label="agent-1", ancestor="agent-0", generation=1, description="add a tool")
    (tmp_path / "experiment" / "eval-and-optimize" / "agents" / "agent-1").mkdir(parents=True)
    config = EvolutionaryOptimizerConfig(storage=CandidateStorageConfig(publish_winner=True))
    runner, backend = _make_runner(
        tmp_path, strategy=RecordingStrategy(winner=winner), config=config, monkeypatch=monkeypatch
    )

    await runner.run()

    assert backend.published == ["agent-1"]


@pytest.mark.asyncio
async def test_the_baseline_is_never_published(monkeypatch, tmp_path) -> None:
    """``ancestor is None`` means the baseline — there is nothing to open a PR against."""
    baseline = make_candidate(run_id="run-1", label="agent-0", ancestor=None, generation=0, description="baseline")
    (tmp_path / "experiment" / "eval-and-optimize" / "agents" / "agent-0").mkdir(parents=True)
    config = EvolutionaryOptimizerConfig(storage=CandidateStorageConfig(publish_winner=True))
    runner, backend = _make_runner(
        tmp_path, strategy=RecordingStrategy(winner=baseline), config=config, monkeypatch=monkeypatch
    )

    await runner.run()

    assert backend.published == []


@pytest.mark.asyncio
async def test_the_winner_is_copied_out_without_any_owners_scaffolding(monkeypatch, tmp_path) -> None:
    winner_dir = tmp_path / "experiment" / "eval-and-optimize" / "agents" / "agent-1"
    winner_dir.mkdir(parents=True)
    winner = make_candidate(
        run_id="run-1",
        label="agent-1",
        ancestor="agent-0",
        generation=1,
        description="add a tool",
        artifact=winner_dir.as_uri(),
    )
    (winner_dir / "main.py").write_text("print('hello')\n")
    (winner_dir / "architecture.md").write_text("# arch")  # the strategy's
    (winner_dir / "harbor_wrapper.py").write_text("# harness")  # the evaluator's
    (winner_dir / "metadata.json").write_text("{}")  # a third-party strategy's real output
    runner, _ = _make_runner(tmp_path, strategy=RecordingStrategy(winner=winner), monkeypatch=monkeypatch)

    await runner.run()

    root = tmp_path / "experiment"
    assert (root / "main.py").read_text() == "print('hello')\n"
    for scaffolding in ("architecture.md", "harbor_wrapper.py"):
        assert not (root / scaffolding).exists(), scaffolding
    # Candidate metadata lives in its own store, so this name is not the host's
    # to strip — a strategy that writes one has produced real output.
    assert (root / "metadata.json").exists()


@pytest.mark.asyncio
async def test_a_run_with_no_winner_still_completes(monkeypatch, tmp_path) -> None:
    runner, backend = _make_runner(tmp_path, strategy=RecordingStrategy(winner=None), monkeypatch=monkeypatch)

    result = await runner.run()

    assert result.winner is None
    assert "winner=none" in result.summary
    assert backend.results == [result]


@pytest.mark.asyncio
async def test_a_missing_strategy_report_falls_back_to_the_compact_summary(monkeypatch, tmp_path) -> None:
    runner, _ = _make_runner(tmp_path, strategy=RecordingStrategy(winner=None), monkeypatch=monkeypatch)

    await runner.run()

    report = (tmp_path / "experiment" / "eval-and-optimize" / "OPTIMIZATION.md").read_text()
    assert "Compact Run Summary" in report
    assert "Optimization complete" in report


@pytest.mark.asyncio
async def test_an_existing_report_is_left_alone(monkeypatch, tmp_path) -> None:
    eo = tmp_path / "experiment" / "eval-and-optimize"
    eo.mkdir(parents=True)
    (eo / "OPTIMIZATION.md").write_text("# The strategy's own report\n")
    runner, _ = _make_runner(tmp_path, strategy=RecordingStrategy(winner=None), monkeypatch=monkeypatch)

    await runner.run()

    assert (eo / "OPTIMIZATION.md").read_text() == "# The strategy's own report\n"


@pytest.mark.asyncio
async def test_resume_refuses_loudly_for_a_strategy_that_cannot(monkeypatch, tmp_path) -> None:
    """These runs cost hours, so a silent restart is the expensive failure."""
    eo = tmp_path / "experiment" / "eval-and-optimize"
    eo.mkdir(parents=True)
    (eo / "run.json").write_text(
        json.dumps(
            {"id": "run-existing", "name": "run-existing", "workspace": "default", "agent": "a", "status": "running"}
        )
    )
    runner, _ = _make_runner(tmp_path, strategy=NonResumableStrategy(), monkeypatch=monkeypatch)

    with pytest.raises(ValueError, match="does not support resume"):
        await runner.run()


@pytest.mark.asyncio
async def test_resume_reopens_the_existing_run_rather_than_creating_one(monkeypatch, tmp_path) -> None:
    eo = tmp_path / "experiment" / "eval-and-optimize"
    eo.mkdir(parents=True)
    (eo / "run.json").write_text(
        json.dumps(
            {"id": "run-existing", "name": "run-existing", "workspace": "default", "agent": "a", "status": "failed"}
        )
    )
    strategy = RecordingStrategy()
    runner, backend = _make_runner(tmp_path, strategy=strategy, monkeypatch=monkeypatch)

    await runner.run()

    assert strategy.ctx is not None
    assert strategy.ctx.run_id == "run-existing"
    assert strategy.ctx.resuming is True
    assert backend.runs == []  # nothing was created


@pytest.mark.asyncio
async def test_persistence_without_git_fails_before_any_work(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runner_module.shutil, "which", lambda _: None)
    config = EvolutionaryOptimizerConfig(storage=CandidateStorageConfig(archive_candidates=True))
    strategy = RecordingStrategy()
    runner, _ = _make_runner(tmp_path, strategy=strategy, config=config, monkeypatch=monkeypatch)

    with pytest.raises(ValueError, match="'git' is not on PATH"):
        await runner.run()

    assert strategy.ctx is None


@pytest.mark.asyncio
async def test_a_failed_run_is_visible_on_disk(monkeypatch, tmp_path) -> None:
    """The local backend writes run.json, so a failed run survives the process."""
    _stub_factories(monkeypatch)
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    runner = ExperimentRunner(
        backend=LocalExperimentalistBackend(client=fake_client(), path=tmp_path / "experiment"),
        strategy=RecordingStrategy(error=ValueError("baseline failed")),
        config=EvolutionaryOptimizerConfig(),
        workspace="default",
        root=tmp_path / "experiment",
        agent=agent_dir,
        train_dataset=DatasetRef(uri="train"),
        validation_dataset=DatasetRef(uri="validation"),
    )

    with pytest.raises(ValueError, match="baseline failed"):
        await runner.run()

    saved = json.loads((tmp_path / "experiment" / "eval-and-optimize" / "run.json").read_text(encoding="utf-8"))
    assert saved["status"] == "failed"


def _insight_dataset(identity_char: str = "a") -> Dataset:
    suite = Path("/experiment/eval-and-optimize/eval_author/insight-1/insight-suite")
    task = Task(id="task-a", uri=(suite / "task-a").as_uri())
    return Dataset(
        id="insight",
        source=ResourceRef(uri=suite.as_uri()),
        tasks=[task],
        metadata={
            "insight_suite_identity": f"sha256:{identity_char * 64}",
            "insight_suite_scorer_identity": f"sha256:{'b' * 64}",
            "insight_suite_task_hashes": {
                "task-a": {"content_hash": f"sha256:{'c' * 64}", "verifier_hash": f"sha256:{'d' * 64}"}
            },
        },
    )


def _insight_candidate(label: str, *, round_num: int, insight: float, validation: float) -> Candidate:
    return make_candidate(
        run_id="run-1",
        label=label,
        ancestor=None if round_num == 0 else "agent-0",
        generation=round_num,
        description="baseline" if round_num == 0 else "improve required tool use",
        rewards={
            "insight": RewardRecord(
                metrics={"reward": validation, "uses_required_tool": insight},
                metadata={
                    "suite_identity": f"sha256:{'a' * 64}",
                    "metric_keys": ["reward", "uses_required_tool"],
                },
            ),
            "validation": RewardRecord(metrics={"reward": validation}),
        },
    )


async def _run_with_insight_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    dataset: Dataset,
    candidates: list[Candidate],
) -> None:
    """Drive a run whose Eval Author authored *dataset*, with *candidates* already scored."""
    backend = FakeBackend(
        client=fake_client(),  # an Insight run needs a platform client to load traces
        insight=Insight(
            workspace="default",
            agent="agent-source",
            title="tool use",
            description="the agent skips the required tool",
        ),
    )
    for candidate in candidates:
        await backend.create_candidate(workspace="default", candidate=candidate)
    winner = candidates[-1]
    (tmp_path / "experiment" / "eval-and-optimize" / "agents" / winner.label).mkdir(parents=True)

    _stub_factories(monkeypatch)
    monkeypatch.setattr(
        runner_module,
        "stage_eval_author_inputs",
        _async(lambda _root, **refs: SimpleNamespace(**refs)),
    )
    monkeypatch.setattr(
        "nemo_experimentalist_plugin.eval_author.agent.EvalAuthor",
        lambda **_: SimpleNamespace(
            run=_async(
                lambda **kwargs: SimpleNamespace(
                    train_dataset=kwargs["train_dataset"],
                    validation_dataset=kwargs["validation_dataset"],
                    insight_suite=dataset,
                )
            )
        ),
    )
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(exist_ok=True)
    await ExperimentRunner(
        backend=backend,
        strategy=RecordingStrategy(winner=winner),
        config=EvolutionaryOptimizerConfig(),
        workspace="default",
        root=tmp_path / "experiment",
        agent=agent_dir,
        insight="insight-1",
        train_dataset=DatasetRef(uri="train"),
        validation_dataset=DatasetRef(uri="validation"),
        task_template=DatasetRef(uri="template"),
    ).run()


def _async(fn):
    async def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def test_a_run_needs_either_an_agent_or_an_insight(tmp_path) -> None:
    with pytest.raises(ValueError, match="One of 'insight' or 'agent'"):
        ExperimentRunner(
            backend=FakeBackend(),
            strategy=RecordingStrategy(),
            config=EvolutionaryOptimizerConfig(),
            workspace="default",
            root=tmp_path,
            agent=None,
            train_dataset=DatasetRef(uri="train"),
            validation_dataset=DatasetRef(uri="validation"),
        )


def test_an_insight_run_needs_a_task_template(tmp_path) -> None:
    with pytest.raises(ValueError, match="'task_template' is required"):
        ExperimentRunner(
            backend=FakeBackend(),
            strategy=RecordingStrategy(),
            config=EvolutionaryOptimizerConfig(),
            workspace="default",
            root=tmp_path,
            agent=None,
            insight="insight-1",
            train_dataset=DatasetRef(uri="train"),
            validation_dataset=DatasetRef(uri="validation"),
        )


def test_the_run_entity_carries_a_progress_counter_not_a_round_count() -> None:
    """Not every strategy has rounds, so the entity counts units and names the unit."""
    run = ExperimentRun(workspace="default", agent="a")
    assert (run.progress_completed, run.progress_total, run.progress_unit) == (0, None, "step")


@pytest.mark.asyncio
async def test_the_run_record_never_carries_the_credential(monkeypatch, tmp_path) -> None:
    """It is written to run.json on disk and mirrored to the platform."""
    monkeypatch.setenv("NEMO_EXPERIMENTALIST_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("NEMO_EXPERIMENTALIST_API_KEY", "super-secret-key")
    for tier in ("SMART", "MID", "FAST"):
        monkeypatch.setenv(f"NEMO_EXPERIMENTALIST_MODELS_{tier}", "vendor/m")
    Configuration.clear_cache()

    strategy = RecordingStrategy()
    runner, _ = _make_runner(tmp_path, strategy=strategy, monkeypatch=monkeypatch)
    await runner.run()

    assert strategy.ctx is not None
    assert "super-secret-key" not in json.dumps(strategy.ctx._run.config_snapshot)


@pytest.mark.asyncio
async def test_the_strategy_receives_the_runs_metric_contract(monkeypatch, tmp_path) -> None:
    """A strategy reads the run's objectives off the context it was handed.

    Reading them from a module-level getter instead would make a run report metrics other
    than the ones it optimized, and `config_snapshot` would record the config's while the
    loop used something else.
    """
    objectives = [MetricTarget(name="pass_rate", direction="maximize")]
    guardrails = [MetricTarget(name="latency", direction="minimize")]
    strategy = RecordingStrategy()
    runner, _ = _make_runner(
        tmp_path,
        strategy=strategy,
        monkeypatch=monkeypatch,
        config=EvolutionaryOptimizerConfig(objective_function=objectives, regression_metrics=guardrails),
    )

    await runner.run()

    assert strategy.ctx is not None
    assert strategy.ctx.objective_metrics == objectives, "the strategy was handed different objectives"
    assert strategy.ctx.regression_metrics == guardrails


@pytest.mark.asyncio
async def test_the_winner_copy_out_never_overwrites_the_runs_own_layout(monkeypatch, tmp_path) -> None:
    """A winner artifact containing `eval-and-optimize` must not delete the run.

    `fork` strips the run layout on the way in, but `commit_candidate` accepts any
    artifact under the candidate root, so a strategy that builds without forking -- the
    documented extension point -- can commit one that carries it. The copy-out `rmtree`s
    each destination first, so without this the run deletes its own records while
    finalizing, and there is nothing left to resume.
    """
    artifact = tmp_path / "artifact" / "agent-1"
    (artifact / "eval-and-optimize").mkdir(parents=True)
    # the winner carries a directory named like the run's own
    (artifact / "eval-and-optimize" / "planted.txt").write_text("from the candidate", encoding="utf-8")
    (artifact / "agent.py").write_text("real source", encoding="utf-8")
    winner = make_candidate(
        run_id="run-1",
        label="agent-1",
        ancestor="agent-0",
        generation=1,
        description="x",
        artifact=artifact.as_uri(),
    )

    runner, _ = _make_runner(tmp_path, strategy=RecordingStrategy(winner=winner), monkeypatch=monkeypatch)

    await runner.run()

    # The run's own directory is left alone; had it been copied, rmtree would have taken
    # the live one with it and this file would be sitting inside the result.
    assert not (tmp_path / "experiment" / "eval-and-optimize" / "planted.txt").exists(), (
        "the winner's eval-and-optimize was copied over the run's own"
    )
    # ...and the rest of the winner still copies, so the guard is not simply skipping.
    assert (tmp_path / "experiment" / "agent.py").read_text(encoding="utf-8") == "real source"
