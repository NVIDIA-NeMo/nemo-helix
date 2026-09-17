# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interrupt a run, start it again over the same directory, and check what survives.

Everything here is real except the model and the evaluator: a real ``ExperimentRunner``,
a real ``ExperimentContext``, and a real ``LocalExperimentalistBackend`` writing to a real
directory. The strategy is scripted rather than the evolutionary loop, because the loop
needs an LLM for every step — but it is scripted to make the same calls in the same order,
so the host-side half of resume is exercised end to end.

Three failures are reproduced below: a fresh run silently minted over a populated
candidate store, a run stranded in ``status="running"`` when finalizing threw, and a torn
``run.json``. Each test states the failure it pins.
"""

import json
import os
from pathlib import Path
from typing import Any, ClassVar

import pytest
from doubles import FakeEvaluator, fake_client
from nemo_experimentalist_plugin.config import CandidateStorageConfig, EvolutionaryOptimizerConfig
from nemo_experimentalist_plugin.entities import Candidate, Dataset, DatasetRef, RewardRecord, Task
from nemo_experimentalist_plugin.experimentalist import runner as runner_module
from nemo_experimentalist_plugin.experimentalist.context import ExperimentContext
from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import LocalExperimentalistBackend
from nemo_experimentalist_plugin.experimentalist.runner import ExperimentRunner


class ScriptedStrategy:
    """Stands in for the loop, calling the context the way the loop does.

    ``fail_after`` names the step to raise on, so a test can cut the run at the point a
    crash would realistically land: after the baseline is committed, after it is scored,
    or not at all.
    """

    supports_resume: ClassVar[bool] = True

    def __init__(self, fail_after: str | None = None) -> None:
        self.fail_after = fail_after
        self.saw_resuming: bool | None = None
        self.population_at_start: list[Candidate] = []
        self.run_id: str | None = None

    async def run(self, ctx: ExperimentContext) -> Candidate | None:
        self.saw_resuming = ctx.resuming
        self.run_id = ctx.run_id
        self.population_at_start = await ctx.candidates()

        baseline = next((c for c in self.population_at_start if c.is_baseline), None)
        if baseline is None:
            baseline = await _import_baseline(ctx)
        if self.fail_after == "baseline":
            raise RuntimeError("interrupted right after committing the baseline")

        if "validation" not in baseline.rewards:
            await ctx.record_reward(baseline, channel="validation", result=RewardRecord(metrics={"reward": 0.25}))
        await ctx.report_progress(completed=1, total=2, unit="round")
        if self.fail_after == "scored":
            raise RuntimeError("interrupted after the baseline was scored")

        return baseline


def _make_runner(tmp_path: Path, strategy: Any, monkeypatch: pytest.MonkeyPatch) -> ExperimentRunner:
    """A runner over a real local backend, with only the model and evaluator faked."""
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(exist_ok=True)
    (agent_dir / "main.py").write_text("print('agent')\n")
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir(exist_ok=True)

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"id": "task-a"}\n')

    # Only the two things that would otherwise need Harbor and a live endpoint.
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
                "build_task_template": staticmethod(lambda _type, ref, **_options: Task(id="template", uri=ref.uri)),
            },
        )(),
    )

    config = EvolutionaryOptimizerConfig(storage=CandidateStorageConfig(archive_candidates=False, publish_winner=False))
    return ExperimentRunner(
        backend=LocalExperimentalistBackend(client=fake_client(), path=experiment_dir),
        strategy=strategy,
        config=config,
        workspace="default",
        root=experiment_dir,
        agent=agent_dir,
        train_dataset=DatasetRef(uri=str(dataset_path), description="train"),
        validation_dataset=DatasetRef(uri=str(dataset_path), description="validation"),
    )


async def _import_baseline(ctx, description: str = "baseline") -> Candidate:
    """Build the baseline the way a strategy does: an import Proposal through its Builder."""
    from nemo_experimentalist_plugin.experimentalist.components.importer import ImportBuilder, import_proposal

    return await ImportBuilder().build(ctx, import_proposal(description))


@pytest.mark.asyncio
async def test_a_second_start_reopens_the_run_and_sees_the_population(tmp_path, monkeypatch) -> None:
    """The whole point of resume: hours of committed work stay reachable."""
    first = ScriptedStrategy(fail_after="scored")
    with pytest.raises(RuntimeError, match="after the baseline was scored"):
        await _make_runner(tmp_path, first, monkeypatch).run()

    second = ScriptedStrategy()
    result = await _make_runner(tmp_path, second, monkeypatch).run()

    assert second.saw_resuming is True, "the runner should have re-opened, not started over"
    assert second.run_id == first.run_id, "a new run id would orphan every candidate already on disk"
    assert [c.label for c in second.population_at_start] == ["agent-0"]
    assert second.population_at_start[0].rewards["validation"].metrics == {"reward": 0.25}
    assert result.winner is not None


@pytest.mark.asyncio
async def test_an_interrupted_first_round_does_not_mint_a_second_baseline(tmp_path, monkeypatch) -> None:
    """Crashing between committing the baseline and scoring it is the likeliest cut point.

    A second baseline would be re-evaluated, offered to the Proposer and published, and
    discarding either would delete the artifact the other still addresses.
    """
    with pytest.raises(RuntimeError, match="after committing the baseline"):
        await _make_runner(tmp_path, ScriptedStrategy(fail_after="baseline"), monkeypatch).run()

    second = ScriptedStrategy()
    await _make_runner(tmp_path, second, monkeypatch).run()

    records = list((tmp_path / "experiment" / "eval-and-optimize" / "candidates").glob("*.json"))
    assert len(records) == 1, f"expected one baseline, found {[p.name for p in records]}"


@pytest.mark.asyncio
async def test_a_failed_run_is_marked_failed_not_left_running(tmp_path, monkeypatch) -> None:
    """Including when it is *finalizing* that throws, not the strategy.

    Finalizing resolves the winner's artifact, copies it out and persists the result. When
    that ran outside the failure handler, a run whose work had actually finished stayed
    `running` forever with no result written.
    """

    class _WinnerWithNoArtifact(ScriptedStrategy):
        async def run(self, ctx: ExperimentContext) -> Candidate | None:
            winner = await super().run(ctx)
            assert winner is not None
            import shutil

            shutil.rmtree(ctx.candidate_dir(winner))  # the artifact goes missing mid-run
            return winner

    with pytest.raises(ValueError, match="has no artifact at"):
        await _make_runner(tmp_path, _WinnerWithNoArtifact(), monkeypatch).run()

    run = json.loads((tmp_path / "experiment" / "eval-and-optimize" / "run.json").read_text())
    assert run["status"] == "failed", "a run that could not be closed out must not read as still running"


@pytest.mark.asyncio
async def test_a_torn_run_json_refuses_rather_than_orphaning_the_population(tmp_path, monkeypatch) -> None:
    """run.json is rewritten on every progress report, so a kill can truncate it.

    Starting fresh would mint a new run id, and `ctx.candidates()` filters by it — every
    candidate already built would be on disk and invisible. Refusing is recoverable;
    silently starting over is not.
    """
    with pytest.raises(RuntimeError, match="after the baseline was scored"):
        await _make_runner(tmp_path, ScriptedStrategy(fail_after="scored"), monkeypatch).run()

    run_path = tmp_path / "experiment" / "eval-and-optimize" / "run.json"
    run_path.write_text(run_path.read_text()[: len(run_path.read_text()) // 2])  # a torn write

    with pytest.raises(ValueError, match="still.*holds candidate records"):
        await _make_runner(tmp_path, ScriptedStrategy(), monkeypatch).run()


@pytest.mark.asyncio
async def test_run_json_appears_by_rename_not_by_truncating_in_place(tmp_path, monkeypatch) -> None:
    """The guard above is a backstop; this is the fix that keeps it from being needed.

    Asserting on the mechanism rather than the outcome, because the outcome is identical
    either way: a plain `write_text` also leaves valid JSON when nothing interrupts it.
    What matters is that a reader interrupting this can only ever see the old file or the
    new one, which is what `os.replace` buys and truncate-then-write does not.
    """
    from nemo_experimentalist_plugin.experimentalist import experimentalist_backend as backend_module

    renamed: list[str] = []
    real_replace = backend_module.os.replace

    def _recording_replace(
        src: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        dst: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        renamed.append(Path(str(dst)).name)
        real_replace(src, dst)

    monkeypatch.setattr(backend_module.os, "replace", _recording_replace)

    await _make_runner(tmp_path, ScriptedStrategy(), monkeypatch).run()

    eo = tmp_path / "experiment" / "eval-and-optimize"
    assert "run.json" in renamed, "run.json was written in place, so a kill mid-write can tear it"
    assert json.loads((eo / "run.json").read_text())["status"] == "completed"
    assert not list(eo.glob(".run.json.tmp")), "the temp file must be renamed away, not left behind"


@pytest.mark.asyncio
async def test_a_round_zero_resume_records_no_baseline_reward_and_does_not_raise(tmp_path) -> None:
    """The narrowest cut point of all: crash after the baseline was scored, before round-0's analysis.

    `_detect_last_round` finds no `round-0.md`, so the loop takes its *fresh* branch — but
    `_ensure_baseline` keeps the already-scored baseline and `_evaluate_validation_candidates`
    returns nothing pending. Indexing that empty map by label raised `KeyError: 'agent-0'`
    and failed the run on restart.
    """
    from doubles import FakeBackend, make_context
    from nemo_experimentalist_plugin.experimentalist.strategies.evolutionary import EvolutionaryStrategy

    ctx = make_context(root=tmp_path, backend=FakeBackend())
    baseline = await _import_baseline(ctx)
    await ctx.record_reward(baseline, channel="validation", result=RewardRecord(metrics={"reward": 0.4}))

    await EvolutionaryStrategy._record_baseline_validation(ctx=ctx, baseline=baseline, results={})

    (stored,) = await ctx.candidates()
    assert stored.rewards["validation"].metrics == {"reward": 0.4}, "the earlier measurement must survive"


@pytest.mark.asyncio
async def test_a_completed_run_is_not_reopened(tmp_path, monkeypatch) -> None:
    """Resume is for a run that was interrupted, not one that finished.

    A completed run has its winner chosen, its report written and possibly a PR opened.
    Re-opening set its status back to `running` and let the strategy rewrite all of it,
    so pointing `--experiment-dir` at a finished run silently destroyed its result. It
    now refuses and says where to point instead.
    """
    await _make_runner(tmp_path, ScriptedStrategy(), monkeypatch).run()

    run = json.loads((tmp_path / "experiment" / "eval-and-optimize" / "run.json").read_text())
    assert run["status"] == "completed", "the first run did not finish; this test needs it to"

    with pytest.raises(ValueError, match="already completed"):
        await _make_runner(tmp_path, ScriptedStrategy(), monkeypatch).run()

    after = json.loads((tmp_path / "experiment" / "eval-and-optimize" / "run.json").read_text())
    assert after["status"] == "completed", "the refused run still reopened the record"
