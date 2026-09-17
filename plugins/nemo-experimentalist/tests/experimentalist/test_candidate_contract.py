# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The candidate contract: fork → build → commit, and what commit refuses.

A Candidate exists only once its artifact does, so nothing durable can point at partial
work. These cover the ways that could be violated — committing a missing artifact,
committing one outside the run's candidate root, letting the derived projections drift
from the Proposal they came from, and updating a Candidate into existence without ever
having built one.
"""

import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest
from doubles import FakeBackend, fake_client, make_candidate, make_context
from nemo_experimentalist_plugin.entities import (
    Candidate,
    Dataset,
    EvaluationResult,
    Proposal,
    ResourceRef,
    RewardRecord,
    Task,
    TrialResult,
)
from nemo_experimentalist_plugin.experimentalist.context import ExperimentContext


def _proposal(ancestor: str | None = "agent-0") -> Proposal:
    return Proposal(
        ancestor=ancestor,
        description="add a retrieval step before the planner",
        kind="code-change",
        payload={"optimization_type": "add_method", "task_ids": ["task-1"], "root_cause": "no retrieval step"},
    )


async def _import_baseline(ctx: ExperimentContext, description: str = "baseline") -> Candidate:
    """Build the baseline the way a strategy does: an import Proposal through its Builder."""
    from nemo_experimentalist_plugin.experimentalist.components.importer import ImportBuilder, import_proposal

    return await ImportBuilder().build(ctx, import_proposal(description))


@pytest.mark.asyncio
async def test_importing_the_baseline_copies_the_agent_under_test(tmp_path: Path) -> None:
    ctx = make_context(root=tmp_path)
    (ctx.agent_dir / "main.py").write_text("print('hello')\n")

    baseline = await _import_baseline(ctx)

    assert baseline.label == "agent-0", "the first fork takes the first free handle; nothing names it"
    assert baseline.is_baseline
    assert baseline.generated_from.kind == "import"
    assert (ctx.candidate_dir(baseline) / "main.py").read_text() == "print('hello')\n"


@pytest.mark.asyncio
async def test_a_baseline_import_carries_no_owners_scaffolding(tmp_path: Path) -> None:
    """The ignore set is composed from three owners, so no one list can strip too much."""
    ctx = make_context(root=tmp_path)
    (ctx.agent_dir / "main.py").write_text("keep me\n")
    for name in ("__pycache__", ".venv", "dataset", "eval-and-optimize", "my-traces"):
        (ctx.agent_dir / name).mkdir()

    imported = ctx.candidate_dir(await _import_baseline(ctx))

    assert (imported / "main.py").exists()
    for name in ("__pycache__", ".venv", "dataset", "eval-and-optimize", "my-traces"):
        assert not (imported / name).exists(), name


@pytest.mark.asyncio
async def test_forking_a_proposal_branches_from_its_ancestors_artifact(tmp_path: Path) -> None:
    """The ancestor is resolved through its stored artifact, not by treating its id as a path."""
    ctx = make_context(root=tmp_path, backend=FakeBackend())
    baseline = await _import_baseline(ctx)
    (ctx.candidate_dir(baseline) / "main.py").write_text("ancestor code\n")

    fork = await ctx.fork(_proposal(ancestor=baseline.id))

    assert fork.workdir.name == "agent-1"
    assert (fork.workdir / "main.py").read_text() == "ancestor code\n"


@pytest.mark.asyncio
async def test_a_fork_reports_the_upstream_it_was_taken_from(tmp_path: Path) -> None:
    """A Builder cannot diff its finished work without a pristine parent to diff against.

    Once it starts editing, the copy in its workdir is no longer the parent — and
    architecture.md is excluded from the seeding entirely, so the upstream directory is
    the only place it can be read.
    """
    ctx = make_context(root=tmp_path, backend=FakeBackend())
    baseline = await _import_baseline(ctx)

    from_parent = await ctx.fork(_proposal(ancestor=baseline.id))
    from_scratch = await ctx.fork(_proposal(ancestor=None))

    assert from_parent.upstream == ctx.candidate_dir(baseline)
    assert from_scratch.upstream is None, "forked from the agent under test, which is not a candidate"


@pytest.mark.asyncio
async def test_committing_derives_lineage_and_description_from_the_proposal(tmp_path: Path) -> None:
    ctx = make_context(root=tmp_path)
    baseline = await _import_baseline(ctx)
    # The ancestor is the baseline's durable id, which is not its display handle.
    proposal = _proposal(ancestor=baseline.id)
    fork = await ctx.fork(proposal)

    candidate = await ctx.commit_candidate(proposal=proposal, artifact=fork.workdir, generation=1)

    assert candidate.ancestor == baseline.id
    assert candidate.ancestor != baseline.label
    assert candidate.description == proposal.description
    assert candidate.generated_from == proposal
    assert candidate.generation == 1
    assert candidate.artifact.uri == fork.workdir.resolve().as_uri()
    assert not candidate.is_baseline


@pytest.mark.asyncio
async def test_committing_a_missing_artifact_is_refused(tmp_path: Path) -> None:
    ctx = make_context(root=tmp_path)

    with pytest.raises(ValueError, match="does not exist"):
        await ctx.commit_candidate(
            proposal=_proposal(ancestor=None),
            artifact=tmp_path / "eval-and-optimize" / "agents" / "never-written",
        )


@pytest.mark.asyncio
async def test_an_artifact_outside_the_candidate_root_is_refused(tmp_path: Path) -> None:
    """The runner must be able to archive and publish every candidate without help."""
    ctx = make_context(root=tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with pytest.raises(ValueError, match="must live under"):
        await ctx.commit_candidate(proposal=_proposal(ancestor=None), artifact=elsewhere)


@pytest.mark.asyncio
async def test_a_candidate_cannot_be_updated_into_existence(tmp_path: Path) -> None:
    """``commit_candidate`` and ``import_baseline`` are the only ways one is born.

    A create-or-update verb would happily persist a record whose artifact had never been
    validated, which is exactly what "nothing durable points at partial work" forbids.
    """
    ctx = make_context(root=tmp_path)
    # Assembled by hand rather than committed, so it has no store id — exactly the shape
    # a component could otherwise smuggle into the store.
    never_committed = make_candidate(label="agent-7", ancestor=None)
    never_committed._id = ""

    with pytest.raises(ValueError, match="never by updating one into existence"):
        await ctx.update_candidate(never_committed, killed_generation=3)


@pytest.mark.asyncio
async def test_updating_a_committed_candidate_persists_the_change(tmp_path: Path) -> None:
    """Through the on-disk backend deliberately.

    The in-memory double stores the very object the context hands back, so ``setattr``
    alone would satisfy this and deleting the backend write would leave it green.
    """
    from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import LocalExperimentalistBackend

    ctx = make_context(root=tmp_path, backend=LocalExperimentalistBackend(client=fake_client(), path=tmp_path))
    baseline = await _import_baseline(ctx)

    await ctx.update_candidate(baseline, killed_generation=2)

    reread = json.loads((tmp_path / "eval-and-optimize" / "candidates" / f"{baseline.id}.json").read_text())
    assert reread["killed_generation"] == 2


def test_a_candidate_whose_projections_drift_from_its_origin_is_invalid() -> None:
    proposal = _proposal()

    with pytest.raises(ValueError, match="disagrees with its Proposal"):
        Candidate(
            label="agent-1",
            run_id="run-1",
            ancestor="somewhere-else",
            generated_from=proposal,
            description=proposal.description,
            artifact=ResourceRef(uri="file:///tmp/agent-1"),
        )


def test_the_baseline_is_the_one_without_an_ancestor() -> None:
    assert make_candidate(label="agent-0").is_baseline
    assert not make_candidate(label="agent-1", ancestor="agent-0").is_baseline


@pytest.mark.asyncio
async def test_candidates_are_listed_from_the_store_not_from_a_directory_walk(tmp_path: Path) -> None:
    """A strategy that does not produce one directory per candidate can still list them."""
    from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import LocalExperimentalistBackend

    backend = LocalExperimentalistBackend(client=fake_client(), path=tmp_path)
    ctx = make_context(root=tmp_path, backend=backend)
    committed = await _import_baseline(ctx)

    listed = await ctx.candidates()

    assert [c.id for c in listed] == [committed.id]
    assert (tmp_path / "eval-and-optimize" / "candidates" / f"{committed.id}.json").is_file()
    # Metadata never lands inside the artifact it describes.
    assert not (ctx.candidate_dir(committed) / "metadata.json").exists()


@pytest.mark.asyncio
async def test_a_fork_does_not_inherit_the_ancestors_architecture_doc(tmp_path: Path) -> None:
    """The Builder is told its directory holds the ancestor's source and nothing else.

    `architecture.md` describes the ancestor, not the candidate. Inheriting it makes that
    statement false at the moment the Builder reads it, and hands a code-writing agent a
    description that no longer matches what it is about to change. The CodeEditBuilder re-seeds it
    from ``Fork.upstream`` afterwards, to edit in place against the finished source.
    """
    ctx = make_context(root=tmp_path, backend=FakeBackend())
    baseline = await _import_baseline(ctx)
    ancestor_dir = ctx.candidate_dir(baseline)
    (ancestor_dir / "main.py").write_text("print('ancestor')\n")
    (ancestor_dir / "architecture.md").write_text("# describes agent-0\n")

    fork = await ctx.fork(_proposal(ancestor=baseline.id))

    assert (fork.workdir / "main.py").read_text() == "print('ancestor')\n"
    assert not (fork.workdir / "architecture.md").exists()
    assert fork.upstream is not None
    assert (fork.upstream / "architecture.md").exists(), "still reachable through the fork's upstream"


@pytest.mark.asyncio
async def test_a_record_whose_artifact_is_gone_is_refused_not_guessed(tmp_path: Path) -> None:
    """Resolving it to the parent yields the shared candidate root.

    Callers copy out of and push whatever that returns, so a stale record would take
    every other candidate's code with it.
    """
    ctx = make_context(root=tmp_path)
    baseline = await _import_baseline(ctx)
    shutil.rmtree(ctx.candidate_dir(baseline))

    with pytest.raises(ValueError, match="has no artifact at"):
        ctx.candidate_dir(baseline)


@pytest.mark.asyncio
async def test_a_discarded_candidate_is_hidden_but_kept(tmp_path: Path) -> None:
    """Rolling back a round is the only thing that discards, and that work is redone.

    Keeping both halves is what stops `_reserve` handing out a label a discarded record
    still claims, and leaves a wrong rollback recoverable.
    """
    ctx = make_context(root=tmp_path, backend=FakeBackend())
    baseline = await _import_baseline(ctx)
    artifact = ctx.candidate_dir(baseline)

    await ctx.discard_candidate(baseline)

    assert await ctx.candidates() == [], "nothing may evaluate, rank or publish it"
    assert artifact.exists(), "the artifact survives, so its directory name stays taken"
    (kept,) = await ctx.candidates(include_discarded=True)
    assert kept.discarded is True


@pytest.mark.asyncio
async def test_every_ancestorless_proposal_gets_its_own_directory(tmp_path: Path) -> None:
    """Only the baseline owns the baseline handle.

    A strategy that builds from the agent under test rather than from a parent — the HPO
    case — emits many ancestor-less proposals; sharing one directory would have them
    overwrite each other and the baseline while still reporting distinct rewards.
    """
    ctx = make_context(root=tmp_path)
    baseline_dir = ctx.candidate_dir(await _import_baseline(ctx))

    first = await ctx.fork(_proposal(ancestor=None))
    second = await ctx.fork(_proposal(ancestor=None))

    assert first.workdir != second.workdir
    assert first.workdir != baseline_dir
    assert second.workdir != baseline_dir


@pytest.mark.asyncio
async def test_concurrent_forks_never_share_a_directory(tmp_path: Path) -> None:
    """Builds run under ``asyncio.gather``, so several forks are in flight at once.

    ``_reserve`` picks ``max(existing) + 1`` by listing the directory, which is only safe
    because nothing awaits between choosing the name and creating it — two candidates
    sharing a directory would overwrite each other's source while still reporting
    separate rewards.
    """
    import asyncio

    ctx = make_context(root=tmp_path, backend=FakeBackend())
    baseline = await _import_baseline(ctx)
    proposals = [_proposal(ancestor=baseline.id) for _ in range(8)]

    forks = await asyncio.gather(*(ctx.fork(p) for p in proposals))

    workdirs = [fork.workdir for fork in forks]
    assert len(set(workdirs)) == len(workdirs), f"reservations collided: {sorted(p.name for p in workdirs)}"
    assert ctx.candidate_dir(baseline) not in workdirs


@pytest.mark.asyncio
async def test_a_cancelled_build_unwinds_the_round_rather_than_being_dropped(tmp_path: Path) -> None:
    """Cancellation is not a build failure, and the difference is invisible to `Exception`.

    `asyncio.gather(..., return_exceptions=True)` hands back `CancelledError` as a *value*.
    It derives from `BaseException`, so a filter written as `isinstance(outcome, Exception)`
    lets a cancelled build through: the round carries on with a partial population and
    ranks candidates that were never built. This covered `_build_candidates` before the
    Builder refactor moved it; without it the guard can be dropped and nothing fails.
    """
    import asyncio

    from nemo_experimentalist_plugin.config import EvolutionaryOptimizerConfig
    from nemo_experimentalist_plugin.experimentalist.strategies.evolutionary import EvolutionaryStrategy

    class _CancellingBuilder:
        accepts = frozenset({"code-change"})

        async def build(self, ctx: object, proposal: Proposal, *, generation: int) -> None:
            raise asyncio.CancelledError

    optimizer = object.__new__(EvolutionaryStrategy)
    optimizer.working_dir = tmp_path
    optimizer._framework_skills_dirs = []
    optimizer._models = None
    optimizer._new_builder = lambda **_: _CancellingBuilder()  # type: ignore[method-assign]

    ctx = make_context(root=tmp_path, backend=FakeBackend())
    config = EvolutionaryOptimizerConfig()

    with pytest.raises(asyncio.CancelledError):
        await optimizer._build_candidates(
            ctx=ctx,
            dataset=ctx.datasets["validation"],
            proposals=[_proposal(ancestor=None)],
            generation=1,
            config=config,
        )


@pytest.mark.asyncio
async def test_a_nested_architecture_doc_is_the_agents_own_and_survives(tmp_path: Path) -> None:
    """Only the one beside the agent is ours to strip.

    `architecture.md` is generated *about* the agent and sits at its root, and the fork
    leaves it out so its absence tells the Proposer it has no model of this candidate.
    An agent whose own source ships `docs/architecture.md` is a different file entirely —
    stripping it at every depth silently deletes the user's documentation from every
    candidate, and from the winner copied back over their workspace.
    """
    ctx = make_context(root=tmp_path, backend=FakeBackend())
    (ctx.agent_dir / "architecture.md").write_text("# generated about the agent\n")
    (ctx.agent_dir / "docs").mkdir()
    (ctx.agent_dir / "docs" / "architecture.md").write_text("# the agent's own docs\n")

    imported = ctx.candidate_dir(await _import_baseline(ctx))

    assert not (imported / "architecture.md").exists(), "the generated one is ours to leave out"
    assert (imported / "docs" / "architecture.md").read_text() == "# the agent's own docs\n"


@pytest.mark.asyncio
async def test_the_store_itself_hides_discarded_candidates(tmp_path: Path) -> None:
    """Through the real backend, not the double.

    The filter lives in `list_candidates` rather than at each call site precisely so a
    consumer cannot forget it — which means the check has to go through the backend that
    ships, or it only proves the double agrees with itself.
    """
    from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import LocalExperimentalistBackend

    ctx = make_context(root=tmp_path, backend=LocalExperimentalistBackend(client=fake_client(), path=tmp_path))
    baseline = await _import_baseline(ctx)
    await ctx.discard_candidate(baseline)

    assert await ctx.candidates() == []
    assert [c.id for c in await ctx.candidates(include_discarded=True)] == [baseline.id]
    # The record is still on disk, marked — not removed.
    stored = json.loads((tmp_path / "eval-and-optimize" / "candidates" / f"{baseline.id}.json").read_text())
    assert stored["discarded"] is True


def test_a_candidate_keeps_its_identity_through_serialization() -> None:
    """Selection crosses a JSON boundary, and identity has to survive it.

    Survivors are the return value of an LLM method, so they arrive validated from the
    model's JSON rather than as the objects that went in. `id` is a computed field backed
    by a private attribute, so without restoring it a survivor comes back with an empty
    id while its label looks fine — and `survived = {s.id for s in survivors}` becomes
    `{""}`, which marks every candidate in the round killed.
    """
    candidate = make_candidate(label="agent-1", ancestor="id-agent-0", candidate_id="a-real-uuid")

    assert Candidate.model_validate(candidate.model_dump()).id == "a-real-uuid"
    assert Candidate.model_validate_json(candidate.model_dump_json()).id == "a-real-uuid"
    assert candidate.slim().id == "a-real-uuid"


@pytest.mark.asyncio
async def test_a_slim_copy_cannot_be_persisted(tmp_path: Path) -> None:
    """`slim()` empties trials so a candidate can go into an LLM prompt. Persisting one
    writes that loss back for every channel at once, and trajectory scoring — which reads
    validation traces — then reports 0.0 for a candidate that has them. Refused instead.
    """
    ctx = make_context(root=tmp_path, backend=FakeBackend())
    candidate = await _import_baseline(ctx)
    measured = RewardRecord(
        metrics={"reward": 0.5}, trials=[TrialResult(id="t1", task_id="task-a", status="completed")]
    )
    await ctx.record_reward(candidate, channel="validation", result=measured)

    with pytest.raises(ValueError, match="slim"):
        await ctx.record_reward(candidate.slim(), channel="train", result=RewardRecord(metrics={"reward": 1.0}))

    assert (await ctx.candidates())[0].rewards["validation"].trials


@pytest.mark.asyncio
async def test_a_builder_documents_the_candidates_it_builds(tmp_path: Path) -> None:
    """`describe` is a Builder verb so a subclass can replace it. The CodeEditBuilder called
    `create_architecture_doc` directly, which honoured an override for the baseline and
    bypassed it for every candidate built after.
    """
    from nemo_experimentalist_plugin.experimentalist.components.coder import CodeEditBuilder
    from nemo_experimentalist_plugin.experimentalist.components.evaluator.base import Evaluator
    from nemo_experimentalist_plugin.experimentalist.components.proposer import CodeChange

    described: list[Path] = []

    class QuietEditor(CodeEditBuilder):
        """Create and modify agent source code as part of the optimization loop."""

        async def describe(self, artifact: Path) -> None:
            described.append(artifact)

        async def apply_change(self, workdir: Path, optimization: str, change: CodeChange) -> None:
            return None

        async def wire_up_change(self, workdir: Path, optimization: str, change: CodeChange) -> None:
            return None

        async def run_pyright(self, workdir: Path) -> str:
            return ""

        async def optimize_subproblem(
            self,
            workdir: Path,
            optimization: str,
            change: CodeChange,
            dataset: Dataset,
            evaluator: Evaluator,
        ) -> None:
            return None

        async def integration_check(
            self,
            workdir: Path,
            dataset: Dataset,
            evaluator: Evaluator,
            max_fix_attempts: int | None = None,
            task_ids: Sequence[str] | None = None,
        ) -> bool:
            return True

    ctx = make_context(root=tmp_path, backend=FakeBackend())
    baseline = await _import_baseline(ctx)
    proposal = Proposal(
        ancestor=baseline.id,
        description="a change",
        kind="code-change",
        payload={"root_cause": "none", "optimization_type": "edit_config", "task_ids": []},
    )

    builder = QuietEditor(workspace=tmp_path, evaluator=ctx.outcome_evaluator, dataset=ctx.datasets["train"])
    candidate = await builder.build(ctx, proposal, generation=1)

    assert described == [ctx.candidate_dir(candidate)]


@pytest.mark.asyncio
async def test_the_smoke_check_evaluates_the_tasks_the_change_claims_to_fix(tmp_path: Path) -> None:
    """Otherwise a build that repairs nothing passes on the first attempt, and the repair
    loop the check exists to drive never runs.

    Observed: a candidate adding an aggregation handler was smoke-checked against a
    lookup control it does not touch, passed, and shipped with both targeted tasks at 0.
    """
    from nemo_experimentalist_plugin.experimentalist.components.coder import CodeEditBuilder

    seen: list[list[str]] = []

    class RecordingEditor(CodeEditBuilder):
        """Create and modify agent source code as part of the optimization loop."""

        async def run_smoke_eval(self, workdir, tasks, dataset, evaluator):  # type: ignore[no-untyped-def]
            seen.append(sorted(t.id for t in tasks))
            return EvaluationResult(id="smoke", trials=[])

        def _is_smoke_results_healthy(self, evaluation, tasks) -> bool:  # type: ignore[no-untyped-def]
            return True

    dataset = Dataset(id="train", tasks=[Task(id="targeted-a"), Task(id="targeted-b"), Task(id="control")])
    ctx = make_context(root=tmp_path, backend=FakeBackend(), datasets={"train": dataset, "validation": dataset})
    coder = RecordingEditor(workspace=tmp_path, evaluator=ctx.outcome_evaluator, dataset=dataset)
    wanted = ["targeted-a", "targeted-b"]

    assert await coder.integration_check(tmp_path, dataset, ctx.outcome_evaluator, task_ids=wanted)

    assert seen == [sorted(wanted)], "the smoke check ignored the tasks the change targeted"


@pytest.mark.asyncio
async def test_the_smoke_check_still_runs_when_a_proposal_names_no_tasks(tmp_path: Path) -> None:
    """A Proposal with no task_ids keeps the old "does it run at all" check."""
    from nemo_experimentalist_plugin.experimentalist.components.coder import CodeEditBuilder

    seen: list[list[str]] = []

    class RecordingEditor(CodeEditBuilder):
        """Create and modify agent source code as part of the optimization loop."""

        async def run_smoke_eval(self, workdir, tasks, dataset, evaluator):  # type: ignore[no-untyped-def]
            seen.append(sorted(t.id for t in tasks))
            return EvaluationResult(id="smoke", trials=[])

        def _is_smoke_results_healthy(self, evaluation, tasks) -> bool:  # type: ignore[no-untyped-def]
            return True

    dataset = Dataset(id="train", tasks=[Task(id="a"), Task(id="b"), Task(id="c")])
    ctx = make_context(root=tmp_path, backend=FakeBackend(), datasets={"train": dataset, "validation": dataset})
    coder = RecordingEditor(workspace=tmp_path, evaluator=ctx.outcome_evaluator, dataset=dataset)

    assert await coder.integration_check(tmp_path, dataset, ctx.outcome_evaluator, task_ids=None)

    assert len(seen) == 1 and len(seen[0]) == 1, "expected the single-random-task fallback"
