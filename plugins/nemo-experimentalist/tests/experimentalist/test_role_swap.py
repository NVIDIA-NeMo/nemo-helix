# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M1's acceptance bar: every seam is swappable from config, with no code change.

The milestone claims the core is done. The test of that claim is that a new
optimization paradigm — HPO next — needs no edit to the strategy or to any existing
component. If a role cannot be named in config and resolved, that claim is false, and
this file says so rather than leaving it to be discovered when HPO is written.

Resolution alone is not the bar: a component that resolves and then fails on its
constructor is not swappable, so construction through the context is asserted too.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from nemo_experimentalist_plugin.config import EvolutionaryOptimizerConfig
from nemo_experimentalist_plugin.entities import Candidate, EvaluationResult, Proposal, RewardRecord
from nemo_experimentalist_plugin.experimentalist.registry import (
    Component,
    get_component,
    load_plugins,
    registered,
    resolve,
)
from nemo_experimentalist_plugin.experimentalist.roles import (
    Analyzer,
    Builder,
    OutcomeEvaluator,
    Proposer,
    Selector,
    Strategy,
    Terminator,
    TrajectoryScorer,
)

#: Every role the evolutionary strategy delegates to, and the config key naming it.
ROLES = {
    "strategy": "strategy",
    "builder": "builder",
    "outcome-evaluator": "outcome_evaluator",
    "proposer": "proposer",
    "selector": "selector",
    "terminator": "terminator",
    "root-cause-analyzer": "analyzer",
    "trajectory-scorer": "trajectory_scorer",
}


async def _no_evaluations(*_args: Any, **_kwargs: Any) -> dict[str, EvaluationResult]:
    """Stand in for an evaluation step that needs a model."""
    return {}


async def _no_analysis(*_args: Any, **_kwargs: Any) -> str:
    """Stand in for the round analysis, which needs a model."""
    return ""


async def _nothing(*_args: Any, **_kwargs: Any) -> None:
    """Stand in for a per-round step whose result nothing here reads."""
    return None


async def _no_proposals(*_args: Any, **_kwargs: Any) -> list[Proposal]:
    """One proposal that builds nothing.

    An *empty* list ends the loop -- there is nothing to build, so another round would
    re-analyze the same population and pay for it again. These tests want a round that
    runs and produces no candidate, which is a different thing, so they pair this with
    `_builds_nothing`.
    """
    return [Proposal(ancestor=None, description="proposed", kind="code-change", payload={})]


async def _builds_nothing(*_args: Any, **_kwargs: Any) -> list[Any]:
    """Accept the proposal and produce no candidate."""
    return []


async def _one_baseline(*, ctx: Any, config: Any) -> None:
    proposal = Proposal(ancestor=None, description="baseline", kind="import", payload={})
    await ctx.component("builder", "import").build(ctx, proposal, generation=0)


@pytest.fixture
def isolated_registry() -> Iterator[None]:
    """Register stand-ins without leaking them into other tests.

    Two hazards, both of which have bitten:

    Restore the mapping's *contents*, never the object. Entry-point discovery writes into
    whichever dict is bound at the time and runs once per process, so replacing the object
    sends the real components into a copy that is then discarded.

    And discover *before* snapshotting. Taking the snapshot first captures an empty
    registry, and the teardown then restores that emptiness over the real components.
    """
    load_plugins()
    before = dict(Component._registry)
    yield
    Component._registry.clear()
    Component._registry.update(before)


def test_the_defaults_name_a_component_that_exists(isolated_registry: None) -> None:
    """A default that does not resolve is a run that dies after the user waits for it."""
    config = EvolutionaryOptimizerConfig()
    for role, key in ROLES.items():
        name = getattr(config, key)
        assert name is not None, f"{key} has no default"
        assert resolve(role, name) is not None, f"default {key}={name!r} does not resolve"


def test_every_role_can_be_swapped_for_one_this_repo_does_not_know(isolated_registry: None) -> None:
    """The whole milestone in one assertion.

    Each stand-in registers the way a `pip install`ed package's would, and config names
    it. Nothing in the strategy or in any existing component changes.
    """

    class Swapped:
        """Marks a class as the replacement, so resolution can be told from a default."""

    class SwappedStrategy(Swapped, Strategy):
        name = "acme-bandit"
        supports_resume: ClassVar[bool] = False

    class SwappedBuilder(Swapped, Builder):
        name = "acme-overlay"
        accepts: ClassVar[frozenset[str]] = frozenset({"parameters"})

    class SwappedEvaluation(Swapped, OutcomeEvaluator):
        name = "acme-scorer"

    class SwappedProposer(Swapped, Proposer):
        name = "acme-hpo"
        produces: ClassVar[frozenset[str]] = frozenset({"parameters"})

    class SwappedSelector(Swapped, Selector):
        name = "acme-crowding"

    class SwappedTerminator(Swapped, Terminator):
        name = "acme-budget"

    class SwappedAnalyzer(Swapped, Analyzer):
        name = "acme-blind"

    class SwappedScorer(Swapped, TrajectoryScorer):
        name = "acme-steps"

    config = EvolutionaryOptimizerConfig(
        strategy="acme-bandit",
        builder="acme-overlay",
        outcome_evaluator="acme-scorer",
        proposer="acme-hpo",
        selector="acme-crowding",
        terminator="acme-budget",
        analyzer="acme-blind",
        trajectory_scorer="acme-steps",
    )

    for role, key in ROLES.items():
        resolved = resolve(role, getattr(config, key))
        assert issubclass(resolved, Swapped), f"{role} resolved to {resolved.__name__}, not the configured one"


@pytest.mark.asyncio
async def test_turning_off_the_analyzer_skips_the_work_that_feeds_it(
    tmp_path, isolated_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`analyzer: null` must skip the train evaluation, which is the expensive half.

    Driven through the loop: the config echoing back what it was handed says nothing
    about whether the loop reads it.
    """
    from doubles import FakeBackend, make_context
    from nemo_experimentalist_plugin.experimentalist.strategies.evolutionary import EvolutionaryStrategy

    trained: list[int] = []

    async def train_eval(**kwargs: Any) -> dict[str, Any]:
        trained.append(kwargs["round_num"])
        return {}

    config = EvolutionaryOptimizerConfig(max_rounds=1, analyzer=None, terminator=None, trajectory_scorer=None)
    loop = EvolutionaryStrategy(working_dir=tmp_path, config=config)
    monkeypatch.setattr(loop, "_ensure_baseline", _one_baseline)
    monkeypatch.setattr(loop, "_evaluate_train_candidates", train_eval)
    monkeypatch.setattr(loop, "_evaluate_validation_candidates", _no_evaluations)
    monkeypatch.setattr(loop, "_record_baseline_validation", _nothing)
    monkeypatch.setattr(loop, "_analyze_round", _no_analysis)
    monkeypatch.setattr(loop, "_propose_improvements", _no_proposals)
    monkeypatch.setattr(loop, "_build_candidates", _builds_nothing)

    await loop._run(make_context(root=tmp_path, backend=FakeBackend()))

    assert trained == [], "the train evaluation ran even though nothing consumes it"


def test_resolution_constructs_with_what_the_caller_passes(isolated_registry: None) -> None:
    """A replacement is built by the strategy that resolves it, on that strategy's terms."""
    seen: dict[str, Any] = {}

    class RecordingBuilder(Builder):
        name = "acme-recording"
        accepts: ClassVar[frozenset[str]] = frozenset({"code-change"})

        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

        async def build(self, ctx: Any, proposal: Proposal, *, generation: int) -> Candidate:
            raise NotImplementedError

    get_component("builder", "acme-recording", workspace="/tmp", config=None, models=None)

    assert seen == {"workspace": "/tmp", "config": None, "models": None}


def test_the_installed_defaults_are_discoverable_without_importing_them() -> None:
    """What `nemo agents experimentalist components` reads, and what a plugin author
    checks their package against."""
    for role in ROLES:
        assert registered(role), f"no component registered for {role}"


def test_an_installed_out_of_tree_package_is_discovered() -> None:
    """The developer journey of §1, checked rather than described.

    `examples/acme-strategies` is a separate package with its own pyproject and one entry
    point. When it is installed, its strategy resolves by name here with no change to this
    repository — which is the only evidence that both discovery levels work for someone
    who is not us.

    This one does not skip when the package is missing, unlike the two below. It is the
    canary for the wiring itself: the package is a locked dev dependency, so absent means
    someone unwired it, and three tests quietly skipping is how a broken one survived
    unnoticed before.
    """
    import acme_strategies  # noqa: F401 - importing is the assertion

    resolved = resolve("strategy", "random-search")

    assert resolved.__module__.startswith("acme_strategies"), "resolved to something in this repo"
    assert issubclass(resolved, Strategy)
    assert "random-search" in registered("strategy")


@pytest.mark.asyncio
async def test_an_out_of_tree_strategy_runs_and_produces_a_winner(tmp_path: Path, isolated_registry: None) -> None:
    """Discovery is half of it. This drives the installed package end to end — it imports
    a baseline, builds through `ctx.component`, scores, and returns a winner — because
    resolving without running is what let a Builder that raised on construction pass.
    """
    pytest.importorskip("acme_strategies", reason="out-of-tree example package is not installed")
    from doubles import FakeBackend, make_context

    ctx = make_context(root=tmp_path, backend=FakeBackend())

    class NoopBuilder(Builder):
        name = "acme-noop-build"
        accepts: ClassVar[frozenset[str]] = frozenset({"code-change"})

        def __init__(self, **kwargs: Any) -> None:
            pass

        async def build(self, ctx: Any, proposal: Proposal, *, generation: int = 0) -> Candidate:
            fork = await ctx.fork(proposal)
            return await ctx.commit_candidate(proposal=proposal, artifact=fork.workdir, generation=generation)

    config = EvolutionaryOptimizerConfig(max_rounds=1, max_candidates=1, builder="acme-noop-build")
    # resolve() is typed to the role protocol, which is the point: the host knows a
    # Strategy, not this package's class. The role declares no __init__, so the
    # constructor's arguments are the resolving strategy's contract, not the role's --
    # hence the cast to build it.
    strategy_class = cast(Any, resolve("strategy", "random-search"))
    strategy = cast("Strategy", strategy_class(working_dir=tmp_path, config=config))

    winner = await strategy.run(ctx)

    assert winner is not None
    assert len(await ctx.candidates()) == 2, "a baseline and one variant"


@pytest.mark.asyncio
async def test_the_round_budget_bounds_the_loop_without_a_terminator(
    tmp_path, isolated_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`max_rounds` must hold even when no terminator is selected.

    A component's opinion must not be the only thing between a config and an unbounded
    run, so the loop is driven here with `terminator: null` and every per-round step
    stubbed: it has to stop on the budget alone, and the count has to be exact.
    """
    from doubles import FakeBackend, make_context
    from nemo_experimentalist_plugin.experimentalist.strategies.evolutionary import EvolutionaryStrategy

    config = EvolutionaryOptimizerConfig(max_rounds=3, terminator=None, analyzer=None, trajectory_scorer=None)
    loop = EvolutionaryStrategy(working_dir=tmp_path, config=config)
    ctx = make_context(root=tmp_path, backend=FakeBackend())
    rounds = 0

    async def one_round(*_args: Any, **kwargs: Any) -> list[Proposal]:
        nonlocal rounds
        rounds += 1
        if rounds > config.max_rounds:  # the bound is broken; stop rather than spin
            raise AssertionError(f"loop ran round {rounds} with max_rounds={config.max_rounds}")
        # Proposing nothing would end the loop on its own, which would pass this test
        # without the budget doing any work.
        return [Proposal(ancestor=None, description="proposed", kind="code-change", payload={})]

    monkeypatch.setattr(loop, "_ensure_baseline", _one_baseline)
    monkeypatch.setattr(loop, "_propose_improvements", one_round)
    monkeypatch.setattr(loop, "_build_candidates", _builds_nothing)
    monkeypatch.setattr(loop, "_evaluate_validation_candidates", _no_evaluations)
    monkeypatch.setattr(loop, "_record_baseline_validation", _nothing)
    monkeypatch.setattr(loop, "_analyze_round", _no_analysis)

    await loop._run(ctx)

    assert rounds == config.max_rounds


@pytest.mark.asyncio
async def test_every_default_constructs_through_the_context(tmp_path, isolated_registry: None) -> None:
    """Resolution is not the bar; construction is."""
    from doubles import FakeBackend, make_context

    ctx = make_context(root=tmp_path, backend=FakeBackend())

    for role, key in ROLES.items():
        if role == "strategy":
            continue  # built by the runner, not by a strategy through the context
        name = getattr(EvolutionaryOptimizerConfig(), key)
        built = ctx.component(role, name)
        assert built is not None, f"{role}={name!r} resolved but could not be constructed"


def test_the_context_supplies_every_run_scoped_argument(tmp_path, isolated_registry: None) -> None:
    """A component cannot know these for itself, so the context is the only thing that
    can supply them — and a default that quietly falls back hides their absence."""
    from doubles import FakeBackend, make_context

    seen: dict[str, Any] = {}

    class Greedy(Builder):
        name = "acme-greedy"

        def __init__(self, workspace, working_dir, evaluator, dataset, **kwargs: Any) -> None:
            seen.update(workspace=workspace, working_dir=working_dir, evaluator=evaluator, dataset=dataset)

        async def build(self, ctx: Any, proposal: Proposal, *, generation: int) -> Candidate:
            raise NotImplementedError

    ctx = make_context(root=tmp_path, backend=FakeBackend())

    ctx.component("builder", "acme-greedy")

    assert seen == {
        "workspace": ctx.root,
        "working_dir": ctx.root,
        "evaluator": ctx.outcome_evaluator,
        "dataset": ctx.datasets["train"],
    }


def test_the_built_in_coder_gets_what_it_needs_to_verify_a_build(tmp_path, isolated_registry: None) -> None:
    """The out-of-tree example builds through `ctx.component`, and a CodeEditBuilder without an
    evaluator or a dataset raises on its first build rather than at construction."""
    from doubles import FakeBackend, make_context

    ctx = make_context(root=tmp_path, backend=FakeBackend())

    coder = ctx.component("builder", "code-edit")

    assert coder._evaluator is ctx.outcome_evaluator
    # Train, never validation: a Builder repairs against what it is handed, and the
    # winner is chosen on validation.
    assert coder._dataset is ctx.datasets["train"]


@pytest.mark.asyncio
async def test_a_swapped_builder_runs_through_the_context(tmp_path, isolated_registry: None) -> None:
    """The narrowest end-to-end check of the seam: config names it, the context builds
    it, and it produces a Candidate through the same verbs the built-in CodeEditBuilder uses."""
    from doubles import FakeBackend, make_context

    class StubBuilder(Builder):
        name = "acme-noop"
        accepts: ClassVar[frozenset[str]] = frozenset({"code-change"})

        def __init__(self, **kwargs: Any) -> None:
            pass

        async def build(self, ctx: Any, proposal: Proposal, *, generation: int = 0) -> Candidate:
            fork = await ctx.fork(proposal)
            (fork.workdir / "changed.py").write_text("# built by an out-of-tree builder\n")
            return await ctx.commit_candidate(proposal=proposal, artifact=fork.workdir, generation=generation)

    ctx = make_context(root=tmp_path, backend=FakeBackend())
    baseline = await ctx.component("builder", "import").build(
        ctx, Proposal(ancestor=None, description="baseline", kind="import", payload={}), generation=0
    )
    proposal = Proposal(ancestor=baseline.id, description="a change", kind="code-change", payload={})

    candidate = await ctx.component("builder", "acme-noop").build(ctx, proposal, generation=1)

    assert candidate.ancestor == baseline.id
    assert (ctx.candidate_dir(candidate) / "changed.py").exists()


def test_registering_a_component_in_a_test_does_not_leak(isolated_registry: None) -> None:
    """The registry is global mutable state, and every leak from it has been invisible to
    the full suite — visible only in isolation or in reverse order."""
    before = set(Component._registry)

    class Ephemeral(Terminator):
        name = "acme-ephemeral"

    assert ("terminator", "acme-ephemeral") in Component._registry
    # The fixture's teardown is what must remove it; this records the contract.
    assert set(Component._registry) - before == {("terminator", "acme-ephemeral")}


@pytest.mark.asyncio
async def test_a_proposer_and_builder_that_cannot_work_together_fail_before_the_run(
    tmp_path, isolated_registry: None
) -> None:
    """A Proposer emitting only kinds the Builder rejects produces empty rounds that look
    like a run doing work. Hours later it finishes with the baseline as winner."""
    from doubles import FakeBackend, make_context
    from nemo_experimentalist_plugin.experimentalist.strategies.evolutionary import EvolutionaryStrategy

    class ParameterProposer(Proposer):
        name = "acme-hpo-only"
        produces: ClassVar[frozenset[str]] = frozenset({"parameters"})

    config = EvolutionaryOptimizerConfig(proposer="acme-hpo-only", builder="code-edit")
    loop = EvolutionaryStrategy(working_dir=tmp_path, config=config)

    with pytest.raises(ValueError, match="no proposal could be built"):
        await loop._run(make_context(root=tmp_path, backend=FakeBackend()))


@pytest.mark.asyncio
async def test_a_proposal_no_builder_accepts_is_dropped_not_raised(tmp_path, isolated_registry: None) -> None:
    """One unbuildable proposal must not end a run that has already spent hours."""
    from doubles import FakeBackend, make_context
    from nemo_experimentalist_plugin.experimentalist.strategies.evolutionary import EvolutionaryStrategy

    config = EvolutionaryOptimizerConfig(builder="code-edit")
    loop = EvolutionaryStrategy(working_dir=tmp_path, config=config)
    ctx = make_context(root=tmp_path, backend=FakeBackend())
    unbuildable = Proposal(ancestor=None, description="tune a knob", kind="parameters", payload={})

    built = await loop._build_candidates(
        ctx=ctx, dataset=ctx.datasets["train"], proposals=[unbuildable], generation=1, config=config
    )

    assert built == []


@pytest.mark.asyncio
async def test_naming_a_trajectory_scorer_reaches_it(
    tmp_path, isolated_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate that skips the scoring step is indistinguishable from `trajectory_scorer:
    null`, so the run silently loses a reward channel it was configured to measure."""
    from doubles import FakeBackend, make_context
    from nemo_experimentalist_plugin.experimentalist.strategies.evolutionary import EvolutionaryStrategy

    reached: list[str] = []

    class StepCountScorer(TrajectoryScorer):
        name = "acme-steps"

        def __init__(self, **kwargs: Any) -> None:
            pass

        async def run(self, ctx: Any, *, candidates: list[Candidate], **kwargs: Any) -> dict[str, RewardRecord]:
            reached.append("acme-steps")
            # A real record, not an empty dict: returning {} never exercises the key type,
            # which is how `dict[Candidate, ...]` shipped despite Candidate being unhashable.
            return {c.id: RewardRecord(metrics={"aggregate": 0.5}) for c in candidates}

    config = EvolutionaryOptimizerConfig(max_rounds=1, analyzer=None, terminator=None, trajectory_scorer="acme-steps")
    loop = EvolutionaryStrategy(working_dir=tmp_path, config=config)

    monkeypatch.setattr(loop, "_ensure_baseline", _one_baseline)
    monkeypatch.setattr(loop, "_evaluate_validation_candidates", _no_evaluations)
    monkeypatch.setattr(loop, "_record_baseline_validation", _nothing)
    monkeypatch.setattr(loop, "_analyze_round", _no_analysis)
    monkeypatch.setattr(loop, "_propose_improvements", _no_proposals)
    monkeypatch.setattr(loop, "_build_candidates", _builds_nothing)

    ctx = make_context(root=tmp_path, backend=FakeBackend())
    await loop._run(ctx)

    assert reached == ["acme-steps"]
    scored = [c for c in await ctx.candidates() if c.rewards["validation-trajectory"].metrics]
    assert scored, "the scorer's records never reached the candidates"


@pytest.mark.asyncio
async def test_the_architecture_doc_comes_from_the_configured_builder(tmp_path, isolated_registry: None) -> None:
    """Not from the CodeEditBuilder: a builder that writes no architecture doc must not have one
    written for it by a component the config never named."""
    from doubles import FakeBackend, make_context
    from nemo_experimentalist_plugin.experimentalist.strategies.evolutionary import EvolutionaryStrategy

    described: list[Path] = []

    class DescribingBuilder(Builder):
        name = "acme-describes"
        accepts: ClassVar[frozenset[str]] = frozenset({"code-change"})

        def __init__(self, **kwargs: Any) -> None:
            pass

        async def build(self, ctx: Any, proposal: Proposal, *, generation: int) -> Candidate:
            raise NotImplementedError

        async def describe(self, artifact: Path) -> None:
            described.append(artifact)

    config = EvolutionaryOptimizerConfig(builder="acme-describes")
    loop = EvolutionaryStrategy(working_dir=tmp_path, config=config)
    ctx = make_context(root=tmp_path, backend=FakeBackend())

    await loop._generate_architecture_doc(ctx=ctx, agent_dir=tmp_path / "candidate", config=config)

    assert described == [tmp_path / "candidate"]


@pytest.mark.asyncio
async def test_the_loop_carries_full_candidates_forward_not_the_selectors_slim_copies(
    tmp_path, isolated_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selector is handed `slim()` copies so trials never reach a prompt. What it
    returns are those same copies, and persisting one empties every channel's trials.
    """
    from doubles import FakeBackend, make_context
    from nemo_experimentalist_plugin.experimentalist.strategies.evolutionary import EvolutionaryStrategy

    class KeepEverything(Selector):
        name = "acme-keep-all"

        def __init__(self, **kwargs: Any) -> None:
            pass

        async def survivors(self, candidates: list[Candidate], *, k: int) -> list[Candidate]:
            return list(candidates)

        def winner(self, candidates: list[Candidate]) -> Candidate | None:
            return candidates[0] if candidates else None

    config = EvolutionaryOptimizerConfig(
        max_rounds=1, selector="acme-keep-all", terminator=None, analyzer=None, trajectory_scorer=None
    )
    loop = EvolutionaryStrategy(working_dir=tmp_path, config=config)
    ctx = make_context(root=tmp_path, backend=FakeBackend())

    async def two_candidates(*, ctx: Any, config: Any) -> None:
        for description in ("baseline", "a sibling"):
            proposal = Proposal(ancestor=None, description=description, kind="import", payload={})
            await ctx.component("builder", "import").build(ctx, proposal, generation=0)

    async def scored(*, ctx: Any, candidates: list[Candidate]) -> dict[str, Any]:
        from nemo_experimentalist_plugin.entities import RewardRecord

        return {c.label: RewardRecord(metrics={"reward": 0.5}) for c in candidates}

    monkeypatch.setattr(loop, "_ensure_baseline", two_candidates)
    monkeypatch.setattr(loop, "_evaluate_validation_candidates", scored)
    monkeypatch.setattr(loop, "_record_baseline_validation", _nothing)
    monkeypatch.setattr(loop, "_analyze_round", _no_analysis)
    monkeypatch.setattr(loop, "_propose_improvements", _no_proposals)
    monkeypatch.setattr(loop, "_build_candidates", _builds_nothing)

    await loop._run(ctx)

    assert all(c.rewards["validation"].metrics for c in await ctx.candidates())


@pytest.mark.asyncio
async def test_the_analyzer_is_built_with_the_runs_trace_loader(tmp_path, isolated_registry: None) -> None:
    """Without them the analyzer cannot load `intake://` traces, and a trace-starved
    diagnosis reads as a finding about the agent rather than about the analyzer.

    Observed: it reported the agent's root cause as "the analyzer lacks the Intake
    client/workspace configuration", and the round built a candidate against that.
    """
    from doubles import FakeBackend, make_context
    from nemo_experimentalist_plugin.experimentalist.strategies.evolutionary import EvolutionaryStrategy

    seen: dict[str, Any] = {}

    class RecordingAnalyzer(Analyzer):
        name = "acme-recording-analyzer"

        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

        async def run(self, **kwargs: Any) -> Any:
            raise AssertionError("construction is what this pins")

    config = EvolutionaryOptimizerConfig(analyzer="acme-recording-analyzer")
    loop = EvolutionaryStrategy(working_dir=tmp_path, config=config)
    ctx = make_context(root=tmp_path, backend=FakeBackend())

    baseline = await _one(ctx)
    with pytest.raises(AssertionError):
        await loop._analyze_round(
            analysis_dir=tmp_path,
            dataset=ctx.datasets["train"],
            evaluations={baseline.label: cast(Any, object())},
            survivors=[baseline],
            round_num=0,
            config=config,
            load_trace=ctx.load_trace,
            ethos_path=None,
        )

    assert "load_trace" in seen, "analyzer built without a trace loader; intake traces are unreadable"
    assert seen["load_trace"] == ctx.load_trace


async def _one(ctx: Any) -> Any:
    proposal = Proposal(ancestor=None, description="baseline", kind="import", payload={})
    return await ctx.component("builder", "import").build(ctx, proposal, generation=0)


@pytest.mark.asyncio
async def test_the_builtin_scorer_returns_records_keyed_by_candidate_id(
    tmp_path, isolated_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Candidate` is a pydantic model and unhashable, so a dict keyed by one raises at
    the moment of return — after the round has already paid for the scoring."""
    from doubles import FakeBackend, make_context
    from nemo_experimentalist_plugin.experimentalist.components.trace_scorer import GoalTreeTrajectoryScorer

    ctx = make_context(root=tmp_path, backend=FakeBackend())
    baseline = await _one(ctx)
    scorer = GoalTreeTrajectoryScorer(workspace=tmp_path)

    async def scored(**_: Any) -> dict[str, Any]:
        return {baseline.label: {"details": {"n1": {}}, "reward": {"aggregate": 0.5}}}

    monkeypatch.setattr(scorer, "_ensure_goal_tree", _nothing)
    monkeypatch.setattr(scorer, "_update_goal_tree", _nothing)
    monkeypatch.setattr(scorer, "_reward_trajectories", scored)

    result = await scorer.run(ctx, candidates=[baseline], round_num=0, analysis="x")

    assert list(result) == [baseline.id], "records must be keyed by candidate id, not by the Candidate"
    assert baseline.trajectory_detail == {"n1": {}}


@pytest.mark.asyncio
async def test_the_scorer_is_told_the_round_its_analysis_describes(
    tmp_path, isolated_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loop's counter has already advanced when the scorer is called, and the scorer
    names its persisted state after the round — so passing the raw counter skips a number
    (observed: a round-1 goal tree written as `round-2-goal.json`)."""
    from doubles import FakeBackend, make_context
    from nemo_experimentalist_plugin.experimentalist.strategies.evolutionary import EvolutionaryStrategy

    rounds: list[int] = []

    class RecordingScorer(TrajectoryScorer):
        name = "acme-round-recorder"

        def __init__(self, **kwargs: Any) -> None:
            pass

        async def run(self, ctx: Any, *, candidates: list[Candidate], round_num: int = 0, **kw: Any) -> dict[str, Any]:
            rounds.append(round_num)
            return {}

    config = EvolutionaryOptimizerConfig(
        max_rounds=1, analyzer=None, terminator=None, trajectory_scorer="acme-round-recorder"
    )
    loop = EvolutionaryStrategy(working_dir=tmp_path, config=config)
    monkeypatch.setattr(loop, "_ensure_baseline", _one_baseline)
    monkeypatch.setattr(loop, "_evaluate_validation_candidates", _no_evaluations)
    monkeypatch.setattr(loop, "_record_baseline_validation", _nothing)
    monkeypatch.setattr(loop, "_analyze_round", _no_analysis)
    monkeypatch.setattr(loop, "_propose_improvements", _no_proposals)
    monkeypatch.setattr(loop, "_build_candidates", _builds_nothing)

    await loop._run(make_context(root=tmp_path, backend=FakeBackend()))

    assert rounds == [0], f"scorer told round {rounds}, but it scored round 0's analysis"


@pytest.mark.asyncio
async def test_the_terminator_sees_eliminated_candidates_too(
    tmp_path, isolated_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Convergence reads the whole history, not this round's working set.

    It counts distinct generations and compares an older Pareto front against the
    current one. `candidates` inside the loop holds only survivors plus the previous
    round's builds, so passing that hides every eliminated candidate: the generation
    window can never fill, and an otherwise converged run pays for every remaining
    round. Driven through the loop, because the call site is what was wrong.
    """
    from doubles import FakeBackend, make_context
    from nemo_experimentalist_plugin.experimentalist.strategies.evolutionary import EvolutionaryStrategy

    seen: list[list[str]] = []

    from nemo_experimentalist_plugin.experimentalist.components.terminator import TerminationDecision

    class RecordingTerminator(Terminator):
        name = "acme-recording-terminator"

        def __init__(self, **kwargs: Any) -> None: ...

        async def run(self, *, round_num: int, candidates: list[Any], prior_analysis: Any = None) -> Any:
            seen.append([c.label for c in candidates])
            return TerminationDecision(stop=False)

    async def baseline_plus_an_eliminated_one(*, ctx: Any, config: Any) -> None:
        """Two candidates, one already eliminated — the case the working set hides."""
        for description in ("baseline", "killed in an earlier round"):
            proposal = Proposal(ancestor=None, description=description, kind="import", payload={})
            candidate = await ctx.component("builder", "import").build(ctx, proposal, generation=0)
            if description != "baseline":
                await ctx.update_candidate(candidate, killed_generation=0)

    config = EvolutionaryOptimizerConfig(
        max_rounds=1, analyzer=None, trajectory_scorer=None, terminator="acme-recording-terminator"
    )
    loop = EvolutionaryStrategy(working_dir=tmp_path, config=config)
    monkeypatch.setattr(loop, "_ensure_baseline", baseline_plus_an_eliminated_one)
    monkeypatch.setattr(loop, "_evaluate_validation_candidates", _no_evaluations)
    monkeypatch.setattr(loop, "_record_baseline_validation", _nothing)
    monkeypatch.setattr(loop, "_analyze_round", _no_analysis)
    monkeypatch.setattr(loop, "_propose_improvements", _no_proposals)
    monkeypatch.setattr(loop, "_build_candidates", _builds_nothing)

    ctx = make_context(root=tmp_path, backend=FakeBackend())
    await loop._run(ctx)

    committed = {c.label for c in await ctx.candidates(include_discarded=True)}
    assert seen, "the terminator was never consulted"
    assert set(seen[0]) == committed, (
        f"terminator saw {seen[0]}, but the run has {sorted(committed)} — "
        "eliminated candidates must still count toward convergence"
    )


def test_no_component_constructor_names_a_platform_type(isolated_registry: None) -> None:
    """A component must be constructible by anyone holding a context, and nothing else.

    Trace readers used to take an async platform client plus a workspace name, which
    put a platform type in the public signature of a role a third party is meant to
    implement: to write one you had to obtain a platform client, and the host had to hand
    its client to a component. They take `load_trace` from the context instead.

    The check is over the registry rather than a list of files, so a component added later
    is covered without anyone remembering to add it here.
    """
    import inspect

    from nemo_experimentalist_plugin.experimentalist.registry import Component, load_plugins

    load_plugins()
    assert Component._registry, "nothing registered; the check would pass vacuously"

    offenders = {
        f"{role}:{name}": [
            parameter
            for parameter, value in inspect.signature(cls.__init__).parameters.items()
            if "NemoClient" in str(value.annotation) or "NeMoPlatform" in str(value.annotation)
        ]
        for (role, name), cls in Component._registry.items()
    }
    named = {key: value for key, value in offenders.items() if value}

    assert not named, f"components naming a platform type in their constructor: {named}"


async def test_the_context_loads_a_trace_by_reference(tmp_path, isolated_registry: None) -> None:
    """The verb that replaced the client: a `file://` ref resolves without a platform."""
    import json

    from doubles import FakeBackend, make_context
    from nemo_experimentalist_plugin.entities import ResourceRef

    span = {"spanId": "abc", "traceId": "def", "name": "llm", "attributes": {"openinference.span.kind": "LLM"}}
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]}) + "\n", encoding="utf-8")
    ctx = make_context(root=tmp_path, backend=FakeBackend())

    explorer = await ctx.load_trace(
        ResourceRef(uri=f"file://{trace}", description="", metadata={"trace_format": "otlp"})
    )

    assert explorer is not None


@pytest.mark.asyncio
async def test_the_example_strategy_resumes_where_the_store_left_off(
    tmp_path: Path, isolated_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The out-of-tree example declares `supports_resume`, so it must honour it.

    Restarting at generation 1 would rebuild finished work and report progress that goes
    backwards. It is the package a third party copies, so the resume contract has to be
    demonstrated rather than merely declared.
    """
    pytest.importorskip("acme_strategies", reason="out-of-tree example package is not installed")
    import importlib

    from doubles import FakeBackend, make_context
    from nemo_experimentalist_plugin.config import EvolutionaryOptimizerConfig

    # Imported by name: the package is optional here, so a static import would make the
    # type checker resolve something that is not installed.
    strategy_class = cast(Any, importlib.import_module("acme_strategies.random_search").RandomSearch)

    ctx = make_context(root=tmp_path, backend=FakeBackend())
    baseline = await _one(ctx)
    assert baseline.generation == 0

    # A run whose configured rounds are already in the store returns its winner and
    # builds nothing more.
    built: list[str] = []
    original = ctx.component

    def _recording_component(role: str, name: str, **kwargs: Any) -> Any:
        component = original(role, name, **kwargs)
        if role == "builder":
            built.append(name)
        return component

    monkeypatch.setattr(ctx, "component", _recording_component)
    strategy = strategy_class(working_dir=tmp_path, config=EvolutionaryOptimizerConfig(max_rounds=0))

    winner = await strategy.run(ctx)

    assert winner is not None, "a resumed run with nothing left to do still has a winner"
    assert built == [], f"a completed run rebuilt candidates: {built}"
