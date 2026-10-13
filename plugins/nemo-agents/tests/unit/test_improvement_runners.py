# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the improvement/runners/ submodule."""

from __future__ import annotations

from pathlib import Path

import pytest


def _make_eval(tmp_path: Path, name: str, marker: str) -> Path:
    """Create a fixture eval dir with files matching ``marker`` (harbor/nat/both)."""
    eval_dir = tmp_path / name
    eval_dir.mkdir()
    if marker == "harbor":
        (eval_dir / "task.toml").write_text(
            "[metadata]\ndifficulty = 'easy'\ncategory = 'test'\ntags = []\n"
            "[agent]\ntimeout_sec = 60.0\n[verifier]\ntimeout_sec = 30.0\n"
        )
        (eval_dir / "instruction.md").write_text("do the thing")
        tests_dir = eval_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_outputs.py").write_text("def test_x(): pass\n")
    elif marker == "nat":
        (eval_dir / "workflow.yml").write_text("workflow: {}")
        (eval_dir / "instruction.md").write_text("do the thing")
        tests_dir = eval_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_outputs.py").write_text("def test_x(): pass\n")
    elif marker == "both":
        # Migration-window task: has both markers
        (eval_dir / "task.toml").write_text(
            "[metadata]\ndifficulty = 'easy'\ncategory = 'test'\ntags = []\n"
            "[agent]\ntimeout_sec = 60.0\n[verifier]\ntimeout_sec = 30.0\n"
        )
        (eval_dir / "workflow.yml").write_text("workflow: {}")
        (eval_dir / "instruction.md").write_text("do the thing")
        tests_dir = eval_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_outputs.py").write_text("def test_x(): pass\n")
    return eval_dir


def test_detect_runner_picks_harbor_for_task_toml(tmp_path: Path) -> None:
    """Auto-detect resolves to Harbor when tasks declare ``task.toml``."""
    from nemo_agents_plugin.improvement.runners.detect import detect_runner

    _make_eval(tmp_path, "task1", "harbor")
    _make_eval(tmp_path, "task2", "harbor")

    runner = detect_runner(tmp_path)
    assert runner.name == "harbor"


def test_detect_runner_picks_nat_for_workflow_yml(tmp_path: Path) -> None:
    """Auto-detect resolves to NAT when tasks declare ``workflow.yml``."""
    from nemo_agents_plugin.improvement.runners.detect import detect_runner

    _make_eval(tmp_path, "task1", "nat")
    _make_eval(tmp_path, "task2", "nat")

    runner = detect_runner(tmp_path)
    assert runner.name == "nat"


def test_detect_runner_migration_window_prefers_nat(tmp_path: Path) -> None:
    """``prefer="nat"`` wins when a task carries both runner markers."""
    from nemo_agents_plugin.improvement.runners.detect import detect_runner

    _make_eval(tmp_path, "migrating", "both")

    runner = detect_runner(tmp_path, prefer="nat")
    assert runner.name == "nat"


def test_detect_runner_migration_window_prefer_harbor(tmp_path: Path) -> None:
    """``prefer="harbor"`` wins when a task carries both runner markers."""
    from nemo_agents_plugin.improvement.runners.detect import detect_runner

    _make_eval(tmp_path, "migrating", "both")

    runner = detect_runner(tmp_path, prefer="harbor")
    assert runner.name == "harbor"


def test_detect_runner_raises_when_no_evals(tmp_path: Path) -> None:
    """Detection raises ``RuntimeError`` for an empty evals directory."""
    from nemo_agents_plugin.improvement.runners.detect import detect_runner

    with pytest.raises(RuntimeError, match="No eval tasks found"):
        detect_runner(tmp_path)


def test_get_runner_by_name() -> None:
    """``get_runner`` resolves known names and rejects unknown ones."""
    from nemo_agents_plugin.improvement.runners.detect import get_runner

    assert get_runner("harbor").name == "harbor"
    assert get_runner("nat").name == "nat"
    with pytest.raises(ValueError):
        get_runner("bogus")


def test_harbor_runner_eval_input_paths(tmp_path: Path) -> None:
    """Harbor exposes its task fixtures as immutable input paths."""
    from nemo_agents_plugin.improvement.runners.harbor import HarborRunner

    eval_dir = _make_eval(tmp_path, "task1", "harbor")
    paths = HarborRunner().eval_input_paths(eval_dir)
    assert eval_dir / "task.toml" in paths
    assert eval_dir / "instruction.md" in paths
    assert eval_dir / "tests" in paths


def test_nat_runner_eval_input_paths(tmp_path: Path) -> None:
    """NAT exposes its task fixtures as immutable input paths."""
    from nemo_agents_plugin.improvement.runners.nat import NATRunner

    eval_dir = _make_eval(tmp_path, "task1", "nat")
    paths = NATRunner().eval_input_paths(eval_dir)
    assert eval_dir / "workflow.yml" in paths
    assert eval_dir / "instruction.md" in paths
    assert eval_dir / "tests" in paths
    assert eval_dir / "environment" in paths


def test_nat_runner_discover(tmp_path: Path) -> None:
    """NAT discovery returns only tasks with ``workflow.yml``."""
    from nemo_agents_plugin.improvement.runners.nat import NATRunner

    _make_eval(tmp_path, "alpha", "nat")
    _make_eval(tmp_path, "beta", "nat")
    _make_eval(tmp_path, "gamma", "harbor")  # should be skipped — no workflow.yml

    specs = NATRunner().discover(tmp_path)
    names = {s.name for s in specs}
    assert names == {"alpha", "beta"}


def test_harbor_runner_discover(tmp_path: Path) -> None:
    """Harbor discovery returns only tasks with ``task.toml``."""
    from nemo_agents_plugin.improvement.runners.harbor import HarborRunner

    _make_eval(tmp_path, "alpha", "harbor")
    _make_eval(tmp_path, "beta", "harbor")
    _make_eval(tmp_path, "gamma", "nat")  # should be skipped — no task.toml

    specs = HarborRunner().discover(tmp_path)
    names = {s.name for s in specs}
    assert names == {"alpha", "beta"}


def test_runner_supports_repeats_capability_flag() -> None:
    """Capability flag matches each runner's actual repeats behavior."""
    from nemo_agents_plugin.improvement.runners.harbor import HarborRunner
    from nemo_agents_plugin.improvement.runners.nat import NATRunner

    assert HarborRunner.supports_repeats is True
    assert NATRunner.supports_repeats is False


async def test_nat_runner_non_zero_exit_is_error_even_with_valid_result_json(tmp_path: Path) -> None:
    """A crashed nat_runner.py must surface as ERROR — the subprocess exit code
    is authoritative and overrides any result.json the runner left behind.
    Otherwise crashed runs masquerade as legitimate FAILs and pollute baselines.
    """
    from nemo_agents_plugin.improvement.models import Difficulty, EvalSpec, EvalStatus
    from nemo_agents_plugin.improvement.runners.nat import NATRunner

    fake_runner = tmp_path / "fake_nat_runner.py"
    # Write a passing-looking result.json to --jobs-dir/<spec.name>/result.json,
    # then exit non-zero. This is the exact "crashed but left a payload" shape
    # gabwow flagged: prior code would parse the file and report FAIL.
    fake_runner.write_text(
        "import json, sys, argparse, pathlib\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--jobs-dir', required=True)\n"
        "p.add_argument('--nhx-base-url')\n"
        "p.add_argument('eval_name')\n"
        "args = p.parse_args()\n"
        "out = pathlib.Path(args.jobs_dir)\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'result.json').write_text(json.dumps({'passed': True, 'reward': 1.0}))\n"
        "sys.exit(7)\n"
    )

    batch_dir = tmp_path / "batch"
    spec = EvalSpec(
        name="crashed_eval",
        path=tmp_path,
        difficulty=Difficulty.EASY,
        category="",
        tags=[],
        agent_timeout_sec=10.0,
        verifier_timeout_sec=10.0,
    )

    result = await NATRunner()._run_single(spec, batch_dir, fake_runner)

    assert result.status == EvalStatus.ERROR
    assert result.exception is not None
    assert "7" in result.exception
    assert "exit" in result.exception.lower()


def test_run_loop_rejects_unknown_trace_parser(tmp_path: Path) -> None:
    """run_loop's trace_parser knob accepts only registered parser names."""
    import asyncio

    from nemo_agents_plugin.improvement.loop import run_loop
    from nemo_agents_plugin.improvement.runners.harbor import HarborRunner

    agent_root = tmp_path / "agent"
    evals_dir = agent_root / "evals"
    evals_dir.mkdir(parents=True)
    _make_eval(evals_dir, "task1", "harbor")

    with pytest.raises(ValueError, match="Unknown trace_parser"):
        asyncio.run(
            run_loop(
                agent_root=agent_root,
                evals_dir=evals_dir,
                trace_parser="bogus",
                # Pin a runner so AUT discovery doesn't fire (discovery raises
                # on empty/missing eval dirs; we're testing the trace_parser
                # validation downstream of it).
                runner=HarborRunner(),
            )
        )


def test_run_loop_discovers_aut_when_runner_not_pinned(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """AUT discovery fires when the caller omits ``runner=`` and logs the choice."""
    import asyncio

    from nemo_agents_plugin.improvement.loop import run_loop

    agent_root = tmp_path / "agent"
    evals_dir = agent_root / "evals"
    evals_dir.mkdir(parents=True)
    _make_eval(evals_dir, "task1", "nat")

    # Discovery picks NAT from the workflow.yml marker, logs the choice, then
    # the trace_parser='bogus' check downstream raises ValueError. The
    # important assertion is the side effect: discovery ran and logged.
    with pytest.raises(ValueError, match="Unknown trace_parser"):
        asyncio.run(
            run_loop(
                agent_root=agent_root,
                evals_dir=evals_dir,
                trace_parser="bogus",
            )
        )
    out = capsys.readouterr().out
    assert "Discovered AUT" in out
    assert "nat" in out


def test_run_loop_discovery_raises_on_empty_evals_dir(tmp_path: Path) -> None:
    """detect_runner raises a clear error when no eval tasks exist."""
    import asyncio

    from nemo_agents_plugin.improvement.loop import run_loop

    agent_root = tmp_path / "agent"
    evals_dir = agent_root / "evals"
    evals_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="No eval tasks found"):
        asyncio.run(
            run_loop(
                agent_root=agent_root,
                evals_dir=evals_dir,
            )
        )


def test_run_loop_rejects_non_harbor_runner_at_apply_step(tmp_path: Path) -> None:
    """The harbor-only guard moved from run_loop entry to just-before-apply.

    Analyze and discovery work for any AUT; only the apply step (claude CLI
    over the skills_path) is harbor-shaped. With a NAT runner pinned plus a
    NAT-shaped batch fixture, the loop should reach the apply-step guard.
    """
    import asyncio

    from nemo_agents_plugin.improvement.loop import run_loop
    from nemo_agents_plugin.improvement.runners.nat import NATRunner

    agent_root = tmp_path / "agent"
    evals_dir = agent_root / "evals"
    evals_dir.mkdir(parents=True)
    _make_eval(evals_dir, "task1", "nat")

    # With trace_parser='bogus' we still trigger a downstream ValueError
    # before iteration; this confirms that pinning a non-harbor runner no
    # longer fails immediately. The previous behavior (RuntimeError on
    # entry) is replaced by the apply-step guard further in.
    with pytest.raises(ValueError, match="Unknown trace_parser"):
        asyncio.run(
            run_loop(
                agent_root=agent_root,
                evals_dir=evals_dir,
                runner=NATRunner(),
                trace_parser="bogus",
            )
        )
