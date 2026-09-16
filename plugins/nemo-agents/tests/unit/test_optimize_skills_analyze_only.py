# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``--analyze-only`` mode of optimize-skills.

Analyze-only consumes a pre-existing batch directory, runs the gap-analysis
pipeline (mechanical + LLM clustering + hypothesis generation), writes the
suggestions to ``<batch_dir>/optimize-suggestions.json``, and exits — no
worktree, no apply, no verify, no MR. Works with any AUT (Harbor or NAT).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_eval(eval_dir: Path, name: str, marker: str) -> Path:
    """Tiny eval fixture — mirrors the helper in test_improvement_runners.py."""
    d = eval_dir / name
    d.mkdir(parents=True)
    if marker == "harbor":
        (d / "task.toml").write_text(
            "[metadata]\ndifficulty = 'easy'\ncategory = 'test'\ntags = []\n"
            "[agent]\ntimeout_sec = 60.0\n[verifier]\ntimeout_sec = 30.0\n"
        )
    elif marker == "nat":
        (d / "workflow.yml").write_text("workflow: {}")
    (d / "instruction.md").write_text("do the thing")
    tests = d / "tests"
    tests.mkdir()
    (tests / "test_outputs.py").write_text("def test_x(): pass\n")
    return d


def _fake_gap_analysis(batch_id: str):
    """Build a minimal GapAnalysis without calling the LLM."""
    from nemo_agents_plugin.improvement.models import (
        EvalCluster,
        GapAnalysis,
        GapCategory,
        Hypothesis,
        MechanicalAnalysis,
    )

    mechanical = MechanicalAnalysis(
        failing_evals=["eval-a"],
        slowest_evals=[("eval-a", 12.0)],
        highest_tool_count=[("eval-a", 5)],
        error_patterns={},
        regressions=[],
        tool_usage_distribution={},
    )
    cluster = EvalCluster(
        cluster_id="cluster-1",
        eval_names=["eval-a"],
        shared_patterns=["missing skill X"],
        signal_type="missing_skill",
        description="Agent lacks skill X",
    )
    hypothesis = Hypothesis(
        cluster_id="cluster-1",
        eval_names=["eval-a"],
        root_cause="Agent has no skill for X",
        category=GapCategory.MISSING_SKILL,
        proposed_fix="Add skill X to .agents/skills/",
        affected_files=[".agents/skills/x.md"],
        expected_impact="eval-a passes",
        confidence=0.8,
    )
    return GapAnalysis(
        batch_id=batch_id,
        mechanical=mechanical,
        clusters=[cluster],
        hypotheses=[hypothesis],
        generated_at=datetime.now(timezone.utc),
    )


def _make_batch_fixture(tmp_path: Path, batch_id: str = "batch-test-001") -> Path:
    """Write a minimal Harbor-shaped batch directory.

    Just enough for ``parse_batch_results`` to return a non-empty BatchResult
    (one trial-level result.json with task_name + reward).
    """
    batch_dir = tmp_path / "jobs" / batch_id
    batch_dir.mkdir(parents=True)

    (batch_dir / "batch_config.json").write_text(
        json.dumps(
            {
                "model": "test-model",
                "agent": "claude-code",
                "started_at": "2026-05-19T00:00:00+00:00",
            }
        )
    )

    job = batch_dir / f"{batch_id}__eval-a"
    job.mkdir()
    trial = job / "trial-1"
    trial.mkdir()
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "eval-a",
                "started_at": "2026-05-19T00:00:01+00:00",
                "finished_at": "2026-05-19T00:00:13+00:00",
                "verifier_result": {"rewards": {"reward": 0.0}},
                "agent_execution": {
                    "started_at": "2026-05-19T00:00:02+00:00",
                    "finished_at": "2026-05-19T00:00:12+00:00",
                },
                "agent_result": {
                    "n_input_tokens": 100,
                    "n_output_tokens": 50,
                    "n_cache_tokens": 0,
                },
            }
        )
    )
    return batch_dir


def test_analyze_only_writes_suggestions_for_nat_aut(tmp_path: Path) -> None:
    """Analyze-only consumes a NAT-shaped suite + batch, writes suggestions.

    Confirms (a) AUT discovery runs without raising on a NAT suite (which
    the old harbor-only guard rejected), (b) no apply step is invoked, and
    (c) the suggestions file lands in the batch dir.
    """
    from nemo_agents_plugin.improvement.loop import run_analyze_only

    evals_dir = tmp_path / "evals"
    _make_eval(evals_dir, "eval-a", "nat")  # NAT suite — would fail old guard
    batch_dir = _make_batch_fixture(tmp_path)

    fake_gap = _fake_gap_analysis(batch_dir.name)
    with (
        patch(
            "nemo_agents_plugin.improvement.loop.generate_gap_analysis",
            return_value=fake_gap,
        ) as mock_analyze,
        patch("nemo_agents_plugin.improvement.loop.apply_hypothesis") as mock_apply,
    ):
        state = asyncio.run(
            run_analyze_only(
                evals_dir=evals_dir,
                initial_batch_dir=batch_dir,
            )
        )

    # generate_gap_analysis was invoked exactly once on the loaded batch
    assert mock_analyze.call_count == 1
    # apply_hypothesis was NEVER invoked
    assert mock_apply.call_count == 0

    # Suggestions file lands in batch_dir
    suggestions = batch_dir / "optimize-suggestions.json"
    assert suggestions.exists(), "expected optimize-suggestions.json in batch dir"
    payload = json.loads(suggestions.read_text())
    assert payload["batch_id"] == batch_dir.name
    assert len(payload["hypotheses"]) == 1
    assert payload["hypotheses"][0]["cluster_id"] == "cluster-1"

    # Returned state is minimal — batch id set, no iterations
    assert state.current_baseline_batch == batch_dir.name
    assert state.iterations == []
    assert state.iteration == 0


def test_analyze_only_works_for_harbor_aut(tmp_path: Path) -> None:
    """The same flow works for a Harbor-shaped suite — analyze is AUT-agnostic."""
    from nemo_agents_plugin.improvement.loop import run_analyze_only

    evals_dir = tmp_path / "evals"
    _make_eval(evals_dir, "eval-a", "harbor")
    batch_dir = _make_batch_fixture(tmp_path)

    fake_gap = _fake_gap_analysis(batch_dir.name)
    with patch(
        "nemo_agents_plugin.improvement.loop.generate_gap_analysis",
        return_value=fake_gap,
    ):
        asyncio.run(
            run_analyze_only(
                evals_dir=evals_dir,
                initial_batch_dir=batch_dir,
            )
        )

    assert (batch_dir / "optimize-suggestions.json").exists()


def test_analyze_only_raises_when_batch_dir_missing(tmp_path: Path) -> None:
    """Missing batch directory surfaces a clear error rather than silently empty."""
    from nemo_agents_plugin.improvement.loop import run_analyze_only

    evals_dir = tmp_path / "evals"
    _make_eval(evals_dir, "eval-a", "harbor")
    missing = tmp_path / "does-not-exist"

    with pytest.raises(RuntimeError, match="does not exist"):
        asyncio.run(
            run_analyze_only(
                evals_dir=evals_dir,
                initial_batch_dir=missing,
            )
        )


def test_optimize_skills_job_analyze_only_requires_initial_batch() -> None:
    """The job's analyze_only branch validates initial_batch up front."""
    from nemo_agents_plugin.jobs.optimize_skills import OptimizeSkillsJob

    cfg = {
        "evals": "/tmp/nonexistent-evals",
        "agent": "/tmp/nonexistent-agent",
        "analyze_only": True,
        "initial_batch": None,
    }
    with pytest.raises(ValueError, match="initial_batch"):
        OptimizeSkillsJob().run(cfg, ctx=MagicMock())


def _agents_cli_with_jobs():
    """Build the agents CLI with ``nemo.jobs`` commands mounted.

    ``AgentsCLI.get_cli()`` alone does not mount job subcommands — the platform
    CLI loader injects them via :func:`add_job_commands`.  Replicate that here so
    ``optimize-skills`` is exercised through the real generated CLI surface
    rather than a hand-written wrapper.
    """
    from nemo_agents_plugin.cli import AgentsCLI
    from nemo_agents_plugin.jobs.optimize_skills import OptimizeSkillsJob
    from nemo_helix_plugin.commands import add_job_commands

    app = AgentsCLI().get_cli()
    add_job_commands(app, {"optimize-skills": OptimizeSkillsJob})
    return app


def _guard_message(result) -> str:
    """CLI output plus any surfaced exception message, for guard assertions."""
    exc = "" if result.exception is None else str(result.exception)
    return f"{result.output}\n{exc}"


def test_cli_analyze_only_flag_flows_through_direct_command() -> None:
    """`optimize-skills --analyze-only` without --initial-batch is submitted for server validation."""
    from nemo_agents_plugin.jobs.optimize_skills import OptimizeSkillsJob
    from typer.testing import CliRunner

    captured: dict[str, object] = {}

    def _submit_remote(_self, job_cls, spec, **kwargs):
        captured["job_cls"] = job_cls
        captured["spec"] = spec
        captured["base_url"] = kwargs["base_url"]
        return {"name": "optimize-skills-123"}

    app = _agents_cli_with_jobs()
    with patch("nemo_helix_plugin.scheduler.NemoJobScheduler.submit_remote", _submit_remote):
        result = CliRunner().invoke(
            app,
            [
                "optimize-skills",
                "--agent",
                "/tmp/agent",
                "--evals",
                "/tmp/x",
                "--analyze-only",
                "--base-url",
                "http://test",
            ],
        )

    assert result.exit_code == 0, _guard_message(result)
    assert captured["job_cls"] is OptimizeSkillsJob
    assert captured["base_url"] == "http://test"
    assert captured["spec"] == {
        "agent": "/tmp/agent",
        "evals": "/tmp/x",
        "analyze_only": True,
    }

    legacy_result = CliRunner().invoke(app, ["optimize-skills", "run"])
    assert legacy_result.exit_code != 0


def test_cli_analyze_only_from_spec_file_flows_through_direct_command(tmp_path: Path) -> None:
    """analyze_only=true in a --spec-file YAML flows through the direct command."""
    from nemo_agents_plugin.jobs.optimize_skills import OptimizeSkillsJob
    from typer.testing import CliRunner

    config = tmp_path / "config.yml"
    config.write_text("analyze_only: true\nagent: /tmp/agent\nevals: /tmp/x\n")

    captured: dict[str, object] = {}

    def _submit_remote(_self, job_cls, spec, **kwargs):
        captured["job_cls"] = job_cls
        captured["spec"] = spec
        captured["base_url"] = kwargs["base_url"]
        return {"name": "optimize-skills-123"}

    app = _agents_cli_with_jobs()
    with patch("nemo_helix_plugin.scheduler.NemoJobScheduler.submit_remote", _submit_remote):
        result = CliRunner().invoke(
            app,
            [
                "optimize-skills",
                "--spec-file",
                str(config),
                "--base-url",
                "http://test",
            ],
        )

    assert result.exit_code == 0, _guard_message(result)
    assert captured["job_cls"] is OptimizeSkillsJob
    assert captured["base_url"] == "http://test"
    assert captured["spec"] == {
        "agent": "/tmp/agent",
        "evals": "/tmp/x",
        "analyze_only": True,
    }
