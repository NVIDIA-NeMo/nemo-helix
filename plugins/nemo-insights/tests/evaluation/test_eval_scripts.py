# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess

import pytest
from evaluation.eval import plan, prep, run_subjects, stack

TAU2_CONFIG = """DEFAULT_LLM_NL_ASSERTIONS = "gpt-4.1-2025-04-14"
DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": 0.0}
DEFAULT_LLM_ENV_INTERFACE = "gpt-4.1-2025-04-14"
DEFAULT_LLM_EVAL_USER_SIMULATOR = "claude-opus-4-5"
"""


@pytest.mark.parametrize(
    ("clickhouse_url", "expected_port"),
    [
        (stack.CLICKHOUSE_URL, 8123),
        ("http://localhost:55123", 55123),
    ],
)
def test_stack_resolves_labeled_clickhouse_on_configured_port(monkeypatch, clickhouse_url, expected_port):
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="nhx-intake-clickhouse-managed\n", stderr="")

    monkeypatch.setattr(stack.subprocess, "run", fake_run)

    assert stack._intake_clickhouse_container(clickhouse_url) == "nhx-intake-clickhouse-managed"
    assert "label=nhx.nvidia.com/managed-by=nemo-helix" in commands[0]
    assert "label=nhx.nvidia.com/component=intake-clickhouse" in commands[0]
    assert f"publish={expected_port}" in commands[0]


def test_stack_rejects_nonlocal_clickhouse_url(monkeypatch):
    monkeypatch.setattr(stack.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("docker must not run"))

    with pytest.raises(RuntimeError, match="ClickHouse URL is not local"):
        stack._intake_clickhouse_container("https://clickhouse.example.internal:8443")


def test_stack_rejects_multiple_matching_clickhouse_containers(monkeypatch):
    monkeypatch.setattr(
        stack.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="nhx-intake-clickhouse-one\nnhx-intake-clickhouse-two\n",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="expected one labeled Intake ClickHouse container.*found.*one.*two"):
        stack._intake_clickhouse_container("http://localhost:55123")


def test_stack_does_not_select_unlabeled_same_name_container(monkeypatch):
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        # An unrelated nhx-intake-clickhouse is running, but the label query
        # intentionally excludes it.
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(stack.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="labeled Intake ClickHouse.*none found"):
        stack._intake_clickhouse_container()

    assert len(commands) == 1
    assert "docker" == commands[0][0] and "ps" == commands[0][1]


def test_repoint_judge_rewrites_exactly_three():
    text, count = prep.repoint_judge(TAU2_CONFIG, "openai/x/judge")
    assert count == 3
    assert text.count('"openai/x/judge"') == 3
    assert 'DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": 0.0}' in text  # untouched


def test_repoint_judge_counts_misses():
    _, count = prep.repoint_judge("UNRELATED = 1\n", "j")
    assert count == 0


def test_repoint_judge_preserves_backslashes():
    text, count = prep.repoint_judge(TAU2_CONFIG, r"openai\1\judge")
    assert count == 3
    assert text.count(r'"openai\1\judge"') == 3


def test_mode_pr_event_forces_analyze():
    assert plan.resolve_mode("pull_request", "produce") == "analyze"


def test_mode_dispatch_input_wins():
    assert plan.resolve_mode("workflow_dispatch", "produce") == "produce"


def test_plan_ignores_push_vars_removed():
    """The vars.EVALUATION_* push fallback is gone: push events resolve to plain defaults."""
    import inspect

    assert plan.resolve_mode("push", "") == "analyze"
    assert plan.resolve_subjects("push", "") == ["tau2-airline"]
    assert list(inspect.signature(plan.resolve_mode).parameters) == ["event", "input_mode"]
    assert list(inspect.signature(plan.resolve_subjects).parameters) == ["event", "input_subjects"]


def test_subjects_split_trim_dropempty():
    assert plan.resolve_subjects("workflow_dispatch", " tau2-airline , tau2-retail ,,") == [
        "tau2-airline",
        "tau2-retail",
    ]


def test_subjects_pr_hardcoded():
    assert plan.resolve_subjects("pull_request", "tau2-retail") == ["tau2-airline"]


def test_build_overrides():
    assert run_subjects.build_overrides("2", "") == ["--set", "num_tasks=2"]
    assert run_subjects.build_overrides("", "3") == ["--set", "num_trials=3"]
    assert run_subjects.build_overrides("", "") == []


def test_subjects_from_env_json_array():
    assert run_subjects.subjects_from_env({"SUBJECTS": '["tau2-airline", "tau2-retail"]'}) == [
        "tau2-airline",
        "tau2-retail",
    ]


def test_run_subjects_uses_live_base_for_analyze(monkeypatch):
    """The produce-loop analyze call targets the in-job stack: --live --base <local>."""
    calls = []
    monkeypatch.setattr(run_subjects, "_evaluation", lambda *a: calls.append(a))
    run_subjects.analyze_with_retry("tau2-airline", "evaluation/tmp/summary.md")
    assert calls == [
        (
            "analyze",
            "tau2-airline",
            "--live",
            "--base",
            "http://localhost:8080",
            "--summary-md",
            "evaluation/tmp/summary.md",
        )
    ]


def test_run_subjects_run_uses_base(monkeypatch, tmp_path):
    """The produce-loop run call uses explicit --base <local>; --local is gone."""
    calls = []
    monkeypatch.setattr(run_subjects, "_evaluation", lambda *a: calls.append(a))

    insights = tmp_path / "insights_tau2-airline.yaml"
    insights.write_text("insights: []\n", encoding="utf-8")
    monkeypatch.setattr(run_subjects, "Path", lambda *_a: insights)

    monkeypatch.setenv("SUBJECTS", '["tau2-airline"]')
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    monkeypatch.delenv("NUM_TASKS", raising=False)
    monkeypatch.delenv("NUM_TRIALS", raising=False)
    run_subjects.main()

    run_call = next(c for c in calls if c[0] == "run")
    assert run_call == ("run", "tau2-airline", "--base", "http://localhost:8080")
    assert "--local" not in [arg for call in calls for arg in call]


def test_analyze_with_retry_recovers_after_transient(monkeypatch):
    import subprocess as sp

    calls = {"n": 0}

    def flaky(*args):
        calls["n"] += 1
        if calls["n"] < 3:
            raise sp.CalledProcessError(1, args)

    monkeypatch.setattr(run_subjects, "_evaluation", flaky)
    monkeypatch.setattr(run_subjects.time, "sleep", lambda _s: None)
    run_subjects.analyze_with_retry("tau2-airline", "evaluation/tmp/summary.md")
    assert calls["n"] == 3


def test_analyze_with_retry_exhausts_and_raises(monkeypatch):
    import subprocess as sp

    import pytest

    def always_fail(*args):
        raise sp.CalledProcessError(1, args)

    monkeypatch.setattr(run_subjects, "_evaluation", always_fail)
    monkeypatch.setattr(run_subjects.time, "sleep", lambda _s: None)
    with pytest.raises(sp.CalledProcessError):
        run_subjects.analyze_with_retry("tau2-airline", "evaluation/tmp/summary.md")
