# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the agent_matrix_benchmark orchestration/reporting layer."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import cast

import pytest
from agent_matrix_benchmark import (
    AgentMatrixConfig,
    Candidate,
    CandidateConfig,
    CodexParams,
    _render_leaderboard_header,
    build_matrix_summary,
    build_nat_runner_command,
    load_matrix_config,
    main,
    parse_candidate,
    parse_candidates,
    render_html,
    render_markdown,
    run_candidates,
    write_reports,
)


class TestCandidateParsing:
    def test_parses_backend_without_model(self) -> None:
        candidate = parse_candidate("codex")

        assert candidate == Candidate(backend="codex", model=None, candidate_id="codex")

    def test_parses_backend_with_model(self) -> None:
        candidate = parse_candidate("codex:gpt-5.1")

        assert candidate == Candidate(backend="codex", model="gpt-5.1", candidate_id="codex-gpt-5.1")

    def test_sanitizes_model_for_candidate_id(self) -> None:
        candidate = parse_candidate("cursor-agent:vendor/model name")

        assert candidate.candidate_id == "cursor-agent-vendor-model-name"

    def test_defaults_to_direct_agent_backends(self) -> None:
        candidates = parse_candidates(None)

        assert [candidate.backend for candidate in candidates] == ["codex", "claude-code", "cursor-agent"]

    def test_expands_agent_model_specs_under_selected_backends(self) -> None:
        candidates = parse_candidates(
            ["codex", "claude-code"],
            ["codex:gpt-5.1", "codex:gpt-5.2", "claude-code:sonnet"],
        )

        assert candidates == [
            Candidate(backend="codex", model="gpt-5.1", candidate_id="codex-gpt-5.1"),
            Candidate(backend="codex", model="gpt-5.2", candidate_id="codex-gpt-5.2"),
            Candidate(backend="claude-code", model="sonnet", candidate_id="claude-code-sonnet"),
        ]

    def test_keeps_backend_default_when_no_sub_models_are_specified(self) -> None:
        candidates = parse_candidates(["cursor-agent"], ["codex:gpt-5.1"])

        assert candidates == [
            Candidate(backend="codex", model="gpt-5.1", candidate_id="codex-gpt-5.1"),
            Candidate(backend="cursor-agent", model=None, candidate_id="cursor-agent"),
        ]

    def test_rejects_agent_model_spec_without_model(self) -> None:
        with pytest.raises(ValueError, match="must use backend:model"):
            parse_candidates(None, ["codex"])

    def test_rejects_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            parse_candidate("aut")

    def test_rejects_duplicate_candidate_ids(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            parse_candidates(["codex", "codex"])


class TestPythonMatrixConfig:
    def test_candidate_config_builds_typed_candidate_params(self) -> None:
        candidate = CandidateConfig(
            backend="codex",
            model="gpt-5.5",
            params=CodexParams(intelligence="high", speed="fast"),
        ).to_candidate()

        assert candidate == Candidate(
            backend="codex",
            model="gpt-5.5",
            candidate_id="codex-gpt-5.5-intelligence-high-speed-fast",
            params={"intelligence": "high", "speed": "fast", "config": {}},
        )

    def test_load_matrix_config_from_python_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "matrix.py"
        config_file.write_text(
            """
from agent_matrix_benchmark import AgentMatrixConfig, CandidateConfig, CodexParams

MATRIX = AgentMatrixConfig(
    manifest="manifests/evaluator_agent_benchmark_mvp.txt",
    candidates=[
        CandidateConfig(
            id="codex-fast",
            backend="codex",
            model="gpt-5.5",
            params=CodexParams(intelligence="high", speed="fast"),
        )
    ],
)
""",
            encoding="utf-8",
        )

        config = load_matrix_config(config_file)

        assert isinstance(config, AgentMatrixConfig)
        assert config.manifest == "manifests/evaluator_agent_benchmark_mvp.txt"
        assert config.candidates[0].to_candidate().candidate_id == "codex-fast"

    def test_config_rejects_backend_param_mismatch(self) -> None:
        with pytest.raises(ValueError, match="ClaudeCodeParams"):
            CandidateConfig(backend="claude-code", model="sonnet", params=CodexParams(intelligence="high"))


def test_leaderboard_resize_handle_only_renders_for_resizable_columns() -> None:
    assert "resize-handle" in _render_leaderboard_header("candidate", "Candidate", True, frozen=False)
    assert "resize-handle" not in _render_leaderboard_header("passed", "Passed", True, frozen=False)


def test_build_nat_runner_command_forwards_matrix_flags(tmp_path: Path) -> None:
    command = build_nat_runner_command(
        candidate=Candidate(
            backend="codex",
            model="gpt-5.1",
            candidate_id="codex-gpt-5.1",
            params={"intelligence": "high", "speed": "fast"},
        ),
        jobs_dir=tmp_path / "jobs" / "codex-gpt-5.1",
        manifest=Path("manifests/evaluator_agent_benchmark_mvp.txt"),
        tasks=[],
        all_tasks=False,
        skip_build=True,
        allow_dirty=True,
        timeout=123,
        codex_auth_json=Path("/tmp/auth.json"),
        nhx_base_url="http://localhost:8080",
        anthropic_base_url="https://anthropic.example",
        python_executable="python-test",
    )

    assert command[:2] == ["python-test", str(Path(__file__).parents[1] / "nat_runner.py")]
    assert command[command.index("--agent-backend") + 1] == "codex"
    assert command[command.index("--candidate-id") + 1] == "codex-gpt-5.1"
    assert command[command.index("--agent-model") + 1] == "gpt-5.1"
    assert json.loads(command[command.index("--candidate-params") + 1]) == {
        "intelligence": "high",
        "speed": "fast",
    }
    assert command[command.index("--timeout") + 1] == "123"
    assert command[command.index("--codex-auth-json") + 1] == "/tmp/auth.json"
    assert "--skip-build" in command
    assert "--allow-dirty" in command
    assert command[command.index("--manifest") + 1] == "manifests/evaluator_agent_benchmark_mvp.txt"


def test_build_nat_runner_command_uses_task_args_when_manifest_omitted(tmp_path: Path) -> None:
    command = build_nat_runner_command(
        candidate=Candidate(backend="claude-code", model=None, candidate_id="claude-code"),
        jobs_dir=tmp_path / "jobs" / "claude-code",
        manifest=None,
        tasks=["task-a", "task-b"],
        all_tasks=False,
        skip_build=False,
        allow_dirty=False,
        timeout=600,
        codex_auth_json=None,
        nhx_base_url="http://localhost:8080",
        anthropic_base_url="https://anthropic.example",
        python_executable="python-test",
    )

    assert "--manifest" not in command
    assert command[-2:] == ["task-a", "task-b"]
    assert command[command.index("--anthropic-base-url") + 1] == "https://anthropic.example"


def test_main_rejects_empty_task_selection(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.argv", ["agent_matrix_benchmark.py", "--agent", "codex"])

    assert main() == 1

    captured = capsys.readouterr()
    assert "No tasks specified" in captured.err


def test_main_rejects_parallel_candidates_without_skip_build(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent_matrix_benchmark.py",
            "--parallel-candidates",
            "2",
            "--agent",
            "codex",
            "evaluator-standalone-sdk-surface-discovery",
        ],
    )

    assert main() == 1

    captured = capsys.readouterr()
    assert "--parallel-candidates > 1 requires --skip-build" in captured.err


def test_main_rejects_zero_parallel_candidates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent_matrix_benchmark.py",
            "--parallel-candidates",
            "0",
            "--agent",
            "codex",
            "evaluator-standalone-sdk-surface-discovery",
        ],
    )

    assert main() == 1

    captured = capsys.readouterr()
    assert "--parallel-candidates must be >= 1" in captured.err


def test_run_candidates_parallelizes_across_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()
    two_active = threading.Event()

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            if active == 2:
                two_active.set()
        two_active.wait(timeout=1)
        with lock:
            active -= 1
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("agent_matrix_benchmark.subprocess.run", fake_run)

    metadata = run_candidates(
        candidates=[
            Candidate(backend="codex", model=None, candidate_id="codex"),
            Candidate(backend="cursor-agent", model=None, candidate_id="cursor-agent"),
        ],
        run_dir=tmp_path,
        selected_tasks=["task-a"],
        manifest=None,
        tasks=["task-a"],
        all_tasks=False,
        skip_build=True,
        allow_dirty=True,
        timeout=30,
        codex_auth_json=None,
        nhx_base_url="http://localhost:8080",
        anthropic_base_url="https://anthropic.example",
        parallel_candidates=2,
    )

    assert max_active == 2
    assert [entry["candidate_id"] for entry in metadata] == ["codex", "cursor-agent"]


class TestAggregationAndReports:
    def test_build_matrix_summary_counts_missing_tasks_and_ranks_candidates(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "matrix"
        _write_result(
            run_dir / "codex" / "run-a" / "result.json",
            task="task-a",
            candidate_id="codex",
            backend="codex",
            model="default",
            passed=True,
            total_tokens=100,
            runtime_sec=10.0,
        )
        _write_result(
            run_dir / "codex" / "run-b" / "result.json",
            task="task-b",
            candidate_id="codex",
            backend="codex",
            model="default",
            passed=False,
            total_tokens=40,
            runtime_sec=20.0,
            verifier_text="assertion failed",
        )
        _write_result(
            run_dir / "cursor-agent" / "run-a" / "result.json",
            task="task-a",
            candidate_id="cursor-agent",
            backend="cursor-agent",
            model="default",
            passed=True,
            total_tokens=80,
            runtime_sec=12.0,
        )
        _write_result(
            run_dir / "cursor-agent" / "run-b" / "result.json",
            task="task-b",
            candidate_id="cursor-agent",
            backend="cursor-agent",
            model="default",
            passed=False,
            total_tokens=30,
            runtime_sec=15.0,
        )
        _write_result(
            run_dir / "claude-code" / "run-a" / "result.json",
            task="task-a",
            candidate_id="claude-code",
            backend="claude-code",
            model="default",
            passed=True,
            total_tokens=None,
            runtime_sec=8.0,
        )

        summary = build_matrix_summary(
            run_dir=run_dir,
            candidates=[
                Candidate(backend="codex", model=None, candidate_id="codex"),
                Candidate(backend="cursor-agent", model=None, candidate_id="cursor-agent"),
                Candidate(backend="claude-code", model=None, candidate_id="claude-code"),
            ],
            tasks=["task-a", "task-b"],
        )

        ranking = cast(list[dict[str, object]], summary["ranking"])
        assert [candidate["candidate_id"] for candidate in ranking] == ["cursor-agent", "codex", "claude-code"]
        assert ranking[0]["pass_rate"] == 0.5
        assert ranking[0]["total_tokens_sum"] == 110
        assert ranking[0]["prompt_tokens_sum"] == 110
        assert ranking[0]["completion_tokens_sum"] == 0
        assert ranking[0]["cache_read_tokens_sum"] == 0
        assert ranking[0]["cache_creation_tokens_sum"] == 0
        candidate_groups = cast(dict[str, list[dict[str, object]]], summary["candidate_groups"])
        assert [candidate["candidate_id"] for candidate in candidate_groups["cursor-agent"]] == ["cursor-agent"]
        rows = cast(list[dict[str, object]], summary["rows"])
        missing = [row for row in rows if row["candidate_id"] == "claude-code" and row["task"] == "task-b"]
        assert missing[0]["status"] == "missing"

    def test_candidate_token_sums_include_partial_usage_with_coverage(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "matrix"
        _write_result(
            run_dir / "cursor-agent" / "run-a" / "result.json",
            task="task-a",
            candidate_id="cursor-agent",
            backend="cursor-agent",
            model="default",
            passed=True,
            total_tokens=140,
            runtime_sec=12.0,
        )
        _write_result(
            run_dir / "cursor-agent" / "run-b" / "result.json",
            task="task-b",
            candidate_id="cursor-agent",
            backend="cursor-agent",
            model="default",
            passed=False,
            total_tokens=None,
            runtime_sec=15.0,
        )

        summary = build_matrix_summary(
            run_dir=run_dir,
            candidates=[Candidate(backend="cursor-agent", model=None, candidate_id="cursor-agent")],
            tasks=["task-a", "task-b"],
        )

        ranking = cast(list[dict[str, object]], summary["ranking"])
        assert ranking[0]["total_tokens_sum"] == 140
        assert ranking[0]["prompt_tokens_sum"] == 140
        assert ranking[0]["completion_tokens_sum"] == 0
        assert ranking[0]["cache_read_tokens_sum"] == 0
        assert ranking[0]["cache_creation_tokens_sum"] == 0
        assert ranking[0]["token_metrics_coverage"] == 0.5

    def test_report_renderers_emit_json_markdown_and_static_html(self, tmp_path: Path) -> None:
        summary = {
            "schema_version": "1.0",
            "generated_at": "2026-05-12T00:00:00+00:00",
            "run_dir": str(tmp_path),
            "manifest": "manifests/demo.txt",
            "tasks": ["task-a"],
            "winner": "codex",
            "ranking": [
                {
                    "rank": 1,
                    "candidate_id": "codex",
                    "agent_backend": "codex",
                    "agent_model": "default",
                    "total_tasks": 1,
                    "passed_tasks": 1,
                    "pass_rate": 1.0,
                    "total_tokens_sum": 100,
                    "prompt_tokens_sum": 90,
                    "completion_tokens_sum": 10,
                    "cache_read_tokens_sum": 0,
                    "cache_creation_tokens_sum": None,
                    "token_metrics_coverage": 1.0,
                    "runtime_sec_sum": 9.0,
                }
            ],
            "candidate_groups": {
                "codex": [
                    {
                        "rank": 1,
                        "candidate_id": "codex",
                        "agent_backend": "codex",
                        "agent_model": "default",
                        "total_tasks": 1,
                        "passed_tasks": 1,
                        "pass_rate": 1.0,
                        "total_tokens_sum": 100,
                        "prompt_tokens_sum": 90,
                        "completion_tokens_sum": 10,
                        "cache_read_tokens_sum": 0,
                        "cache_creation_tokens_sum": None,
                        "token_metrics_coverage": 1.0,
                        "runtime_sec_sum": 9.0,
                    }
                ]
            },
            "rows": [
                {
                    "candidate_id": "codex",
                    "agent_backend": "codex",
                    "agent_model": "default",
                    "task": "task-a",
                    "status": "verify_failed",
                    "passed": False,
                    "reward": 0.0,
                    "runtime_sec": 9.0,
                    "total_tokens": 100,
                    "prompt_tokens": 90,
                    "completion_tokens": 10,
                    "cache_read_tokens": 0,
                    "cache_creation_tokens": None,
                    "token_metrics_status": "available",
                    "verifier_scores": {
                        "aggregate_scores": [
                            {"name": "agent_eval/task_success.task_success", "mean": 0.0},
                            {"name": "agent_eval/verification_score.verification_score", "mean": 0.75},
                            {"name": "agent_eval/output_schema_valid.output_schema_valid", "mean": 0.0},
                            {"name": "agent_eval/surface_adherence.surface_adherence", "mean": 1.0},
                        ],
                    },
                    "result_path": str(tmp_path / "result.json"),
                    "output_dir": str(tmp_path),
                    "failure_excerpt": "<script>alert('x')</script>",
                    "provenance": {"commit_short": "abc123", "commit_dirty": True},
                }
            ],
            "run_metadata": [],
        }

        reports = write_reports(summary, tmp_path)
        markdown = render_markdown(summary)
        rendered_html = render_html(summary)

        assert Path(reports["json"]).exists()
        assert json.loads(Path(reports["json"]).read_text())["winner"] == "codex"
        assert "| Rank | Candidate |" in markdown
        assert "Cache Read" in markdown
        assert "Task Matrix" in markdown
        assert "Agent Matrix Benchmark" in rendered_html
        assert "Columns" in rendered_html
        assert 'data-role="candidate-leaderboard"' in rendered_html
        assert "Agents And Models" in rendered_html
        assert 'class="task-result-cell"' in rendered_html
        assert 'class="token-buckets"' in rendered_html
        assert "dirty worktree" in rendered_html
        assert "n/a" in rendered_html
        assert "This backend did not emit this token bucket." in rendered_html
        assert "This backend emitted this token bucket with value zero." in rendered_html
        assert "Verifier rejected the completed agent attempt" in rendered_html
        assert "Output Contract" in rendered_html
        assert "Verification Score" in rendered_html
        assert "Failed or blocking metrics" in rendered_html
        assert "All metric scores" in rendered_html
        assert "Benchmark success" in rendered_html
        assert "Verifier failed" in rendered_html
        assert "Agent failed" in rendered_html
        assert "task_success" in rendered_html
        assert "surface_adherence" in rendered_html
        assert "&lt;script&gt;alert" in rendered_html
        assert "script src=" not in rendered_html
        assert "http://" not in rendered_html
        assert Path(reports["html"]).read_text() == rendered_html


def _write_result(
    path: Path,
    *,
    task: str,
    candidate_id: str,
    backend: str,
    model: str,
    passed: bool,
    total_tokens: int | None,
    runtime_sec: float,
    verifier_text: str | None = None,
) -> None:
    output_dir = path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if verifier_text is not None:
        verifier_dir = output_dir / "verifier"
        verifier_dir.mkdir()
        (verifier_dir / "test-stdout.txt").write_text(verifier_text, encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "task": task,
                "candidate_id": candidate_id,
                "agent_backend": backend,
                "agent_model": model,
                "output_dir": str(output_dir),
                "agent": "ok",
                "verify": "ok" if passed else "failed",
                "passed": passed,
                "reward": 1 if passed else 0,
                "runtime_sec": runtime_sec,
                "metrics": {
                    "total_tokens": total_tokens,
                    "prompt_tokens": total_tokens,
                    "completion_tokens": 0 if total_tokens is not None else None,
                    "cache_read_tokens": 0 if total_tokens is not None else None,
                    "cache_creation_tokens": 0 if total_tokens is not None else None,
                    "token_metrics_status": "available" if total_tokens is not None else "unavailable",
                },
                "provenance": {"commit_short": "abc123", "commit_dirty": False},
            }
        ),
        encoding="utf-8",
    )
