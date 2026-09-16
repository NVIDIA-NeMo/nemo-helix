# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the scaled-evals CLI client.

No live server: a click ``CliRunner`` drives the commands while an httpx
``MockTransport`` stands in for the API, so we can assert the exact method,
path, headers, and body sent and the output/exit code returned.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import httpx
import pytest

pytest.importorskip("scaled_evals")
from click.testing import CliRunner
from scaled_evals.cli import client as client_module
from scaled_evals.cli.main import cli


def runner_with(monkeypatch, handler) -> CliRunner:
    """Patch make_client so the CLI talks to ``handler`` via a MockTransport.

    The real make_client still runs (preserving /v1 prefix and auth header
    resolution); only the transport is swapped.
    """
    monkeypatch.setenv("SCALED_EVALS_BASE_URL", "https://api.example.com")
    real = client_module.make_client

    def patched(base_url, token, transport=None, **kwargs):
        return real(base_url, token, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("scaled_evals.cli.main.make_client", patched)
    return CliRunner()


def test_cli_import_does_not_require_server_settings() -> None:
    env = os.environ.copy()
    env.pop("CREDENTIALS_ENCRYPTION_KEY", None)
    proc = subprocess.run(
        [sys.executable, "-c", "from scaled_evals.cli.main import cli; print(cli.name)"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "cli" in proc.stdout


# ---------- successful creates --------------------------------------------


def test_task_create_success(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            201,
            json={
                "id": "task_1",
                "revision": 1,
                "status": "pending",
                "upload": {
                    "method": "PUT",
                    "workspace": "default",
                    "fileset": "se-task-task_1",
                    "path": "rev/1/tarball.tar.gz",
                    "remote_path": "default/se-task-task_1#rev/1/tarball.tar.gz",
                },
            },
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["task", "create", "--name", "My Bench"])
    assert result.exit_code == 0, result.output
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/tasks"
    assert seen["body"] == {"name": "My Bench"}
    assert "task_1" in result.output
    assert "default/se-task-task_1#rev/1/tarball.tar.gz" in result.output


def test_json_option_before_or_after_subcommands_matches(monkeypatch) -> None:
    envelope = {
        "data": [{"id": "task_1", "visibility": "private", "slug": "smoke", "name": "Smoke"}],
        "next_cursor": None,
    }
    seen_params = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.append(dict(request.url.params))
        return httpx.Response(200, json=envelope)

    runner = runner_with(monkeypatch, handler)
    root_result = runner.invoke(cli, ["--json", "task", "list", "--limit", "1"])
    trailing_result = runner.invoke(cli, ["task", "list", "--limit", "1", "--json"])

    assert root_result.exit_code == 0, root_result.output
    assert trailing_result.exit_code == 0, trailing_result.output
    assert json.loads(root_result.output) == envelope
    assert json.loads(trailing_result.output) == envelope
    assert root_result.output == trailing_result.output
    assert seen_params == [{"limit": "1"}, {"limit": "1"}]


def test_json_option_rejects_duplicate_root_and_subcommand(monkeypatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("request must not be sent")

    result = runner_with(monkeypatch, handler).invoke(cli, ["--json", "task", "list", "--json"])

    assert result.exit_code != 0
    assert "--json may be supplied only once" in result.output


def test_json_option_help_mentions_trailing_position() -> None:
    runner = CliRunner()

    root_result = runner.invoke(cli, ["--help"])
    command_result = runner.invoke(cli, ["task", "list", "--help"])

    assert root_result.exit_code == 0, root_result.output
    assert command_result.exit_code == 0, command_result.output
    root_help = " ".join(root_result.output.split())
    command_help = " ".join(command_result.output.split())
    assert "accepted before or after subcommands" in root_help
    assert "accepted before or after subcommands" in command_help


def test_credential_create_does_not_echo_secret(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            201,
            json={
                "id": "cred_1",
                "name": "openai key",
                "provider": "openai",
                "payload_kind": "key",
                "fingerprint": "sha256:abcd",
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "credential",
            "create",
            "--name",
            "openai key",
            "--provider",
            "openai",
            "--key",
            "sk-secret",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["body"]["key"] == "sk-secret"
    assert "sk-secret" not in result.output
    assert "sha256:abcd" in result.output


def test_config_profile_create_parses_config_json(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "cfg_1", "name": "ws", "type": "intake"})

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "config-profile",
            "create",
            "--name",
            "ws",
            "--type",
            "intake",
            "--config",
            '{"workspace": "w1"}',
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["body"] == {"name": "ws", "type": "intake", "config": {"workspace": "w1"}}
    assert "cfg_1" in result.output


def test_config_profile_create_accepts_gym_type(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "cfg_gym", "name": "gym", "type": "gym"})

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "config-profile",
            "create",
            "--name",
            "gym",
            "--type",
            "gym",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["body"] == {"name": "gym", "type": "gym"}
    assert "cfg_gym" in result.output


def test_benchmark_qualify_builds_body(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "bm_1",
                "name": "Dataset",
                "slug": "dataset",
                "description": None,
                "visibility": "private",
                "qualification_status": "qualified",
                "qualification_evidence": {"oracle": "ev_1"},
                "current_revision": 2,
                "created_at": "2026-07-18T00:00:00Z",
                "updated_at": "2026-07-18T00:00:00Z",
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "--json",
            "benchmark",
            "qualify",
            "bm_1",
            "--status",
            "qualified",
            "--evidence",
            '{"oracle":"ev_1"}',
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/benchmarks/bm_1/qualification"
    assert seen["body"] == {"status": "qualified", "evidence": {"oracle": "ev_1"}}


def test_benchmark_promote_posts_to_promote_endpoint(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "id": "bm_1",
                "name": "Dataset",
                "slug": "dataset",
                "description": None,
                "visibility": "public",
                "qualification_status": "qualified",
                "qualification_evidence": {},
                "current_revision": 2,
                "created_at": "2026-07-18T00:00:00Z",
                "updated_at": "2026-07-18T00:00:00Z",
            },
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["--json", "benchmark", "promote", "bm_1"])

    assert result.exit_code == 0, result.output
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/benchmarks/bm_1/promote"


def test_evaluation_create_builds_body(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={
                "id": "ev_1",
                "name": "run 7",
                "status": "queued",
                "task_id": "task_1",
                "task_revision": 2,
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "evaluation",
            "create",
            "--name",
            "run 7",
            "--task-id",
            "task_1",
            "--task-revision",
            "2",
            "--framework-version",
            "stable",
            "--harbor-profile-id",
            "cfg_h",
            "--agent-bundle",
            "ab_claude",
            "--extra-skill-object-key",
            "skills/review/SKILL.md",
            "--instruction-prefix",
            "inspect first",
            "--instruction-postfix",
            "summarize last",
            "--initial-user-turn",
            "Initialize",
            "--n-attempts",
            "3",
            "--credential",
            "anthropic=cred_1",
            "--credential",
            "intake=cred_2",
            "--parallelism",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["path"] == "/v1/evaluations"
    assert seen["body"] == {
        "name": "run 7",
        "task_id": "task_1",
        "task_revision": 2,
        "framework_version": "stable",
        "credentials": {"anthropic": "cred_1", "intake": "cred_2"},
        "agent_bundle_id": "ab_claude",
        "extra_skill_object_keys": ["skills/review/SKILL.md"],
        "instruction_prefix": "inspect first",
        "instruction_postfix": "summarize last",
        "initial_user_turns": ["Initialize"],
        "n_attempts": 3,
        "harbor_profile_id": "cfg_h",
        "parallelism": 4,
    }
    assert "ev_1" in result.output


def test_evaluation_preflight_posts_complete_request(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "kind": "evaluation",
                "runnable": True,
                "checked_at": "2026-08-06T12:00:00Z",
                "checks": [],
                "member_summary": None,
            },
        )

    request_body = {"name": "check", "task_id": "task_1", "task_revision": 2}
    result = runner_with(monkeypatch, handler).invoke(
        cli,
        ["evaluation", "preflight", "--request", json.dumps(request_body)],
    )

    assert result.exit_code == 0, result.output
    assert seen == {"path": "/v1/evaluations/preflight", "body": request_body}
    assert "Preflight: runnable" in result.output


def test_evaluation_create_preflights_exact_body_before_creation(monkeypatch) -> None:
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        if request.url.path.endswith("/preflight"):
            return httpx.Response(
                200,
                json={
                    "kind": "evaluation",
                    "runnable": True,
                    "checked_at": "2026-08-06T12:00:00Z",
                    "checks": [],
                    "member_summary": None,
                },
            )
        return httpx.Response(
            202,
            json={
                "id": "ev_1",
                "name": "checked run",
                "status": "queued",
                "task_id": "task_1",
                "task_revision": 2,
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "evaluation",
            "create",
            "--preflight",
            "--name",
            "checked run",
            "--task-id",
            "task_1",
            "--task-revision",
            "2",
            "--credential",
            "anthropic=cred_1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [path for path, _ in seen] == [
        "/v1/evaluations/preflight",
        "/v1/evaluations",
    ]
    assert seen[0][1] == seen[1][1]
    assert "Preflight: runnable" in result.output
    assert "ev_1" in result.output


def test_evaluation_create_preflight_stops_blocked_request(monkeypatch) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "kind": "evaluation",
                "runnable": False,
                "checked_at": "2026-08-06T12:00:00Z",
                "checks": [
                    {
                        "prerequisite": "task_revision",
                        "state": "unavailable",
                        "blocking": True,
                        "code": "task_not_ready",
                        "message": "task revision is not ready",
                        "details": {},
                    }
                ],
                "member_summary": None,
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "evaluation",
            "create",
            "--preflight",
            "--name",
            "blocked run",
            "--task-id",
            "task_1",
            "--task-revision",
            "2",
        ],
    )

    assert result.exit_code == 1, result.output
    assert seen == ["/v1/evaluations/preflight"]
    assert "Preflight: not runnable" in result.output
    assert "task revision is not ready" in result.output


def test_agent_bundle_create_builds_private_registration_body(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            201,
            json={
                "id": "ab_custom",
                "bundle_name": "my-codex",
                "agent_name": "codex",
                "agent_version": "0.142.5",
                "visibility": "private",
                "qualification_status": "registered",
                "image_ref": "registry.example/codex:0.142.5",
                "image_digest": "registry.example/codex@sha256:" + "a" * 64,
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "agent-bundle",
            "create",
            "--bundle-name",
            "my-codex",
            "--agent-name",
            "codex",
            "--agent-version",
            "0.142.5",
            "--image-digest",
            "registry.example/codex@sha256:" + "a" * 64,
            "--image-ref",
            "registry.example/codex:0.142.5",
            "--entrypoint",
            "bin/codex",
            "--source-lock-digest",
            "sha256:" + "b" * 64,
            "--fingerprint",
            "sha256:" + "c" * 64,
            "--builder-profile",
            "native-cli-v1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["path"] == "/v1/agent-bundles"
    assert seen["body"] == {
        "bundle_name": "my-codex",
        "agent_name": "codex",
        "agent_version": "0.142.5",
        "image_ref": "registry.example/codex:0.142.5",
        "image_digest": "registry.example/codex@sha256:" + "a" * 64,
        "entrypoint": "bin/codex",
        "source_lock_digest": "sha256:" + "b" * 64,
        "fingerprint": "sha256:" + "c" * 64,
        "builder_profile": "native-cli-v1",
    }
    assert "ab_custom" in result.output


def test_benchmark_run_create_builds_body(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={
                "id": "bmr_1",
                "name": "suite",
                "status": "running",
                "benchmark_id": "bm_1",
                "benchmark_revision": 2,
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "benchmark-run",
            "create",
            "--name",
            "suite",
            "--benchmark-id",
            "bm_1",
            "--benchmark-revision",
            "2",
            "--framework-version",
            "0.6.3",
            "--framework-profile-id",
            "cfg_h",
            "--member-framework-profile",
            "task_0=cfg_member",
            "--agent-bundle",
            "ab_codex",
            "--extra-skill-object-key",
            "skills/review/SKILL.md",
            "--instruction-prefix",
            "inspect first",
            "--instruction-postfix",
            "summarize last",
            "--initial-user-turn",
            "Initialize",
            "--n-attempts",
            "3",
            "--parallelism",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["path"] == "/v1/benchmark-runs"
    assert seen["body"] == {
        "name": "suite",
        "benchmark_id": "bm_1",
        "benchmark_revision": 2,
        "framework_version": "0.6.3",
        "framework_profile_id": "cfg_h",
        "member_framework_profile_ids": {"task_0": "cfg_member"},
        "agent_bundle_id": "ab_codex",
        "extra_skill_object_keys": ["skills/review/SKILL.md"],
        "instruction_prefix": "inspect first",
        "instruction_postfix": "summarize last",
        "initial_user_turns": ["Initialize"],
        "n_attempts": 3,
        "parallelism": 2,
    }
    assert "bmr_1" in result.output


def test_benchmark_run_preflight_posts_complete_request(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "kind": "benchmark_run",
                "runnable": False,
                "checked_at": "2026-08-06T12:00:00Z",
                "checks": [
                    {
                        "prerequisite": "benchmark_members",
                        "state": "unavailable",
                        "blocking": True,
                        "code": "task_not_ready",
                        "message": "1 benchmark member is not runnable",
                        "details": {},
                    }
                ],
                "member_summary": {
                    "total": 2,
                    "ready": 1,
                    "blocked": 1,
                    "failures": [],
                    "failures_truncated": False,
                },
            },
        )

    request_body = {"name": "check", "benchmark_id": "bm_1"}
    result = runner_with(monkeypatch, handler).invoke(
        cli,
        ["benchmark-run", "preflight", "--request", json.dumps(request_body)],
    )

    assert result.exit_code == 0, result.output
    assert seen == {"path": "/v1/benchmark-runs/preflight", "body": request_body}
    assert "Preflight: not runnable" in result.output
    assert "1 ready, 1 blocked, 2 total" in result.output


def test_benchmark_run_create_preflights_exact_body_before_creation(monkeypatch) -> None:
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        if request.url.path.endswith("/preflight"):
            return httpx.Response(
                200,
                json={
                    "kind": "benchmark_run",
                    "runnable": True,
                    "checked_at": "2026-08-06T12:00:00Z",
                    "checks": [],
                    "member_summary": {
                        "total": 2,
                        "ready": 2,
                        "blocked": 0,
                        "failures": [],
                        "failures_truncated": False,
                    },
                },
            )
        return httpx.Response(
            202,
            json={
                "id": "bmr_1",
                "name": "checked suite",
                "status": "queued",
                "benchmark_id": "bm_1",
                "benchmark_revision": 2,
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "benchmark-run",
            "create",
            "--preflight",
            "--name",
            "checked suite",
            "--benchmark-id",
            "bm_1",
            "--benchmark-revision",
            "2",
            "--runtime",
            "sandbox_k8s",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [path for path, _ in seen] == [
        "/v1/benchmark-runs/preflight",
        "/v1/benchmark-runs",
    ]
    assert seen[0][1] == seen[1][1]
    assert "Preflight: runnable" in result.output
    assert "bmr_1" in result.output


def test_benchmark_run_create_help_documents_member_cap() -> None:
    result = CliRunner().invoke(cli, ["benchmark-run", "create", "--help"])
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "Maximum active benchmark member evaluations" in output
    assert "one managed Switchyard gateway" in output
    assert "requires parallelism=1" not in output


def test_benchmark_run_get_shows_per_task_breakdown(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/benchmark-runs/bmr_1"
        return httpx.Response(
            200,
            json={
                "id": "bmr_1",
                "name": "suite",
                "status": "succeeded",
                "benchmark_id": "bm_1",
                "benchmark_revision": 1,
                "reward": 1.0,
                "n_trials": 2,
                "result": {
                    "kind": "benchmark",
                    "per_task": [
                        {
                            "task_slug": "t0",
                            "status": "succeeded",
                            "reward": 1.0,
                            "evaluation_id": "ev_a",
                        },
                    ],
                },
            },
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["benchmark-run", "get", "bmr_1"])
    assert result.exit_code == 0, result.output
    assert "bmr_1" in result.output
    assert "ev_a" in result.output  # member execution id in the per-task breakdown


def test_benchmark_run_get_filters_member_breakdown_by_failure_code(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/benchmark-runs/bmr_1"
        return httpx.Response(
            200,
            json={
                "id": "bmr_1",
                "name": "suite",
                "status": "failed",
                "benchmark_id": "bm_1",
                "benchmark_revision": 1,
                "result": {
                    "kind": "benchmark",
                    "per_task": [
                        {
                            "task_slug": "passed",
                            "status": "succeeded",
                            "evaluation_id": "ev_passed",
                        },
                        {
                            "task_slug": "agent-exit",
                            "status": "failed",
                            "failure_category": "task",
                            "failure_code": "NonZeroAgentExitCodeError",
                            "exception_counts": {"NonZeroAgentExitCodeError": 1},
                            "evaluation_id": "ev_agent_exit",
                        },
                        {
                            "task_slug": "sandbox",
                            "status": "failed",
                            "failure_category": "infrastructure",
                            "failure_code": "SandboxCreationError",
                            "evaluation_id": "ev_sandbox",
                        },
                    ],
                },
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "benchmark-run",
            "get",
            "bmr_1",
            "--failure-code",
            "NonZeroAgentExitCodeError",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "members:   1 of 3 matched filters" in result.output
    assert "agent-exit" in result.output
    assert "failure=task/NonZeroAgentExitCodeError" in result.output
    assert "ev_passed" not in result.output
    assert "ev_sandbox" not in result.output


def test_benchmark_run_evaluations_pages_then_filters_with_diagnostics(monkeypatch) -> None:
    seen_cursors: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/benchmark-runs/bmr_1/evaluations"
        assert request.url.params["limit"] == "200"
        assert request.url.params["order"] == "asc"
        cursor = request.url.params.get("cursor")
        seen_cursors.append(cursor)
        if cursor is None:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "ev_agent_exit",
                            "status": "failed",
                            "task_id": "task_agent_exit",
                            "reward": None,
                            "current_execution": 1,
                            "max_executions": 3,
                            "last_failure_category": "task",
                            "last_failure_code": "NonZeroAgentExitCodeError",
                            "exception_counts": {"NonZeroAgentExitCodeError": 1},
                        }
                    ],
                    "next_cursor": "page-2",
                },
            )
        assert cursor == "page-2"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "ev_passed",
                        "status": "succeeded",
                        "task_id": "task_passed",
                        "reward": 1.0,
                    }
                ],
                "next_cursor": None,
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        ["benchmark-run", "evaluations", "bmr_1", "--failure-category", "task"],
    )

    assert result.exit_code == 0, result.output
    assert seen_cursors == [None, "page-2"]
    assert "ev_agent_exit" in result.output
    assert "failure=task/NonZeroAgentExitCodeError" in result.output
    assert "attempt=1/3" in result.output
    assert "auto-retryable" in result.output
    assert "ev_passed" not in result.output


def test_benchmark_run_member_commands_document_filters() -> None:
    runner = CliRunner()
    for command in ("get", "evaluations"):
        result = runner.invoke(cli, ["benchmark-run", command, "--help"])
        assert result.exit_code == 0, result.output
        assert "--status" in result.output
        assert "--failure-code" in result.output
        assert "--failure-category" in result.output


def test_benchmark_run_reproduce_prints_complete_command(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/benchmark-runs/bmr_1/reproduce"
        return httpx.Response(
            200,
            json={
                "benchmark_run_id": "bmr_1",
                "source_status": "failed",
                "request": {
                    "name": "rerun of suite",
                    "benchmark_id": "bm_1",
                    "benchmark_revision": 2,
                    "framework": "harbor",
                    "framework_version": "0.6.3",
                    "credentials": {"openai": "cred_openai"},
                    "extra_skill_object_keys": ["skills/review/SKILL.md"],
                    "instruction_prefix": "inspect first",
                    "instruction_postfix": "summarize last",
                    "initial_user_turns": ["Initialize"],
                    "runtime": "sandbox_k8s",
                    "network_policy": "unrestricted",
                    "network_policy_config": {},
                    "n_attempts": 3,
                    "parallelism": 2,
                    "max_concurrent_members": 4,
                    "visibility": "private",
                },
                "cli_command": [],
                "notes": ["Secret material is not exported."],
            },
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["benchmark-run", "reproduce", "bmr_1"])

    assert result.exit_code == 0, result.output
    assert "rerun command:" in result.output
    assert "benchmark-run create" in result.output
    assert "--n-attempts 3" in result.output
    assert "--max-concurrent-members 4" in result.output
    assert "openai=cred_openai" in result.output


def test_benchmark_run_reproduce_rerun_posts_request(monkeypatch) -> None:
    seen: list[tuple[str, str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.content))
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "benchmark_run_id": "bmr_1",
                    "source_status": "failed",
                    "request": {
                        "name": "rerun of suite",
                        "benchmark_id": "bm_1",
                        "benchmark_revision": 2,
                        "runtime": "sandbox_k8s",
                        "parallelism": 2,
                    },
                    "cli_command": [],
                    "notes": [],
                },
            )
        return httpx.Response(
            202,
            json={
                "id": "bmr_2",
                "name": "debug rerun",
                "status": "running",
                "benchmark_id": "bm_1",
                "benchmark_revision": 2,
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        ["benchmark-run", "reproduce", "bmr_1", "--rerun", "--name", "debug rerun"],
    )

    assert result.exit_code == 0, result.output
    assert seen[0][0:2] == ("GET", "/v1/benchmark-runs/bmr_1/reproduce")
    assert seen[1][0:2] == ("POST", "/v1/benchmark-runs")
    assert json.loads(seen[1][2])["name"] == "debug rerun"
    assert "bmr_2" in result.output


def test_evaluation_create_builds_framework_profile_body(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={
                "id": "ev_1",
                "name": "gym run",
                "status": "queued",
                "task_id": "task_1",
                "task_revision": 1,
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "evaluation",
            "create",
            "--name",
            "gym run",
            "--framework",
            "nemo_gym",
            "--task-id",
            "task_1",
            "--task-revision",
            "1",
            "--framework-profile-id",
            "cfg_g",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["path"] == "/v1/evaluations"
    assert seen["body"] == {
        "name": "gym run",
        "framework": "nemo_gym",
        "task_id": "task_1",
        "task_revision": 1,
        "framework_profile_id": "cfg_g",
    }
    assert "ev_1" in result.output


# ---------- broker upload (direct to Files) --------------------------------


def test_task_create_then_upload(monkeypatch, tmp_path) -> None:
    tarball = tmp_path / "pack.tar.gz"
    tarball.write_bytes(b"TARBYTES")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "id": "task_1",
                    "revision": 1,
                    "status": "pending",
                    "upload": {
                        "method": "PUT",
                        "workspace": "default",
                        "fileset": "se-task-task_1",
                        "path": "rev/1/tarball.tar.gz",
                        "remote_path": "default/se-task-task_1#rev/1/tarball.tar.gz",
                    },
                },
            )
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["content_length"] = request.headers.get("Content-Length")
        seen["has_auth"] = "authorization" in request.headers
        seen["content"] = request.content
        return httpx.Response(200)

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        ["--token", "nvapi-xyz", "task", "create", "--name", "B", "--tarball", str(tarball)],
    )
    assert result.exit_code == 0, result.output
    assert seen["method"] == "PUT"
    # Uploads go directly to the Files service's v2 fileset endpoint on the same host.
    assert seen["url"] == "https://api.example.com/v2/workspaces/default/filesets/se-task-task_1/-/rev/1/tarball.tar.gz"
    assert seen["content_length"] == str(len(b"TARBYTES"))
    assert seen["content"] == b"TARBYTES"
    # Unlike the old presigned URL, the Files endpoint authenticates via the bearer
    # token, so it MUST be kept on the upload request.
    assert seen["has_auth"] is True


def test_task_upload_mints_revision(monkeypatch, tmp_path) -> None:
    tarball = tmp_path / "pack.tar.gz"
    tarball.write_bytes(b"NEWREV")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen["revisions_path"] = request.url.path
            return httpx.Response(
                201,
                json={
                    "id": "task_1",
                    "revision": 2,
                    "status": "uploading",
                    "upload": {
                        "method": "PUT",
                        "workspace": "default",
                        "fileset": "se-task-task_1",
                        "path": "rev/2/tarball.tar.gz",
                        "remote_path": "default/se-task-task_1#rev/2/tarball.tar.gz",
                    },
                },
            )
        seen["put_url"] = str(request.url)
        seen["content"] = request.content
        return httpx.Response(200)

    result = runner_with(monkeypatch, handler).invoke(cli, ["task", "upload", "task_1", str(tarball)])
    assert result.exit_code == 0, result.output
    assert seen["revisions_path"] == "/v1/tasks/task_1/revisions"
    assert seen["put_url"] == "https://api.example.com/v2/workspaces/default/filesets/se-task-task_1/-/rev/2/tarball.tar.gz"
    assert seen["content"] == b"NEWREV"
    assert "revision 2" in result.output


def test_task_upload_json_reports_local_result(monkeypatch, tmp_path) -> None:
    tarball = tmp_path / "pack.tar.gz"
    tarball.write_bytes(b"NEWREV")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "id": "task_1",
                    "revision": 2,
                    "upload": {
                        "method": "PUT",
                        "workspace": "default",
                        "fileset": "se-task-task_1",
                        "path": "rev/2/tarball.tar.gz",
                        "remote_path": "default/se-task-task_1#rev/2/tarball.tar.gz",
                    },
                },
            )
        return httpx.Response(200)

    result = runner_with(monkeypatch, handler).invoke(cli, ["--json", "task", "upload", "task_1", str(tarball)])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "task_id": "task_1",
        "revision": 2,
        "path": str(tarball),
        "uploaded": True,
    }


def test_task_finalize(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(202, json={"id": "task_1", "revision": 1, "status": "building"})

    result = runner_with(monkeypatch, handler).invoke(cli, ["task", "finalize", "task_1"])
    assert result.exit_code == 0, result.output
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/tasks/task_1/finalize"
    assert "building" in result.output


def test_task_finalize_sends_reuse_image_body(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(202, json={"id": "task_1", "revision": 1, "status": "building"})

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "task",
            "finalize",
            "task_1",
            "--image-ref",
            "registry.example.com/team/task:signed",
            "--image-digest",
            "sha256:" + "a" * 64,
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["body"] == {
        "image_ref": "registry.example.com/team/task:signed",
        "image_digest": "sha256:" + "a" * 64,
    }


def test_task_finalize_rejects_digest_without_image_ref(monkeypatch) -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "task",
            "finalize",
            "task_1",
            "--image-digest",
            "sha256:" + "a" * 64,
        ],
    )

    assert result.exit_code == 1
    assert "--image-digest requires --image-ref" in result.output
    assert called is False


def test_task_get_shows_build_status(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(
            200,
            json={
                "id": "task_1",
                "name": "B",
                "slug": "b",
                "visibility": "private",
                "revision": 1,
                "status": "ready",
                "image_ref": "registry:5000/task_1:1",
                "image_digest": "sha256:deadbeef",
            },
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["task", "get", "task_1"])
    assert result.exit_code == 0, result.output
    assert "ready" in result.output
    assert "registry:5000/task_1:1" in result.output
    assert "sha256:deadbeef" in result.output


def test_task_get_by_slug(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "id": "task_1",
                "name": "B",
                "slug": "bench-slug",
                "visibility": "private",
                "revision": 3,
                "status": "ready",
            },
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["task", "get-by-slug", "bench-slug"])
    assert result.exit_code == 0, result.output
    assert seen["method"] == "GET"
    assert seen["path"] == "/v1/tasks/by-slug/bench-slug"
    assert "task_1" in result.output
    assert "ready" in result.output


def test_task_list_update_delete(monkeypatch) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, dict(request.url.params), request.content))
        if request.method == "PATCH":
            return httpx.Response(
                200,
                json={
                    "id": "task_1",
                    "name": "New",
                    "slug": "new",
                    "visibility": "team",
                    "revision": 1,
                    "status": "ready",
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": "task_1", "deleted": True})
        return httpx.Response(
            200,
            json={
                "data": [{"id": "task_1", "name": "Bench", "slug": "bench", "visibility": "private"}],
                "next_cursor": "next",
            },
        )

    runner = runner_with(monkeypatch, handler)
    listed = runner.invoke(cli, ["task", "list", "--limit", "2", "--order", "asc"])
    updated = runner.invoke(
        cli,
        [
            "task",
            "update",
            "task_1",
            "--name",
            "New",
            "--slug",
            "new",
            "--visibility",
            "team",
        ],
    )
    deleted = runner.invoke(cli, ["task", "delete", "task_1"])

    assert listed.exit_code == 0, listed.output
    assert updated.exit_code == 0, updated.output
    assert deleted.exit_code == 0, deleted.output
    assert seen[0][0:3] == ("GET", "/v1/tasks", {"limit": "2", "order": "asc"})
    assert seen[1][0:2] == ("PATCH", "/v1/tasks/task_1")
    assert json.loads(seen[1][3]) == {"name": "New", "slug": "new", "visibility": "team"}
    assert seen[2][0:2] == ("DELETE", "/v1/tasks/task_1")
    assert "next_cursor: next" in listed.output
    assert "New" in updated.output
    assert "deleted task task_1" in deleted.output


def test_credential_list_filters_by_provider(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "cred_1",
                        "name": "anthropic prod",
                        "provider": "anthropic",
                        "payload_kind": "key",
                        "fingerprint": "sha256:abcd",
                    }
                ],
                "next_cursor": "next",
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        ["credential", "list", "--provider", "anthropic", "--limit", "5", "--order", "asc"],
    )
    assert result.exit_code == 0, result.output
    assert seen["path"] == "/v1/credentials"
    assert seen["params"]["provider"] == "anthropic"
    assert seen["params"]["limit"] == "5"
    assert seen["params"]["order"] == "asc"
    assert "cred_1" in result.output
    assert "sha256:abcd" in result.output
    assert "next_cursor: next" in result.output


def test_credential_create_accepts_openshift_provider(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            201,
            json={
                "id": "cred_os",
                "name": "cluster token",
                "provider": "openshift",
                "payload_kind": "key",
                "fingerprint": "sha256:os",
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "credential",
            "create",
            "--name",
            "cluster token",
            "--provider",
            "openshift",
            "--key",
            "openshift-token",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["body"] == {
        "name": "cluster token",
        "provider": "openshift",
        "key": "openshift-token",
    }
    assert "cred_os" in result.output
    assert "openshift-token" not in result.output


def test_credential_get_does_not_echo_secret(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/credentials/cred_1"
        return httpx.Response(
            200,
            json={
                "id": "cred_1",
                "name": "openai key",
                "provider": "openai",
                "payload_kind": "key",
                "fingerprint": "sha256:abcd",
            },
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["credential", "get", "cred_1"])
    assert result.exit_code == 0, result.output
    assert "cred_1" in result.output
    assert "sha256:abcd" in result.output
    assert "sk-" not in result.output


def test_credential_verify_posts(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(
            200,
            json={"id": "cred_1", "verified": True, "reason": "ok"},
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["credential", "verify", "cred_1"])
    assert result.exit_code == 0, result.output
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/credentials/cred_1/verify"
    assert "verified: True" in result.output


def test_credential_rename_rotate_delete(monkeypatch) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": "cred_1", "deleted": True})
        return httpx.Response(
            200,
            json={
                "id": "cred_1",
                "name": body.get("name") if body and "name" in body else "renamed",
                "provider": "openai",
                "payload_kind": "key",
                "fingerprint": "sha256:new",
            },
        )

    runner = runner_with(monkeypatch, handler)
    renamed = runner.invoke(cli, ["credential", "rename", "cred_1", "--name", "renamed"])
    rotated = runner.invoke(cli, ["credential", "rotate", "cred_1", "--key", "sk-new-secret"])
    deleted = runner.invoke(cli, ["credential", "delete", "cred_1"])

    assert renamed.exit_code == 0, renamed.output
    assert rotated.exit_code == 0, rotated.output
    assert deleted.exit_code == 0, deleted.output
    assert seen[0] == ("PATCH", "/v1/credentials/cred_1", {"name": "renamed"})
    assert seen[1] == ("POST", "/v1/credentials/cred_1/rotate", {"key": "sk-new-secret"})
    assert seen[2] == ("DELETE", "/v1/credentials/cred_1", None)
    assert "sk-new-secret" not in rotated.output
    assert "sha256:new" in rotated.output


def test_config_profile_list_and_get(monkeypatch) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, dict(request.url.params)))
        if request.url.path.endswith("/cfg_1"):
            return httpx.Response(200, json={"id": "cfg_1", "name": "intake", "type": "intake"})
        return httpx.Response(
            200,
            json={
                "data": [{"id": "cfg_1", "name": "intake", "type": "intake"}],
                "next_cursor": None,
            },
        )

    runner = runner_with(monkeypatch, handler)
    listed = runner.invoke(cli, ["config-profile", "list", "--type", "intake"])
    assert listed.exit_code == 0, listed.output
    got = runner.invoke(cli, ["config-profile", "get", "cfg_1"])
    assert got.exit_code == 0, got.output
    assert seen[0] == ("GET", "/v1/config-profiles", {"type": "intake"})
    assert seen[1] == ("GET", "/v1/config-profiles/cfg_1", {})
    assert "cfg_1" in listed.output
    assert "intake" in got.output


def test_config_profile_create_accepts_yaml_file(monkeypatch, tmp_path) -> None:
    config = tmp_path / "switchyard.yaml"
    config.write_text(
        """
switchyard_routing_profiles_yaml: |
  profiles:
    default:
      model: strong
model: nemotron
"""
    )
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "cfg_sw", "name": "sw", "type": "switchyard"})

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "config-profile",
            "create",
            "--name",
            "sw",
            "--type",
            "switchyard",
            "--config-yaml",
            f"@{config}",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["body"]["config"]["model"] == "nemotron"
    assert "profiles:" in seen["body"]["config"]["switchyard_routing_profiles_yaml"]


def test_config_profile_update_delete(monkeypatch) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": "cfg_1", "deleted": True})
        return httpx.Response(200, json={"id": "cfg_1", "name": "renamed", "type": "intake"})

    runner = runner_with(monkeypatch, handler)
    updated = runner.invoke(
        cli,
        [
            "config-profile",
            "update",
            "cfg_1",
            "--name",
            "renamed",
            "--config",
            '{"workspace": "w2"}',
        ],
    )
    deleted = runner.invoke(cli, ["config-profile", "delete", "cfg_1"])

    assert updated.exit_code == 0, updated.output
    assert deleted.exit_code == 0, deleted.output
    assert seen[0] == (
        "PATCH",
        "/v1/config-profiles/cfg_1",
        {"name": "renamed", "config": {"workspace": "w2"}},
    )
    assert seen[1] == ("DELETE", "/v1/config-profiles/cfg_1", None)


def test_config_profile_list_passes_order(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={"data": [{"id": "cfg_1", "name": "h", "type": "harbor"}]},
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli, ["config-profile", "list", "--type", "harbor", "--order", "asc"]
    )
    assert result.exit_code == 0, result.output
    assert seen["params"] == {"type": "harbor", "order": "asc"}
    assert "cfg_1" in result.output


def test_config_profile_create_rejects_invalid_yaml(monkeypatch) -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "config-profile",
            "create",
            "--name",
            "bad",
            "--type",
            "switchyard",
            "--config-yaml",
            "items:\n  - [",
        ],
    )
    assert result.exit_code != 0
    assert "not valid YAML" in result.output
    assert called is False


# ---------- evaluation run + pull-down ------------------------------------


def test_evaluation_create_includes_runtime(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(202, json={"id": "ev_1", "status": "queued"})

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "evaluation",
            "create",
            "--name",
            "r",
            "--task-id",
            "task_1",
            "--task-revision",
            "1",
            "--runtime",
            "sandbox_k8s",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["body"]["runtime"] == "sandbox_k8s"


def test_evaluation_create_rejects_bad_framework(monkeypatch) -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "evaluation",
            "create",
            "--name",
            "r",
            "--task-id",
            "task_1",
            "--task-revision",
            "1",
            "--framework",
            "typo",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid value for '--framework'" in result.output
    assert called is False


def test_evaluation_create_help_documents_api_defaults() -> None:
    result = CliRunner().invoke(cli, ["evaluation", "create", "--help"])
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "API default: sandbox_k8s" in output
    assert "API default: harbor" in result.output
    assert "--framework [harbor|nemo_gym]" in result.output


def test_evaluation_create_includes_switchyard_and_intake_profiles(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(202, json={"id": "ev_1", "status": "queued"})

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "evaluation",
            "create",
            "--name",
            "r",
            "--task-id",
            "task_1",
            "--task-revision",
            "1",
            "--framework-profile-id",
            "cfg_h",
            "--switchyard-profile-id",
            "cfg_sw",
            "--intake-profile-id",
            "cfg_intake",
            "--runtime",
            "sandbox_k8s",
            "--visibility",
            "team",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["body"] == {
        "name": "r",
        "task_id": "task_1",
        "task_revision": 1,
        "framework_profile_id": "cfg_h",
        "switchyard_profile_id": "cfg_sw",
        "intake_profile_id": "cfg_intake",
        "runtime": "sandbox_k8s",
        "visibility": "team",
    }


def test_evaluation_list_passes_filters_and_shows_cursor(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "data": [{"id": "ev_1", "status": "running", "name": "r1"}],
                "next_cursor": "abc123",
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "evaluation",
            "list",
            "--status",
            "running",
            "--team-id",
            "team_1",
            "--limit",
            "5",
            "--order",
            "asc",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["path"] == "/v1/evaluations"
    assert seen["params"]["status"] == "running"
    assert seen["params"]["team_id"] == "team_1"
    assert seen["params"]["limit"] == "5"
    assert seen["params"]["order"] == "asc"
    assert "ev_1" in result.output
    assert "next_cursor: abc123" in result.output


def test_evaluation_get_shows_reward_summary(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(
            200,
            json={
                "id": "ev_1",
                "name": "r",
                "status": "succeeded",
                "task_id": "task_1",
                "task_revision": 1,
                "reward": 0.75,
                "n_trials": 4,
                "n_completed": 4,
                "n_errored": 0,
                "n_failed_solve": 0,
                "outcome": {"category": "completed", "exception_counts": {}},
                "current_execution": 2,
                "finished_at": "2026-06-05T00:00:00Z",
            },
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "get", "ev_1"])
    assert result.exit_code == 0, result.output
    assert "succeeded" in result.output
    assert "outcome:   completed" in result.output
    assert "0.75" in result.output
    assert "4 (0 errored)" in result.output
    assert "4 completed, 0 failed solve" in result.output
    assert "retries:    1" in result.output


def test_evaluation_get_hides_retry_detail_before_a_retry(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(
            200,
            json={
                "id": "ev_1",
                "name": "r",
                "status": "running",
                "task_id": "task_1",
                "task_revision": 1,
                "current_execution": 1,
                "outcome": {"category": "in_progress", "exception_counts": {}},
            },
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "get", "ev_1"])

    assert result.exit_code == 0, result.output
    assert "retries:" not in result.output


def test_evaluation_retry_posts_same_evaluation_retry_endpoint(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(
            202,
            json={
                "id": "ev_1",
                "name": "r",
                "status": "queued",
                "status_detail": "manual retry scheduled; execution 2",
                "task_id": "task_1",
                "task_revision": 1,
                "benchmark_run_id": "bmr_1",
                "current_execution": 2,
                "outcome": {"category": "in_progress", "exception_counts": {}},
            },
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "retry", "ev_1"])

    assert result.exit_code == 0, result.output
    assert seen == {"method": "POST", "path": "/v1/evaluations/ev_1/retry"}
    assert "benchmark run: bmr_1" in result.output
    assert "retries:    1" in result.output


def test_evaluation_reproduce_prints_command_and_request(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/evaluations/ev_1/reproduce"
        return httpx.Response(
            200,
            json={
                "evaluation_id": "ev_1",
                "source_status": "failed",
                "request": {
                    "name": "rerun of run 7",
                    "task_id": "task_1",
                    "task_revision": 2,
                    "framework": "harbor",
                    "framework_profile_id": "cfg_h",
                    "harbor_profile_id": "cfg_h",
                    "switchyard_profile_id": None,
                    "intake_profile_id": None,
                    "credentials": {"openai": "cred_openai"},
                    "extra_skill_object_keys": [],
                    "runtime": "sandbox_k8s",
                    "parallelism": 4,
                    "visibility": "private",
                },
                "cli_command": [
                    "scaled-evals",
                    "evaluation",
                    "create",
                    "--name",
                    "rerun of run 7",
                    "--credential",
                    "openai=cred_openai",
                ],
                "notes": ["Secret material is not exported."],
            },
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "reproduce", "ev_1"])
    assert result.exit_code == 0, result.output
    assert "rerun command:" in result.output
    assert "openai=cred_openai" in result.output
    assert '"task_id": "task_1"' in result.output
    assert "Secret material" in result.output


def test_evaluation_reproduce_rerun_posts_request(monkeypatch) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.content))
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "evaluation_id": "ev_1",
                    "source_status": "failed",
                    "request": {
                        "name": "rerun of run 7",
                        "task_id": "task_1",
                        "task_revision": 2,
                        "framework": "harbor",
                        "credentials": {"openai": "cred_openai"},
                        "extra_skill_object_keys": [],
                        "runtime": "sandbox_k8s",
                        "parallelism": 4,
                        "visibility": "private",
                    },
                    "cli_command": [],
                    "notes": [],
                },
            )
        return httpx.Response(
            202,
            json={
                "id": "ev_2",
                "name": "debug rerun",
                "status": "queued",
                "task_id": "task_1",
                "task_revision": 2,
                "runtime": "sandbox_k8s",
                "parallelism": 4,
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli, ["evaluation", "reproduce", "ev_1", "--rerun", "--name", "debug rerun"]
    )
    assert result.exit_code == 0, result.output
    assert seen[0][0:2] == ("GET", "/v1/evaluations/ev_1/reproduce")
    assert seen[1][0:2] == ("POST", "/v1/evaluations")
    assert json.loads(seen[1][2])["name"] == "debug rerun"
    assert "ev_2" in result.output


def test_evaluation_reproduce_reports_missing_required_fields(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/evaluations/ev_1/reproduce"
        return httpx.Response(
            200,
            json={
                "evaluation_id": "ev_1",
                "request": {"name": "incomplete", "task_id": "task_1"},
                "cli_command": [],
            },
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "reproduce", "ev_1"])

    assert result.exit_code != 0
    assert "missing required field(s): task_revision, runtime, parallelism" in result.output
    assert "KeyError" not in result.output


def test_evaluation_cancel(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(200, json={"id": "ev_1", "status": "cancelled"})

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "cancel", "ev_1"])
    assert result.exit_code == 0, result.output
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/evaluations/ev_1/cancel"
    assert "cancelled" in result.output


def test_evaluation_logs_prints_lines(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "evaluation_id": "ev_1",
                "lines": ["l1", "l2"],
                "status": "running",
                "complete": False,
            },
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "logs", "ev_1", "--tail", "50"])
    assert result.exit_code == 0, result.output
    assert seen["params"]["tail_lines"] == "50"
    assert "l1" in result.output
    assert "l2" in result.output


def test_evaluation_logs_follow_reads_sse_until_terminal(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/evaluations/ev_1/logs/stream"
        body = (
            'event: log\ndata: {"line": "started"}\n\n'
            "event: ping\ndata: {}\n\n"
            'event: status\ndata: {"type": "status", "status": "succeeded"}\n\n'
        )
        return httpx.Response(200, content=body.encode())

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "logs", "ev_1", "--follow"])
    assert result.exit_code == 0, result.output
    assert "started" in result.output
    assert "-- status: succeeded" in result.output


def test_evaluation_logs_follow_rejects_tail(monkeypatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("request must not be sent")

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "logs", "ev_1", "--follow", "--tail", "5"])

    assert result.exit_code != 0
    assert "--tail cannot be combined with --follow" in result.output


def test_evaluation_logs_follow_fails_if_stream_ends_before_terminal(monkeypatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'event: log\ndata: {"line":"started"}\n\n')

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "logs", "ev_1", "--follow"])

    assert result.exit_code != 0
    assert "stream ended before a terminal status" in result.output


def test_evaluation_logs_follow_surfaces_structured_stream_error(monkeypatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"detail": {"error": {"code": "forbidden", "message": "no stream"}}},
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "logs", "ev_1", "--follow"])

    assert result.exit_code != 0
    assert "forbidden: no stream" in result.output


def test_evaluation_events_serializes_pagination(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "evaluation_id": "ev_1",
                        "type": "status",
                        "status": "running",
                        "detail": "started",
                        "at": "2026-06-23T00:00:00Z",
                    }
                ],
                "next_cursor": "cursor-2",
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "evaluation",
            "events",
            "ev_1",
            "--limit",
            "1",
            "--cursor",
            "cursor-1",
            "--offset",
            "10",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["params"] == {"limit": "1", "cursor": "cursor-1", "offset": "10"}
    assert "started" in result.output
    assert "next_cursor: cursor-2" in result.output


def test_evaluation_events_json_preserves_api_envelope(monkeypatch) -> None:
    envelope = {
        "data": [
            {
                "evaluation_id": "ev_1",
                "type": "status",
                "status": "queued",
                "detail": None,
                "at": "2026-06-23T00:00:00Z",
            }
        ],
        "next_cursor": "cursor-2",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/evaluations/ev_1/events"
        return httpx.Response(200, json=envelope)

    result = runner_with(monkeypatch, handler).invoke(cli, ["--json", "evaluation", "events", "ev_1", "--limit", "1"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == envelope


def test_evaluation_events_follow_reads_sse_until_terminal(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/evaluations/ev_1/events/stream"
        body = (
            'event: status\ndata: {"type": "status", "status": "running"}\n\n'
            "event: ping\ndata: {}\n\n"
            'event: status\ndata: {"type": "status", "status": "succeeded"}\n\n'
        )
        return httpx.Response(200, content=body.encode())

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "events", "ev_1", "--follow"])
    assert result.exit_code == 0, result.output
    assert "running" in result.output
    assert "succeeded" in result.output


def test_evaluation_events_follow_rejects_pagination(monkeypatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("request must not be sent")

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "events", "ev_1", "--follow", "--limit", "1"])

    assert result.exit_code != 0
    assert "cannot be combined with --follow" in result.output


def test_evaluation_wait_succeeds_after_polling(monkeypatch) -> None:
    statuses = iter(
        [
            {"id": "ev_1", "name": "r", "status": "running"},
            {
                "id": "ev_1",
                "name": "r",
                "status": "succeeded",
                "task_id": "task_1",
                "task_revision": 1,
                "reward": 1.0,
                "n_trials": 1,
                "n_errored": 0,
                "links": {
                    "artifacts": "/evaluations/ev_1/artifacts",
                    "provenance": "/evaluations/ev_1/artifacts/scaled-evals-provenance.json",
                },
            },
        ]
    )
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json=next(statuses))

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "wait", "ev_1", "--interval", "0"])
    assert result.exit_code == 0, result.output
    assert seen == [("GET", "/v1/evaluations/ev_1"), ("GET", "/v1/evaluations/ev_1")]
    assert "ev_1: running" in result.output
    assert "ev_1: succeeded" in result.output
    assert "reward:    1.0" in result.output
    assert "provenance:" in result.output


def test_evaluation_wait_nonzero_for_failed_cancelled_and_blocked(monkeypatch) -> None:
    for status in ("failed", "cancelled", "blocked"):
        seen = {}

        def handler(
            request: httpx.Request,
            *,
            terminal_status: str = status,
            seen_paths: dict[str, str] = seen,
        ) -> httpx.Response:
            seen_paths["path"] = request.url.path
            return httpx.Response(
                200,
                json={
                    "id": "ev_1",
                    "name": "r",
                    "status": terminal_status,
                    "status_detail": "done badly",
                    "task_id": "task_1",
                    "task_revision": 1,
                },
            )

        result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "wait", "ev_1", "--interval", "0"])
        assert result.exit_code == 1, result.output
        assert seen["path"] == "/v1/evaluations/ev_1"
        assert status in result.output
        assert "done badly" in result.output


def test_evaluation_wait_times_out(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/evaluations/ev_1"
        return httpx.Response(200, json={"id": "ev_1", "name": "r", "status": "running"})

    result = runner_with(monkeypatch, handler).invoke(
        cli, ["evaluation", "wait", "ev_1", "--interval", "0", "--timeout", "0"]
    )
    assert result.exit_code != 0
    assert "timed out waiting for ev_1" in result.output


def test_evaluation_wait_interrupted(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/evaluations/ev_1"
        return httpx.Response(200, json={"id": "ev_1", "name": "r", "status": "running"})

    def interrupt(_interval: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("scaled_evals.cli.main.time.sleep", interrupt)
    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "wait", "ev_1"])
    assert result.exit_code == 130
    assert "interrupted" in result.output


def test_evaluation_create_wait_prints_final_json(monkeypatch) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.content))
        if request.method == "POST":
            return httpx.Response(202, json={"id": "ev_1", "name": "r", "status": "queued"})
        return httpx.Response(
            200,
            json={
                "id": "ev_1",
                "name": "r",
                "status": "succeeded",
                "task_id": "task_1",
                "task_revision": 1,
                "reward": 0.5,
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "--json",
            "evaluation",
            "create",
            "--name",
            "r",
            "--task-id",
            "task_1",
            "--task-revision",
            "1",
            "--wait",
            "--wait-interval",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen[0][0:2] == ("POST", "/v1/evaluations")
    assert seen[1][0:2] == ("GET", "/v1/evaluations/ev_1")
    assert json.loads(seen[0][2]) == {
        "name": "r",
        "task_id": "task_1",
        "task_revision": 1,
    }
    body = json.loads(result.output)
    assert body["id"] == "ev_1"
    assert body["status"] == "succeeded"
    assert body["reward"] == 0.5


def test_evaluation_artifacts_lists(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [{"path": "result.json", "size_bytes": 12, "updated_at": "t"}],
                "next_cursor": None,
            },
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "artifacts", "ev_1"])
    assert result.exit_code == 0, result.output
    assert "result.json" in result.output


def test_evaluation_download_follows_redirect_and_strips_auth(monkeypatch, tmp_path) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "store":
            seen["dl_auth"] = "authorization" in request.headers
            return httpx.Response(200, content=b"ARTIFACT-BYTES")
        # API artifact route → presigned redirect.
        seen["api_path"] = request.url.path
        return httpx.Response(307, headers={"location": "https://store/result.json?sig=x"})

    dest = tmp_path / "out.json"
    result = runner_with(monkeypatch, handler).invoke(
        cli,
        ["--token", "nvapi-xyz", "evaluation", "download", "ev_1", "result.json", "-o", str(dest)],
    )
    assert result.exit_code == 0, result.output
    assert seen["api_path"] == "/v1/evaluations/ev_1/artifacts/result.json"
    assert dest.read_bytes() == b"ARTIFACT-BYTES"
    # The API bearer token must not be forwarded to object storage.
    assert seen["dl_auth"] is False


def test_evaluation_download_json_reports_local_result(monkeypatch, tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "store":
            return httpx.Response(200, content=b"ARTIFACT-BYTES")
        return httpx.Response(307, headers={"location": "https://store/result.json?sig=x"})

    dest = tmp_path / "out.json"
    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "--json",
            "evaluation",
            "download",
            "ev_1",
            "result.json",
            "-o",
            str(dest),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "evaluation_id": "ev_1",
        "artifact_path": "result.json",
        "path": str(dest),
        "downloaded": True,
    }
    assert dest.read_bytes() == b"ARTIFACT-BYTES"


def test_evaluation_download_error_preserves_existing_destination(monkeypatch, tmp_path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"detail": {"error": {"code": "store_failed", "message": "nope"}}},
        )

    dest = tmp_path / "out.json"
    dest.write_bytes(b"KEEP")
    result = runner_with(monkeypatch, handler).invoke(
        cli, ["evaluation", "download", "ev_1", "result.json", "-o", str(dest)]
    )

    assert result.exit_code != 0
    assert "store_failed" in result.output
    assert dest.read_bytes() == b"KEEP"


def test_evaluation_archive_downloads_presigned_url_without_auth(monkeypatch, tmp_path) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "store":
            seen["dl_auth"] = "authorization" in request.headers
            return httpx.Response(200, content=b"ARCHIVE-BYTES")
        seen["api_path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "evaluation_id": "ev_1",
                "status": "ready",
                "format": "tar.gz",
                "download": {"method": "GET", "url": "https://store/results.tar.gz?sig=x"},
            },
        )

    dest = tmp_path / "results.tar.gz"
    result = runner_with(monkeypatch, handler).invoke(
        cli,
        ["--token", "nvapi-xyz", "evaluation", "archive", "ev_1", "--download", "-o", str(dest)],
    )
    assert result.exit_code == 0, result.output
    assert seen["api_path"] == "/v1/evaluations/ev_1/archive"
    assert dest.read_bytes() == b"ARCHIVE-BYTES"
    assert seen["dl_auth"] is False


def test_evaluation_archive_downloads_api_relative_url_with_auth(monkeypatch, tmp_path) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/evaluations/ev_1/archive/download":
            seen["dl_auth"] = request.headers.get("authorization")
            return httpx.Response(200, content=b"ARCHIVE-BYTES")
        seen["api_path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "evaluation_id": "ev_1",
                "status": "ready",
                "format": "tar.gz",
                "download": {
                    "method": "GET",
                    "url": "/evaluations/ev_1/archive/download",
                },
            },
        )

    dest = tmp_path / "results.tar.gz"
    result = runner_with(monkeypatch, handler).invoke(
        cli,
        ["--token", "nvapi-xyz", "evaluation", "archive", "ev_1", "--download", "-o", str(dest)],
    )

    assert result.exit_code == 0, result.output
    assert seen["api_path"] == "/v1/evaluations/ev_1/archive"
    assert seen["dl_auth"] == "Bearer nvapi-xyz"
    assert dest.read_bytes() == b"ARCHIVE-BYTES"


def test_evaluation_archive_force_posts(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={"evaluation_id": "ev_1", "status": "building", "format": "tar.gz"},
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "archive", "ev_1", "--force"])
    assert result.exit_code == 0, result.output
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/evaluations/ev_1/archive"
    assert seen["body"] == {"force": True}
    assert "building" in result.output


def test_evaluation_archive_handles_null_download_before_ready(monkeypatch) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, json.loads(request.content) if request.content else None))
        return httpx.Response(
            202 if request.method == "POST" else 200,
            json={
                "evaluation_id": "ev_1",
                "status": "building" if request.method == "POST" else "pending",
                "format": "tar.gz",
                "download": None,
            },
        )

    runner = runner_with(monkeypatch, handler)
    status_result = runner.invoke(cli, ["evaluation", "archive", "ev_1"])
    build_result = runner.invoke(cli, ["evaluation", "archive", "ev_1", "--build"])

    assert status_result.exit_code == 0, status_result.output
    assert "pending" in status_result.output
    assert "url:" not in status_result.output
    assert build_result.exit_code == 0, build_result.output
    assert "building" in build_result.output
    assert "url:" not in build_result.output
    assert seen == [("GET", None), ("POST", {"force": False})]


def test_evaluation_harbor_viewer_shows_manual_upload_command(monkeypatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "ev_1",
                "links": {
                    "harbor_viewer": None,
                    "harbor_viewer_archive": "/evaluations/ev_1/harbor-viewer/archive",
                    "harbor_viewer_upload": ("https://viewer.example/api/uploads/jobs?overwrite=true"),
                },
            },
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "harbor-viewer", "ev_1"])

    assert result.exit_code == 0, result.output
    assert "/evaluations/ev_1/harbor-viewer/archive" in result.output
    assert "https://viewer.example/api/uploads/jobs?overwrite=true" in result.output
    assert "scaled-evals evaluation harbor-viewer ev_1 --upload" in result.output


def test_evaluation_harbor_viewer_upload_bridges_from_workstation_without_api_auth(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "viewer.example":
            seen["upload_auth"] = request.headers.get("authorization")
            seen["upload_content_type"] = request.headers.get("content-type")
            seen["upload_body"] = request.content
            return httpx.Response(200, json={"job_name": "ev_1"})
        if request.url.path == "/v1/evaluations/ev_1/harbor-viewer/archive":
            seen["download_auth"] = request.headers.get("authorization")
            return httpx.Response(200, content=b"VIEWER-ARCHIVE")
        return httpx.Response(
            200,
            json={
                "id": "ev_1",
                "links": {
                    "harbor_viewer": None,
                    "harbor_viewer_archive": "/evaluations/ev_1/harbor-viewer/archive",
                    "harbor_viewer_upload": ("https://viewer.example/api/uploads/jobs?overwrite=true"),
                },
            },
        )

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        ["--token", "api-token", "evaluation", "harbor-viewer", "ev_1", "--upload"],
    )

    assert result.exit_code == 0, result.output
    assert seen["download_auth"] == "Bearer api-token"
    assert seen["upload_auth"] is None
    assert str(seen["upload_content_type"]).startswith("multipart/form-data; boundary=")
    assert b"VIEWER-ARCHIVE" in seen["upload_body"]
    assert "uploaded:   ev_1" in result.output


def test_evaluation_provenance_prints_json_from_artifact_redirect(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "store":
            seen["dl_auth"] = "authorization" in request.headers
            return httpx.Response(
                200,
                json={"schema_version": "scaled-evals-run-provenance-v2", "evaluation_id": "ev_1"},
            )
        if request.url.path == "/v1/evaluations/ev_1":
            return httpx.Response(
                200,
                json={
                    "id": "ev_1",
                    "status": "succeeded",
                    "links": {"provenance": "/evaluations/ev_1/artifacts/scaled-evals-provenance.json"},
                },
            )
        seen["artifact_path"] = request.url.path
        return httpx.Response(307, headers={"location": "https://store/provenance.json?sig=x"})

    result = runner_with(monkeypatch, handler).invoke(
        cli, ["--token", "nvapi-xyz", "--json", "evaluation", "provenance", "ev_1"]
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["schema_version"] == "scaled-evals-run-provenance-v2"
    assert seen["artifact_path"] == "/v1/evaluations/ev_1/artifacts/scaled-evals-provenance.json"
    assert seen["dl_auth"] is False


def test_evaluation_sbom_prints_cyclonedx_from_artifact_redirect(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "store":
            seen["dl_auth"] = "authorization" in request.headers
            return httpx.Response(
                200,
                json={"bomFormat": "CycloneDX", "specVersion": "1.6"},
            )
        if request.url.path == "/v1/evaluations/ev_1":
            return httpx.Response(
                200,
                json={
                    "id": "ev_1",
                    "status": "succeeded",
                    "links": {"sbom": "/evaluations/ev_1/artifacts/scaled-evals-sbom.cdx.json"},
                },
            )
        seen["artifact_path"] = request.url.path
        return httpx.Response(307, headers={"location": "https://store/sbom.json?sig=x"})

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        ["--token", "nvapi-xyz", "--json", "evaluation", "sbom", "ev_1"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["bomFormat"] == "CycloneDX"
    assert seen["artifact_path"] == "/v1/evaluations/ev_1/artifacts/scaled-evals-sbom.cdx.json"
    assert seen["dl_auth"] is False


def test_evaluation_provenance_reports_not_available(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/evaluations/ev_1":
            return httpx.Response(
                200,
                json={
                    "id": "ev_1",
                    "status": "running",
                    "links": {"provenance": "/evaluations/ev_1/artifacts/scaled-evals-provenance.json"},
                },
            )
        return httpx.Response(
            404,
            json={"detail": {"error": {"code": "not_found", "message": "not found"}}},
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "provenance", "ev_1"])
    assert result.exit_code != 0
    assert "not available yet" in result.output


def test_evaluation_provenance_does_not_match_not_found_message_text(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/evaluations/ev_1":
            return httpx.Response(200, json={"id": "ev_1", "links": {}})
        return httpx.Response(
            403,
            json={
                "detail": {
                    "error": {
                        "code": "permission_denied",
                        "message": "not_found text is not an error code",
                    }
                }
            },
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["evaluation", "provenance", "ev_1"])

    assert result.exit_code != 0
    assert "permission_denied" in result.output
    assert "not available yet" not in result.output


def test_evaluation_provenance_downloads_output(monkeypatch, tmp_path) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "store":
            seen["dl_auth"] = "authorization" in request.headers
            return httpx.Response(200, content=b'{"evaluation_id":"ev_1"}')
        if request.url.path == "/v1/evaluations/ev_1":
            return httpx.Response(
                200,
                json={
                    "id": "ev_1",
                    "status": "succeeded",
                    "links": {"provenance": "/evaluations/ev_1/artifacts/scaled-evals-provenance.json"},
                },
            )
        return httpx.Response(307, headers={"location": "https://store/provenance.json?sig=x"})

    dest = tmp_path / "provenance.json"
    result = runner_with(monkeypatch, handler).invoke(
        cli,
        ["--token", "nvapi-xyz", "evaluation", "provenance", "ev_1", "-o", str(dest)],
    )
    assert result.exit_code == 0, result.output
    assert dest.read_text() == '{"evaluation_id":"ev_1"}'
    assert seen["dl_auth"] is False
    assert "downloaded provenance" in result.output


# ---------- error handling ------------------------------------------------


def test_error_envelope_surfaced(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"detail": {"error": {"code": "not_found", "message": "nope", "details": {}}}},
        )

    result = runner_with(monkeypatch, handler).invoke(cli, ["task", "create", "--name", "B"])
    assert result.exit_code != 0
    assert "not_found: nope" in result.output


def test_network_error_surfaced(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    result = runner_with(monkeypatch, handler).invoke(cli, ["task", "create", "--name", "B"])
    assert result.exit_code != 0
    assert "failed" in result.output.lower()


# ---------- config / auth resolution --------------------------------------


def test_base_url_and_token_from_env(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(201, json={"id": "cfg_1", "name": "n", "type": "harbor"})

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        ["config-profile", "create", "--name", "n", "--type", "harbor"],
        env={
            "SCALED_EVALS_BASE_URL": "https://api.example.com",
            "SCALED_EVALS_TOKEN": "env-token",
        },
    )
    assert result.exit_code == 0, result.output
    assert seen["url"] == "https://api.example.com/v1/config-profiles"
    assert seen["auth"] == "Bearer env-token"


def test_flag_overrides_env(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(201, json={"id": "cfg_1", "name": "n", "type": "harbor"})

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "--base-url",
            "https://flag.example.com",
            "--token",
            "flag-token",
            "config-profile",
            "create",
            "--name",
            "n",
            "--type",
            "harbor",
        ],
        env={
            "SCALED_EVALS_BASE_URL": "https://api.example.com",
            "SCALED_EVALS_TOKEN": "env-token",
        },
    )
    assert result.exit_code == 0, result.output
    assert seen["url"] == "https://flag.example.com/v1/config-profiles"
    assert seen["auth"] == "Bearer flag-token"
