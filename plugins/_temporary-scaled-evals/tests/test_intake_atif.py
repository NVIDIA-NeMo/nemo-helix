# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for post-run NHX Intake ATIF upload."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import call, patch

import pytest

try:
    from scaled_evals.intake.atif_payload import IntakeError, trial_payloads
    from scaled_evals.intake.client import (
        atif_ingest_url,
        create_evaluation,
        create_experiment_group,
        post_atif_payload,
        request_json,
    )
    from scaled_evals.intake.config import (
        resolve_intake_target,
        resolve_routing_task,
        validate_intake_profile_config,
    )
    from scaled_evals.intake.experiments import ExperimentRequest, build_experiment_name
    from scaled_evals.intake.upload import upload_job_atif, upload_job_atif_warn
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _sample_job(tmp_path: Path) -> Path:
    job_dir = tmp_path / "ev_test123"
    trial_dir = job_dir / "task-a__trial1"
    _write_json(
        job_dir / "config.json",
        {
            "job_name": "ev_test123",
            "agents": [{"name": "oracle", "model_name": "noop"}],
            "datasets": [{"name": "hello-task"}],
        },
    )
    _write_json(
        job_dir / "result.json",
        {"id": "job-harbor", "n_total_trials": 1, "stats": {"n_errored_trials": 0}},
    )
    _write_json(
        trial_dir / "result.json",
        {
            "id": "trial-1",
            "task_name": "task-a",
            "trial_name": "task-a__trial1",
            "verifier_result": {"rewards": {"reward": 1.0}},
        },
    )
    _write_json(
        trial_dir / "agent" / "trajectory.json",
        {
            "schema_version": "ATIF-v1.6",
            "session_id": "tool-session",
            "steps": [
                {"step_id": 1, "source": "user", "message": "go"},
                {"step_id": 2, "source": "agent", "message": "done"},
            ],
        },
    )
    return job_dir


def test_resolve_intake_target_defaults_workspace_and_source() -> None:
    target = resolve_intake_target(
        {},
        task_slug="my-bench",
        base_url="https://platform.example",
    )
    assert target.workspace == "default"
    assert target.app == "my-bench"
    assert target.source == "scaled-evals"
    assert target.base_url == "https://platform.example"


def test_resolve_intake_target_honors_profile_overrides() -> None:
    target = resolve_intake_target(
        {"workspace": "team-ws", "app": "custom-app"},
        task_slug="ignored-slug",
        base_url="https://platform.example/",
    )
    assert target.workspace == "team-ws"
    assert target.app == "custom-app"


def test_resolve_intake_target_accepts_prefixed_aliases() -> None:
    target = resolve_intake_target(
        {
            "intake_base_url": "https://intake.example/apis/intake/v2/",
            "intake_workspace": "team-ws",
            "intake_app": "switchyard-app",
        },
        task_slug="ignored-slug",
        base_url="https://platform.example/",
    )

    assert target.base_url == "https://intake.example/apis/intake/v2"
    assert target.workspace == "team-ws"
    assert target.app == "switchyard-app"


@pytest.mark.parametrize("config", [{}, {"workspace": " "}, {"workspace": 42}])
def test_intake_profile_validation_requires_typed_workspace(config: dict) -> None:
    with pytest.raises(ValueError):
        validate_intake_profile_config(config)


def test_resolve_intake_target_accepts_inert_switchyard_capture_keys() -> None:
    target = resolve_intake_target(
        {
            "capture_content": True,
            "switchyard_intake_capture_content": True,
        },
        task_slug="my-bench",
        base_url="https://platform.example/",
    )

    assert target.workspace == "default"
    assert target.app == "my-bench"


def test_resolve_routing_task_supports_switchyard_header_keys() -> None:
    assert resolve_routing_task({"task": "custom-task"}, task_slug="fallback") == "custom-task"
    assert resolve_routing_task({"intake_task": "prefixed-task"}, task_slug="fallback") == "prefixed-task"
    assert resolve_routing_task({}, task_slug="fallback") == "fallback"


def test_broken_python_smoke_uses_gym_smoke_task_slug() -> None:
    """``task_gym_smoke`` slug from a broken-python example defaults Intake app."""
    target = resolve_intake_target(
        {"workspace": "default"},
        task_slug="gym-smoke",
        base_url="https://platform.example",
    )
    assert target.app == "gym-smoke"


def test_trial_payloads_keep_run_metadata_without_context_by_default(tmp_path: Path) -> None:
    job_dir = _sample_job(tmp_path)
    payloads = trial_payloads(
        job_dir,
        "team-ws",
        "auto",
        "scaled-evals",
        evaluation_run_id="ev_abc",
    )
    assert len(payloads) == 1
    payload = payloads[0].payload
    assert payloads[0].external_id == "scaled-evals:ev_abc:trial-1"
    assert "evaluation_context" not in payload
    assert "experiment_context" not in payload
    assert payload["extra"]["job_id"] == "ev_abc"
    assert payload["extra"]["experiment"]["dataset_name"] == "hello-task"
    assert payload["session_id"] == "task-a__trial1"
    assert payload["extra"]["app"] == "team-ws/hello-task"


def _write_switchyard_session(job_dir: Path, session: dict) -> None:
    _write_json(
        job_dir / "switchyard" / "routing_stats_final.json",
        {
            "requested_session_ids": [session["session_id"]],
            "sessions": {session["session_id"]: session},
        },
    )


def _write_native_atif_metrics(
    job_dir: Path,
    *,
    model: str,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    cost_usd: float,
) -> None:
    trajectory_path = job_dir / "task-a__trial1" / "agent" / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text())
    trajectory["agent"] = {"name": "oracle", "version": "1", "model_name": model}
    trajectory["final_metrics"] = {
        "total_prompt_tokens": prompt_tokens,
        "total_cached_tokens": cached_tokens,
        "total_completion_tokens": completion_tokens,
        "total_cost_usd": cost_usd,
    }
    _write_json(trajectory_path, trajectory)


def test_trial_payloads_preserve_matching_native_single_model_metrics(tmp_path: Path) -> None:
    job_dir = _sample_job(tmp_path)
    _write_native_atif_metrics(
        job_dir,
        model="claude-opus-4-8",
        prompt_tokens=155199,
        cached_tokens=81834,
        completion_tokens=7648,
        cost_usd=0.6036695,
    )
    _write_switchyard_session(
        job_dir,
        {
            "session_id": "ev_abc",
            "total_calls": 6,
            "total_prompt_tokens": 155199,
            "total_cached_tokens": 81834,
            "total_cache_creation_tokens": 3782,
            "total_completion_tokens": 7648,
            "models": {
                "azure/anthropic/claude-opus-4-8": {
                    "calls": 6,
                    "prompt_tokens": 155199,
                    "cached_tokens": 81834,
                    "cache_creation_tokens": 3782,
                    "completion_tokens": 7648,
                }
            },
        },
    )

    payload = trial_payloads(
        job_dir,
        "team-ws",
        "auto",
        "scaled-evals",
        evaluation_run_id="ev_abc",
    )[0].payload

    assert payload["agent"]["model_name"] == "claude-opus-4-8"
    assert payload["extra"]["experiment"]["model"] == "claude-opus-4-8"
    assert payload["final_metrics"] == {
        "total_prompt_tokens": 155199,
        "total_cached_tokens": 81834,
        "total_completion_tokens": 7648,
        "total_cost_usd": 0.6036695,
        "total_steps": 2,
    }
    assert "switchyard_routing" not in payload["extra"]


def test_trial_payloads_replace_gateway_model_name_during_hydration(tmp_path: Path) -> None:
    job_dir = _sample_job(tmp_path)
    _write_native_atif_metrics(
        job_dir,
        model="openai/switchyard",
        prompt_tokens=1,
        cached_tokens=0,
        completion_tokens=1,
        cost_usd=0.01,
    )
    _write_switchyard_session(
        job_dir,
        {
            "session_id": "ev_abc",
            "total_calls": 1,
            "total_prompt_tokens": 100,
            "total_cached_tokens": 0,
            "total_cache_creation_tokens": 0,
            "total_completion_tokens": 10,
            "models": {
                "nvidia/nvidia/nemotron-3-super-v3": {
                    "calls": 1,
                    "prompt_tokens": 100,
                    "cached_tokens": 0,
                    "cache_creation_tokens": 0,
                    "completion_tokens": 10,
                }
            },
        },
    )

    payload = trial_payloads(
        job_dir,
        "team-ws",
        "auto",
        "scaled-evals",
        evaluation_run_id="ev_abc",
    )[0].payload

    assert payload["agent"]["model_name"] == "nvidia/nvidia/nemotron-3-super-v3"
    assert payload["extra"]["experiment"]["model"] == "nvidia/nvidia/nemotron-3-super-v3"
    assert payload["final_metrics"]["total_prompt_tokens"] == 100


def test_trial_payloads_hydrate_single_model_root_totals_and_cache_cost(tmp_path: Path) -> None:
    job_dir = _sample_job(tmp_path)
    trajectory_path = job_dir / "task-a__trial1" / "agent" / "trajectory.json"
    native_trajectory = json.loads(trajectory_path.read_text())
    _write_switchyard_session(
        job_dir,
        {
            "session_id": "ev_abc",
            "total_calls": 2,
            "total_prompt_tokens": 1000,
            "total_cached_tokens": 600,
            "total_cache_creation_tokens": 100,
            "total_completion_tokens": 20,
            "models": {
                "nvidia/switchyard/gpt-5.4": {
                    "calls": 2,
                    "prompt_tokens": 1000,
                    "cached_tokens": 600,
                    "cache_creation_tokens": 100,
                    "completion_tokens": 20,
                }
            },
        },
    )

    payload = trial_payloads(
        job_dir,
        "team-ws",
        "auto",
        "scaled-evals",
        evaluation_run_id="ev_abc",
    )[0].payload

    assert payload["agent"]["model_name"] == "nvidia/switchyard/gpt-5.4"
    assert payload["final_metrics"]["total_prompt_tokens"] == 1000
    assert payload["final_metrics"]["total_cached_tokens"] == 600
    assert payload["final_metrics"]["total_completion_tokens"] == 20
    assert payload["final_metrics"]["total_cost_usd"] == pytest.approx(0.00145)
    assert all("metrics" not in step for step in payload["steps"])
    routing = payload["extra"]["switchyard_routing"]
    assert routing["total_cache_creation_tokens"] == 100
    assert routing["cost_status"] == "complete"
    assert routing["models"]["nvidia/switchyard/gpt-5.4"]["calls"] == 2
    assert json.loads(trajectory_path.read_text()) == payload
    assert json.loads(trajectory_path.with_name("trajectory.json.bak").read_text()) == (native_trajectory)
    retry_payload = trial_payloads(
        job_dir,
        "team-ws",
        "auto",
        "scaled-evals",
        evaluation_run_id="ev_abc",
    )[0].payload
    assert json.loads(trajectory_path.read_text()) == retry_payload
    assert json.loads(trajectory_path.with_name("trajectory.json.bak").read_text()) == (native_trajectory)


def test_trial_payloads_price_switchyard_gpt_5_4_from_pinned_catalog(tmp_path: Path) -> None:
    job_dir = _sample_job(tmp_path)
    _write_switchyard_session(
        job_dir,
        {
            "session_id": "ev_abc",
            "total_calls": 2,
            "total_prompt_tokens": 1000,
            "total_cached_tokens": 600,
            "total_cache_creation_tokens": 0,
            "total_completion_tokens": 20,
            "models": {
                "nvidia/switchyard/gpt-5.4": {
                    "calls": 2,
                    "prompt_tokens": 1000,
                    "cached_tokens": 600,
                    "cache_creation_tokens": 0,
                    "completion_tokens": 20,
                }
            },
        },
    )

    payload = trial_payloads(
        job_dir,
        "team-ws",
        "auto",
        "scaled-evals",
        evaluation_run_id="ev_abc",
    )[0].payload

    assert payload["agent"]["model_name"] == "nvidia/switchyard/gpt-5.4"
    assert payload["final_metrics"]["total_cost_usd"] == pytest.approx(0.00145)
    routing = payload["extra"]["switchyard_routing"]
    assert routing["cost_status"] == "complete"
    assert routing["models"]["nvidia/switchyard/gpt-5.4"]["pricing"]["matched_model"] == ("gpt-5.4")


def test_trial_payloads_price_claude_opus_4_8_with_cache_creation(tmp_path: Path) -> None:
    job_dir = _sample_job(tmp_path)
    _write_switchyard_session(
        job_dir,
        {
            "session_id": "ev_abc",
            "total_calls": 1,
            "total_prompt_tokens": 155199,
            "total_cached_tokens": 81834,
            "total_cache_creation_tokens": 3782,
            "total_completion_tokens": 7648,
            "models": {
                "claude-opus-4-8": {
                    "calls": 1,
                    "prompt_tokens": 155199,
                    "cached_tokens": 81834,
                    "cache_creation_tokens": 3782,
                    "completion_tokens": 7648,
                }
            },
        },
    )

    payload = trial_payloads(
        job_dir,
        "team-ws",
        "auto",
        "scaled-evals",
        evaluation_run_id="ev_abc",
    )[0].payload

    assert payload["agent"]["model_name"] == "claude-opus-4-8"
    assert payload["final_metrics"]["total_cost_usd"] == pytest.approx(0.6036695)
    pricing = payload["extra"]["switchyard_routing"]["models"]["claude-opus-4-8"]["pricing"]
    assert pricing["matched_model"] == "claude-opus-4-8"
    assert pricing["cache_creation_cost_usd"] == pytest.approx(0.0236375)


def test_trial_payloads_preserve_native_model_for_mixed_routing_and_sum_model_costs(
    tmp_path: Path,
) -> None:
    job_dir = _sample_job(tmp_path)
    _write_native_atif_metrics(
        job_dir,
        model="nvidia/switchyard/gpt-5.4",
        prompt_tokens=100,
        cached_tokens=50,
        completion_tokens=10,
        cost_usd=0.00001,
    )
    _write_switchyard_session(
        job_dir,
        {
            "session_id": "ev_abc",
            "total_calls": 2,
            "total_prompt_tokens": 300,
            "total_cached_tokens": 50,
            "total_cache_creation_tokens": 0,
            "total_completion_tokens": 30,
            "models": {
                "nvidia/switchyard/gpt-5.4": {
                    "calls": 1,
                    "prompt_tokens": 100,
                    "cached_tokens": 50,
                    "cache_creation_tokens": 0,
                    "completion_tokens": 10,
                },
                "claude-opus-4-8": {
                    "calls": 1,
                    "prompt_tokens": 200,
                    "cached_tokens": 0,
                    "cache_creation_tokens": 0,
                    "completion_tokens": 20,
                },
            },
        },
    )

    payload = trial_payloads(
        job_dir,
        "team-ws",
        "auto",
        "scaled-evals",
        evaluation_run_id="ev_abc",
    )[0].payload

    assert payload["agent"]["model_name"] == "nvidia/switchyard/gpt-5.4"
    assert payload["extra"]["experiment"]["model"] == "nvidia/switchyard/gpt-5.4"
    assert payload["final_metrics"]["total_cost_usd"] == pytest.approx(0.0017875)
    assert set(payload["extra"]["switchyard_routing"]["models"]) == {
        "nvidia/switchyard/gpt-5.4",
        "claude-opus-4-8",
    }


def test_trial_payloads_keep_tokens_but_omit_cost_for_unknown_pricing(tmp_path: Path) -> None:
    job_dir = _sample_job(tmp_path)
    _write_switchyard_session(
        job_dir,
        {
            "session_id": "ev_abc",
            "total_calls": 1,
            "total_prompt_tokens": 100,
            "total_cached_tokens": 0,
            "total_cache_creation_tokens": 0,
            "total_completion_tokens": 10,
            "models": {
                "nvidia/not-in-pinned-catalog": {
                    "calls": 1,
                    "prompt_tokens": 100,
                    "cached_tokens": 0,
                    "cache_creation_tokens": 0,
                    "completion_tokens": 10,
                }
            },
        },
    )

    payload = trial_payloads(
        job_dir,
        "team-ws",
        "auto",
        "scaled-evals",
        evaluation_run_id="ev_abc",
    )[0].payload

    assert payload["agent"]["model_name"] == "nvidia/not-in-pinned-catalog"
    assert payload["final_metrics"]["total_prompt_tokens"] == 100
    assert "total_cost_usd" not in payload["final_metrics"]
    assert payload["extra"]["switchyard_routing"]["cost_status"] == "unknown_pricing"


@pytest.mark.parametrize("stats_kind", ["missing", "mismatched"])
def test_trial_payloads_do_not_hydrate_missing_or_mismatched_session_stats(tmp_path: Path, stats_kind: str) -> None:
    job_dir = _sample_job(tmp_path)
    if stats_kind == "mismatched":
        _write_switchyard_session(
            job_dir,
            {
                "session_id": "ev_abc",
                "total_calls": 2,
                "total_prompt_tokens": 999,
                "total_cached_tokens": 0,
                "total_cache_creation_tokens": 0,
                "total_completion_tokens": 1,
                "models": {
                    "nvidia/nvidia/nemotron-3-super-v3": {
                        "calls": 1,
                        "prompt_tokens": 1,
                        "cached_tokens": 0,
                        "cache_creation_tokens": 0,
                        "completion_tokens": 1,
                    }
                },
            },
        )

    payload = trial_payloads(
        job_dir,
        "team-ws",
        "auto",
        "scaled-evals",
        evaluation_run_id="ev_abc",
    )[0].payload

    assert payload["agent"]["model_name"] == "noop"
    assert "switchyard_routing" not in payload["extra"]
    assert payload["final_metrics"]["total_prompt_tokens"] == 0
    assert "total_cost_usd" not in payload["final_metrics"]


def test_trial_payloads_emit_canonical_evaluation_context(tmp_path: Path) -> None:
    job_dir = _sample_job(tmp_path)
    payloads = trial_payloads(
        job_dir,
        "team-ws",
        "auto",
        "scaled-evals",
        evaluation_run_id="ev_abc",
        evaluation_id="terminal-bench-2-noop-ev_abc",
    )

    payload = payloads[0].payload
    # Only evaluation_id + test_case_id survive ingest; no sha/run_id/metadata.
    assert payload["evaluation_context"] == {
        "evaluation_id": "terminal-bench-2-noop-ev_abc",
        "test_case_id": "task-a",
    }
    assert "experiment_context" not in payload


def test_trial_payloads_test_case_id_override(tmp_path: Path) -> None:
    job_dir = _sample_job(tmp_path)
    payloads = trial_payloads(
        job_dir,
        "team-ws",
        "auto",
        "scaled-evals",
        evaluation_run_id="ev_abc",
        evaluation_id="exp-1",
        test_case_id="hello-task",
    )

    assert payloads[0].payload["evaluation_context"] == {
        "evaluation_id": "exp-1",
        "test_case_id": "hello-task",
    }


def test_trial_payloads_do_not_stringify_structured_harbor_task_id(tmp_path: Path) -> None:
    job_dir = _sample_job(tmp_path)
    result_path = job_dir / "task-a__trial1" / "result.json"
    result = json.loads(result_path.read_text())
    result["task_id"] = {"path": "/tmp/sandbox/task"}
    result_path.write_text(json.dumps(result))

    payload = trial_payloads(
        job_dir,
        "team-ws",
        "auto",
        "scaled-evals",
        evaluation_run_id="ev_abc",
        evaluation_id="exp-1",
    )[0].payload

    assert payload["evaluation_context"]["test_case_id"] == "task-a"


def test_build_experiment_name_is_slugged_and_short() -> None:
    name = build_experiment_name("Terminal-Bench 2", "bmr_1f145b10aaaabbbb")
    assert name == "terminal-bench-2-bmr-1f145b10aaaabbbb"


def test_build_experiment_name_fits_intake_limit_for_long_benchmark_name() -> None:
    name = build_experiment_name(
        "freetona-shared-switchyard-smoke-20260722224341",
        "bmr_3697264bf4904850a24637c120",
    )

    assert len(name) <= 63
    assert name[0].isalpha()
    assert "--" not in name
    assert not name.endswith("-")
    assert name.endswith("-bmr-3697264bf4904850a24637c120")


def test_atif_ingest_url() -> None:
    url = atif_ingest_url("https://platform.example", "default")
    assert url == "https://platform.example/apis/intake/v2/workspaces/default/ingest/atif"


class _Response:
    def __init__(self, status: int, body: bytes = b"") -> None:
        self.status = status
        self.body = body

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *_args):  # noqa: ANN002, ANN204
        return False

    def read(self) -> bytes:
        return self.body


def _http_error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://platform.example/intake",
        status,
        "transient",
        hdrs=None,
        fp=io.BytesIO(b'{"detail":"try again"}'),
    )


def test_request_json_retries_transient_http_errors() -> None:
    with (
        patch(
            "scaled_evals.intake.client.urllib.request.urlopen",
            side_effect=[_http_error(503), _Response(201, b'{"id":"ok"}')],
        ) as urlopen,
        patch("scaled_evals.intake.client.time.sleep") as sleep,
    ):
        status, body = request_json("POST", "https://platform.example/intake", {}, 5)

    assert (status, body) == (201, {"id": "ok"})
    assert urlopen.call_count == 2
    assert sleep.call_args_list == [call(1.0)]


def test_request_json_retries_transport_errors_then_succeeds() -> None:
    with (
        patch(
            "scaled_evals.intake.client.urllib.request.urlopen",
            side_effect=[urllib.error.URLError("connection reset"), _Response(204)],
        ) as urlopen,
        patch("scaled_evals.intake.client.time.sleep") as sleep,
    ):
        status, body = request_json("POST", "https://platform.example/intake", {}, 5)

    assert (status, body) == (204, None)
    assert urlopen.call_count == 2
    assert sleep.call_args_list == [call(1.0)]


def test_request_json_does_not_retry_permanent_http_errors() -> None:
    with (
        patch(
            "scaled_evals.intake.client.urllib.request.urlopen",
            side_effect=_http_error(422),
        ) as urlopen,
        patch("scaled_evals.intake.client.time.sleep") as sleep,
    ):
        status, body = request_json("POST", "https://platform.example/intake", {}, 5)

    assert (status, body) == (422, {"detail": "try again"})
    assert urlopen.call_count == 1
    sleep.assert_not_called()


def test_request_json_exhausts_transport_retries_as_intake_error() -> None:
    with (
        patch(
            "scaled_evals.intake.client.urllib.request.urlopen",
            side_effect=urllib.error.URLError("offline"),
        ) as urlopen,
        patch("scaled_evals.intake.client.time.sleep") as sleep,
        pytest.raises(IntakeError, match="transport failed after 3 attempts"),
    ):
        request_json("POST", "https://platform.example/intake", {}, 5)

    assert urlopen.call_count == 3
    assert sleep.call_args_list == [call(1.0), call(2.0)]


def test_upload_warning_persists_sanitized_evaluation_diagnostic(tmp_path: Path) -> None:
    job_dir = _sample_job(tmp_path)
    target = resolve_intake_target({"workspace": "default"}, task_slug=None, base_url="https://platform.example")

    with patch(
        "scaled_evals.intake.upload.upload_job_atif",
        side_effect=IntakeError("ATIF ingest POST failed: HTTP 413; attempts=1: api_key=sk-secret123456"),
    ):
        note = upload_job_atif_warn(
            job_dir,
            target,
            evaluation_run_id="ev_test123",
            experiment=ExperimentRequest(benchmark="terminal-bench", run_key="bmr_123"),
        )

    diagnostic = json.loads((job_dir / "intake-upload.json").read_text())
    assert diagnostic["status"] == "failed"
    assert diagnostic["attempts"] == 1
    assert diagnostic["error_type"] == "IntakeError"
    assert "sk-secret123456" not in diagnostic["error"]
    assert "HTTP 413" in (note or "")


def test_create_experiment_group_reads_id_on_conflict() -> None:
    calls: list[tuple[str, str]] = []

    def fake_request_json(method, url, payload, timeout):  # noqa: ANN001
        calls.append((method, url))
        if method == "POST":
            return 409, {"detail": "exists"}
        return 200, {"id": "eg-existing"}

    with patch("scaled_evals.intake.client.request_json", fake_request_json):
        group_id = create_experiment_group("https://platform.example", "default", "terminal-bench-2", {}, 5)

    assert group_id == "eg-existing"
    assert calls[0][0] == "POST"
    assert calls[1][0] == "GET"
    assert calls[1][1].endswith("/experiments/terminal-bench-2")


def test_create_evaluation_treats_conflict_as_success() -> None:
    with patch("scaled_evals.intake.client.request_json", lambda *a: (409, {"detail": "exists"})):
        create_evaluation("https://platform.example", "default", {"name": "exp-1"}, 5)  # no raise


def test_post_atif_payload_posts_without_auth(tmp_path: Path) -> None:
    job_dir = _sample_job(tmp_path)
    payloads = trial_payloads(job_dir, "default", "auto", "scaled-evals", evaluation_run_id="ev_x")
    calls: list[tuple[str, str]] = []

    def fake_request_json(method, url, payload, timeout):  # noqa: ANN001
        calls.append((method, url))
        return 201, None

    with patch("scaled_evals.intake.client.request_json", fake_request_json):
        post_atif_payload("https://platform.example", "default", payloads[0], 5)

    assert calls[0][0] == "POST"
    assert calls[0][1].endswith("/workspaces/default/ingest/atif")


def test_upload_job_atif_creates_experiment_then_logs_trials(tmp_path: Path) -> None:
    job_dir = _sample_job(tmp_path)
    target = resolve_intake_target(
        {"workspace": "default"},
        task_slug=None,
        base_url="https://platform.example",
    )
    seen: list[tuple[str, str, dict | None]] = []

    def fake_request_json(method, url, payload, timeout):  # noqa: ANN001
        seen.append((method, url, payload))
        if url.endswith("/experiments"):
            return 201, {"id": "eg-1"}
        return 201, None

    with patch("scaled_evals.intake.client.request_json", fake_request_json):
        result = upload_job_atif(
            job_dir,
            target,
            evaluation_run_id="ev_test123",
            experiment=ExperimentRequest(benchmark="hello-task", run_key="ev_test123"),
        )

    assert result.uploaded == 1
    urls = [url for _, url, _ in seen]
    assert any(u.endswith("/experiments") for u in urls)
    assert any(u.endswith("/evaluations") for u in urls)
    # The Evaluation records which intake contract version produced it.
    exp_body = next(p for _, u, p in seen if u.endswith("/evaluations"))
    assert exp_body["metadata"]["intake_contract_ref"].startswith("nemo-helix@")
    # It belongs to its parent Experiment via the canonical experiment_ids list (not the
    # deprecated experiment_group_id scalar).
    assert exp_body["experiment_ids"] == ["eg-1"]
    assert "experiment_group_id" not in exp_body
    atif_calls = [(u, p) for m, u, p in seen if u.endswith("/ingest/atif")]
    assert len(atif_calls) == 1
    # The trial is tagged with the created experiment's stable run identity.
    assert atif_calls[0][1]["evaluation_context"]["evaluation_id"] == "hello-task-ev-test123"
    assert atif_calls[0][1]["evaluation_context"]["test_case_id"] == "task-a"


def test_upload_job_atif_records_switchyard_per_model_metrics_in_evaluation_metadata(
    tmp_path: Path,
) -> None:
    job_dir = _sample_job(tmp_path)
    _write_switchyard_session(
        job_dir,
        {
            "session_id": "ev_test123",
            "total_calls": 2,
            "total_prompt_tokens": 300,
            "total_cached_tokens": 50,
            "total_cache_creation_tokens": 0,
            "total_completion_tokens": 30,
            "models": {
                "model-a": {
                    "calls": 1,
                    "prompt_tokens": 100,
                    "cached_tokens": 50,
                    "cache_creation_tokens": 0,
                    "completion_tokens": 10,
                },
                "model-b": {
                    "calls": 1,
                    "prompt_tokens": 200,
                    "cached_tokens": 0,
                    "cache_creation_tokens": 0,
                    "completion_tokens": 20,
                },
            },
        },
    )
    target = resolve_intake_target({"workspace": "default"}, task_slug=None, base_url="https://platform.example")

    def fake_request_json(method, url, payload, timeout):  # noqa: ANN001
        if url.endswith("/experiments"):
            return 201, {"id": "eg-1"}
        if url.endswith("/evaluations"):
            assert payload is not None
            metadata = payload["metadata"]
            assert json.loads(metadata["input_tokens_by_model"]) == {"model-a": 100, "model-b": 200}
            assert json.loads(metadata["output_tokens_by_model"]) == {"model-a": 10, "model-b": 20}
            assert json.loads(metadata["cache_hit_rate_by_model"]) == {
                "model-a": 0.5,
                "model-b": 0.0,
            }
        return 201, None

    with patch("scaled_evals.intake.client.request_json", fake_request_json):
        upload_job_atif(
            job_dir,
            target,
            evaluation_run_id="ev_test123",
            experiment=ExperimentRequest(benchmark="hello-task", run_key="ev_test123"),
        )


def test_upload_job_atif_omits_partial_switchyard_per_model_metadata(tmp_path: Path) -> None:
    job_dir = _sample_job(tmp_path)
    _write_switchyard_session(
        job_dir,
        {
            "session_id": "ev_test123",
            "total_calls": 2,
            "total_prompt_tokens": 300,
            "total_cached_tokens": 50,
            "total_cache_creation_tokens": 0,
            "total_completion_tokens": 30,
            "models": {
                "model-a": {
                    "calls": 1,
                    "prompt_tokens": 100,
                    "cached_tokens": 50,
                    "cache_creation_tokens": 0,
                    "completion_tokens": 10,
                },
                "model-b": {
                    "calls": 1,
                    "prompt_tokens": 200,
                    "cached_tokens": -1,
                    "cache_creation_tokens": 0,
                    "completion_tokens": 20,
                },
            },
        },
    )
    target = resolve_intake_target({"workspace": "default"}, task_slug=None, base_url="https://platform.example")

    def fake_request_json(method, url, payload, timeout):  # noqa: ANN001
        if url.endswith("/experiments"):
            return 201, {"id": "eg-1"}
        if url.endswith("/evaluations"):
            assert payload is not None
            assert "input_tokens_by_model" not in payload["metadata"]
            assert "output_tokens_by_model" not in payload["metadata"]
            assert "cache_hit_rate_by_model" not in payload["metadata"]
        return 201, None

    with patch("scaled_evals.intake.client.request_json", fake_request_json):
        upload_job_atif(
            job_dir,
            target,
            evaluation_run_id="ev_test123",
            experiment=ExperimentRequest(benchmark="hello-task", run_key="ev_test123"),
        )


def test_upload_job_atif_groups_different_member_models_by_full_run_id(
    tmp_path: Path,
) -> None:
    first_job = _sample_job(tmp_path / "first")
    second_job = _sample_job(tmp_path / "second")
    _write_native_atif_metrics(
        first_job,
        model="claude-opus-4-8",
        prompt_tokens=10,
        cached_tokens=0,
        completion_tokens=1,
        cost_usd=0.1,
    )
    _write_native_atif_metrics(
        second_job,
        model="azure/anthropic/claude-opus-4-8",
        prompt_tokens=20,
        cached_tokens=0,
        completion_tokens=2,
        cost_usd=0.2,
    )
    target = resolve_intake_target(
        {"workspace": "default"},
        task_slug=None,
        base_url="https://platform.example",
    )
    request = ExperimentRequest(
        benchmark="terminal-bench-2-1",
        run_key="bmr_62ab9a2443e145f9848b99fff5",
    )
    seen: list[tuple[str, str, dict | None]] = []

    def fake_request_json(method, url, payload, timeout):  # noqa: ANN001
        seen.append((method, url, payload))
        if url.endswith("/experiments"):
            return 201, {"id": "eg-1"}
        return 201, None

    with patch("scaled_evals.intake.client.request_json", fake_request_json):
        for job_dir, evaluation_run_id in (
            (first_job, "ev_first"),
            (second_job, "ev_second"),
        ):
            result = upload_job_atif(
                job_dir,
                target,
                evaluation_run_id=evaluation_run_id,
                experiment=request,
            )
            assert result.uploaded == 1

    expected_id = "terminal-bench-2-1-bmr-62ab9a2443e145f9848b99fff5"
    experiment_bodies = [payload for _, url, payload in seen if url.endswith("/evaluations")]
    assert [body["name"] for body in experiment_bodies] == [expected_id, expected_id]

    atif_payloads = [payload for _, url, payload in seen if url.endswith("/ingest/atif")]
    assert {payload["evaluation_context"]["evaluation_id"] for payload in atif_payloads} == {expected_id}
    assert {payload["agent"]["model_name"] for payload in atif_payloads} == {
        "claude-opus-4-8",
        "azure/anthropic/claude-opus-4-8",
    }


def test_upload_job_atif_errored_job_without_trajectory_creates_experiment(
    tmp_path: Path,
) -> None:
    errored_job = tmp_path / "ev_errored"
    _write_json(
        errored_job / "result.json",
        {"n_total_trials": 1, "stats": {"n_completed_trials": 0, "n_errored_trials": 1}},
    )
    _write_json(
        errored_job / "task-a__trial1" / "result.json",
        {
            "id": "trial-1",
            "task_name": "task-a",
            "trial_name": "task-a__trial1",
            "exception_info": {"type": "AgentError", "message": "agent exited 1"},
        },
    )
    target = resolve_intake_target({"workspace": "default"}, task_slug=None, base_url="https://platform.example")
    seen: list[tuple[str, dict | None]] = []

    with patch(
        "scaled_evals.intake.client.request_json",
        lambda m, u, p, t: seen.append((u, p)) or (201, {"id": "eg-1"}),
    ):
        result = upload_job_atif(
            errored_job,
            target,
            evaluation_run_id="ev_errored",
            experiment=ExperimentRequest(benchmark="hello-task", run_key="ev_errored"),
        )

    assert result.uploaded == 1
    assert any(url.endswith("/experiments") for url, _ in seen)
    experiment_body = next(payload for url, payload in seen if url.endswith("/evaluations"))
    assert experiment_body["name"] == "hello-task-ev-errored"
    task_records = [payload for url, payload in seen if url.endswith("/ingest/atif")]
    assert len(task_records) == 1
    assert task_records[0]["evaluation_context"] == {
        "evaluation_id": "hello-task-ev-errored",
        "test_case_id": "task-a",
    }
    assert task_records[0]["steps"] == []
    assert task_records[0]["extra"]["trajectory_status"] == "missing"
    assert task_records[0]["extra"]["trial_result"]["exception_info"]["type"] == "AgentError"


def test_upload_job_atif_hard_failed_tasks_without_results_each_write_record(
    tmp_path: Path,
) -> None:
    target = resolve_intake_target({"workspace": "default"}, task_slug=None, base_url="https://platform.example")
    seen: list[tuple[str, dict | None]] = []

    with patch(
        "scaled_evals.intake.client.request_json",
        lambda m, u, p, t: seen.append((u, p)) or (201, {"id": "eg-1"}),
    ):
        for evaluation_run_id, test_case_id in (
            ("ev_hard_failed_a", "task-hard-failed-a"),
            ("ev_hard_failed_b", "task-hard-failed-b"),
        ):
            errored_job = tmp_path / evaluation_run_id
            errored_job.mkdir()
            result = upload_job_atif(
                errored_job,
                target,
                evaluation_run_id=evaluation_run_id,
                experiment=ExperimentRequest(benchmark="terminal-bench", run_key="bmr_123"),
                test_case_id=test_case_id,
            )
            assert result.uploaded == 1

    task_records = [payload for url, payload in seen if url.endswith("/ingest/atif")]
    assert [record["evaluation_context"] for record in task_records] == [
        {
            "evaluation_id": "terminal-bench-bmr-123",
            "test_case_id": "task-hard-failed-a",
        },
        {
            "evaluation_id": "terminal-bench-bmr-123",
            "test_case_id": "task-hard-failed-b",
        },
    ]
    assert [record["session_id"] for record in task_records] == [
        "ev_hard_failed_a",
        "ev_hard_failed_b",
    ]
    assert all(record["steps"] == [] for record in task_records)
    assert all(record["extra"]["trajectory_status"] == "missing" for record in task_records)
