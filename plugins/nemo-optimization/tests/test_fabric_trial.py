# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult, AgentEvalSummary
from nemo_evaluator_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus
from nemo_evaluator_sdk.enums import ModelFormat
from nemo_evaluator_sdk.metrics.protocol import MetricOutput
from nemo_evaluator_sdk.metrics.tunable_rag_evaluator import TunableRagEvaluatorMetric
from nemo_evaluator_sdk.values.evidence import (
    EVIDENCE_FORMAT_ATIF,
    EVIDENCE_TRACE,
    CandidateEvidence,
    EvidenceDescriptor,
)
from nemo_optimization.backends.optuna.fabric_trial import (
    FabricTrialEvaluator,
    _model_from_fabric,
    build_agent_eval_tasks,
    reduce_agent_eval_scores,
)
from nemo_optimization.backends.optuna.study_driver import StudyDriverError


def _payload(dataset: Path) -> dict[str, Any]:
    return {
        "schema_version": "fabric.agent/v1alpha1",
        "metadata": {"name": "demo"},
        "harness": {"adapter_id": "nvidia.fabric.hermes"},
        "models": {
            "default": {"provider": "openai", "model": "agent", "base_url": "http://agent/v1", "temperature": 0.0},
            "judge": {"provider": "openai", "model": "judge", "base_url": "http://judge/v1"},
        },
        "optimizer": {
            "numeric": {"enabled": True, "n_trials": 1},
            "eval_metrics": {"average_score": {"direction": "maximize"}},
            "search_space": {
                "temperature": {
                    "type": "fabric",
                    "path": "models.default.temperature",
                    "values": [0.0, 0.2],
                }
            },
        },
        "eval": {
            "general": {
                "dataset": {"file_path": str(dataset)},
                "max_concurrency": 1,
            },
            "fabric": {
                "profiles": [{"schema_version": "fabric.profile/v1alpha1", "metadata": {"name": "base"}}],
                "capture_trajectory": True,
                "timeout_s": 30,
            },
            "evaluators": {
                "accuracy": {
                    "_type": "tunable_rag_evaluator",
                    "llm_name": "judge",
                    "default_scoring": True,
                    "judge_llm_prompt": "",
                }
            },
        },
    }


def test_build_agent_eval_tasks_from_json_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "rows.json"
    dataset.write_text('[{"id": "1", "question": "q?", "answer": "a"}]\n', encoding="utf-8")

    tasks = build_agent_eval_tasks(_payload(dataset))

    assert len(tasks) == 1
    assert tasks[0].id == "1"
    assert tasks[0].inputs == {"instruction": "q?"}
    assert tasks[0].reference == {"answer": "a"}
    assert isinstance(tasks[0].metrics[0], TunableRagEvaluatorMetric)
    assert tasks[0].metrics[0].model.format == ModelFormat.OPEN_AI


def test_model_from_fabric_maps_providers_to_model_format() -> None:
    payload = {
        "models": {
            "openai_judge": {
                "provider": "openai",
                "model": "gpt",
                "base_url": "http://judge/v1",
            },
            "nim_judge": {
                "provider": "nim",
                "model": "nim-model",
                "url": "http://nim/v1",
            },
            "nvidia_judge": {
                "provider": "nvidia",
                "model": "nv-model",
                "base_url": "http://nv/v1",
            },
        }
    }
    assert _model_from_fabric(payload, "openai_judge").format == ModelFormat.OPEN_AI
    assert _model_from_fabric(payload, "nim_judge").format == ModelFormat.NVIDIA_NIM
    assert _model_from_fabric(payload, "nvidia_judge").format == ModelFormat.OPEN_AI


def test_model_from_fabric_rejects_unknown_provider() -> None:
    payload = {
        "models": {
            "judge": {
                "provider": "anthropic",
                "model": "claude",
                "base_url": "http://judge/v1",
            }
        }
    }
    with pytest.raises(StudyDriverError, match="unsupported provider 'anthropic'"):
        _model_from_fabric(payload, "judge")


def test_build_agent_eval_tasks_accepts_body_label(tmp_path: Path) -> None:
    dataset = tmp_path / "rows.json"
    dataset.write_text(
        '[{"id": "mail-1", "body": "Send password now", "label": "phishing"}]\n',
        encoding="utf-8",
    )

    tasks = build_agent_eval_tasks(_payload(dataset))

    assert tasks[0].inputs == {"instruction": "Send password now"}
    assert tasks[0].reference == {"answer": "phishing"}


def test_build_agent_eval_tasks_composes_subject_and_body(tmp_path: Path) -> None:
    dataset = tmp_path / "rows.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "mail-1",
                    "subject": "Urgent",
                    "body": "Click this link",
                    "label": "phishing",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    tasks = build_agent_eval_tasks(_payload(dataset))

    assert tasks[0].inputs == {"instruction": "Urgent\n\nClick this link"}
    assert tasks[0].reference == {"answer": "phishing"}


def test_build_agent_eval_tasks_preserves_judge_api_key_env(tmp_path: Path) -> None:
    dataset = tmp_path / "rows.json"
    dataset.write_text('[{"id": "1", "question": "q?", "answer": "a"}]\n', encoding="utf-8")
    payload = _payload(dataset)
    payload["models"]["judge"]["api_key_env"] = "NVIDIA_API_KEY"

    tasks = build_agent_eval_tasks(payload)

    metric = tasks[0].metrics[0]
    assert isinstance(metric, TunableRagEvaluatorMetric)
    assert metric.model.api_key_secret is not None
    assert metric.model.api_key_secret.root == "NVIDIA_API_KEY"


def test_reduce_agent_eval_scores_averages_requested_output() -> None:
    scores = [
        AgentEvalTaskScore(
            id="s1",
            run_id="r",
            task_id="1",
            trial_id="t1",
            metric_type="tunable-rag-evaluator",
            status=AgentEvalScoreStatus.COMPLETED,
            outputs=[MetricOutput(name="average_score", value=0.25)],
        ),
        AgentEvalTaskScore(
            id="s2",
            run_id="r",
            task_id="2",
            trial_id="t2",
            metric_type="tunable-rag-evaluator",
            status=AgentEvalScoreStatus.COMPLETED,
            outputs=[MetricOutput(name="average_score", value=0.75)],
        ),
    ]

    assert reduce_agent_eval_scores(scores, ["average_score"]) == {"average_score": 0.5}


def test_reduce_agent_eval_scores_skips_failed_task_scores() -> None:
    """One failed dataset row must not fail the Optuna trial reduction."""
    scores = [
        AgentEvalTaskScore(
            id="s1",
            run_id="r",
            task_id="ok",
            trial_id="t1",
            metric_type="tunable-rag-evaluator",
            status=AgentEvalScoreStatus.COMPLETED,
            outputs=[MetricOutput(name="average_score", value=1.0)],
        ),
        AgentEvalTaskScore(
            id="s2",
            run_id="r",
            task_id="urgent-your-account-has-been-suspended",
            trial_id="t2",
            metric_type="tunable-rag-evaluator",
            status=AgentEvalScoreStatus.FAILED,
            outputs=[],
            diagnostics=[],
        ),
    ]

    assert reduce_agent_eval_scores(scores, ["average_score"]) == {"average_score": 1.0}


def test_reduce_agent_eval_scores_rejects_when_all_failed() -> None:
    scores = [
        AgentEvalTaskScore(
            id="s1",
            run_id="r",
            task_id="1",
            trial_id="t1",
            metric_type="tunable-rag-evaluator",
            status=AgentEvalScoreStatus.FAILED,
            outputs=[],
        ),
    ]
    with pytest.raises(StudyDriverError, match="did not produce"):
        reduce_agent_eval_scores(scores, ["average_score"])


def test_reduce_agent_eval_scores_rejects_missing_metric() -> None:
    with pytest.raises(StudyDriverError, match="did not produce"):
        reduce_agent_eval_scores([], ["average_score"])


def test_fabric_trial_evaluator_invokes_agent_evaluator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = tmp_path / "rows.json"
    dataset.write_text('[{"id": "1", "question": "q?", "answer": "a"}]\n', encoding="utf-8")
    captured: dict[str, Any] = {}

    class FakeRuntime:
        def __init__(self, **kwargs: Any) -> None:
            captured["runtime"] = kwargs

    class FakeAgentEvaluator:
        def run_sync(self, *, tasks, target, config):  # noqa: ANN001
            captured["tasks"] = tasks
            captured["target"] = target
            captured["config"] = config
            trial = AgentEvalTrial(
                id="1:fabric",
                task_id="1",
                status=AgentEvalTrialStatus.COMPLETED,
                evidence=CandidateEvidence(
                    descriptors={
                        EVIDENCE_TRACE: EvidenceDescriptor(
                            kind=EVIDENCE_TRACE,
                            format=EVIDENCE_FORMAT_ATIF,
                            ref="/tmp/trace.atif.json",
                        )
                    }
                ),
                output={"output_text": "answer"},
            )
            score = AgentEvalTaskScore(
                id="s",
                run_id="r",
                task_id="1",
                trial_id="1:fabric",
                metric_type="tunable-rag-evaluator",
                status=AgentEvalScoreStatus.COMPLETED,
                outputs=[MetricOutput(name="average_score", value=0.9)],
            )
            return AgentEvalResult(
                run_id="r",
                tasks=list(tasks),
                trials=[trial],
                scores=[score],
                summary=AgentEvalSummary.from_scores([score], tasks=tasks),
                # The real evaluator carries the config's work_dir onto the result; the trial
                # evaluator persists into it, so the fake has to do the same to stay faithful.
                work_dir=config.work_dir,
            )

    monkeypatch.setattr("nemo_optimization.backends.optuna.fabric_trial.FabricAgentRuntime", FakeRuntime)
    monkeypatch.setattr("nemo_optimization.backends.optuna.fabric_trial.AgentEvaluator", FakeAgentEvaluator)

    evaluator = FabricTrialEvaluator(
        payload=_payload(dataset),
        metric_names=["average_score"],
        output_dir=tmp_path / "out",
        experiment_id="exp-test",
    )

    scores = evaluator.evaluate(
        trial_number=7,
        suggestions={"models.default.temperature": 0.2},
        trial_overlay={"metadata": {"name": "trial-007"}},
        rep=0,
    )

    assert scores == {"average_score": 0.9}
    assert captured["runtime"]["trajectory_extra"] == {
        "nemo.optimizer.experiment_id": "exp-test",
        "nemo.optimizer.trial_number": 7,
        "nemo.optimizer.rep": 0,
    }
    assert "profiles" not in captured["runtime"]
    assert captured["runtime"]["config"]["models"]["default"]["temperature"] == 0.2
    assert "optimizer" not in captured["runtime"]["config"]
    assert "eval" not in captured["runtime"]["config"]
    assert (tmp_path / "out" / "trial_trace_map.json").is_file()
    # Storing is now an explicit persist() call, so assert the per-trial bundle still lands.
    bundle = tmp_path / "out" / "agent_eval" / "trial-007" / "rep-000"
    assert (bundle / "run.json").is_file()
    assert not (bundle / "report.html").exists()  # write_dashboard=False
    trace_map = json.loads((tmp_path / "out" / "trial_trace_map.json").read_text(encoding="utf-8"))
    assert trace_map[0]["experiment_id"] == "exp-test"
    assert trace_map[0]["row_id"] == "1"


def test_fabric_trial_evaluator_rejects_a_removed_run_hook(tmp_path: Path) -> None:
    """A config written for the retired per-task hook must fail at construction, not run hook-less.

    Silently dropping ``eval.run_hook`` would score an agent whose MCP binding was never set up
    and report it as the tuned configuration's result.
    """
    dataset = tmp_path / "rows.json"
    dataset.write_text('[{"id": "1", "question": "q?", "answer": "a"}]\n', encoding="utf-8")
    payload = _payload(dataset)
    payload["eval"]["run_hook"] = {"type": "mcp_run_binding", "bindings": []}

    with pytest.raises(StudyDriverError, match="eval.run_hook is no longer supported"):
        FabricTrialEvaluator(
            payload=payload, metric_names=["average_score"], output_dir=tmp_path / "out", experiment_id="exp"
        )


def test_build_metrics_rejects_a_tool_call_count_evaluator_without_a_tool_name() -> None:
    from nemo_optimization.backends.optuna.fabric_trial import _build_metrics

    with pytest.raises(StudyDriverError, match="requires a non-empty tool_name"):
        _build_metrics({}, {"evaluators": {"once": {"_type": "tool_call_count", "expected_calls": 1}}})


def test_build_metrics_rejects_a_negative_tool_call_count_expectation() -> None:
    from nemo_optimization.backends.optuna.fabric_trial import _build_metrics

    with pytest.raises(StudyDriverError, match="non-negative integer"):
        _build_metrics(
            {}, {"evaluators": {"once": {"_type": "tool_call_count", "tool_name": "t", "expected_calls": -1}}}
        )


@pytest.mark.parametrize("value", [1.9, "1", True])
def test_build_metrics_rejects_a_non_integer_tool_call_count_expectation(value: object) -> None:
    from nemo_optimization.backends.optuna.fabric_trial import _build_metrics

    with pytest.raises(StudyDriverError, match="non-negative integer"):
        _build_metrics(
            {}, {"evaluators": {"once": {"_type": "tool_call_count", "tool_name": "t", "expected_calls": value}}}
        )


def test_resolve_mcp_server_paths_absolutizes_only_bundle_files(tmp_path: Path) -> None:
    from nemo_optimization.backends.optuna.fabric_trial import resolve_mcp_server_paths

    (tmp_path / "mcps").mkdir()
    (tmp_path / "mcps" / "server.py").write_text("print(1)\n", encoding="utf-8")
    config: dict[str, Any] = {
        "mcp": {
            "servers": {
                "bundled": {
                    "transport": "stdio",
                    "url": "python3",
                    "args": ["mcps/server.py", "--verbose", "missing.py"],
                },
                "script": {"transport": "stdio", "url": "mcps/server.py"},
                "remote": {"transport": "http", "url": "mcps/server.py"},
            }
        }
    }
    resolve_mcp_server_paths(config, root=tmp_path)

    servers = config["mcp"]["servers"]
    assert servers["bundled"]["url"] == "python3"  # a command on PATH is not a bundle file
    assert servers["bundled"]["args"] == [str(tmp_path.resolve() / "mcps" / "server.py"), "--verbose", "missing.py"]
    assert servers["script"]["url"] == str(tmp_path.resolve() / "mcps" / "server.py")
    assert servers["remote"]["url"] == "mcps/server.py"  # only stdio servers are launched from a path


def test_build_metrics_accepts_tool_argument_matches_input_and_rejects_bad_normalize() -> None:
    from nemo_evaluator_sdk.agent_eval.metrics import ToolArgumentMatchesInputMetric
    from nemo_optimization.backends.optuna.fabric_trial import _build_metrics

    (metric,) = _build_metrics(
        {},
        {"evaluators": {"v": {"_type": "tool_argument_matches_input", "tool_name": "t", "input_key": "email"}}},
    )
    assert isinstance(metric, ToolArgumentMatchesInputMetric)
    assert (metric.tool_name, metric.argument, metric.input_key, metric.normalize) == (
        "t",
        "text",
        "email",
        "whitespace",
    )
    with pytest.raises(StudyDriverError, match="tool_argument_matches_input evaluator is invalid"):
        _build_metrics(
            {}, {"evaluators": {"v": {"_type": "tool_argument_matches_input", "tool_name": "t", "normalize": "fuzzy"}}}
        )
    with pytest.raises(StudyDriverError, match="requires a non-empty tool_name"):
        _build_metrics({}, {"evaluators": {"v": {"_type": "tool_argument_matches_input"}}})
