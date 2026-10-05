# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ATIF v1.7 domain and mapper tests."""

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from nhx.intake.spans.api.spans_schemas import Span
from nhx.intake.spans.domain import NEMO_STEP_ID_ATTRIBUTE, SpanKind, SpanStatus
from nhx.intake.spans.ingest.atif import AtifIngestRequest
from nhx.intake.spans.ingest.atif_domain import (
    AtifAgent,
    AtifStepAgent,
    AtifStepUser,
    AtifSubagentTrajectoryRef,
    AtifTrajectory,
)
from nhx.intake.spans.ingest.atif_mapping import AtifTrajectoryDepthError, _step_observation, trajectory_to_spans
from nhx.intake.spans.ingest.evaluation_context import EvaluationContext
from pydantic import ValidationError

EVALUATION_CONTEXT: dict[str, Any] = {
    "evaluation_name": "eval-sample-agent-baseline",
    "test_case_name": "sample-test-case",
}


def _nested_trajectory(depth: int) -> AtifTrajectory:
    """Build a timestamp-free ATIF trajectory tree with the requested depth."""
    payload: dict[str, Any] = {
        "schema_version": "ATIF-v1.7",
        "trajectory_id": f"trajectory-{depth}",
        "agent": {"name": f"agent-{depth}", "version": "1.0"},
        "steps": [],
    }
    for level in reversed(range(1, depth)):
        payload = {
            "schema_version": "ATIF-v1.7",
            "trajectory_id": f"trajectory-{level}",
            "agent": {"name": f"agent-{level}", "version": "1.0"},
            "steps": [],
            "subagent_trajectories": [payload],
        }
    payload["session_id"] = "depth-run"
    return AtifTrajectory.model_validate(payload)


def test_atif_v17_models_accept_new_fields() -> None:
    trajectory = AtifTrajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "trace-session-id",
            "trajectory_id": "root-trajectory",
            "agent": {"name": "sample-agent", "version": "1.0.0"},
            "steps": [
                {
                    "step_id": 1,
                    "source": "agent",
                    "message": "delegating",
                    "llm_call_count": 1,
                    "observation": {
                        "results": [
                            {
                                "content": "delegated",
                                "extra": {"retrieval_score": 0.92},
                                "subagent_trajectory_ref": [
                                    {"trajectory_id": "sub-trajectory", "session_id": "trace-session-id"}
                                ],
                            }
                        ]
                    },
                }
            ],
            "subagent_trajectories": [
                {
                    "schema_version": "ATIF-v1.7",
                    "session_id": "trace-session-id",
                    "trajectory_id": "sub-trajectory",
                    "agent": {"name": "subagent", "version": "1.0"},
                    "steps": [],
                }
            ],
        }
    )

    assert trajectory.schema_version == "ATIF-v1.7"
    assert trajectory.trajectory_id == "root-trajectory"
    assert trajectory.steps[0].llm_call_count == 1
    assert trajectory.subagent_trajectories is not None
    assert trajectory.subagent_trajectories[0].trajectory_id == "sub-trajectory"

    request = AtifIngestRequest.model_validate(trajectory.to_json_dict())
    request_trajectory = request.to_trajectory()
    assert request_trajectory.trajectory_id == "root-trajectory"
    assert request_trajectory.subagent_trajectories is not None
    assert request_trajectory.subagent_trajectories[0].trajectory_id == "sub-trajectory"


# ATIF v1.7 reference semantics:
# https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
def test_atif_v17_embedded_subagents_expand_recursively_in_the_parent_trace() -> None:
    trajectory = AtifTrajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "run-123",
            "trajectory_id": "root-trajectory",
            "agent": {"name": "root-agent", "version": "1.0"},
            "steps": [
                {
                    "step_id": 1,
                    "timestamp": "2026-05-18T10:00:00Z",
                    "source": "user",
                    "message": "Research and review this answer.",
                },
                {
                    "step_id": 2,
                    "timestamp": "2026-05-18T10:00:01Z",
                    "source": "agent",
                    "message": "Delegating research.",
                    "observation": {
                        "results": [
                            {
                                "content": "Research completed.",
                                "subagent_trajectory_ref": [
                                    {"trajectory_id": "research-trajectory", "session_id": "run-123"},
                                    {
                                        "trajectory_path": "subagents/external-trajectory.json",
                                        "session_id": "run-123",
                                    },
                                ],
                            }
                        ]
                    },
                },
                {
                    "step_id": 3,
                    "timestamp": "2026-05-18T10:00:06Z",
                    "source": "agent",
                    "message": "Here is the reviewed answer.",
                },
            ],
            "subagent_trajectories": [
                {
                    "schema_version": "ATIF-v1.7",
                    "trajectory_id": "research-trajectory",
                    "agent": {"name": "research-agent", "version": "1.0"},
                    "steps": [
                        {
                            "step_id": 1,
                            "timestamp": "2026-05-18T10:00:02Z",
                            "source": "user",
                            "message": "Research the answer.",
                        },
                        {
                            "step_id": 2,
                            "timestamp": "2026-05-18T10:00:03Z",
                            "source": "agent",
                            "message": "Delegating review.",
                            "observation": {
                                "results": [
                                    {
                                        "subagent_trajectory_ref": [
                                            {
                                                "trajectory_id": "review-trajectory",
                                                "trajectory_path": "subagents/review-trajectory.json",
                                                "session_id": "run-123",
                                            }
                                        ]
                                    }
                                ]
                            },
                        },
                    ],
                    "subagent_trajectories": [
                        {
                            "schema_version": "ATIF-v1.7",
                            "session_id": "run-123",
                            "trajectory_id": "review-trajectory",
                            "agent": {"name": "review-agent", "version": "1.0"},
                            "steps": [
                                {
                                    "step_id": 1,
                                    "timestamp": "2026-05-18T10:00:04Z",
                                    "source": "user",
                                    "message": "Review the research.",
                                },
                                {
                                    "step_id": 2,
                                    "timestamp": "2026-05-18T10:00:08Z",
                                    "source": "agent",
                                    "message": "The research is invalid.",
                                    "tool_calls": [
                                        {
                                            "tool_call_id": "validate-1",
                                            "function_name": "validate_research",
                                            "arguments": {},
                                        }
                                    ],
                                    "observation": {
                                        "results": [
                                            {
                                                "source_call_id": "validate-1",
                                                "content": "[error] unsupported claim",
                                            }
                                        ]
                                    },
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    )

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
    )

    root = next(span for span in spans if span.name == "root-agent")
    root_delegation = next(
        span for span in spans if span.name == "root-agent" and span.external_parent_span_id == root.external_span_id
    )
    research = next(span for span in spans if span.name == "research-agent")
    research_delegation = next(
        span
        for span in spans
        if span.name == "research-agent" and span.external_parent_span_id == research.external_span_id
    )
    review = next(span for span in spans if span.name == "review-agent")
    failed_tool = next(span for span in spans if span.name == "validate_research")
    external_ref = next(span for span in spans if span.name == "subagent-subagents/external-trajectory.json")

    assert len(spans) == 12
    assert len({span.external_span_id for span in spans}) == len(spans)
    assert {span.trace_id for span in spans} == {"run-123"}
    assert {span.session_id for span in spans} == {"run-123"}
    assert root.external_parent_span_id == ""
    assert research.external_parent_span_id == root_delegation.external_span_id
    assert review.external_parent_span_id == research_delegation.external_span_id
    assert external_ref.external_parent_span_id == root_delegation.external_span_id
    assert not any(span.name == "subagent-subagents/review-trajectory.json" for span in spans)
    assert root.start_time == datetime(2026, 5, 18, 10, tzinfo=timezone.utc)
    assert root.end_time == datetime(2026, 5, 18, 10, 0, 8, tzinfo=timezone.utc)
    # A tool error stays on its TOOL span. No trajectory AGENT span inherits it,
    # regardless of where that trajectory sits in the embedded subagent tree.
    assert root.status == SpanStatus.SUCCESS
    assert research.status == SpanStatus.SUCCESS
    assert review.status == SpanStatus.SUCCESS
    assert failed_tool.status == SpanStatus.ERROR

    root_raw = json.loads(root.attributes_string["atif.raw"])
    research_raw = json.loads(research.attributes_string["atif.raw"])
    assert "subagent_trajectories" not in root_raw
    assert research_raw["trajectory_id"] == "research-trajectory"
    assert "subagent_trajectories" not in research_raw


def test_atif_v17_subagent_depth_is_bounded_before_mapping() -> None:
    trajectory = _nested_trajectory(depth=3)

    with pytest.raises(
        AtifTrajectoryDepthError,
        match="ATIF trajectory depth 3 exceeds configured maximum 2",
    ):
        trajectory_to_spans(
            workspace="default",
            trajectory=trajectory,
            ingested_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
            max_subagent_depth=2,
        )

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
        max_subagent_depth=3,
    )
    assert [span.name for span in spans] == ["agent-1", "agent-2", "agent-3"]


def test_atif_v17_timestamp_free_subagent_starts_at_delegating_step() -> None:
    ingested_at = datetime(2026, 5, 18, tzinfo=timezone.utc)
    trajectory = AtifTrajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "fallback-run",
            "trajectory_id": "root-trajectory",
            "agent": {"name": "root-agent", "version": "1.0"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "Plan first."},
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "Delegate second.",
                    "observation": {
                        "results": [
                            {
                                "subagent_trajectory_ref": [
                                    {"trajectory_id": "worker-trajectory", "session_id": "fallback-run"}
                                ]
                            }
                        ]
                    },
                },
            ],
            "subagent_trajectories": [
                {
                    "schema_version": "ATIF-v1.7",
                    "trajectory_id": "worker-trajectory",
                    "agent": {"name": "worker-agent", "version": "1.0"},
                    "steps": [{"step_id": 1, "source": "user", "message": "Do the work."}],
                }
            ],
        }
    )

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=ingested_at,
    )
    delegation = next(span for span in spans if span.name == "root-agent" and span.kind == SpanKind.LLM)
    worker = next(span for span in spans if span.name == "worker-agent")
    worker_step = next(
        span for span in spans if span.name == "user-1" and span.external_parent_span_id == worker.external_span_id
    )

    assert delegation.start_time == ingested_at + timedelta(milliseconds=1)
    assert worker.start_time == delegation.start_time
    assert worker_step.start_time == delegation.start_time


def test_atif_v17_sibling_subagents_can_share_a_session_without_span_id_collisions() -> None:
    trajectory = AtifTrajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "shared-run",
            "agent": {"name": "orchestrator", "version": "1.0"},
            "steps": [
                {
                    "step_id": 1,
                    "source": "agent",
                    "message": "Dispatch both workers.",
                    "observation": {
                        "results": [
                            {
                                "subagent_trajectory_ref": [
                                    {"trajectory_id": "worker-a", "session_id": "shared-run"},
                                    {"trajectory_id": "worker-b", "session_id": "shared-run"},
                                ]
                            }
                        ]
                    },
                }
            ],
            "subagent_trajectories": [
                {
                    "schema_version": "ATIF-v1.7",
                    "session_id": "shared-run",
                    "trajectory_id": "worker-a",
                    "agent": {"name": "worker", "version": "1.0"},
                    "steps": [{"step_id": 1, "source": "user", "message": "work"}],
                },
                {
                    "schema_version": "ATIF-v1.7",
                    "session_id": "shared-run",
                    "trajectory_id": "worker-b",
                    "agent": {"name": "worker", "version": "1.0"},
                    "steps": [{"step_id": 1, "source": "user", "message": "work"}],
                },
            ],
        }
    )

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
    )

    worker_roots = [span for span in spans if span.name == "worker"]
    worker_steps = [
        span
        for span in spans
        if span.name == "user-1" and span.external_parent_span_id in {s.external_span_id for s in worker_roots}
    ]
    assert len(worker_roots) == 2
    assert len(worker_steps) == 2
    assert len({span.external_span_id for span in [*worker_roots, *worker_steps]}) == 4
    assert {span.trace_id for span in spans} == {"shared-run"}


def test_atif_v17_subagent_ref_requires_resolution_key() -> None:
    with pytest.raises(ValidationError, match="trajectory_id, trajectory_path, or session_id"):
        AtifSubagentTrajectoryRef()

    assert AtifSubagentTrajectoryRef(session_id="trace-session-id").session_id == "trace-session-id"

    with pytest.raises(ValidationError, match="trajectory_id or trajectory_path"):
        AtifTrajectory.model_validate(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "trace-session-id",
                "agent": {"name": "root", "version": "1.0"},
                "steps": [
                    {
                        "step_id": 1,
                        "source": "agent",
                        "message": "delegating",
                        "observation": {
                            "results": [
                                {"subagent_trajectory_ref": [{"session_id": "trace-session-id"}]},
                            ]
                        },
                    }
                ],
            }
        )

    legacy = AtifTrajectory.model_validate(
        {
            "schema_version": "ATIF-v1.6",
            "session_id": "trace-session-id",
            "agent": {"name": "root", "version": "1.0"},
            "steps": [
                {
                    "step_id": 1,
                    "source": "agent",
                    "message": "delegating",
                    "observation": {
                        "results": [
                            {"subagent_trajectory_ref": [{"session_id": "trace-session-id"}]},
                        ]
                    },
                }
            ],
        }
    )
    assert isinstance(legacy.steps[0], AtifStepAgent)
    assert legacy.steps[0].observation is not None
    legacy_refs = legacy.steps[0].observation.results[0].subagent_trajectory_ref
    assert legacy_refs is not None
    legacy_ref = legacy_refs[0]
    assert legacy_ref.session_id == "trace-session-id"

    assert AtifSubagentTrajectoryRef(trajectory_id="sub-trajectory").trajectory_id == "sub-trajectory"
    assert AtifSubagentTrajectoryRef(trajectory_path="subagents/sub-trajectory.json").trajectory_path is not None


def test_evaluation_context_normalizes_deprecated_fields_and_ignores_retired_fields() -> None:
    # Retired keys (evaluation_sha, evaluation_run_id, metadata) are accepted and dropped rather
    # than rejected, so stale producers keep ingesting without ingest erroring on unknown fields.
    context = EvaluationContext.model_validate(
        {
            "evaluation_id": "eval-1",
            "test_case_id": "case-1",
            "evaluation_sha": "abc132901",
            "evaluation_run_id": "evalrun-1",
            "metadata": {"attempt": 1},
        }
    )
    assert context.evaluation_name == "eval-1"
    assert context.test_case_name == "case-1"
    assert context.model_dump() == {
        "evaluation_name": "eval-1",
        "test_case_name": "case-1",
        "evaluation_id": "eval-1",
        "test_case_id": "case-1",
    }


def test_atif_ingest_request_rejects_legacy_top_level_project() -> None:
    body = {
        "schema_version": "ATIF-v1.7",
        "session_id": "trace-session-id",
        "project": "legacy-project",
        "agent": {"name": "sample-agent", "version": "1.0.0"},
        "steps": [],
    }

    with pytest.raises(ValidationError) as exc_info:
        AtifIngestRequest.model_validate(body)

    error_locations = {tuple(error["loc"]) for error in exc_info.value.errors()}
    assert ("project",) in error_locations


def test_atif_mapping_writes_evaluation_context_only_on_root_span() -> None:
    trajectory = AtifTrajectory(
        schema_version="ATIF-v1.5",
        session_id="trace-session-id",
        evaluation_context=EvaluationContext.model_validate(EVALUATION_CONTEXT),
        agent=AtifAgent(
            name="sample-agent",
            version="1.0.0",
            model_name="provider/sample-model",
        ),
        steps=[AtifStepUser(step_id=1, source="user", message="solve")],
    )

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
    )

    root = next(span for span in spans if span.name == "sample-agent")
    child = next(span for span in spans if span.name == "user-1")
    assert root.attributes_string["nemo.evaluation.name"] == EVALUATION_CONTEXT["evaluation_name"]
    # sha/run_id/metadata are dropped by the trimmed ingest EvaluationContext, so they never
    # reach the span from the JSON evaluation_context path.
    assert "nemo.experiment.sha" not in root.attributes_string
    assert "nemo.experiment.run_id" not in root.attributes_string
    assert "evaluation.id" not in root.attributes_string
    assert root.attributes_string["nemo.test_case.name"] == EVALUATION_CONTEXT["test_case_name"]
    assert "nemo.experiment.metadata" not in root.attributes_string

    root_response = Span.from_domain(root)
    assert root_response.evaluation_context is not None
    assert root_response.evaluation_context.evaluation_name == EVALUATION_CONTEXT["evaluation_name"]
    assert root_response.evaluation_context.test_case_name == EVALUATION_CONTEXT["test_case_name"]
    assert root_response.raw_attributes is not None
    root_raw = json.loads(root_response.raw_attributes)
    assert "evaluation_context" not in root_raw
    assert "evaluation.metadata" not in root_raw
    assert "nemo.experiment.metadata" not in root_raw

    child_response = Span.from_domain(child)
    assert child_response.evaluation_context is None
    assert "evaluation.id" not in child.attributes_string
    assert "nemo.evaluation.name" not in child.attributes_string
    assert "nemo.test_case.name" not in child.attributes_string


def test_atif_mapping_persists_step_order_when_timestamps_match() -> None:
    timestamp = datetime(2026, 8, 21, 13, 32, 49, 843000, tzinfo=timezone.utc)
    trajectory = AtifTrajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "trace-session-id",
            "agent": {"name": "sample-agent", "version": "1.0.0"},
            "steps": [
                {"step_id": 1, "timestamp": timestamp.isoformat(), "source": "system", "message": "rules"},
                {"step_id": 2, "timestamp": timestamp.isoformat(), "source": "user", "message": "task"},
            ],
        }
    )

    spans = trajectory_to_spans(workspace="default", trajectory=trajectory, ingested_at=timestamp)
    steps = [span for span in spans if NEMO_STEP_ID_ATTRIBUTE in span.attributes_number]

    assert [(span.name, span.start_time, span.attributes_number[NEMO_STEP_ID_ATTRIBUTE]) for span in steps] == [
        ("system-1", timestamp, 1),
        ("user-2", timestamp, 2),
    ]


def test_atif_mapping_uses_root_final_metrics_when_steps_have_no_metrics() -> None:
    trajectory = AtifTrajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "trace-session-id",
            "agent": {"name": "sample-agent", "version": "1.0.0"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "solve the task"},
                {"step_id": 2, "source": "agent", "message": "final answer"},
            ],
            "final_metrics": {
                "total_prompt_tokens": 10,
                "total_completion_tokens": 5,
                "total_cached_tokens": 2,
                "total_cost_usd": 0.25,
            },
        }
    )

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
    )

    root_response = Span.from_domain(spans[0])
    child_response = Span.from_domain(
        next(span for span in spans if span.name == "sample-agent" and span.kind == SpanKind.LLM)
    )
    assert root_response.input_tokens == 10
    assert root_response.output_tokens == 5
    assert root_response.cached_tokens == 2
    assert root_response.total_tokens == 15
    assert root_response.cost_total_usd == 0.25
    assert child_response.input_tokens is None
    assert child_response.cost_total_usd is None


def test_atif_mapping_does_not_duplicate_final_metrics_when_step_metrics_exist() -> None:
    trajectory = AtifTrajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "trace-session-id",
            "agent": {"name": "sample-agent", "version": "1.0.0"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "solve the task"},
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "final answer",
                    "metrics": {"prompt_tokens": 3, "completion_tokens": 2, "cost_usd": 0.04},
                },
            ],
            "final_metrics": {
                "total_prompt_tokens": 10,
                "total_completion_tokens": 5,
                "total_cost_usd": 0.25,
            },
        }
    )

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
    )

    root_response = Span.from_domain(spans[0])
    child_response = Span.from_domain(
        next(span for span in spans if span.name == "sample-agent" and span.kind == SpanKind.LLM)
    )
    assert root_response.input_tokens is None
    assert root_response.output_tokens is None
    assert root_response.cost_total_usd is None
    assert child_response.input_tokens == 3
    assert child_response.output_tokens == 2
    assert child_response.total_tokens == 5
    assert child_response.cost_total_usd == 0.04


def test_atif_mapping_uses_root_cost_when_step_metrics_have_tokens_only() -> None:
    trajectory = AtifTrajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "trace-session-id",
            "agent": {"name": "sample-agent", "version": "1.0.0"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "solve the task"},
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "final answer",
                    "metrics": {"prompt_tokens": 3, "completion_tokens": 2, "cached_tokens": 1},
                },
            ],
            "final_metrics": {
                "total_prompt_tokens": 10,
                "total_completion_tokens": 5,
                "total_cached_tokens": 2,
                "total_cost_usd": 0.25,
            },
        }
    )

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
    )

    root_response = Span.from_domain(spans[0])
    child_response = Span.from_domain(
        next(span for span in spans if span.name == "sample-agent" and span.kind == SpanKind.LLM)
    )
    assert root_response.input_tokens is None
    assert root_response.output_tokens is None
    assert root_response.cached_tokens is None
    assert root_response.cost_total_usd == 0.25
    assert child_response.input_tokens == 3
    assert child_response.output_tokens == 2
    assert child_response.cached_tokens == 1
    assert child_response.cost_total_usd is None


def test_atif_mapping_keeps_tool_error_on_tool_span() -> None:
    trajectory = AtifTrajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "trace-session-id",
            "agent": {"name": "sample-agent", "version": "1.0.0"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "solve the task"},
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "using a tool",
                    "tool_calls": [{"tool_call_id": "call-1", "function_name": "Bash"}],
                    "observation": {
                        "results": [
                            {
                                "source_call_id": "call-1",
                                "content": "Exit code 1\n[error] failed",
                            }
                        ]
                    },
                },
                {"step_id": 3, "source": "agent", "message": "final answer"},
            ],
        }
    )

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
    )

    root = spans[0]
    llm = next(span for span in spans if span.kind == SpanKind.LLM)
    tool = next(span for span in spans if span.kind == SpanKind.TOOL)
    assert root.name == "sample-agent"
    assert root.input == "solve the task"
    assert root.output == "final answer"
    assert root.status == SpanStatus.SUCCESS
    assert llm.status == SpanStatus.SUCCESS
    assert tool.status == SpanStatus.ERROR


def test_atif_mapping_promotes_root_extra_error() -> None:
    trajectory = AtifTrajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "trace-session-id",
            "agent": {"name": "sample-agent", "version": "1.0.0"},
            "extra": {
                "error": {
                    "type": "AgentTimeoutError",
                    "message": "Agent execution timed out after 900.0 seconds",
                }
            },
            "steps": [
                {"step_id": 1, "source": "user", "message": "solve the task"},
                {"step_id": 2, "source": "agent", "message": "partial answer"},
            ],
        }
    )

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
    )

    root = spans[0]
    root_response = Span.from_domain(root)
    llm = next(span for span in spans if span.kind == SpanKind.LLM)
    assert root.status == SpanStatus.ERROR
    assert root_response.error_type == "AgentTimeoutError"
    assert root_response.error_message == "Agent execution timed out after 900.0 seconds"
    assert json.loads(root_response.raw_attributes or "{}")["extra"]["error"] == {
        "type": "AgentTimeoutError",
        "message": "Agent execution timed out after 900.0 seconds",
    }
    assert llm.status == SpanStatus.SUCCESS


@pytest.mark.parametrize("error", ["timeout", [], 42])
def test_atif_mapping_ignores_non_object_root_extra_error(error: Any) -> None:
    trajectory = AtifTrajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "trace-session-id",
            "agent": {"name": "sample-agent", "version": "1.0.0"},
            "extra": {"error": error},
            "steps": [{"step_id": 1, "source": "agent", "message": "answer"}],
        }
    )

    root = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
    )[0]

    assert root.status == SpanStatus.SUCCESS


def test_atif_mapping_span_ids_are_trace_native_and_ignore_evaluation_run_id() -> None:
    base = {
        "schema_version": "ATIF-v1.5",
        "session_id": "trace-session-id",
        "agent": {"name": "sample-agent", "version": "1.0.0"},
        "steps": [{"step_id": 1, "source": "user", "message": "solve"}],
    }
    first = AtifTrajectory.model_validate(
        {**base, "evaluation_context": {**EVALUATION_CONTEXT, "evaluation_run_id": "evalrun-a"}}
    )
    second = AtifTrajectory.model_validate(
        {**base, "evaluation_context": {**EVALUATION_CONTEXT, "evaluation_run_id": "evalrun-b"}}
    )

    ingested_at = datetime(2026, 5, 18, tzinfo=timezone.utc)
    first_ids = {
        span.name: span.external_span_id
        for span in trajectory_to_spans(workspace="default", trajectory=first, ingested_at=ingested_at)
    }
    second_ids = {
        span.name: span.external_span_id
        for span in trajectory_to_spans(workspace="default", trajectory=second, ingested_at=ingested_at)
    }

    assert first_ids == second_ids


def _timed_trajectory(steps: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> AtifTrajectory:
    return AtifTrajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "trace-session-id",
            "agent": {"name": "sample-agent", "version": "1.0.0"},
            "steps": steps,
            **({"extra": extra} if extra is not None else {}),
        }
    )


def test_atif_v17_tool_call_extra_is_accepted_and_preserved() -> None:
    trajectory = _timed_trajectory(
        [
            {
                "step_id": 1,
                "source": "agent",
                "message": "running a tool",
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "function_name": "search",
                        "arguments": {"query": "x"},
                        "extra": {
                            "ancestry": {"function_id": "fn-1", "function_name": "searcher"},
                            "invocation": {"start_timestamp": 100.0, "end_timestamp": 101.5},
                            "producer_specific": {"anything": True},
                        },
                    }
                ],
            }
        ]
    )
    assert isinstance(trajectory.steps[0], AtifStepAgent)
    assert trajectory.steps[0].tool_calls is not None
    extra = trajectory.steps[0].tool_calls[0].extra
    assert extra is not None
    assert extra["ancestry"]["function_name"] == "searcher"

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
    )
    tool = next(span for span in spans if span.name == "search")
    assert json.loads(tool.input)["extra"]["invocation"]["end_timestamp"] == 101.5


def test_atif_mapping_uses_invocation_timing_for_step_and_tool_spans() -> None:
    base = datetime(2026, 6, 1, tzinfo=timezone.utc)
    trajectory = _timed_trajectory(
        [
            {
                "step_id": 1,
                "source": "agent",
                "message": "working",
                "extra": {
                    "invocation": {
                        "start_timestamp": base.timestamp() + 10,
                        "end_timestamp": base.timestamp() + 40,
                    }
                },
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "function_name": "bash",
                        "extra": {
                            "invocation": {
                                "start_timestamp": base.timestamp() + 12,
                                "end_timestamp": base.timestamp() + 30,
                            }
                        },
                    }
                ],
            }
        ]
    )

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
    )
    root = next(span for span in spans if span.name == "sample-agent")
    step = next(span for span in spans if span.name == "sample-agent" and span.kind == SpanKind.LLM)
    tool = next(span for span in spans if span.name == "bash")

    assert step.start_time == base + timedelta(seconds=10)
    assert step.end_time == base + timedelta(seconds=40)
    assert tool.start_time == base + timedelta(seconds=12)
    assert tool.end_time == base + timedelta(seconds=30)
    assert root.start_time == base + timedelta(seconds=10)
    assert root.end_time == base + timedelta(seconds=40)


def test_atif_mapping_root_covers_tool_call_only_invocation_timing() -> None:
    base = datetime(2026, 6, 1, tzinfo=timezone.utc)
    trajectory = _timed_trajectory(
        [
            {
                "step_id": 1,
                "source": "agent",
                "message": "working",
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "function_name": "bash",
                        "extra": {
                            "invocation": {
                                "start_timestamp": base.timestamp() + 10,
                                "end_timestamp": base.timestamp() + 20,
                            }
                        },
                    }
                ],
            }
        ]
    )

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=base + timedelta(seconds=15),
    )
    root = next(span for span in spans if span.name == "sample-agent")

    assert root.start_time == base + timedelta(seconds=10)
    assert root.end_time == base + timedelta(seconds=20)


def test_atif_mapping_drops_root_end_time_before_root_start_time() -> None:
    base = datetime(2026, 6, 1, tzinfo=timezone.utc)
    trajectory = _timed_trajectory(
        [
            {
                "step_id": 1,
                "source": "agent",
                "message": "out of order",
                "extra": {
                    "invocation": {
                        "start_timestamp": base.timestamp() + 10,
                        "end_timestamp": base.timestamp() + 5,
                    }
                },
            }
        ]
    )

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=base,
    )
    root = next(span for span in spans if span.name == "sample-agent")

    assert root.start_time == base + timedelta(seconds=10)
    assert root.end_time is None


def test_atif_mapping_does_not_infer_step_or_tool_ends() -> None:
    base = datetime(2026, 6, 1, tzinfo=timezone.utc)
    trajectory = _timed_trajectory(
        [
            {"step_id": 1, "source": "user", "message": "solve", "timestamp": base.isoformat()},
            {
                "step_id": 2,
                "source": "agent",
                "message": "on it",
                "timestamp": (base + timedelta(seconds=5)).isoformat(),
                "tool_calls": [{"tool_call_id": "call-1", "function_name": "bash"}],
            },
        ],
        extra={
            "verifier": {
                "started_at": (base + timedelta(seconds=50)).isoformat(),
                "finished_at": (base + timedelta(seconds=60)).isoformat(),
            }
        },
    )

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
    )
    user = next(span for span in spans if span.name == "user-1")
    step = next(span for span in spans if span.name == "sample-agent" and span.kind == SpanKind.LLM)
    tool = next(span for span in spans if span.name == "bash")
    root = next(span for span in spans if span.name == "sample-agent" and span.kind == SpanKind.AGENT)

    assert user.start_time == base
    assert user.end_time is None
    assert step.start_time == base + timedelta(seconds=5)
    assert step.end_time is None
    assert tool.start_time == step.start_time
    assert tool.end_time is None
    assert root.end_time == base + timedelta(seconds=60)


def test_atif_mapping_leaves_end_time_none_when_underivable() -> None:
    base = datetime(2026, 6, 1, tzinfo=timezone.utc)
    trajectory = _timed_trajectory(
        [
            {"step_id": 1, "source": "user", "message": "solve", "extra": {"invocation": "yesterday"}},
            {
                "step_id": 2,
                "source": "agent",
                "message": "on it",
                "extra": {"invocation": {"start_timestamp": "noon", "end_timestamp": None}},
                "tool_calls": [{"tool_call_id": "call-1", "function_name": "bash"}],
            },
            {
                "step_id": 3,
                "source": "agent",
                "message": "out of order",
                "extra": {
                    "invocation": {
                        "start_timestamp": base.timestamp() + 10,
                        "end_timestamp": base.timestamp() + 5,
                    }
                },
            },
        ]
    )

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
    )
    user = next(span for span in spans if span.name == "user-1")
    tool = next(span for span in spans if span.name == "bash")
    llm_steps = [span for span in spans if span.name == "sample-agent" and span.kind == SpanKind.LLM]
    last = llm_steps[-1]

    # Malformed invocation blocks never fail ingest and never fabricate ends
    # from later steps. The out-of-order explicit end is also dropped.
    assert user.end_time is None
    assert tool.end_time is None
    assert last.start_time == base + timedelta(seconds=10)
    assert last.end_time is None


def test_atif_mapping_tool_end_requires_explicit_tool_invocation_end() -> None:
    base = datetime(2026, 6, 1, tzinfo=timezone.utc)
    trajectory = _timed_trajectory(
        [
            {
                "step_id": 1,
                "source": "agent",
                "message": "working",
                "extra": {
                    "invocation": {
                        "start_timestamp": base.timestamp() + 10,
                        "end_timestamp": base.timestamp() + 40,
                    }
                },
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "function_name": "bash",
                        "extra": {
                            "invocation": {
                                "start_timestamp": base.timestamp() + 12,
                            }
                        },
                    }
                ],
            }
        ]
    )

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=base,
    )
    step = next(span for span in spans if span.name == "sample-agent" and span.kind == SpanKind.LLM)
    tool = next(span for span in spans if span.name == "bash")

    assert step.end_time == base + timedelta(seconds=40)
    assert tool.start_time == base + timedelta(seconds=12)
    assert tool.end_time is None


def test_atif_mapping_end_time_none_when_no_timing_exists_at_all() -> None:
    trajectory = _timed_trajectory(
        [
            {"step_id": 1, "source": "user", "message": "solve"},
            {
                "step_id": 2,
                "source": "agent",
                "message": "on it",
                "tool_calls": [{"tool_call_id": "call-1", "function_name": "bash"}],
            },
        ]
    )

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
    )
    assert all(
        span.end_time is None
        for span in spans
        if span.name in {"user-1", "bash"} or (span.name == "sample-agent" and span.kind == SpanKind.LLM)
    )


def test_atif_v17_observation_on_nested_system_step() -> None:
    """Regression: DeepAgents emits an observation on a nested *system* step.

    ``observation`` previously lived only on ``AtifStepAgent``, so a system step
    carrying one was rejected as an extra field and Intake returned HTTP 422,
    storing nothing. It now lives on ``AtifStepBase``; every step type accepts it,
    validators run on all step types, and the mapper reads observations anywhere.
    """
    trajectory = AtifTrajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "deepagents-session",
            "trajectory_id": "root",
            "agent": {"name": "orchestrator", "version": "1.0"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "classify this email"},
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "delegating to phishing-analyzer",
                    "tool_calls": [{"tool_call_id": "task-1", "function_name": "task"}],
                    "observation": {
                        "results": [{"source_call_id": "task-1", "subagent_trajectory_ref": [{"trajectory_id": "sub"}]}]
                    },
                },
            ],
            "subagent_trajectories": [
                {
                    "schema_version": "ATIF-v1.7",
                    "trajectory_id": "sub",
                    "agent": {"name": "phishing-analyzer", "version": "1.0"},
                    "steps": [
                        {
                            "step_id": 1,
                            "source": "system",
                            "message": "You are a careful email phishing analyzer.",
                            "observation": {"results": [{"content": "is_likely_phishing: true"}]},
                        }
                    ],
                }
            ],
        }
    )

    system_step = trajectory.subagent_trajectories[0].steps[0]
    assert system_step.source == "system"
    assert system_step.observation is not None
    assert system_step.observation.results[0].content == "is_likely_phishing: true"

    # Map the whole tree. The nested system step is emitted as its own span;
    # previously the entire POST 422'd before any span could be stored.
    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    system_span = next((s for s in spans if s.name == "system-1"), None)
    assert system_span is not None, "nested system step should map to a span"

    # Guard the mapper fix: `_step_observation` must read observations on
    # non-agent steps. If it regresses to agent-only (returns None), the
    # observation is lost even though the POST is now accepted.
    observation = _step_observation(system_step)
    assert observation is not None
    assert observation.results[0].content == "is_likely_phishing: true"


def test_atif_mapping_stamps_agent_name_on_every_span() -> None:
    trajectory = AtifTrajectory(
        schema_version="ATIF-v1.7",
        session_id="trace-session-id",
        agent=AtifAgent(
            name="sample-agent",
            version="1.0.0",
            model_name="provider/sample-model",
        ),
        steps=[AtifStepUser(step_id=1, source="user", message="solve")],
    )

    spans = trajectory_to_spans(
        workspace="default",
        trajectory=trajectory,
        ingested_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
    )

    # Not just the root, so agent-scoped span queries work too.
    assert spans
    for span in spans:
        assert span.attributes_string["gen_ai.agent.name"] == "sample-agent"
