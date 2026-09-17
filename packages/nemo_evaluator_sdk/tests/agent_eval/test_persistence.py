# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the run-bundle loader (`read_trials`) — the inverse of `persist_run`."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from nemo_evaluator_sdk.agent_eval.persistence import persist_run, read_trials
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult, AgentEvalSummary, TrialMetricValue
from nemo_evaluator_sdk.agent_eval.trials import (
    AgentEvalTrial,
    AgentEvalTrialStatus,
    AgentOutput,
    TrialError,
    TrialMeasurements,
)
from nemo_evaluator_sdk.values.evidence import CandidateEvidence, EvidenceDescriptor
from pydantic import ValidationError


def _write_trial(run_dir: Path, trial: AgentEvalTrial) -> None:
    (run_dir / "trials.jsonl").write_text(json.dumps(trial.model_dump(mode="json")) + "\n", encoding="utf-8")


def _trial_with_workspace(ref: str) -> AgentEvalTrial:
    return AgentEvalTrial(
        id="taskA:fabric",
        task_id="taskA",
        status=AgentEvalTrialStatus.COMPLETED,
        output=AgentOutput(output_text="done"),
        evidence=CandidateEvidence(descriptors={"workspace": EvidenceDescriptor(kind="filesystem", ref=ref)}),
        metadata={"agent_ok": True},
    )


def test_read_trials_loads_trial_and_keeps_valid_evidence_ref(tmp_path: Path) -> None:
    # A stored bundle whose workspace evidence ref still resolves as-is (re-scoring reads the deliverables).
    workspace = tmp_path / "evidence" / "fabric" / "run" / "000000-taskA" / "workspace"
    (workspace / "output").mkdir(parents=True)
    (workspace / "output" / "memo.txt").write_text("deliverable", encoding="utf-8")
    _write_trial(tmp_path, _trial_with_workspace(str(workspace)))

    (trial,) = read_trials(tmp_path)
    assert trial.task_id == "taskA" and trial.status == AgentEvalTrialStatus.COMPLETED
    ws = trial.evidence.require("workspace")
    assert ws.kind == "filesystem" and Path(ws.ref).is_dir()  # type: ignore[arg-type]


def test_read_trials_rebuilds_evidence_ref_for_a_moved_bundle(tmp_path: Path) -> None:
    # The stored ref points at the (now gone) launch dir, but the evidence exists under the given run_dir —
    # read_trials rebuilds the ref under run_dir from the evidence/ tail so a moved/copied bundle still works.
    workspace = tmp_path / "evidence" / "fabric" / "run" / "000000-taskA" / "workspace"
    workspace.mkdir(parents=True)
    stale = "/some/old/launch/dir/evidence/fabric/run/000000-taskA/workspace"
    _write_trial(tmp_path, _trial_with_workspace(stale))

    (trial,) = read_trials(tmp_path)
    ws = trial.evidence.require("workspace")
    assert Path(ws.ref) == workspace and Path(ws.ref).is_dir()  # type: ignore[arg-type]


def test_persist_run_writes_bundle_relative_refs_that_survive_a_move(tmp_path: Path) -> None:
    # persist_run stores evidence refs RELATIVE to the bundle, so the bundle is self-contained: copy it
    # anywhere and read_trials still resolves the deliverables — no launch-CWD dependency.
    bundle = tmp_path / "run"
    workspace = bundle / "evidence" / "fabric" / "r" / "000000-taskA" / "workspace"
    (workspace / "output").mkdir(parents=True)
    (workspace / "output" / "memo.txt").write_text("deliverable", encoding="utf-8")

    result = AgentEvalResult(
        run_id="r",
        tasks=[],
        trials=[_trial_with_workspace(str(workspace))],  # absolute ref under the bundle
        scores=[],
        summary=AgentEvalSummary.from_scores([], tasks=[]),
    )
    persist_run(result, bundle)

    stored = json.loads((bundle / "trials.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert stored["evidence"]["descriptors"]["workspace"]["ref"] == "evidence/fabric/r/000000-taskA/workspace"

    moved = tmp_path / "moved-bundle"
    shutil.copytree(bundle, moved)
    (trial,) = read_trials(moved)
    resolved = Path(trial.evidence.require("workspace").ref)  # type: ignore[arg-type]
    assert resolved.is_dir() and resolved == moved / "evidence" / "fabric" / "r" / "000000-taskA" / "workspace"


def test_persist_run_revalidates_model_copy_corrupted_measurements_before_writing(tmp_path: Path) -> None:
    trial = _trial_with_workspace("workspace").model_copy(
        update={"measurements": TrialMeasurements(prompt_tokens=8, completion_tokens=2)}
    )
    result = AgentEvalResult(
        run_id="r",
        tasks=[],
        trials=[trial],
        scores=[],
        summary=AgentEvalSummary.from_scores([], tasks=[]),
    )
    corrupted_measurements = trial.measurements.model_copy(update={"prompt_tokens": -1})
    corrupted_trial = trial.model_copy(update={"measurements": corrupted_measurements})
    corrupted_result = result.model_copy(update={"trials": [corrupted_trial]})
    bundle = tmp_path / "bundle"

    with pytest.raises(ValidationError):
        persist_run(corrupted_result, bundle, write_html_dashboard=False)

    assert not bundle.exists()


def test_persist_run_writes_per_task_values_to_summary(tmp_path: Path) -> None:
    summary = AgentEvalSummary(
        task_metric_values={
            "task-a": {
                "harbor_reward.reward": [
                    TrialMetricValue(trial_id="task-a__aaa", value=1.0),
                    TrialMetricValue(trial_id="task-a__bbb", value=0.0),
                ]
            },
            "task-b": {
                "harbor_reward.reward": [
                    TrialMetricValue(trial_id="task-b__ccc", value=1.0),
                    TrialMetricValue(trial_id="task-b__ddd", value=None),
                ]
            },
        }
    )
    result = AgentEvalResult(run_id="run", tasks=[], trials=[], scores=[], summary=summary)

    persist_run(result, tmp_path)

    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["task_metric_values"] == {
        "task-a": {
            "harbor_reward.reward": [
                {"trial_id": "task-a__aaa", "value_type": "number", "value": 1.0},
                {"trial_id": "task-a__bbb", "value_type": "number", "value": 0.0},
            ]
        },
        "task-b": {
            "harbor_reward.reward": [
                {"trial_id": "task-b__ccc", "value_type": "number", "value": 1.0},
                {"trial_id": "task-b__ddd", "value_type": "missing", "value": None},
            ]
        },
    }
    assert AgentEvalSummary.model_validate(payload).task_metric_values == summary.task_metric_values


def test_summary_json_round_trips_trial_order_through_sort_keys(tmp_path: Path) -> None:
    # persist_run writes with sort_keys=True. Values are a list precisely so that trial order
    # survives that: keying them by trial id would come back sorted lexicographically instead.
    summary = AgentEvalSummary(
        task_metric_values={
            "task-a": {
                "harbor_reward.reward": [
                    TrialMetricValue(trial_id="z-trial", value=1.0),
                    TrialMetricValue(trial_id="a-trial", value=0.0),
                ]
            }
        }
    )
    result = AgentEvalResult(run_id="run", tasks=[], trials=[], scores=[], summary=summary)

    persist_run(result, tmp_path)
    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    reloaded = AgentEvalSummary.model_validate(payload)

    assert [a.trial_id for a in reloaded.task_metric_values["task-a"]["harbor_reward.reward"]] == [
        "z-trial",
        "a-trial",
    ]


def test_read_trials_rejects_evidence_ref_that_escapes_the_bundle(tmp_path: Path) -> None:
    # A copied/untrusted bundle whose ref uses `..` to escape the bundle must NOT be resolved to the
    # outside path: the rebuilt ref is accepted only when it stays inside run_dir.
    (tmp_path / "outside_secret").mkdir()  # a real dir outside the bundle
    bundle = tmp_path / "run"
    bundle.mkdir()
    escaping = "evidence/../../outside_secret"  # bundle/evidence/../../outside_secret -> tmp_path/outside_secret
    _write_trial(bundle, _trial_with_workspace(escaping))

    (trial,) = read_trials(bundle)
    # left as the stored ref — never rewritten to the escaped (but existing) outside path
    assert trial.evidence.require("workspace").ref == escaping


def test_persist_and_read_keep_external_evidence_refs_absolute(tmp_path: Path) -> None:
    # Evidence that lives OUTSIDE the bundle (e.g. a re-scored bundle referencing the ORIGINAL run's
    # deliverables) must keep its absolute ref through persist + read — not relativized, not dropped.
    external = tmp_path / "original-run" / "evidence" / "fabric" / "r" / "000000-taskA" / "workspace"
    (external / "output").mkdir(parents=True)
    (external / "output" / "memo.txt").write_text("deliverable", encoding="utf-8")
    external_ref = str(external.resolve())

    bundle = tmp_path / "rescored"  # a DIFFERENT bundle that references the external deliverables
    result = AgentEvalResult(
        run_id="r",
        tasks=[],
        trials=[_trial_with_workspace(external_ref)],
        scores=[],
        summary=AgentEvalSummary.from_scores([], tasks=[]),
    )
    persist_run(result, bundle)

    stored = json.loads((bundle / "trials.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert stored["evidence"]["descriptors"]["workspace"]["ref"] == external_ref  # absolute, unchanged

    (trial,) = read_trials(bundle)
    assert trial.evidence.require("workspace").ref == external_ref  # retained as-is


def test_persist_defaults_to_the_work_dir_so_evidence_lands_inside_the_bundle(tmp_path: Path) -> None:
    # Defaulting to work_dir is what keeps a bundle self-contained: the evidence the trials point at
    # is already underneath it, so persist can rewrite the refs bundle-relative.
    workspace = tmp_path / "evidence" / "000000-taskA" / "workspace"
    workspace.mkdir(parents=True)
    result = AgentEvalResult(
        run_id="r",
        tasks=[],
        trials=[_trial_with_workspace(str(workspace))],
        scores=[],
        summary=AgentEvalSummary.from_scores([], tasks=[]),
        work_dir=tmp_path,
    )

    location = result.persist(write_dashboard=False)

    assert location.output_dir == tmp_path
    stored = json.loads((tmp_path / "trials.jsonl").read_text(encoding="utf-8"))["evidence"]["descriptors"]
    assert stored["workspace"]["ref"] == "evidence/000000-taskA/workspace"  # relative => self-contained


def test_persist_without_a_work_dir_or_an_explicit_target_is_an_error() -> None:
    # An in-memory run has nowhere to go; failing loudly beats inventing a directory.
    result = AgentEvalResult(
        run_id="r", tasks=[], trials=[], scores=[], summary=AgentEvalSummary.from_scores([], tasks=[])
    )

    with pytest.raises(ValueError, match="no work_dir"):
        result.persist()


def test_persist_accepts_an_explicit_target_for_an_in_memory_run(tmp_path: Path) -> None:
    # The counterpart to the error above: an in-memory run has no default, but naming one works.
    result = AgentEvalResult(
        run_id="r", tasks=[], trials=[], scores=[], summary=AgentEvalSummary.from_scores([], tasks=[])
    )

    location = result.persist(tmp_path, write_dashboard=False)

    assert location.output_dir == tmp_path
    assert json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))["run_id"] == "r"


def test_persist_run_writes_the_error_rollup_to_summary(tmp_path: Path) -> None:
    summary = AgentEvalSummary(
        error_trial_ids={"RuntimeError": ["task-a__aaa", "task-b__ccc"], "TimeoutError": ["task-a__bbb"]},
        error_count=3,
    )
    result = AgentEvalResult(run_id="run", tasks=[], trials=[], scores=[], summary=summary)

    persist_run(result, tmp_path)

    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["error_trial_ids"] == {
        "RuntimeError": ["task-a__aaa", "task-b__ccc"],
        "TimeoutError": ["task-a__bbb"],
    }
    assert payload["error_count"] == 3
    # Ids keep their order through sort_keys=True, which sorts mapping keys and not list contents.
    assert AgentEvalSummary.model_validate(payload).error_trial_ids == summary.error_trial_ids


def test_trials_jsonl_round_trips_a_typed_error(tmp_path: Path) -> None:
    trial = AgentEvalTrial(
        id="task-a__aaa",
        task_id="task-a",
        status=AgentEvalTrialStatus.PARTIAL,
        error=TrialError(type="RuntimeError", message="boom", traceback="Traceback...\n"),
    )
    result = AgentEvalResult(run_id="run", tasks=[], trials=[trial], scores=[], summary=AgentEvalSummary())

    persist_run(result, tmp_path)
    [reloaded] = read_trials(tmp_path)

    assert reloaded.error == trial.error


@pytest.mark.parametrize(
    ("payload", "expected_measurements"),
    [
        pytest.param(
            {
                "measurements": {
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "runtime_sec": 0,
                    "cost_usd": 0,
                },
                "metadata": {"provenance": "kept"},
            },
            TrialMeasurements(prompt_tokens=8, completion_tokens=2, runtime_sec=0, cost_usd=0),
            id="typed",
        ),
        pytest.param(
            {"metadata": {"prompt_tokens": 8, "completion_tokens": 2, "duration_ms": 1500}},
            TrialMeasurements(),
            id="metadata-only",
        ),
        pytest.param(
            {
                "measurements": {"prompt_tokens": 8, "completion_tokens": 2},
                "metadata": {"prompt_tokens": 999, "completion_tokens": 999, "duration_ms": 1500},
            },
            TrialMeasurements(prompt_tokens=8, completion_tokens=2),
            id="conflicting-metadata",
        ),
    ],
)
def test_trials_jsonl_preserves_typed_measurements_and_opaque_metadata(
    tmp_path: Path,
    payload: dict[str, object],
    expected_measurements: TrialMeasurements,
) -> None:
    row = {"id": "task-a:trial", "task_id": "task-a", "status": "partial", **payload}
    (tmp_path / "trials.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )

    [loaded] = read_trials(tmp_path)
    assert loaded.measurements == expected_measurements
    assert loaded.metadata == payload.get("metadata", {})

    round_trip_dir = tmp_path / "round-trip"
    persist_run(
        AgentEvalResult(run_id="run", tasks=[], trials=[loaded], scores=[], summary=AgentEvalSummary()),
        round_trip_dir,
        write_html_dashboard=False,
    )
    [reloaded] = read_trials(round_trip_dir)
    assert reloaded == loaded


@pytest.mark.parametrize("measurements", [None, {"prompt_tokens": -1}])
def test_trials_jsonl_rejects_invalid_typed_measurements_without_metadata_fallback(
    tmp_path: Path,
    measurements: object,
) -> None:
    (tmp_path / "trials.jsonl").write_text(
        json.dumps(
            {
                "id": "task-a:invalid",
                "task_id": "task-a",
                "status": "partial",
                "measurements": measurements,
                "metadata": {"prompt_tokens": 8, "duration_ms": 1500},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        read_trials(tmp_path)
