# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from pathlib import Path

import pytest
from nemo_evaluator_sdk.agent_eval.trials import (
    AgentEvalTrial,
    AgentEvalTrialStatus,
    TrialError,
    TrialMeasurements,
    resolve_trial_status,
    standard_evidence_descriptors,
)
from nemo_evaluator_sdk.values.evidence import CandidateEvidence, EvidenceDescriptor
from pydantic import ValidationError


def test_trial_measurements_default_empty_and_serialize_on_trial() -> None:
    trial = AgentEvalTrial(id="trial-1", task_id="task-1", status=AgentEvalTrialStatus.PARTIAL)

    assert trial.measurements == TrialMeasurements()
    assert trial.model_dump(mode="json")["measurements"] == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cache_creation_tokens": None,
        "cache_read_tokens": None,
        "runtime_sec": None,
        "cost_usd": None,
    }


def test_trial_measurements_derive_and_validate_total_tokens() -> None:
    measurements = TrialMeasurements(prompt_tokens=80, completion_tokens=10)

    assert measurements.total_tokens == 90
    assert TrialMeasurements(prompt_tokens=80, completion_tokens=10, total_tokens=90) == measurements

    with pytest.raises(ValidationError, match="requires prompt_tokens and completion_tokens"):
        TrialMeasurements(total_tokens=90)
    with pytest.raises(ValidationError, match=r"must equal prompt_tokens \+ completion_tokens"):
        TrialMeasurements(prompt_tokens=80, completion_tokens=10, total_tokens=91)


@pytest.mark.parametrize(
    "field",
    [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_creation_tokens",
        "cache_read_tokens",
    ],
)
@pytest.mark.parametrize("bad", [True, "1", -1, 1.5])
def test_trial_measurements_require_strict_nonnegative_token_counts(field: str, bad: object) -> None:
    with pytest.raises(ValidationError):
        TrialMeasurements.model_validate({field: bad})


@pytest.mark.parametrize("field", ["runtime_sec", "cost_usd"])
@pytest.mark.parametrize("bad", [True, "1.5", -0.1, float("nan"), float("inf"), float("-inf"), 10**1000])
def test_trial_measurements_require_finite_nonnegative_real_values(field: str, bad: object) -> None:
    with pytest.raises(ValidationError):
        TrialMeasurements.model_validate({field: bad})


def test_trial_measurements_normalize_integer_runtime_and_cost_to_float() -> None:
    measurements = TrialMeasurements(runtime_sec=2, cost_usd=0)

    assert measurements.runtime_sec == 2.0
    assert type(measurements.runtime_sec) is float
    assert measurements.cost_usd == 0.0
    assert type(measurements.cost_usd) is float


def test_trial_measurements_reject_direct_mutation() -> None:
    measurements = TrialMeasurements(prompt_tokens=8, completion_tokens=2)

    with pytest.raises(ValidationError, match="Instance is frozen"):
        measurements.prompt_tokens = -1


@pytest.mark.parametrize(
    "corrupted_update",
    [
        {"prompt_tokens": -1},
        {"runtime_sec": float("nan")},
        {"total_tokens": 999},
    ],
)
def test_trial_measurements_can_only_be_replaced_with_a_valid_complete_value(
    corrupted_update: dict[str, object],
) -> None:
    trial = AgentEvalTrial(id="trial-1", task_id="task-1", status=AgentEvalTrialStatus.PARTIAL)
    replacement = TrialMeasurements(prompt_tokens=8, completion_tokens=2)

    trial.measurements = replacement

    assert trial.measurements == replacement
    corrupted = replacement.model_copy(update=corrupted_update)
    with pytest.raises(ValidationError):
        trial.measurements = corrupted
    assert trial.measurements == replacement


def test_trial_measurements_compatibility_import_is_same_class() -> None:
    from nemo_evaluator_sdk.agent_eval.metrics import TrialMeasurements as CompatibilityTrialMeasurements

    assert CompatibilityTrialMeasurements is TrialMeasurements


def test_trial_validation_keeps_measurement_named_metadata_opaque() -> None:
    trial = AgentEvalTrial.model_validate(
        {
            "id": "trial-1",
            "task_id": "task-1",
            "status": "partial",
            "metadata": {"prompt_tokens": 80, "completion_tokens": 10, "duration_ms": 2500},
        }
    )

    assert trial.measurements == TrialMeasurements()
    assert trial.metadata == {"prompt_tokens": 80, "completion_tokens": 10, "duration_ms": 2500}


def test_trial_validation_keeps_typed_measurements_authoritative_and_metadata_unchanged() -> None:
    trial = AgentEvalTrial.model_validate(
        {
            "id": "trial-1",
            "task_id": "task-1",
            "status": "partial",
            "measurements": {"prompt_tokens": 8, "completion_tokens": 2, "runtime_sec": 0},
            "metadata": {
                "prompt_tokens": 999,
                "completion_tokens": 999,
                "duration_ms": 999_000,
            },
        }
    )

    assert trial.measurements == TrialMeasurements(prompt_tokens=8, completion_tokens=2, runtime_sec=0)
    assert trial.metadata == {"prompt_tokens": 999, "completion_tokens": 999, "duration_ms": 999_000}


@pytest.mark.parametrize("measurements", [None, {"prompt_tokens": -1}])
def test_invalid_typed_measurements_do_not_fall_back_to_metadata(measurements: object) -> None:
    with pytest.raises(ValidationError):
        AgentEvalTrial.model_validate(
            {
                "id": "trial-1",
                "task_id": "task-1",
                "status": "partial",
                "measurements": measurements,
                "metadata": {"prompt_tokens": 8, "duration_ms": 2500},
            }
        )


def test_trial_measurement_metadata_survives_round_trip() -> None:
    trial = AgentEvalTrial.model_validate(
        {
            "id": "trial-1",
            "task_id": "task-1",
            "status": "partial",
            "measurements": {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0},
            "metadata": {"prompt_tokens": 999, "duration_ms": 2500, "provenance": "kept"},
        }
    )

    assert AgentEvalTrial.model_validate_json(trial.model_dump_json()) == trial


def test_trial_accepts_mapping_shaped_evidence_and_serializes_descriptors() -> None:
    # Evidence accepts a bare {name: descriptor} mapping (coerced to CandidateEvidence);
    # drive it through model_validate so the mapping shape is exercised end-to-end.
    trial = AgentEvalTrial.model_validate(
        {
            "id": "trial-1",
            "task_id": "task-1",
            "status": "completed",
            "output": {"output_text": "Answer"},
            "evidence": {
                "final_state": {"kind": "filesystem", "ref": "runs/local/final-state"},
                "trace": {"kind": "trace", "format": "atif", "ref": "runs/local/trace.atif.json"},
            },
        }
    )

    assert trial.evidence is not None
    assert trial.evidence.require("final_state", kind="filesystem").ref == "runs/local/final-state"
    assert trial.model_dump(mode="json")["evidence"] == {
        "descriptors": {
            "final_state": {
                "kind": "filesystem",
                "ref": "runs/local/final-state",
                "format": None,
                "data": None,
                "description": None,
                "metadata": {},
            },
            "trace": {
                "kind": "trace",
                "ref": "runs/local/trace.atif.json",
                "format": "atif",
                "data": None,
                "description": None,
                "metadata": {},
            },
        },
        "metadata": {},
    }


def test_completed_trial_requires_output() -> None:
    with pytest.raises(ValueError, match="completed trial requires output"):
        AgentEvalTrial(id="trial-1", task_id="task-1", status=AgentEvalTrialStatus.COMPLETED)


def test_trial_get_evidence_is_named_and_null_safe() -> None:
    descriptor = EvidenceDescriptor(kind="json", format="json", ref="/tmp/result.json")
    trial = AgentEvalTrial(
        id="trial-1",
        task_id="task-1",
        status=AgentEvalTrialStatus.PARTIAL,
        evidence=CandidateEvidence(descriptors={"result": descriptor}),
    )

    assert trial.get_evidence("result") is descriptor
    assert trial.get_evidence("missing") is None
    assert trial.model_copy(update={"evidence": None}).get_evidence("result") is None


def test_resolve_trial_status_maps_ran_but_failed_to_partial() -> None:
    assert resolve_trial_status(True) == AgentEvalTrialStatus.COMPLETED
    # A ran-but-unsuccessful agent stays scorable (PARTIAL), not dropped (FAILED).
    assert resolve_trial_status(False) == AgentEvalTrialStatus.PARTIAL


def test_standard_evidence_descriptors_builds_documented_keys(tmp_path: Path) -> None:
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir()
    # The trace does not need to exist: the fallback is a parser hint inferred from
    # its name, not an eager content-validation step.
    descriptors = standard_evidence_descriptors(
        logs_dir=tmp_path / "agent",
        final_state_dir=tmp_path / "workspace",
        trace_path=tmp_path / "atif-trace.json",
        initial_state_ref="s3://inputs",
        verifier_logs_dir=verifier_dir,
        primary_log="agent.log",
    )
    assert set(descriptors) == {"initial_state", "trace", "logs", "final_state", "verifier_logs"}
    assert descriptors["trace"].format == "atif"
    assert descriptors["logs"].metadata == {"primary_log": "agent.log"}

    # A missing verifier dir is omitted; trace is optional.
    minimal = standard_evidence_descriptors(logs_dir=tmp_path / "a", final_state_dir=tmp_path / "w")
    assert set(minimal) == {"logs", "final_state"}


def test_trial_error_rejects_an_empty_type() -> None:
    # An empty type would become an empty rollup key, which names nothing. The Harbor adapter
    # normalises before it gets here; this is the backstop for every other producer.
    for blank in ("", "   "):
        with pytest.raises(ValidationError, match="must not be empty"):
            TrialError(type=blank)


def test_trial_error_is_frozen_and_forbids_extras() -> None:
    error = TrialError(type="RuntimeError")

    with pytest.raises(ValidationError):
        error.type = "TimeoutError"
    with pytest.raises(ValidationError):
        TrialError(type="RuntimeError", stack="...")  # ty: ignore[unknown-argument]


def test_trial_error_round_trips_every_field_through_json() -> None:
    trial = AgentEvalTrial(
        id="t0",
        task_id="task-a",
        status=AgentEvalTrialStatus.PARTIAL,
        error=TrialError(
            type="RuntimeError",
            message="boom",
            traceback="Traceback (most recent call last):\n",
            occurred_at=datetime(2026, 8, 13, 17, 22, 32, 230852),
        ),
    )

    reloaded = AgentEvalTrial.model_validate(trial.model_dump(mode="json"))

    assert reloaded.error == trial.error
    assert AgentEvalTrial.model_validate(reloaded.model_dump(mode="json")).error == trial.error


def test_error_is_read_only_from_the_typed_field() -> None:
    # `error` is the single carrier. A bundle written before it existed recorded the type in
    # free-form metadata, and that is deliberately NOT interpreted: metadata stays opaque, so a
    # pre-TrialError bundle re-scores with no error rollup rather than a guessed one.
    trial = AgentEvalTrial.model_validate(
        {
            "id": "t0",
            "task_id": "task-a",
            "status": "partial",
            "metadata": {"exception_type": "TimeoutError", "reward": 0.0},
        }
    )

    assert trial.error is None
    assert trial.metadata["exception_type"] == "TimeoutError"  # kept verbatim, just not read


def test_a_trial_without_any_error_signal_loads_unchanged() -> None:
    trial = AgentEvalTrial.model_validate({"id": "t0", "task_id": "task-a", "status": "partial"})

    assert trial.error is None
    assert trial.metadata == {}
