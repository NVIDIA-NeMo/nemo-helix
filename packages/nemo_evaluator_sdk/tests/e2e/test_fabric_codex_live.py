# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The real fabric -> codex -> Relay run, end to end on this host.

Lives under ``tests/e2e`` so the root conftest marks it ``e2e`` (never part of ``-m unit`` or CI).
It also needs the harness adapters, the ``nemo-relay`` gateway, and ``codex`` on ``PATH``; without
them it skips.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
from pathlib import Path

import pytest
from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator
from nemo_evaluator_sdk.agent_eval.runtimes.fabric import runtime as fabric_runtime
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nemo_evaluator_sdk.values.evidence import EVIDENCE_FORMAT_ATIF, EVIDENCE_FORMAT_OTLP, EVIDENCE_TRACE
from nemo_evaluator_sdk.values.otlp import parse_resource_spans, resource_spans_from_text

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.skip_in_ci]


class _TrajectoryEvidenceMetric:
    """Scores 1.0 iff the trial exposes a readable ATIF trajectory with at least one step.

    Exercises exactly the concern under test: a grader receives the candidate evidence and can locate
    + read the captured trajectory (the ``trace`` descriptor) for the task.
    """

    @property
    def type(self) -> str:
        return "has-trajectory"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.boolean("has_trajectory")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:  # noqa: A002 - matches protocol
        steps = 0
        evidence = input.candidate.evidence
        if evidence is not None:
            # Format-qualified: this metric parses ATIF, and the primary trace key is OTLP
            # whenever the runner captured one.
            descriptor = evidence.get(f"{EVIDENCE_TRACE}:{EVIDENCE_FORMAT_ATIF}") or evidence.get(EVIDENCE_TRACE)
            if descriptor is not None and descriptor.ref and descriptor.format == EVIDENCE_FORMAT_ATIF:
                payload = json.loads(Path(descriptor.ref).read_text(encoding="utf-8"))
                steps = len(payload.get("steps") or [])
        return MetricResult(outputs=[MetricOutput(name="has_trajectory", value=steps > 0)])


def _task() -> AgentEvalTask:
    return AgentEvalTask(
        id="say-done",
        intent="Agent follows a trivial instruction and exits cleanly.",
        inputs={"instruction": "Reply with the single word DONE and nothing else."},
        metrics=[_TrajectoryEvidenceMetric()],
    )


def _codex_adapter_installed() -> bool:
    """Whether the codex harness adapter is installed (the ``fabric`` extra, not the base SDK).

    ``nemo_fabric`` itself is a base dependency, so importing it proves nothing about harnesses:
    without the adapters Fabric resolves none and fails with ``available adapters: []``. ``find_spec``
    raises rather than returning None when the parent package is missing, hence the guard.
    """
    try:
        importlib.import_module("nemo_fabric_adapters.codex.adapter")
    except ImportError:  # the adapter package or one of its own dependencies (e.g. openai_codex) is absent
        return False
    return True


# No NeMo-Fabric checkout in the gate: the adapter registry resolves from the installed wheels
# (<sys.prefix>/share/nemo-fabric/adapters), so the package-scoped `fabric` extra is enough.
_LIVE_READY = bool(shutil.which("codex") and shutil.which("nemo-relay") and _codex_adapter_installed())
_LIVE_MODEL = os.environ.get("NEMO_FABRIC_LIVE_MODEL", "gpt-5.6-terra")
requires_live_fabric = pytest.mark.skipif(
    not _LIVE_READY,
    reason=(
        "needs the harness adapters "
        "(uv sync --frozen --package nemo-evaluator-sdk --extra fabric --inexact) + the nemo-relay gateway "
        "(script/dev-install-fabric.sh) + codex on PATH"
    ),
)


@requires_live_fabric
@pytest.mark.timeout(300)
def test_fabric_codex_live_eval_captures_atif_trajectory(tmp_path: Path) -> None:
    codex_config = {
        "schema_version": "fabric.agent/v1alpha1",
        "metadata": {"name": "eval-fabric-live"},
        "harness": {
            "adapter_id": "nvidia.fabric.codex",
            "resolution": "preinstalled",
            "settings": {"sandbox": "workspace-write"},
        },
        "runtime": {
            "input_schema": "text",
            "output_schema": "message",
            "timeout_seconds": 180,
        },
        "environment": {"provider": "local", "workspace": str(tmp_path / "ws")},
        # Fabric's codex adapter requires an explicit model provider — it does not fall back to the
        # Codex CLI's own configured default, and starting without one fails the adapter lifecycle
        # with `codex_invalid_configuration`. Override for an account with different model access.
        "models": {"default": {"provider": "openai", "model": _LIVE_MODEL}},
        "telemetry": {"enabled": False},
    }
    (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
    runtime = fabric_runtime.FabricAgentRuntime(
        config=codex_config,
        work_root=tmp_path / "fabric",
        capture_trajectory=True,
    )

    result = AgentEvaluator().run_sync(
        tasks=[_task()],
        target=runtime,
        config=AgentEvalRunConfig(work_dir=tmp_path / "out", parallelism=1),
    )

    trial = result.trials[0]
    assert trial.status == "completed", f"{trial.metadata.get('error_type')}: {trial.metadata.get('error')}"
    assert trial.evidence is not None
    # OTLP is primary because Relay exported one and the runner captured it; ATIF stays reachable
    # under its own key, so a metric written against either view still finds it.
    trace = trial.evidence.descriptors[EVIDENCE_TRACE]
    assert trace.format == EVIDENCE_FORMAT_OTLP
    assert trace.ref is not None
    otlp = Path(trace.ref)
    assert otlp.exists() and otlp.stat().st_size > 0
    spans = parse_resource_spans(resource_spans_from_text(otlp.read_text(encoding="utf-8")))
    assert [span for rs in spans for ss in rs.scope_spans for span in ss.spans], "captured no spans"

    atif_descriptor = trial.evidence.descriptors[f"{EVIDENCE_TRACE}:{EVIDENCE_FORMAT_ATIF}"]
    assert atif_descriptor.ref is not None
    atif = Path(atif_descriptor.ref)
    assert atif.exists() and atif.stat().st_size > 0
    assert "steps" in json.loads(atif.read_text(encoding="utf-8"))
    # The metric read the real trajectory and scored on it.
    scores = [s for s in result.scores if s.metric_type == "has-trajectory"]
    assert scores and scores[0].outputs[0].value in (True, 1.0)
