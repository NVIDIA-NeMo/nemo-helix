# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Fabric-driven agent eval end-to-end against a fake ``nemo_fabric``, scored by a metric that
consumes the captured trajectory evidence: the runner -> evaluator -> metric -> evidence chain.

The real fabric -> codex -> Relay run lives in ``tests/e2e/test_fabric_codex_live.py``.
"""

from __future__ import annotations

import copy
import json
import sys
import types
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator
from nemo_evaluator_sdk.agent_eval.runtimes.fabric import runtime as fabric_runtime
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nemo_evaluator_sdk.values.evidence import EVIDENCE_FORMAT_ATIF, EVIDENCE_FORMAT_OTLP, EVIDENCE_TRACE
from nemo_evaluator_sdk.values.otlp import parse_resource_spans, resource_spans_from_text
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span


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


# --- hermetic: fake nemo_fabric so CI exercises the runner+evaluator+metric+evidence chain ----------


class _FakeEnvironment:
    def __init__(self, *, provider: str = "local", workspace: str | None = None, artifacts: str | None = None) -> None:
        self.provider = provider
        self.workspace = workspace
        self.artifacts = artifacts


class _FakeRuntimeCfg:
    def __init__(self, artifacts: str | None = None) -> None:
        self.artifacts = artifacts


class _FakeConfig:
    """Stand-in for nemo_fabric.FabricConfig supporting the config-first helpers the runtime uses."""

    def __init__(self) -> None:
        self.environment: _FakeEnvironment | None = None
        self.runtime = _FakeRuntimeCfg()
        self.models: dict[str, Any] = {}
        self.relay: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> _FakeConfig:
        return cls()

    def model_copy(self, *, deep: bool = False) -> _FakeConfig:
        clone = _FakeConfig()
        clone.environment = copy.deepcopy(self.environment)
        clone.runtime = _FakeRuntimeCfg(self.runtime.artifacts)
        clone.models = copy.deepcopy(self.models)
        clone.relay = copy.deepcopy(self.relay)
        return clone

    def enable_relay(
        self,
        *,
        project: str | None = None,
        output_dir: str | None = None,
        observability: Any = None,
        components: Any = None,
        policy: Any = None,
    ) -> _FakeConfig:
        self.relay = {
            "project": project,
            "output_dir": output_dir,
            "observability": observability,
            "components": components,
            "policy": policy,
        }
        return self


class _FakeModelConfig:
    """Stand-in for nemo_fabric.ModelConfig (FabricConfig.models is dict[str, ModelConfig])."""

    def __init__(self, *, provider: str, model: str, **extra: Any) -> None:
        self.provider = provider
        self.model = model
        self.extra = extra


class _FakeRelayModel:
    """Stand-in for nemo_fabric's RelayObservabilityConfig/RelayAtifConfig/RelayAtofConfig (kwargs bag)."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeArtifact:
    def __init__(self, name: str, kind: str, path: Path) -> None:
        self.name = name
        self.kind = kind
        self.path = path
        self.media_type = "application/json" if kind == "atif" else "text/plain"
        self.metadata: dict[str, Any] = {}


class _FakeManifest:
    def __init__(self, artifacts: list[_FakeArtifact]) -> None:
        self.root: Path | None = None
        self.artifacts = artifacts


class _FakeResult:
    def __init__(self, artifacts: list[_FakeArtifact]) -> None:
        self.status = "succeeded"
        self.output = {"adapter": "cli", "response": "DONE"}
        self.error = None
        self.harness = "codex"
        self.adapter_id = "nvidia.fabric.codex"
        self.adapter_kind = "process"
        self.invocation_id = "inv-1"
        self.artifacts = _FakeManifest(artifacts)
        self.telemetry: list[Any] = []
        self.events: list[Any] = []

    def to_mapping(self) -> dict[str, Any]:
        return {"status": self.status, "output": self.output}


@pytest.mark.asyncio
async def test_fabric_runner_eval_exposes_trajectory_to_metric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A real (fake-backed) ATIF file the promoted artifact points at.
    atif_path = tmp_path / "trajectory.atif.json"
    atif_path.write_text(json.dumps({"schema_version": "atif/v1", "steps": [{"kind": "message"}]}), encoding="utf-8")
    artifacts = [_FakeArtifact("relay_atif", "atif", atif_path), _FakeArtifact("stdout", "log", tmp_path / "out.txt")]
    (tmp_path / "out.txt").write_text("DONE\n", encoding="utf-8")

    class _FakeClient:
        # Fabric is a plain reusable facade (not an async context manager).
        async def run(self, agent: Any, **kwargs: Any) -> _FakeResult:
            return _FakeResult(artifacts)

    class _FakeRunRequest:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    module = types.ModuleType("nemo_fabric")
    setattr(module, "Fabric", _FakeClient)
    setattr(module, "FabricConfig", _FakeConfig)
    setattr(module, "EnvironmentConfig", _FakeEnvironment)
    setattr(module, "ModelConfig", _FakeModelConfig)
    setattr(module, "RunRequest", _FakeRunRequest)
    # The runtime builds the relay observability config from Fabric's own typed models (lazy import).
    setattr(module, "RelayObservabilityConfig", _FakeRelayModel)
    setattr(module, "RelayAtifConfig", _FakeRelayModel)
    setattr(module, "RelayAtofConfig", _FakeRelayModel)
    setattr(module, "RelayAtofFileSinkConfig", _FakeRelayModel)
    setattr(module, "RelayOpenTelemetryConfig", _FakeRelayModel)
    setattr(module, "RelayOpenTelemetryEndpointConfig", _FakeRelayModel)
    monkeypatch.setitem(sys.modules, "nemo_fabric", module)
    # nemo_relay stays a hard (installed) dependency here so ``run_tasks``'s capture-trajectory fail-fast
    # (``import nemo_relay.observability``) resolves; only the optional native nemo_fabric SDK is faked.
    # The observability config itself is built from nemo_fabric's typed models (faked above), not nemo_relay's.

    runtime = fabric_runtime.FabricAgentRuntime(
        config={"metadata": {"name": "a"}, "harness": {"adapter_id": "nvidia.fabric.codex"}},
        work_root=tmp_path / "fabric",
    )
    result = AgentEvaluator().run_sync(
        tasks=[_task()],
        target=runtime,
        config=AgentEvalRunConfig(work_dir=tmp_path / "out", parallelism=1),
    )

    trial = result.trials[0]
    assert trial.status == "completed"
    # The trajectory is exposed under the standard trace key, as an existing ATIF file.
    assert trial.evidence is not None
    trace = trial.evidence.descriptors[EVIDENCE_TRACE]
    # ATIF, not OTLP: this test's fake Fabric never runs Relay, so nothing exports a trace to
    # capture. The primary key is OTLP only when one was actually produced.
    assert trace.format == EVIDENCE_FORMAT_ATIF
    assert trace.ref is not None
    assert Path(trace.ref).exists()
    # ATIF stays reachable under its own key; the format decides which view is primary, never which
    # is reachable -- the same rule Harbor follows.
    atif = trial.evidence.descriptors[f"{EVIDENCE_TRACE}:{EVIDENCE_FORMAT_ATIF}"]
    assert atif.format == EVIDENCE_FORMAT_ATIF
    assert atif.ref is not None and Path(atif.ref).exists()
    # The metric received the evidence and scored from the trajectory content.
    scores = [s for s in result.scores if s.metric_type == "has-trajectory"]
    assert scores and scores[0].trial_id == trial.id
    assert scores[0].outputs[0].name == "has_trajectory"
    assert scores[0].outputs[0].value in (True, 1.0)


# --- gated live: real fabric -> codex -> Relay ATIF ------------------------------------------------


@pytest.mark.asyncio
async def test_a_relay_export_is_captured_and_becomes_the_primary_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runtime-to-writer wiring, without needing a live Relay.

    The fake harness does what Relay does: read the OTLP endpoint out of the config it was handed
    and POST an export to it. That covers the parts a writer unit test cannot -- that the endpoint
    reaches Relay's config at all, that the capture is still open while the harness runs, and that
    the resulting file is registered as the primary trace.
    """
    atif_path = tmp_path / "trajectory.atif.json"
    atif_path.write_text(json.dumps({"schema_version": "atif/v1", "steps": [{"kind": "message"}]}), encoding="utf-8")
    artifacts = [_FakeArtifact("relay_atif", "atif", atif_path)]

    def _endpoint_from(config: Any) -> str:
        opentelemetry = config.relay["observability"].kwargs["opentelemetry"]
        return str(opentelemetry.kwargs["endpoints"][0].kwargs["endpoint"])

    class _ExportingClient:
        async def run(self, agent: Any, **kwargs: Any) -> _FakeResult:
            span = Span(trace_id=bytes(range(16)), span_id=bytes(range(8)), name="codex-turn")
            export = ExportTraceServiceRequest(resource_spans=[ResourceSpans(scope_spans=[ScopeSpans(spans=[span])])])
            request = urllib.request.Request(
                _endpoint_from(agent),
                data=export.SerializeToString(),
                headers={"Content-Type": "application/x-protobuf"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                assert response.status == 200
            return _FakeResult(artifacts)

    class _FakeRunRequest:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    module = types.ModuleType("nemo_fabric")
    setattr(module, "Fabric", _ExportingClient)
    setattr(module, "FabricConfig", _FakeConfig)
    setattr(module, "EnvironmentConfig", _FakeEnvironment)
    setattr(module, "ModelConfig", _FakeModelConfig)
    setattr(module, "RunRequest", _FakeRunRequest)
    for name in (
        "RelayObservabilityConfig",
        "RelayAtifConfig",
        "RelayAtofConfig",
        "RelayAtofFileSinkConfig",
        "RelayOpenTelemetryConfig",
        "RelayOpenTelemetryEndpointConfig",
    ):
        setattr(module, name, _FakeRelayModel)
    monkeypatch.setitem(sys.modules, "nemo_fabric", module)

    runtime = fabric_runtime.FabricAgentRuntime(
        config={"metadata": {"name": "a"}, "harness": {"adapter_id": "nvidia.fabric.codex"}},
        work_root=tmp_path / "fabric",
    )
    result = AgentEvaluator().run_sync(
        tasks=[_task()],
        target=runtime,
        config=AgentEvalRunConfig(work_dir=tmp_path / "out", parallelism=1),
    )

    trial = result.trials[0]
    assert trial.status == "completed", f"{trial.metadata.get('error_type')}: {trial.metadata.get('error')}"
    assert trial.evidence is not None
    trace = trial.evidence.descriptors[EVIDENCE_TRACE]
    assert trace.format == EVIDENCE_FORMAT_OTLP
    assert trace.ref is not None
    spans = parse_resource_spans(resource_spans_from_text(Path(trace.ref).read_text(encoding="utf-8")))
    assert [span.name for rs in spans for ss in rs.scope_spans for span in ss.spans] == ["codex-turn"]
    assert trial.evidence.descriptors[f"{EVIDENCE_TRACE}:{EVIDENCE_FORMAT_ATIF}"].format == EVIDENCE_FORMAT_ATIF
