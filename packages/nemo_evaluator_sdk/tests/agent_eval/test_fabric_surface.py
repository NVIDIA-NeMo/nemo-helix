# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests binding ``FabricAgentRuntime`` to the REAL ``nemo_fabric`` API surface.

The other fabric unit tests are hermetic: they replace ``nemo_fabric`` with a hand-written fake, so
they encode whatever call signature the fake author wrote and keep passing even when the real Fabric
API moves underneath them. That is exactly how two breakages reached us unnoticed against a newer
Fabric — the ``enable_relay`` keyword changed and the harness adapter ids dropped their ``.cli``/
``.sdk`` suffixes — because nothing exercised the real package's call sites.

These tests close that gap by driving the runtime's own composition against the installed Fabric:

* **enable_relay signature** — ``_compose_config`` calls
  ``FabricConfig.enable_relay(observability=RelayObservabilityConfig(...))``. The retired ``config=``
  keyword (or a shape Fabric's ``RelayObservabilityConfig`` rejects) would raise here.
* **relay observability shape** — Fabric's relay models are ``extra="allow"``, so a stale field name
  is accepted *silently* and simply never takes effect (Relay's config policy warns on unknown fields
  rather than failing). ``ATOF``'s destination moved onto a typed sink list, and the old flat
  ``output_directory``/``filename``/``mode`` form would export nothing at all. Asserting the composed
  values — not just that the block exists — is what makes that visible.
* **adapter-id resolution** — the bare harness id the runtime forwards must resolve against Fabric's
  adapter registry via the planner. A suffixed ``nvidia.fabric.codex.cli`` would raise instead.

``importorskip('nemo_fabric')`` makes the whole module inert wherever the native Fabric wheels are not
installed (the hermetic-only 3.11 lanes), so it never competes with the fake-backed unit tests. It is
meant to run where the ``fabric`` extra is present — e.g. the Linux ``fabric-wheel-smoke`` CI job, or a
local ``uv sync --extra fabric`` (Fabric publishes a macOS arm64 wheel as of 0.1.0rc2).
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("nemo_fabric")

from nemo_evaluator_sdk.agent_eval.runtimes.fabric import _common
from nemo_evaluator_sdk.agent_eval.runtimes.fabric import _sandbox_execution as crt
from nemo_evaluator_sdk.agent_eval.runtimes.fabric import runtime as fabric_runtime
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime import FabricAgentRuntime
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import SkillSet
from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.providers.docker import DockerSandboxProvider
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask

# The ``fabric`` extra installs claude/codex/deepagents/hermes. Codex and hermes are both covered
# below because they exercise *different* skill routing: codex self-discovers bundles from its
# workspace, hermes accepts them natively through Fabric's ``skills`` config.


def _adapter_installed(name: str) -> bool:
    """Whether a harness adapter is installed (the ``fabric`` extra, not the base SDK).

    ``nemo_fabric`` itself is a base dependency, so importing it proves nothing about harnesses:
    without the adapters Fabric resolves none and fails with ``available adapters: []``. ``find_spec``
    raises rather than returning None when the parent package is missing, hence the guard.
    """
    try:
        return importlib.util.find_spec(f"nemo_fabric_adapters.{name}") is not None
    except ModuleNotFoundError:
        return False


def _codex_adapter_installed() -> bool:
    return _adapter_installed("codex")


requires_harness_adapters = pytest.mark.skipif(
    not _codex_adapter_installed(),
    reason="needs the harness adapters: uv sync --extra fabric",
)

requires_hermes_adapter = pytest.mark.skipif(
    not _adapter_installed("hermes"),
    reason="needs the hermes harness adapter: uv sync --extra fabric",
)

_CODEX_ADAPTER_ID = "nvidia.fabric.codex"
_HERMES_ADAPTER_ID = "nvidia.fabric.hermes"
_SURFACE_TASK = AgentEvalTask(id="surface-1", intent="Answer.", inputs={"instruction": "Ping?"})
_HERMES_CONFIG = {
    "metadata": {"name": "fabric-surface-hermes"},
    "harness": {"adapter_id": _HERMES_ADAPTER_ID, "resolution": "preinstalled"},
    "runtime": {"input_schema": "chat", "output_schema": "message"},
    "environment": {"provider": "local"},
}
_CODEX_CONFIG = {
    "metadata": {"name": "fabric-surface"},
    "harness": {"adapter_id": _CODEX_ADAPTER_ID, "resolution": "preinstalled"},
    "runtime": {"input_schema": "text", "output_schema": "message"},
    "environment": {"provider": "local"},
}


def test_compose_config_enables_relay_via_current_signature(tmp_path: Path) -> None:
    """The runtime's real ``enable_relay(observability=...)`` call is accepted by installed Fabric.

    Exercises ``_compose_config`` (which calls ``enable_relay`` with the observability config built by
    ``_relay_config``) rather than executing a harness, so it needs only the Fabric wheels — no codex
    CLI, relay gateway, or model. The dropped ``config=`` keyword would raise a ``TypeError`` here.
    """
    from nemo_fabric import (  # ty: ignore[unresolved-import]
        FabricConfig,
        RelayConfig,
        RelayObservabilityConfig,
    )

    runtime = FabricAgentRuntime(config=_CODEX_CONFIG, capture_trajectory=True)
    agent_config = FabricConfig.from_mapping(_CODEX_CONFIG)
    evidence_dir = tmp_path / "evidence"
    workspace_dir = evidence_dir / "workspace"
    evidence_dir.mkdir()
    workspace_dir.mkdir()

    composed = runtime._compose_config(agent_config, evidence_dir, workspace_dir, task=_SURFACE_TASK)

    # enable_relay accepted the runtime's call and stored Fabric's own typed relay models: a populated
    # RelayConfig whose observability carries the ATIF/ATOF shape ``_relay_config`` built. The dropped
    # ``config=`` keyword — or an observability shape Fabric rejects — would have raised before this.
    assert isinstance(composed.relay, RelayConfig)
    observability = composed.relay.observability
    assert isinstance(observability, RelayObservabilityConfig)

    # The exporters must land on the *declared fields*, not in the extras bag. Fabric's models allow
    # extras, so asserting `is not None` alone would still pass with a stale field name that exports
    # nothing — these assertions are what actually pin the current schema.
    relay_dir = str(evidence_dir / "relay")
    atif = observability.atif
    assert atif is not None and atif.enabled is True
    assert str(atif.output_directory) == relay_dir
    assert atif.filename_template == _common.ATIF_FILENAME_TEMPLATE

    atof = observability.atof
    assert atof is not None and atof.enabled is True
    (atof_sink,) = atof.sinks or []
    assert str(atof_sink.output_directory) == relay_dir
    assert atof_sink.filename == _common.ATOF_FILENAME


@requires_harness_adapters
def test_compose_config_is_a_complete_config_fabric_accepts(tmp_path: Path) -> None:
    """The composed per-task config is self-contained — no profile layer is needed or possible.

    Fabric 0.1.0rc2 deleted profile overlays: ``FabricProfileConfig`` is gone, ``Fabric.run``/``plan``
    take no ``profiles``, and ``FabricConfig.from_mapping`` raises on a ``profiles`` key. This asserts
    the runtime's composed config carries the evaluator-owned settings itself and round-trips through
    Fabric's own validator, which is the whole contract now.
    """
    from nemo_fabric import Fabric, FabricConfig  # ty: ignore[unresolved-import]

    runtime = FabricAgentRuntime(config=_CODEX_CONFIG, model="openai/gpt-5.4", capture_trajectory=False)
    evidence_dir = tmp_path / "evidence"
    workspace_dir = evidence_dir / "workspace"
    workspace_dir.mkdir(parents=True)

    composed = runtime._compose_config(
        FabricConfig.from_mapping(_CODEX_CONFIG), evidence_dir, workspace_dir, task=_SURFACE_TASK
    )
    composed.add_skill_path(str(tmp_path / "staged-skill"))

    # Evaluator-owned per-task settings live on the config itself, not in a trailing overlay.
    assert composed.environment is not None and str(composed.environment.workspace) == str(workspace_dir)
    assert composed.models["default"].model == "openai/gpt-5.4"
    assert str(tmp_path / "staged-skill") in [str(p) for p in composed.skills.paths]

    # It survives Fabric's own round-trip and planner, so it is a complete config by Fabric's rules.
    assert Fabric().plan(FabricConfig.from_mapping(composed.to_mapping())).adapter.adapter_id == _CODEX_ADAPTER_ID


@requires_harness_adapters
def test_runtime_adapter_id_resolves_against_registry() -> None:
    """The bare harness id the runtime forwards resolves through Fabric's real planner.

    Planning resolves the selected adapter from the registry without starting the runtime, so it needs
    only the installed adapter package. A retired ``nvidia.fabric.codex.cli`` id would raise a
    ``FabricConfigError`` instead of resolving.
    """
    from nemo_fabric import Fabric, FabricConfig  # ty: ignore[unresolved-import]

    plan = Fabric().plan(FabricConfig.from_mapping(_CODEX_CONFIG))

    assert plan.adapter.adapter_id == _CODEX_ADAPTER_ID


@requires_hermes_adapter
def test_hermes_routes_skills_natively_per_the_real_planner() -> None:
    """Fabric really does route skills to the hermes harness natively.

    This is the one capability the hermetic tests cannot check: they fake the planner and hardcode
    ``_NATIVE_SKILL_ADAPTERS``, so the entire native-injection branch — ``SKILL_MODE_NATIVE``,
    ``add_skill_path``, and the native-vs-workspace split in ``install_skills`` — is otherwise
    validated only against our own assumption. If Fabric stopped routing hermes' skills
    ``harness_native``, every fake-backed test would keep passing while real runs silently fell back.

    Drives our own ``resolve_skill_mode`` over a real ``RunPlan`` so the assertion covers the code
    the runtime actually executes, not just Fabric's output shape.
    """
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import SKILL_MODE_NATIVE, resolve_skill_mode
    from nemo_fabric import Fabric, FabricConfig  # ty: ignore[unresolved-import]

    # A skill path must be attached for the planner to emit a skills route at all — the sentinel need
    # not exist on disk, mirroring how FabricAgentRuntime probes capabilities.
    probe = FabricConfig.from_mapping(_HERMES_CONFIG)
    probe.add_skill_path(fabric_runtime._SKILL_PROBE_PATH)
    plan = Fabric().plan(probe)

    assert plan.adapter.adapter_id == _HERMES_ADAPTER_ID
    assert (
        resolve_skill_mode(capability_plan=plan.capability_plan, adapter_id=plan.adapter.adapter_id)
        == SKILL_MODE_NATIVE
    )


@requires_harness_adapters
def test_codex_also_routes_skills_natively_so_the_workspace_branch_is_a_fallback() -> None:
    """The shipped codex adapter accepts the native skills config, so codex plans ``native`` too.

    This is not what our fake-backed tests encode: they hardcode codex as non-native and exercise
    ``SKILL_MODE_CODEX_SKILLS_DIR`` (staging into ``<workspace>/.agents/skills/``). That branch is a
    genuine fallback for a codex-harness adapter which routes skills ``unsupported``, but the adapter
    we ship no longer takes it. Pinning the real answer here keeps the two from silently diverging —
    and would catch a revert, which would change where bundles are staged and whether the workspace
    needs post-run cleanup.
    """
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import SKILL_MODE_NATIVE, resolve_skill_mode
    from nemo_fabric import Fabric, FabricConfig  # ty: ignore[unresolved-import]

    probe = FabricConfig.from_mapping(_CODEX_CONFIG)
    probe.add_skill_path(fabric_runtime._SKILL_PROBE_PATH)
    plan = Fabric().plan(probe)

    assert "skills" in plan.capability_plan.get("routes", [])[0].get("kind", "")
    assert (
        resolve_skill_mode(capability_plan=plan.capability_plan, adapter_id=plan.adapter.adapter_id)
        == SKILL_MODE_NATIVE
    )


# --------------------------------------------------------------------------------------------------
# Sandbox mode. The host mode composes a typed ``FabricConfig`` and so is checked by Fabric's own
# validator on every call; the sandbox mode marshals a plain mapping across the sandbox boundary,
# where nothing validates it. These bind that mapping, and the in-sandbox
# entrypoint it names, back to the installed Fabric.
# --------------------------------------------------------------------------------------------------

_CONTAINER_CONFIG = {
    "metadata": {"name": "fabric-surface-container"},
    "harness": {"adapter_id": _HERMES_ADAPTER_ID, "resolution": "preinstalled"},
    "runtime": {"input_schema": "chat", "output_schema": "message"},
}


def _composed_container_config() -> Any:
    from nemo_fabric import FabricConfig  # ty: ignore[unresolved-import]

    # A real provider, never used: composing the config touches no sandbox.
    execution = crt.SandboxExecution(
        config=_CONTAINER_CONFIG,
        provider=DockerSandboxProvider(),
        image="unused",
        env={},
        model=None,
        timeout_s=600,
        capture_trajectory=True,
        trajectory_extra=None,
        runtime_name="fabric",
        skills=SkillSet(()),
    )
    return FabricConfig.from_mapping(execution._composed_config())


def test_container_composed_config_enables_relay_for_installed_fabric() -> None:
    """The composed mapping actually switches Relay on, by installed Fabric's own reading of it.

    Fabric ignores telemetry keys it does not recognize rather than rejecting them, so a stale
    envelope yields a container trial that runs to success and captures no trajectory at all — no
    ATIF, no ATOF, and no error. Asserting through ``FabricConfig.from_mapping`` is what makes the
    envelope Fabric's to define instead of ours to guess.
    """
    composed = _composed_container_config()

    assert composed.telemetry is not None
    assert "relay" in composed.telemetry.providers
    assert composed.relay is not None and str(composed.relay.output_dir) == crt._RELAY_DIR

    # As on the host: the exporters must land on the declared fields, not in Fabric's extras bag.
    observability = composed.relay.observability
    assert observability is not None
    atif = observability.atif
    assert atif is not None and atif.enabled is True
    assert str(atif.output_directory) == crt._RELAY_DIR
    assert atif.filename_template == _common.ATIF_FILENAME_TEMPLATE
    atof = observability.atof
    assert atof is not None and atof.enabled is True
    (atof_sink,) = atof.sinks or []
    assert str(atof_sink.output_directory) == crt._RELAY_DIR
    assert atof_sink.filename == _common.ATOF_FILENAME


def test_container_composed_config_keeps_caller_and_runtime_keys() -> None:
    """Enabling Relay does not cost the caller's harness or the runtime's own /out wiring.

    The telemetry keys are merged onto the composed mapping rather than replacing it, and both sets
    have to survive: a container trial whose workspace is unset writes outside the retrievable tree.
    """
    composed = _composed_container_config()

    assert composed.harness is not None and composed.harness.adapter_id == _HERMES_ADAPTER_ID
    assert composed.environment is not None
    assert str(composed.environment.workspace) == crt._WORKSPACE_DIR
    assert str(composed.runtime.artifacts) == crt._ARTIFACTS_DIR


def test_sandbox_driver_calls_an_api_installed_fabric_still_has() -> None:
    """The seeded driver's Fabric calls exist, are async, and take the arguments it passes.

    The driver is transported as source and only ever executed inside the sandbox, so nothing else
    imports it and no other test would notice Fabric retiring what it calls. Fabric has already done
    this once — ``FabricClient`` became ``Fabric`` and ``run`` became a coroutine — and the symptom is
    a container trial that fails on a traceback from a file the host wrote.
    """
    # Importing the driver is itself the check: its module-level ``from nemo_fabric import Fabric,
    # FabricConfig`` is the line a rename retires, and nothing else imports this module.
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric import sandbox_driver
    from nemo_fabric import Fabric, RunResult  # ty: ignore[unresolved-import]

    assert callable(sandbox_driver.main)
    assert inspect.iscoroutinefunction(Fabric.run)
    assert {"config", "input", "base_dir"} <= set(inspect.signature(Fabric.run).parameters)
    assert callable(RunResult.to_mapping)


def test_sandbox_dockerfile_smoke_check_imports_resolve() -> None:
    """The image's build-time import check names symbols the pinned Fabric still exports.

    It is the image build's only guard that the wheels are usable, and it runs where nobody sees it
    until a trial fails; ``from nemo_fabric import FabricClient`` outlived that class by two releases.
    """
    dockerfile = Path(crt.__file__).with_name("sandbox.Dockerfile").read_text(encoding="utf-8")
    (check,) = [
        line.split('python -c "', 1)[1].rsplit('"', 1)[0]
        for line in dockerfile.splitlines()
        if line.startswith("RUN python -c ")
    ]
    exec(compile(check, "<sandbox.Dockerfile>", "exec"), {})  # noqa: S102 - the recipe is repo source


def test_otlp_endpoint_lands_on_relays_declared_opentelemetry_field() -> None:
    """A configured OTLP endpoint reaches Fabric's declared field, not its extras bag.

    Both runtimes route their receiver's endpoint through ``relay_observability``. Fabric's relay
    models allow extras, so a stale field name is accepted in silence and simply never exported —
    and because the whole block is optional, dropping it entirely also leaves a trial that runs to
    success with an ATIF trajectory and no OTLP at all.
    """
    from nemo_fabric import RelayOpenTelemetryConfig  # ty: ignore[unresolved-import]

    observability = _common.relay_observability(
        relay_dir="/out/relay",
        agent_name="fabric_container",
        agent_version=_common.FABRIC_AGENT_VERSION,
        otlp_endpoint="http://127.0.0.1:4318/v1/traces",
    )

    opentelemetry = observability.opentelemetry
    assert isinstance(opentelemetry, RelayOpenTelemetryConfig)
    assert opentelemetry.enabled is True
    (endpoint,) = opentelemetry.endpoints or []
    assert str(endpoint.endpoint) == "http://127.0.0.1:4318/v1/traces"
    assert endpoint.service_name == "fabric_container"

    # Omitting it must leave the field unset rather than half-configured: Relay treats a disabled
    # section and an absent one alike, but a populated-but-endpointless one fails at export.
    assert (
        _common.relay_observability(
            relay_dir="/out/relay", agent_name="fabric_container", agent_version=_common.FABRIC_AGENT_VERSION
        ).opentelemetry
        is None
    )


def test_otlp_endpoint_survives_the_serialized_telemetry_fragment() -> None:
    """The endpoint reaches the mapping that crosses the sandbox boundary, not just the typed model.

    ``relay_telemetry_fragment`` is where a container runner's endpoint is handed to Fabric, and it
    accepts ``otlp_endpoint`` as a keyword — so failing to forward it is invisible at the call site
    and costs the container trial its whole OTLP trace while leaving ATIF intact.
    """
    fragment = _common.relay_telemetry_fragment(
        _CONTAINER_CONFIG,
        relay_dir="/out/relay",
        agent_name="fabric_container",
        agent_version=_common.FABRIC_AGENT_VERSION,
        otlp_endpoint="http://127.0.0.1:4318/v1/traces",
    )

    (endpoint,) = fragment["relay"]["observability"]["opentelemetry"]["endpoints"]
    assert endpoint["endpoint"] == "http://127.0.0.1:4318/v1/traces"
