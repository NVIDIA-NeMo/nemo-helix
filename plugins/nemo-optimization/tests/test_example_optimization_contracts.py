# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the supported replacement and rejection of archived NAT optimizer inputs.

No platform, harness process, or model call is needed. In particular, passing a
platform agent must not make the archived NAT overlay a valid Fabric study.
"""

from pathlib import Path
from typing import Any

import pytest
import typer
import yaml
from nemo_fabric import FabricConfig
from nemo_helix_plugin.errors import LocalRunError
from nemo_optimization.agents import _to_fabric_agent_package
from nemo_optimization.bundle import BundlePreflightError, preflight_bundle
from nemo_optimization.config_overlay import get_by_dotted_path
from nemo_optimization.fabric import FabricOptimizeError, build_optimize_payload
from nemo_optimization.fabric_evaluator import _build_metrics, build_agent_eval_tasks
from nemo_optimization.optimize_cli import register_prepare_fileset_command
from nemo_optimization.search_space import parse_numeric_search_space
from typer.testing import CliRunner

_ROOT = Path(__file__).resolve().parents[3]
_BUNDLE = _ROOT / "plugins/nemo-optimization/examples/hermes-optimize"
_ARCHIVES = ("plugins/nemo-agents/examples/react-agent/legacy-nat/react-optimize.yml.legacy",)


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _stored_agent() -> dict[str, Any]:
    return _to_fabric_agent_package(_yaml(_BUNDLE / "agents/chatonly/agent.yaml"), label="hermes-optimize-chatonly")


@pytest.mark.parametrize("archive", _ARCHIVES)
@pytest.mark.parametrize("with_agent", [False, True])
def test_archived_nat_is_rejected_even_with_a_fabric_agent(archive: str, with_agent: bool) -> None:
    path = _ROOT / archive
    config = _yaml(path)
    # Exercise the actual job composition boundary, not just a schema/version assertion.
    with pytest.raises(FabricOptimizeError, match="legacy NAT workflow YAML"):
        build_optimize_payload(agent_config=_stored_agent() if with_agent else None, optimize_config=config)
    with pytest.raises(BundlePreflightError, match="optimizer.search_space is empty"):
        preflight_bundle(path.parent, path.name, agent="hermes-optimize-chatonly" if with_agent else None)
    # There must not be a second copy at the previously advertised runnable location.
    assert not (path.parent.parent / path.name.removesuffix(".legacy")).exists()


def test_supported_overlay_cannot_optimize_the_legacy_react_agent() -> None:
    agent = _yaml(_ROOT / "plugins/nemo-agents/examples/react-agent/react-agent.yml")
    overlay = _yaml(_BUNDLE / "optimize-chatonly-via-agent.yaml")
    with pytest.raises(LocalRunError, match="unsupported config_format"):
        _to_fabric_agent_package(agent, label="react-agent")
    with pytest.raises(FabricOptimizeError, match="legacy NAT workflow YAML"):
        build_optimize_payload(agent_config=agent, optimize_config=overlay)


@pytest.mark.parametrize("via_agent", [False, True])
def test_supported_example_parses_targets_real_fields_and_emits_its_objectives(
    via_agent: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    filename = "optimize-chatonly-via-agent.yaml" if via_agent else "optimize-chatonly.yaml"
    config = preflight_bundle(_BUNDLE, filename, agent="hermes-optimize-chatonly" if via_agent else None)
    payload = build_optimize_payload(agent_config=_stored_agent() if via_agent else None, optimize_config=config)
    # Fabric's parser validates the agent body; planning requires the optional Hermes adapter.
    agent = {key: value for key, value in payload.items() if key not in {"optimizer", "eval"}}
    FabricConfig.from_mapping(agent)
    space = parse_numeric_search_space(payload["optimizer"])
    assert space
    for dimension in space.values():
        assert dimension.path.startswith("models.default.")
        assert isinstance(get_by_dotted_path(payload, dimension.path), float)
    assert payload["eval"]["fabric"]["base_dir"] == "."
    judge = payload["eval"]["evaluators"]["accuracy"]["llm_name"]
    assert payload["models"][judge]["model"]
    metrics = _build_metrics(payload, payload["eval"])
    emitted = {spec.name for metric in metrics for spec in metric.output_spec()}
    objectives = {item["evaluator_name"] for item in payload["optimizer"]["eval_metrics"].values()}
    assert objectives == {"average_score"}
    assert objectives <= emitted
    monkeypatch.chdir(_BUNDLE)
    tasks = build_agent_eval_tasks(payload)
    assert tasks
    assert all(task.intent and task.reference["answer"] and task.metrics for task in tasks)


@pytest.mark.parametrize("via_agent", [False, True])
def test_supported_bundle_cli_dry_run_needs_no_platform(via_agent: bool) -> None:
    app = typer.Typer()

    @app.callback()
    def root() -> None:
        pass

    register_prepare_fileset_command(app)
    args = [
        "prepare-fileset",
        "--source",
        str(_BUNDLE),
        "--optimize-config",
        "optimize-chatonly-via-agent.yaml" if via_agent else "optimize-chatonly.yaml",
        "--fileset",
        "hermes-optimize-chatonly",
        "--workspace",
        "default",
        "--dry-run",
        "--no-check-models",
    ]
    if via_agent:
        args += ["--agent", "hermes-optimize-chatonly"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Preflight passed. Would upload" in result.output
    help_result = CliRunner().invoke(app, ["prepare-fileset", "--help"])
    assert help_result.exit_code == 0
    assert "Fabric optimize bundle" in help_result.output
