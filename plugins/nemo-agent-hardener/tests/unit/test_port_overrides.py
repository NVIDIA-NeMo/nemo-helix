# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from _doubles import make_job_context
from nemo_agent_hardener_plugin.jobs import manifest as manifest_mod
from nemo_agent_hardener_plugin.jobs.errors import AgentHardenerRunError
from nemo_helix_plugin.agent_hardener.types import ManifestInit, ManifestUpdate, WarGameSpec
from pydantic import ValidationError


@pytest.mark.parametrize("schema", ["WarGameSpec", "ManifestInit", "ManifestUpdate"])
def test_studio_generated_port_constraints_match_api(schema: str) -> None:
    source = Path(__file__).parents[2] / "web" / "src" / "generated" / "schema" / f"{schema}.ts"
    assert re.search(r"@minimum 1\s+\* @maximum 65535\s+\*/\s+port\?: number", source.read_text())


@pytest.mark.parametrize("port", [0, -1, 65536])
@pytest.mark.parametrize(
    "model,fields",
    [
        (WarGameSpec, {"manifest_id": "saved"}),
        (ManifestInit, {"name": "saved", "agent": "agent"}),
        (ManifestUpdate, {}),
    ],
)
def test_api_port_range(model: Any, fields: dict[str, Any], port: int) -> None:
    with pytest.raises(ValidationError, match="port"):
        model.model_validate({**fields, "port": port})


@pytest.mark.parametrize("port", [1, 65535, None])
@pytest.mark.parametrize(
    "model,fields",
    [
        (WarGameSpec, {"manifest_id": "saved"}),
        (ManifestInit, {"name": "saved", "agent": "agent"}),
        (ManifestUpdate, {}),
    ],
)
def test_api_valid_ports(model: Any, fields: dict[str, Any], port: int | None) -> None:
    assert model.model_validate({**fields, "port": port}).port == port


@pytest.mark.parametrize("option", ["--port 8000", "--port=8000", "--port '8000'", '--port="8000"'])
def test_numeric_listener_and_env_are_updated(option: str) -> None:
    manifest = {
        "agent": {"port": 8000, "start_command": f"python app.py {option}", "env": {"PORT": "8000", "OTHER": "keep"}}
    }
    manifest_mod._apply_manifest_overrides(manifest, {"port": 9100})
    assert manifest["agent"]["start_command"] == f"python app.py {option.replace('8000', '9100')}"
    assert manifest["agent"]["env"] == {"PORT": "9100", "OTHER": "keep"}
    assert manifest["agent"]["port"] == 9100


@pytest.mark.parametrize("port", [0, -1, 65536, True, "9001", 9001.5])
def test_invalid_worker_port_is_atomic(port: Any) -> None:
    manifest = {"agent": {"port": 8000, "start_command": "python app.py --port 8000"}}
    before = deepcopy(manifest)
    with pytest.raises(AgentHardenerRunError):
        manifest_mod._apply_manifest_overrides(manifest, {"port": port, "attack_intensity": "thorough"})
    assert manifest == before


@pytest.mark.parametrize(
    "agent",
    [
        {"port": 8000},
        {"port": 8000, "start_command": "python app.py"},
        {"port": 8000, "start_command": "python app.py --port $PORT"},
        {"port": 8000, "start_command": "python app.py --port '$PORT'", "env": {"PORT": "8000"}},
        {"port": 8000, "start_command": "python app.py --port $PORT --host $HOST", "env": {"PORT": "8000"}},
        {"port": 8000, "start_command": "python app.py --port ${PORT:-8000}", "env": {"PORT": "8000"}},
        {"port": 8000, "start_command": "python app.py --port $PORT; other-command", "env": {"PORT": "8000"}},
        {"port": 8000, "start_command": "python app.py\n--port $PORT", "env": {"PORT": "8000"}},
        {"port": 8000, "start_command": "python app.py --port $PORT\nother-command", "env": {"PORT": "8000"}},
        {"port": 8000, "start_command": "python app.py --port 8000 --port 8001"},
        {"port": 8000, "start_command": "python app.py -- --port 8000"},
        {"port": 8000, "start_command": "python app.py -- --port=8000"},
        {"port": 8000, "start_command": "python app.py --port"},
        {"port": 8000, "start_command": "python app.py --port 8000; other-command"},
        {"port": 8000, "start_command": "python -c 'text --port 8000 other'"},
        {"port": 8000, "start_command": "python app.py --port '8000"},
        {"port": 8000, "start_command": ["python", "app.py"]},
        {"port": 8000, "env": ["PORT=8000"]},
    ],
)
def test_unsupported_listener_is_atomic(agent: dict[str, Any]) -> None:
    manifest = {"agent": deepcopy(agent)}
    before = deepcopy(manifest)
    with pytest.raises(AgentHardenerRunError):
        manifest_mod._apply_manifest_overrides(manifest, {"port": 9100, "attack_intensity": "thorough"})
    assert manifest == before


def test_listener_before_option_terminator_is_updated() -> None:
    agent = {"port": 8000, "start_command": "python app.py --port 8000 -- positional-argument"}
    manifest_mod._apply_agent_port(agent, 9100)
    assert agent == {"port": 9100, "start_command": "python app.py --port 9100 -- positional-argument"}


def test_env_only_listener() -> None:
    agent = {"port": 8000, "start_command": "python app.py", "env": {"PORT": "8000"}}
    manifest_mod._apply_agent_port(agent, 9100)
    assert agent == {"port": 9100, "start_command": "python app.py", "env": {"PORT": "9100"}}


@pytest.mark.parametrize("stored_port", [8000, "8000", "08000"])
def test_same_port_preserves_custom_listener(stored_port: int | str) -> None:
    agent = {"port": stored_port, "start_command": "custom-launcher"}
    manifest_mod._apply_agent_port(agent, 8000)
    assert agent == {"port": 8000, "start_command": "custom-launcher"}


@pytest.mark.parametrize("stored_port", [8000, "8000", "08000"])
@pytest.mark.parametrize(
    "command",
    [
        "python app.py --port $PORT",
        "python app.py --port ${PORT:-8000}",
        'sh -c "cd app && python main.py --port 8000"',
    ],
)
def test_unchanged_port_preserves_uninterpretable_listener(stored_port: int | str, command: str) -> None:
    agent = {"port": stored_port, "start_command": command}
    manifest = {"agent": agent}
    # Saved ports are reapplied on every run; this must not require interpreting
    # a command that worked before port overrides were introduced.
    manifest_mod._apply_manifest_overrides(manifest, {"port": 8000})
    assert agent == {"port": 8000, "start_command": command}


@pytest.mark.parametrize(
    "command",
    [
        "python app.py --port $PORT",
        "python app.py --port ${PORT:-8000}",
        'sh -c "cd app && python main.py --port 8000"',
    ],
)
def test_changed_port_still_rejects_uninterpretable_listener(command: str) -> None:
    agent = {"port": 8000, "start_command": command}
    before = deepcopy(agent)
    with pytest.raises(AgentHardenerRunError):
        manifest_mod._apply_agent_port(agent, 9100)
    assert agent == before


def test_unchanged_opaque_listener_does_not_partially_update_env() -> None:
    agent = {
        "port": "8000",
        "start_command": "python app.py --port ${PORT:-8000}",
        "env": {"PORT": "9100", "OTHER": "keep"},
    }
    before = deepcopy(agent)
    manifest_mod._apply_agent_port(agent, 8000)
    assert agent == {**before, "port": 8000}


@pytest.mark.parametrize("option", ["--port $PORT", "--port=${PORT}", '--port "$PORT"', '--port="${PORT}"'])
@pytest.mark.parametrize("stored_port", [8000, "8000", 9100])
def test_env_backed_listener_preserves_command(option: str, stored_port: int | str) -> None:
    command = f"python app.py {option}"
    agent = {"port": stored_port, "start_command": command, "env": {"PORT": "8000", "OTHER": "keep"}}
    manifest_mod._apply_agent_port(agent, 9100)
    assert agent == {"port": 9100, "start_command": command, "env": {"PORT": "9100", "OTHER": "keep"}}


@pytest.mark.parametrize("stored_port", ["8000", 9100, "9100"])
def test_stale_listener_is_repaired_even_when_routing_port_matches(stored_port: int | str) -> None:
    agent = {"port": stored_port, "start_command": "python app.py --port 8000", "env": {"PORT": "8000"}}
    manifest_mod._apply_agent_port(agent, 9100)
    assert agent == {"port": 9100, "start_command": "python app.py --port 9100", "env": {"PORT": "9100"}}


def test_legacy_zero_is_ignored_but_run_zero_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = SimpleNamespace(data={"port": 0, "agent_fileset": "default/bundle"})
    client = SimpleNamespace(entities=SimpleNamespace(get_entity_by_name=lambda **kwargs: record))

    def from_bundle(_client: Any, _name: str, data: dict[str, Any], *_args: Any) -> dict[str, Any]:
        manifest = {"agent": {"port": 8000, "start_command": "custom-launcher"}}
        manifest_mod._apply_manifest_overrides(manifest, data)
        return manifest

    monkeypatch.setattr(manifest_mod, "_materialize_from_bundle", from_bundle)
    ctx = make_job_context(tmp_path)
    path = manifest_mod._materialize_manifest(client, "saved", ctx)
    assert yaml.safe_load(Path(path).read_text())["agent"]["port"] == 8000
    with pytest.raises(AgentHardenerRunError):
        manifest_mod._materialize_manifest(client, "saved", ctx, {"port": 0})
    assert record.data["port"] == 0
