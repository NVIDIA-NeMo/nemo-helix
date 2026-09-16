# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import tempfile
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from nemo_agent_optimization_plugin.registration import register_optimized_agent
from nemo_platform_plugin.run_dependencies import LocalRunError


def _config(name: str = "my-agent") -> dict[str, Any]:
    return {
        "config_format": "nemo-agents-spec-v1",
        "name": name,
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes", "model": {"provider": "openai", "model": "m"}}},
    }


class _FakeFiles:
    def __init__(self) -> None:
        self.uploads: list[dict[str, Any]] = []
        self.fail = False
        self._temp_copies: list[Path] = []

    def upload(self, **kwargs: Any) -> Any:
        if self.fail:
            raise RuntimeError("upload exploded")
        # Preserve files from temp directory to persistent location
        local_path_str = kwargs.get("local_path", "").rstrip("/")
        if local_path_str:
            local_path = Path(local_path_str)
            if local_path.exists():
                # Create a persistent copy
                persistent_dir = Path(tempfile.mkdtemp(prefix=f"fake-fileset-{kwargs['fileset']}-"))
                shutil.copytree(local_path, persistent_dir, dirs_exist_ok=True)
                self._temp_copies.append(persistent_dir)
                # Update the kwargs to point to the persistent copy
                kwargs = {**kwargs, "local_path": str(persistent_dir) + "/"}
        self.uploads.append(kwargs)
        return type("R", (), {"name": kwargs["fileset"]})()


class _FakeAgents:
    def __init__(self, conflict: bool = False) -> None:
        self.created: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.conflict = conflict

    def create(self, **kwargs: Any) -> dict[str, Any]:
        if self.conflict:
            request = httpx.Request("POST", "http://x/agents")
            response = httpx.Response(409, request=request)
            raise httpx.HTTPStatusError("conflict", request=request, response=response)
        self.created.append(kwargs)
        return {"name": kwargs["name"]}

    def delete(self, name: str, workspace: str | None = None) -> None:
        self.deleted.append(name)


class _FakeSdk:
    def __init__(self, conflict: bool = False) -> None:
        self.agents = _FakeAgents(conflict=conflict)
        self.files = _FakeFiles()


def test_creates_the_agent_entity_as_nemo_agents_spec_v1() -> None:
    sdk = _FakeSdk()
    result = register_optimized_agent(
        _config("my-agent-opt"), name="my-agent-opt",
        source_agent_config=_config(), workspace="my-ws", sdk=sdk,
    )

    assert result == {"agent": "my-ws/my-agent-opt"}
    created = sdk.agents.created[0]
    assert created["name"] == "my-agent-opt"
    assert created["config_format"] == "nemo-agents-spec-v1"
    assert created["workspace"] == "my-ws"


def test_uploads_the_optimized_agent_yaml_to_the_ethos_fileset() -> None:
    sdk = _FakeSdk()
    register_optimized_agent(
        _config("my-agent-opt"), name="my-agent-opt",
        source_agent_config=_config(), workspace="my-ws", sdk=sdk,
    )

    upload = sdk.files.uploads[0]
    assert upload["fileset"] == "my-agent-opt-ethos"
    assert upload["workspace"] == "my-ws"
    staged = Path(upload["local_path"].rstrip("/")) / "agent.yaml"
    assert yaml.safe_load(staged.read_text(encoding="utf-8"))["name"] == "my-agent-opt"


def test_a_name_conflict_fails_without_overwriting() -> None:
    sdk = _FakeSdk(conflict=True)
    with pytest.raises(LocalRunError, match="already exists"):
        register_optimized_agent(
            _config(), name="taken", source_agent_config=_config(),
            workspace="my-ws", sdk=sdk,
        )
    assert sdk.files.uploads == []


def test_a_failed_ethos_upload_rolls_the_agent_back() -> None:
    sdk = _FakeSdk()
    sdk.files.fail = True
    with pytest.raises(LocalRunError, match="rolled back"):
        register_optimized_agent(
            _config(), name="my-agent-opt", source_agent_config=_config(),
            workspace="my-ws", sdk=sdk,
        )
    assert sdk.agents.deleted == ["my-agent-opt"]


def test_an_invalid_optimized_config_fails_before_any_entity_is_created() -> None:
    sdk = _FakeSdk()
    with pytest.raises(LocalRunError, match="is not a valid nemo-agents-spec-v1"):
        register_optimized_agent(
            {"config_format": "nemo-agents-spec-v1", "name": "x"},
            name="my-agent-opt", source_agent_config=_config(),
            workspace="my-ws", sdk=sdk,
        )
    assert sdk.agents.created == []
