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
    def __init__(self, ethos: str | None = None) -> None:
        self.uploads: list[dict[str, Any]] = []
        self.downloads: list[dict[str, Any]] = []
        self.fail = False
        self.ethos = ethos
        self._temp_copies: list[Path] = []

    def download(self, **kwargs: Any) -> None:
        self.downloads.append(kwargs)
        if self.ethos is None:
            raise FileNotFoundError(f"fileset {kwargs['workspace']}/{kwargs['fileset']} does not exist")
        (Path(kwargs["local_path"]) / kwargs["remote_path"]).write_text(self.ethos, encoding="utf-8")

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
    def __init__(self, conflict: bool = False, ethos: str | None = None) -> None:
        self.agents = _FakeAgents(conflict=conflict)
        self.files = _FakeFiles(ethos=ethos)


def _register(sdk: _FakeSdk, optimized: dict[str, Any] | None = None, name: str = "my-agent-opt") -> Any:
    return register_optimized_agent(
        optimized if optimized is not None else _config(name),
        name=name,
        source_agent="my-agent",
        source_workspace="my-ws",
        workspace="my-ws",
        sdk=sdk,
    )


def test_creates_the_agent_entity_as_nemo_agents_spec_v1() -> None:
    sdk = _FakeSdk()
    result = _register(sdk)

    assert result == {"agent": "my-ws/my-agent-opt"}
    created = sdk.agents.created[0]
    assert created["name"] == "my-agent-opt"
    assert created["config_format"] == "nemo-agents-spec-v1"
    assert created["workspace"] == "my-ws"


def test_uploads_the_optimized_agent_yaml_to_the_ethos_fileset() -> None:
    sdk = _FakeSdk()
    _register(sdk)

    upload = sdk.files.uploads[0]
    assert upload["fileset"] == "my-agent-opt-ethos"
    assert upload["workspace"] == "my-ws"
    staged = Path(upload["local_path"].rstrip("/")) / "agent.yaml"
    assert yaml.safe_load(staged.read_text(encoding="utf-8"))["name"] == "my-agent-opt"


def test_stages_the_source_agents_ethos_alongside_the_optimized_config() -> None:
    """The optimized agent's fileset is meant to hold both halves of the contract."""
    sdk = _FakeSdk(ethos="# Ethos\n\nBe helpful.\n")
    _register(sdk)

    (download,) = sdk.files.downloads
    assert download["fileset"] == "my-agent-ethos"
    assert download["workspace"] == "my-ws"
    assert download["remote_path"] == "ETHOS.md"
    staged = Path(sdk.files.uploads[0]["local_path"].rstrip("/")) / "ETHOS.md"
    assert staged.read_text(encoding="utf-8") == "# Ethos\n\nBe helpful.\n"


def test_a_source_agent_without_an_ethos_fileset_is_a_clean_skip() -> None:
    """Agents registered straight from an agent.yaml have no ethos; the optimized agent
    is still valid, so a missing source ethos must not fail the run."""
    sdk = _FakeSdk(ethos=None)
    result = _register(sdk)

    assert result == {"agent": "my-ws/my-agent-opt"}
    staged_dir = Path(sdk.files.uploads[0]["local_path"].rstrip("/"))
    assert (staged_dir / "agent.yaml").is_file()
    assert not (staged_dir / "ETHOS.md").exists()


def test_a_name_conflict_fails_without_overwriting() -> None:
    sdk = _FakeSdk(conflict=True)
    with pytest.raises(LocalRunError, match="already exists"):
        _register(sdk, _config(), name="taken")
    assert sdk.files.uploads == []


def test_a_failed_ethos_upload_rolls_the_agent_back() -> None:
    sdk = _FakeSdk()
    sdk.files.fail = True
    with pytest.raises(LocalRunError, match="rolled back"):
        _register(sdk, _config())
    assert sdk.agents.deleted == ["my-agent-opt"]


def test_an_invalid_optimized_config_fails_before_any_entity_is_created() -> None:
    sdk = _FakeSdk()
    with pytest.raises(LocalRunError, match="is not a valid nemo-agents-spec-v1"):
        _register(sdk, {"config_format": "nemo-agents-spec-v1", "name": "x"})
    assert sdk.agents.created == []
