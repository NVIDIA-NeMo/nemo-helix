# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Fabric Ethos artifact staging."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import yaml
from nemo_agents_plugin.runner.fabric_artifact_staging import (
    FabricArtifactStagingError,
    FabricEthosFilesetNotFound,
    _download_fileset,
    stage_fabric_ethos_config_files,
    stage_fabric_ethos_dir,
    validate_referenced_skill_paths,
)
from nemo_deployments_plugin.entities import ConfigFile
from nemo_platform_plugin.client.errors import NemoHTTPError
from nemo_platform_plugin.client.errors import NotFoundError as PluginClientNotFoundError
from nemo_platform_plugin.files.client import AsyncFilesClient


def _fabric_config(*, skills_paths: list[str] | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "main",
        "harnesses": {
            "main": {
                "kind": "codex",
                "settings": {},
            }
        },
    }
    if skills_paths is not None:
        config["skills"] = {"paths": skills_paths}
    return config


def _plugin_not_found(message: str = "missing fileset") -> PluginClientNotFoundError:
    response = httpx.Response(
        404,
        request=httpx.Request("GET", "http://platform/filesets/fabric-agent-ethos"),
        json={"detail": message},
    )
    return PluginClientNotFoundError(response)


@pytest.fixture
def patched_download_fileset(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _download_fileset(
        files_client: AsyncMock,
        *,
        workspace: str,
        fileset_name: str,
        local_path: Path,
    ) -> None:
        try:
            await files_client.download(local_path=str(local_path), fileset=fileset_name, workspace=workspace)
        except (FileNotFoundError, PluginClientNotFoundError) as exc:
            raise FabricEthosFilesetNotFound("missing fileset") from exc

    monkeypatch.setattr(
        "nemo_agents_plugin.runner.fabric_artifact_staging._download_fileset",
        _download_fileset,
    )


def _fileset_file(path: str, content: bytes) -> dict[str, Any]:
    return {
        "file_ref": f"fileset://default/fabric-agent-ethos#{path}",
        "file_url": f"fileset://default/fabric-agent-ethos#{path}",
        "path": path,
        "size": len(content),
    }


@pytest.mark.asyncio
async def test_download_fileset_downloads_listed_files_with_typed_client(tmp_path: Path) -> None:
    content = {
        "agent.yaml": b"name: fabric-agent\n",
        "skills/review/SKILL.md": b"# Review\n",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/files"):
            return httpx.Response(200, json={"data": [_fileset_file(path, data) for path, data in content.items()]})
        encoded_path = str(request.url).split("/-/", 1)[1]
        path = encoded_path.replace("%2F", "/")
        return httpx.Response(200, content=content[path])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        files_client = AsyncFilesClient(base_url="http://test", workspace="default", http_client=http_client)
        await _download_fileset(
            files_client,
            workspace="default",
            fileset_name="fabric-agent-ethos",
            local_path=tmp_path,
        )

    assert (tmp_path / "agent.yaml").read_bytes() == b"name: fabric-agent\n"
    assert (tmp_path / "skills" / "review" / "SKILL.md").read_bytes() == b"# Review\n"


@pytest.mark.asyncio
async def test_stage_fabric_ethos_config_files_real_missing_fileset_falls_back() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "missing fileset"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        files_client = AsyncFilesClient(base_url="http://test", workspace="default", http_client=http_client)
        result = await stage_fabric_ethos_config_files(
            workspace="default",
            agent_name="fabric-agent",
            rewritten_agent_config=_fabric_config(),
            agent_yaml_path="/workspace/agent.yaml",
            files_client=files_client,
        )

    assert result == [
        ConfigFile(path="/workspace/agent.yaml", content=yaml.safe_dump(_fabric_config(), sort_keys=False)),
    ]


@pytest.mark.asyncio
async def test_stage_fabric_ethos_config_files_preserves_files_service_errors() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "files service unavailable"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        files_client = AsyncFilesClient(base_url="http://test", workspace="default", http_client=http_client)
        with pytest.raises(NemoHTTPError, match="HTTP 500"):
            await stage_fabric_ethos_config_files(
                workspace="default",
                agent_name="fabric-agent",
                rewritten_agent_config=_fabric_config(),
                agent_yaml_path="/workspace/agent.yaml",
                files_client=files_client,
            )


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_config_files_without_agent_name_returns_inline_yaml() -> None:
    config = _fabric_config()
    rewritten = {**config, "gateway": "http://example"}
    result = await stage_fabric_ethos_config_files(
        workspace="default",
        agent_name="",
        rewritten_agent_config=rewritten,
        agent_yaml_path="/workspace/agent.yaml",
        files_client=AsyncMock(),
    )
    assert result == [
        ConfigFile(path="/workspace/agent.yaml", content=yaml.safe_dump(rewritten, sort_keys=False)),
    ]


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_config_files_missing_fileset_falls_back() -> None:
    config = _fabric_config()
    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=FileNotFoundError("missing"))

    result = await stage_fabric_ethos_config_files(
        workspace="default",
        agent_name="fabric-agent",
        rewritten_agent_config=config,
        agent_yaml_path="/workspace/agent.yaml",
        files_client=files_client,
    )

    assert len(result) == 1
    assert result[0].path == "/workspace/agent.yaml"
    files_client.download.assert_awaited_once()
    await_args = files_client.download.await_args
    assert await_args is not None
    assert await_args.kwargs["fileset"] == "fabric-agent-ethos"
    assert await_args.kwargs["workspace"] == "default"


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_config_files_not_found_error_falls_back() -> None:
    config = _fabric_config()
    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=_plugin_not_found())

    result = await stage_fabric_ethos_config_files(
        workspace="default",
        agent_name="fabric-agent",
        rewritten_agent_config=config,
        agent_yaml_path="/workspace/agent.yaml",
        files_client=files_client,
    )

    assert len(result) == 1
    assert result[0].path == "/workspace/agent.yaml"


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_config_files_plugin_client_not_found_error_falls_back() -> None:
    config = _fabric_config()
    files_client = AsyncMock()
    response = httpx.Response(
        404,
        request=httpx.Request("GET", "http://platform/filesets/fabric-agent-ethos"),
        json={"detail": "Fileset not found"},
    )
    files_client.download = AsyncMock(side_effect=PluginClientNotFoundError(response))

    result = await stage_fabric_ethos_config_files(
        workspace="default",
        agent_name="fabric-agent",
        rewritten_agent_config=config,
        agent_yaml_path="/workspace/agent.yaml",
        files_client=files_client,
    )

    assert len(result) == 1
    assert result[0].path == "/workspace/agent.yaml"


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_config_files_stages_sibling_artifacts() -> None:
    async def _fake_download(*, local_path: str, fileset: str | None, workspace: str | None) -> None:
        del fileset, workspace
        root = Path(local_path)
        (root / "agent.yaml").write_text("stale: true\n", encoding="utf-8")
        (root / "prompts").mkdir()
        (root / "prompts" / "system.md").write_text("You are helpful.\n", encoding="utf-8")
        skill_dir = root / "skills" / "review"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Review\n", encoding="utf-8")
        (root / "ETHOS.md").write_text("# Ethos\n", encoding="utf-8")
        (root / "AGENT-SPEC.md").write_text("x" * 1_000_000, encoding="utf-8")

    rewritten = _fabric_config(skills_paths=["skills/review"])
    rewritten["models"] = {"default": {"provider": "openai", "model": "gpt", "settings": {"base_url": "http://igw"}}}
    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=_fake_download)

    result = await stage_fabric_ethos_config_files(
        workspace="default",
        agent_name="fabric-agent",
        rewritten_agent_config=rewritten,
        agent_yaml_path="/workspace/agent.yaml",
        files_client=files_client,
    )

    by_path = {item.path: item.content for item in result}
    assert "/workspace/agent.yaml" in by_path
    assert "/workspace/prompts/system.md" in by_path
    assert "/workspace/skills/review/SKILL.md" in by_path
    assert "/workspace/ETHOS.md" not in by_path
    assert "/workspace/AGENT-SPEC.md" not in by_path
    loaded = yaml.safe_load(by_path["/workspace/agent.yaml"])
    assert loaded["models"]["default"]["settings"]["base_url"] == "http://igw"
    assert "stale" not in by_path["/workspace/agent.yaml"]


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_config_files_missing_fileset_rejects_configured_skills() -> None:
    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=_plugin_not_found())

    with pytest.raises(FabricArtifactStagingError, match="skills/review"):
        await stage_fabric_ethos_config_files(
            workspace="default",
            agent_name="fabric-agent",
            rewritten_agent_config=_fabric_config(skills_paths=["skills/review"]),
            agent_yaml_path="/workspace/agent.yaml",
            files_client=files_client,
        )


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_config_files_rejects_non_utf8_artifact() -> None:
    async def _fake_download(*, local_path: str, fileset: str | None, workspace: str | None) -> None:
        del fileset, workspace
        root = Path(local_path)
        (root / "agent.yaml").write_text("name: fabric-agent\n", encoding="utf-8")
        (root / "logo.bin").write_bytes(b"\xff\xfe\x00binary")

    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=_fake_download)

    with pytest.raises(FabricArtifactStagingError, match="non-UTF-8"):
        await stage_fabric_ethos_config_files(
            workspace="default",
            agent_name="fabric-agent",
            rewritten_agent_config=_fabric_config(),
            agent_yaml_path="/workspace/agent.yaml",
            files_client=files_client,
        )


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_config_files_rejects_oversized_fileset() -> None:
    async def _fake_download(*, local_path: str, fileset: str | None, workspace: str | None) -> None:
        del fileset, workspace
        root = Path(local_path)
        (root / "agent.yaml").write_text("name: fabric-agent\n", encoding="utf-8")
        (root / "huge.md").write_text("x" * 1_000_000, encoding="utf-8")

    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=_fake_download)

    with pytest.raises(FabricArtifactStagingError, match="exceeding"):
        await stage_fabric_ethos_config_files(
            workspace="default",
            agent_name="fabric-agent",
            rewritten_agent_config=_fabric_config(),
            agent_yaml_path="/workspace/agent.yaml",
            files_client=files_client,
        )


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_config_files_rejects_missing_skill_path() -> None:
    async def _fake_download(*, local_path: str, fileset: str | None, workspace: str | None) -> None:
        del fileset, workspace
        root = Path(local_path)
        (root / "agent.yaml").write_text("name: fabric-agent\n", encoding="utf-8")

    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=_fake_download)

    with pytest.raises(FabricArtifactStagingError, match="skills/review"):
        await stage_fabric_ethos_config_files(
            workspace="default",
            agent_name="fabric-agent",
            rewritten_agent_config=_fabric_config(skills_paths=["skills/review"]),
            agent_yaml_path="/workspace/agent.yaml",
            files_client=files_client,
        )


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_dir_stages_sibling_artifacts(tmp_path: Path) -> None:
    async def _fake_download(*, local_path: str, fileset: str | None, workspace: str | None) -> None:
        del fileset, workspace
        root = Path(local_path)
        (root / "mcps").mkdir()
        (root / "mcps" / "calculator.py").write_text("print(1)\n", encoding="utf-8")
        skill_dir = root / "skills" / "review"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Review\n", encoding="utf-8")

    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=_fake_download)

    await stage_fabric_ethos_dir(
        workspace="default",
        agent_name="fabric-agent",
        agent_config=_fabric_config(skills_paths=["skills/review"]),
        base_dir=tmp_path,
        files_client=files_client,
    )

    assert (tmp_path / "mcps" / "calculator.py").read_text(encoding="utf-8") == "print(1)\n"
    assert (tmp_path / "skills" / "review" / "SKILL.md").exists()
    await_args = files_client.download.await_args
    assert await_args is not None
    assert await_args.kwargs["fileset"] == "fabric-agent-ethos"
    assert Path(await_args.kwargs["local_path"]) != tmp_path


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_dir_missing_fileset_is_not_fatal(tmp_path: Path) -> None:
    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=_plugin_not_found())

    await stage_fabric_ethos_dir(
        workspace="default",
        agent_name="fabric-agent",
        agent_config=_fabric_config(),
        base_dir=tmp_path,
        files_client=files_client,
    )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_dir_missing_fileset_rejects_configured_skills(tmp_path: Path) -> None:
    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=FileNotFoundError("missing"))

    with pytest.raises(FabricArtifactStagingError, match="skills/review"):
        await stage_fabric_ethos_dir(
            workspace="default",
            agent_name="fabric-agent",
            agent_config=_fabric_config(skills_paths=["skills/review"]),
            base_dir=tmp_path,
            files_client=files_client,
        )


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_dir_without_downloader_skips_download(tmp_path: Path) -> None:
    await stage_fabric_ethos_dir(
        workspace="default",
        agent_name="",
        agent_config=_fabric_config(),
        base_dir=tmp_path,
        files_client=None,
    )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_dir_allows_oversized_tree(tmp_path: Path) -> None:
    async def _fake_download(*, local_path: str, fileset: str | None, workspace: str | None) -> None:
        del fileset, workspace
        (Path(local_path) / "huge.md").write_text("x" * 1_000_000, encoding="utf-8")

    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=_fake_download)

    await stage_fabric_ethos_dir(
        workspace="default",
        agent_name="fabric-agent",
        agent_config=_fabric_config(),
        base_dir=tmp_path,
        files_client=files_client,
    )

    assert (tmp_path / "huge.md").exists()


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_dir_removes_files_dropped_from_fileset(tmp_path: Path) -> None:
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "STALE.md").write_text("removed upstream\n", encoding="utf-8")
    (tmp_path / "agent.yaml").write_text("stale: true\n", encoding="utf-8")

    async def _fake_download(*, local_path: str, fileset: str | None, workspace: str | None) -> None:
        del fileset, workspace
        (Path(local_path) / "mcps").mkdir()
        (Path(local_path) / "mcps" / "calculator.py").write_text("print(1)\n", encoding="utf-8")

    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=_fake_download)

    await stage_fabric_ethos_dir(
        workspace="default",
        agent_name="fabric-agent",
        agent_config=_fabric_config(),
        base_dir=tmp_path,
        files_client=files_client,
    )

    assert not (tmp_path / "skills").exists()
    assert not (tmp_path / "agent.yaml").exists()
    assert (tmp_path / "mcps" / "calculator.py").exists()


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_dir_missing_fileset_does_not_accept_stale_skills(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Stale\n", encoding="utf-8")

    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=FileNotFoundError("missing"))

    with pytest.raises(FabricArtifactStagingError, match="skills/review"):
        await stage_fabric_ethos_dir(
            workspace="default",
            agent_name="fabric-agent",
            agent_config=_fabric_config(skills_paths=["skills/review"]),
            base_dir=tmp_path,
            files_client=files_client,
        )


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_dir_preserves_runtime_directories(tmp_path: Path) -> None:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "events.atof.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "scratch.txt").write_text("keep\n", encoding="utf-8")
    (tmp_path / "STALE.md").write_text("drop\n", encoding="utf-8")

    config = _fabric_config()
    config["environment"] = {"workspace": "./workspace", "artifacts": "./artifacts"}

    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=FileNotFoundError("missing"))

    await stage_fabric_ethos_dir(
        workspace="default",
        agent_name="fabric-agent",
        agent_config=config,
        base_dir=tmp_path,
        files_client=files_client,
    )

    assert (tmp_path / "artifacts" / "events.atof.jsonl").exists()
    assert (tmp_path / "workspace" / "scratch.txt").exists()
    assert not (tmp_path / "STALE.md").exists()


def test_validate_referenced_skill_paths_accepts_agent_root() -> None:
    validate_referenced_skill_paths(_fabric_config(skills_paths=["."]), {PurePosixPath("SKILL.md")})


def test_validate_referenced_skill_paths_rejects_agent_root_when_nothing_staged() -> None:
    with pytest.raises(FabricArtifactStagingError, match=r"\."):
        validate_referenced_skill_paths(_fabric_config(skills_paths=["."]), set())


def test_validate_referenced_skill_paths_does_not_match_sibling_prefix() -> None:
    with pytest.raises(FabricArtifactStagingError, match="skills/review"):
        validate_referenced_skill_paths(
            _fabric_config(skills_paths=["skills/review"]),
            {PurePosixPath("skills/review-notes/SKILL.md")},
        )


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_dir_preserves_runtime_dir_through_parent_traversal(tmp_path: Path) -> None:
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "scratch.txt").write_text("keep\n", encoding="utf-8")
    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "stale.txt").write_text("drop\n", encoding="utf-8")

    config = _fabric_config()
    config["environment"] = {"workspace": "foo/../workspace", "artifacts": "../outside"}

    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=FileNotFoundError("missing"))

    await stage_fabric_ethos_dir(
        workspace="default",
        agent_name="fabric-agent",
        agent_config=config,
        base_dir=tmp_path,
        files_client=files_client,
    )

    assert (tmp_path / "workspace" / "scratch.txt").exists()
    assert not (tmp_path / "foo").exists()


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_dir_fails_when_stale_tree_cannot_be_removed(tmp_path: Path) -> None:
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "SKILL.md").write_text("# Stale\n", encoding="utf-8")

    files_client = AsyncMock()
    files_client.download = AsyncMock()

    with patch(
        "nemo_agents_plugin.runner.fabric_artifact_staging.shutil.rmtree",
        side_effect=PermissionError("read-only"),
    ):
        with pytest.raises(FabricArtifactStagingError, match="stale Ethos artifact"):
            await stage_fabric_ethos_dir(
                workspace="default",
                agent_name="fabric-agent",
                agent_config=_fabric_config(),
                base_dir=tmp_path,
                files_client=files_client,
            )

    files_client.download.assert_not_awaited()


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_config_files_missing_fileset_rejects_agent_root_skill() -> None:
    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=_plugin_not_found())

    with pytest.raises(FabricArtifactStagingError, match=r"skills\.paths entry|was not found"):
        await stage_fabric_ethos_config_files(
            workspace="default",
            agent_name="fabric-agent",
            rewritten_agent_config=_fabric_config(skills_paths=["."]),
            agent_yaml_path="/workspace/agent.yaml",
            files_client=files_client,
        )


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_config_files_accepts_agent_root_skill_from_fileset() -> None:
    async def _fake_download(*, local_path: str, fileset: str | None, workspace: str | None) -> None:
        del fileset, workspace
        (Path(local_path) / "SKILL.md").write_text("# Root skill\n", encoding="utf-8")

    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=_fake_download)

    result = await stage_fabric_ethos_config_files(
        workspace="default",
        agent_name="fabric-agent",
        rewritten_agent_config=_fabric_config(skills_paths=["."]),
        agent_yaml_path="/workspace/agent.yaml",
        files_client=files_client,
    )

    assert "/workspace/SKILL.md" in {item.path for item in result}


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_config_files_ethos_md_alone_does_not_satisfy_root_skill() -> None:
    async def _fake_download(*, local_path: str, fileset: str | None, workspace: str | None) -> None:
        del fileset, workspace
        (Path(local_path) / "ETHOS.md").write_text("# Ethos\n", encoding="utf-8")

    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=_fake_download)

    with pytest.raises(FabricArtifactStagingError, match="was not found"):
        await stage_fabric_ethos_config_files(
            workspace="default",
            agent_name="fabric-agent",
            rewritten_agent_config=_fabric_config(skills_paths=["."]),
            agent_yaml_path="/workspace/agent.yaml",
            files_client=files_client,
        )


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_config_files_legacy_contract_alone_does_not_satisfy_root_skill() -> None:
    async def _fake_download(*, local_path: str, fileset: str | None, workspace: str | None) -> None:
        del fileset, workspace
        (Path(local_path) / "AGENT-SPEC.md").write_text("# Legacy contract\n", encoding="utf-8")

    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=_fake_download)

    with pytest.raises(FabricArtifactStagingError, match="was not found"):
        await stage_fabric_ethos_config_files(
            workspace="default",
            agent_name="fabric-agent",
            rewritten_agent_config=_fabric_config(skills_paths=["."]),
            agent_yaml_path="/workspace/agent.yaml",
            files_client=files_client,
        )


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_dir_drops_contract_markdown(tmp_path: Path) -> None:
    async def _fake_download(*, local_path: str, fileset: str | None, workspace: str | None) -> None:
        del fileset, workspace
        root = Path(local_path)
        (root / "ETHOS.md").write_text("# Ethos\n", encoding="utf-8")
        (root / "AGENT-SPEC.md").write_text("# Legacy contract\n", encoding="utf-8")
        (root / "mcps").mkdir()
        (root / "mcps" / "calculator.py").write_text("print(1)\n", encoding="utf-8")

    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=_fake_download)

    await stage_fabric_ethos_dir(
        workspace="default",
        agent_name="fabric-agent",
        agent_config=_fabric_config(),
        base_dir=tmp_path,
        files_client=files_client,
    )

    assert not (tmp_path / "ETHOS.md").exists()
    assert not (tmp_path / "AGENT-SPEC.md").exists()
    assert (tmp_path / "mcps" / "calculator.py").exists()


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_dir_rejects_symlink_escaping_base_dir(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("nope\n", encoding="utf-8")
    base_dir = tmp_path / "base"
    base_dir.mkdir()

    async def _fake_download(*, local_path: str, fileset: str | None, workspace: str | None) -> None:
        del fileset, workspace
        (Path(local_path) / "escape").symlink_to(outside)

    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=_fake_download)

    with pytest.raises(FabricArtifactStagingError, match="escapes the agent base directory"):
        await stage_fabric_ethos_dir(
            workspace="default",
            agent_name="fabric-agent",
            agent_config=_fabric_config(),
            base_dir=base_dir,
            files_client=files_client,
        )


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_dir_rejects_fileset_paths_under_preserved_runtime_dirs(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    base_dir = tmp_path / "base"
    (base_dir / "artifacts").mkdir(parents=True)
    (base_dir / "artifacts" / "link").symlink_to(outside)

    config = _fabric_config()
    config["environment"] = {"artifacts": "./artifacts"}

    async def _fake_download(*, local_path: str, fileset: str | None, workspace: str | None) -> None:
        del fileset, workspace
        target = Path(local_path) / "artifacts" / "link"
        target.mkdir(parents=True)
        (target / "secret.txt").write_text("nope\n", encoding="utf-8")

    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=_fake_download)

    with pytest.raises(FabricArtifactStagingError, match="preserved runtime directory"):
        await stage_fabric_ethos_dir(
            workspace="default",
            agent_name="fabric-agent",
            agent_config=config,
            base_dir=base_dir,
            files_client=files_client,
        )

    assert not (outside / "secret.txt").exists()
    assert (base_dir / "artifacts" / "link").is_symlink()


@pytest.mark.usefixtures("patched_download_fileset")
@pytest.mark.asyncio
async def test_stage_fabric_ethos_dir_allows_symlinks_inside_runtime_directories(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    base_dir = tmp_path / "base"
    (base_dir / "artifacts").mkdir(parents=True)
    (base_dir / "artifacts" / "link").symlink_to(outside)

    config = _fabric_config()
    config["environment"] = {"workspace": "./workspace", "artifacts": "./artifacts"}

    files_client = AsyncMock()
    files_client.download = AsyncMock(side_effect=FileNotFoundError("missing"))

    await stage_fabric_ethos_dir(
        workspace="default",
        agent_name="fabric-agent",
        agent_config=config,
        base_dir=base_dir,
        files_client=files_client,
    )

    assert (base_dir / "artifacts" / "link").is_symlink()
