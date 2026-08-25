# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AgentEnvironment resolution and config merge."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from nemo_agents_plugin.entities import (
    NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    AgentComputeSpec,
    AgentEnvironment,
    AgentEnvironmentInline,
    AgentEnvironmentSpec,
    AgentSandboxSpec,
    ComputeResources,
    ComputeSpecInline,
    EnvironmentSpecInline,
    McpFulfillment,
    ModelProviderOverride,
    SandboxSpecInline,
)
from nemo_agents_plugin.environment_resolution import (
    EnvironmentResolutionError,
    merge_environment_spec_into_agent_config,
    resolve_environment,
)
from nemo_platform_plugin.entity_client import NemoEntityNotFoundError


def _agent_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "config_format": NEMO_AGENTS_SPEC_CONFIG_FORMAT,
        "name": "fabric-agent",
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes"}},
        "models": {"default": {"provider": "openai", "model": "openai/gpt-5.4"}},
        "environment": {"provider": "local"},
    }
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# resolve_environment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_none_returns_empty() -> None:
    resolved = await resolve_environment(None, workspace="default", entity_client=AsyncMock())
    assert resolved.environment_spec is None
    assert resolved.sandbox_spec is None
    assert resolved.compute_spec is None


@pytest.mark.asyncio
async def test_resolve_inline_environment_with_inline_specs() -> None:
    environment = AgentEnvironmentInline(
        environment_spec=EnvironmentSpecInline(env={"FOO": "bar"}),
        sandbox_spec=SandboxSpecInline(provider="openshell", provider_config={"timeout": 30}),
        compute_spec=ComputeSpecInline(resources=ComputeResources(limits={"cpu": "2"})),
    )
    resolved = await resolve_environment(environment, workspace="default", entity_client=AsyncMock())
    assert resolved.environment_spec is not None
    assert resolved.environment_spec.env == {"FOO": "bar"}
    assert resolved.sandbox_spec is not None
    assert resolved.sandbox_spec.provider == "openshell"
    assert resolved.sandbox_spec.provider_config == {"timeout": 30}
    assert resolved.compute_spec is not None
    assert resolved.compute_spec.resources.limits == {"cpu": "2"}


@pytest.mark.asyncio
async def test_resolve_environment_ref_dereferences_all_entities() -> None:
    env_entity = AgentEnvironment(
        name="env1",
        workspace="default",
        environment_spec="default/espec",
        sandbox_spec="default/sspec",
        compute_spec="default/cspec",
    )
    espec = AgentEnvironmentSpec(name="espec", workspace="default", env={"A": "1"})
    sspec = AgentSandboxSpec(name="sspec", workspace="default", provider="openshell")
    cspec = AgentComputeSpec(name="cspec", workspace="default", resources=ComputeResources(requests={"cpu": "1"}))

    entity_client = AsyncMock()
    entity_client.get = AsyncMock(side_effect=[env_entity, espec, sspec, cspec])

    resolved = await resolve_environment("default/env1", workspace="default", entity_client=entity_client)

    assert resolved.environment_spec is not None
    assert resolved.environment_spec.env == {"A": "1"}
    assert resolved.sandbox_spec is not None
    assert resolved.sandbox_spec.provider == "openshell"
    assert resolved.compute_spec is not None
    assert resolved.compute_spec.resources.requests == {"cpu": "1"}
    # AgentEnvironment, then its three specs.
    assert entity_client.get.await_count == 4


@pytest.mark.asyncio
async def test_resolve_missing_environment_ref_raises() -> None:
    entity_client = AsyncMock()
    entity_client.get = AsyncMock(side_effect=NemoEntityNotFoundError("gone"))
    with pytest.raises(EnvironmentResolutionError, match="AgentEnvironment 'env1' not found"):
        await resolve_environment("default/env1", workspace="default", entity_client=entity_client)


@pytest.mark.asyncio
async def test_resolve_missing_environment_spec_ref_raises() -> None:
    env_entity = AgentEnvironment(name="env1", workspace="default", environment_spec="default/missing")
    entity_client = AsyncMock()
    entity_client.get = AsyncMock(side_effect=[env_entity, NemoEntityNotFoundError("gone")])
    with pytest.raises(EnvironmentResolutionError, match="AgentEnvironmentSpec 'missing' not found"):
        await resolve_environment("default/env1", workspace="default", entity_client=entity_client)


async def test_resolve_missing_sandbox_spec_ref_raises() -> None:
    env_entity = AgentEnvironment(name="env1", workspace="default", sandbox_spec="default/missing")
    entity_client = AsyncMock()
    entity_client.get = AsyncMock(side_effect=[env_entity, NemoEntityNotFoundError("gone")])
    with pytest.raises(EnvironmentResolutionError, match="AgentSandboxSpec 'missing' not found"):
        await resolve_environment("default/env1", workspace="default", entity_client=entity_client)


# ---------------------------------------------------------------------------
# merge_environment_spec_into_agent_config
# ---------------------------------------------------------------------------


def test_merge_none_spec_returns_config_unchanged() -> None:
    config = _agent_config()
    result = merge_environment_spec_into_agent_config(config, None)
    assert result.config is config
    assert result.secrets == {}


def test_merge_ignores_non_fabric_config() -> None:
    config = {"config_format": "nat-workflow-v1", "functions": {}}
    spec = EnvironmentSpecInline(env={"FOO": "bar"}, secrets={"TOK": "default/tok"})
    result = merge_environment_spec_into_agent_config(config, spec)
    # Non-fabric config is left unchanged, but top-level secrets are still collected.
    assert result.config is config
    assert result.secrets == {"TOK": "default/tok"}


def test_merge_env_spec_wins_on_collision() -> None:
    config = _agent_config(environment={"provider": "local", "env": {"SHARED": "agent"}})
    spec = EnvironmentSpecInline(env={"SHARED": "spec", "ONLY_SPEC": "spec"})
    result = merge_environment_spec_into_agent_config(config, spec)
    # EnvironmentSpec overrides the Agent's value on collision; unique keys merge.
    assert result.config["environment"]["env"] == {"SHARED": "spec", "ONLY_SPEC": "spec"}
    # Original is not mutated (deep copy).
    assert config["environment"]["env"] == {"SHARED": "agent"}


def test_merge_environment_mirror_fields_override_agent() -> None:
    config = _agent_config(environment={"provider": "docker", "connection": {"agentkey": "a"}})
    spec = EnvironmentSpecInline(
        provider="k8s",
        workspace_path="/ws",
        artifacts_path="/artifacts",
        control_location="in_env_control",
        ownership="fabric_owned",
        connection={"url": "http://x"},
    )
    env = merge_environment_spec_into_agent_config(config, spec).config["environment"]
    # EnvironmentSpec overrides the Agent's explicitly-set provider.
    assert env["provider"] == "k8s"
    # workspace_path/artifacts_path map onto workspace/artifacts.
    assert env["workspace"] == "/ws"
    assert env["artifacts"] == "/artifacts"
    assert env["control_location"] == "in_env_control"
    assert env["ownership"] == "fabric_owned"
    # Dict fields: Agent keys survive, spec keys win on collision.
    assert env["connection"] == {"agentkey": "a", "url": "http://x"}


def test_merge_environment_mirror_field_unset_spec_keeps_agent_default() -> None:
    # A spec that leaves a scalar unset must not clobber the Agent's value.
    config = _agent_config(environment={"workspace": "/agent-ws"})
    spec = EnvironmentSpecInline(workspace_path="/ws")  # provider left unset
    env = merge_environment_spec_into_agent_config(config, spec).config["environment"]
    assert env["workspace"] == "/ws"


def test_merge_omitted_provider_does_not_clobber_agent() -> None:
    # Regression: provider defaults to "local" (non-None). A spec that never set
    # provider must not overwrite the Agent's explicit "docker".
    config = _agent_config(environment={"provider": "docker"})
    spec = EnvironmentSpecInline(env={"FOO": "bar"})  # provider not set
    env = merge_environment_spec_into_agent_config(config, spec).config["environment"]
    assert env["provider"] == "docker"


def test_merge_explicit_provider_overrides_agent() -> None:
    config = _agent_config(environment={"provider": "docker"})
    spec = EnvironmentSpecInline(provider="k8s")  # explicitly set
    env = merge_environment_spec_into_agent_config(config, spec).config["environment"]
    assert env["provider"] == "k8s"


def test_merge_model_provider_override_overrides_agent() -> None:
    config = _agent_config()
    spec = EnvironmentSpecInline(
        model_provider_override=ModelProviderOverride(
            base_url="https://api.example.com",
            provider="anthropic",
            api_key="MY_SECRET",
        )
    )
    model = merge_environment_spec_into_agent_config(config, spec).config["models"]["default"]
    assert model["base_url"] == "https://api.example.com"
    # EnvironmentSpec's provider overrides the Agent's.
    assert model["provider"] == "anthropic"
    assert model["api_key_env"] == "MY_SECRET"


def test_merge_top_level_secrets_collected_not_in_config() -> None:
    config = _agent_config()
    spec = EnvironmentSpecInline(secrets={"APP_TOKEN": "default/app-token"})
    result = merge_environment_spec_into_agent_config(config, spec)
    assert result.secrets == {"APP_TOKEN": "default/app-token"}
    # Secret refs never land in the config as plaintext env values.
    assert "APP_TOKEN" not in result.config.get("environment", {}).get("env", {})


def test_merge_mcp_fulfills_declared_server_env_wins_and_secrets_collected() -> None:
    config = _agent_config(
        mcp={
            "servers": {
                "search": {
                    "transport": "streamable-http",
                    "url": "http://agent-url",
                    "custom_headers": {"Authorization": "Bearer ${SEARCH_TOKEN}"},
                }
            }
        }
    )
    spec = EnvironmentSpecInline(
        mcp={
            "search": McpFulfillment(
                url="http://env-url",
                env={"E": "1"},
                secrets={"SEARCH_TOKEN": "default/search-token"},
            ),
            # Fulfillment for a server the Agent did not declare — must be ignored.
            "undeclared": McpFulfillment(url="http://new-url"),
        }
    )
    result = merge_environment_spec_into_agent_config(config, spec)
    servers = result.config["mcp"]["servers"]
    # Fulfillment url overrides the Agent's; non-secret env merges in.
    assert servers["search"]["url"] == "http://env-url"
    assert servers["search"]["env"] == {"E": "1"}
    assert servers["search"]["custom_headers"] == {"Authorization": "Bearer ${SEARCH_TOKEN}"}
    # The secret ref is collected for process-env injection, NOT written into the
    # server config as a literal reference string (env-var-name indirection).
    assert "SEARCH_TOKEN" not in servers["search"]["env"]
    assert result.secrets == {"SEARCH_TOKEN": "default/search-token"}
    # An environment cannot add MCP servers the Agent never declared.
    assert "undeclared" not in servers


def test_merge_conflicting_secret_reference_raises() -> None:
    # The same env var name bound to two different secret refs is ambiguous.
    config = _agent_config(mcp={"servers": {"search": {"transport": "streamable-http", "url": "http://a"}}})
    spec = EnvironmentSpecInline(
        secrets={"TOKEN": "default/one"},
        mcp={"search": McpFulfillment(url="http://a", secrets={"TOKEN": "default/two"})},
    )
    with pytest.raises(EnvironmentResolutionError, match="conflicting references"):
        merge_environment_spec_into_agent_config(config, spec)


def test_merge_same_secret_reference_twice_is_ok() -> None:
    # Identical ref for the same name is not a conflict.
    config = _agent_config(mcp={"servers": {"search": {"transport": "streamable-http", "url": "http://a"}}})
    spec = EnvironmentSpecInline(
        secrets={"TOKEN": "default/same"},
        mcp={"search": McpFulfillment(url="http://a", secrets={"TOKEN": "default/same"})},
    )
    result = merge_environment_spec_into_agent_config(config, spec)
    assert result.secrets == {"TOKEN": "default/same"}


def test_merge_rejects_top_level_plaintext_and_secret_same_name() -> None:
    # Same name bound both as plaintext env and as a secret is ambiguous.
    config = _agent_config()
    spec = EnvironmentSpecInline(env={"TOKEN": "plain"}, secrets={"TOKEN": "default/token"})
    with pytest.raises(EnvironmentResolutionError, match="both as a secret and as a plaintext"):
        merge_environment_spec_into_agent_config(config, spec)


def test_merge_rejects_agent_plaintext_and_spec_secret_same_name() -> None:
    # The plaintext binding can come from the Agent config, not just the spec.
    config = _agent_config(environment={"env": {"TOKEN": "agent-plain"}})
    spec = EnvironmentSpecInline(secrets={"TOKEN": "default/token"})
    with pytest.raises(EnvironmentResolutionError, match="both as a secret and as a plaintext"):
        merge_environment_spec_into_agent_config(config, spec)


def test_merge_rejects_mcp_plaintext_and_secret_same_name() -> None:
    config = _agent_config(mcp={"servers": {"search": {"transport": "streamable-http", "url": "http://a"}}})
    spec = EnvironmentSpecInline(
        mcp={"search": McpFulfillment(url="http://a", env={"TOKEN": "plain"}, secrets={"TOKEN": "default/token"})},
    )
    with pytest.raises(EnvironmentResolutionError, match="both as a secret and as a plaintext"):
        merge_environment_spec_into_agent_config(config, spec)


def test_merge_no_environment_reference_is_identical_to_today() -> None:
    # Baseline agent config with no environment merged behaves unchanged.
    config = _agent_config()
    result = merge_environment_spec_into_agent_config(config, None)
    assert result.config == config
    assert result.secrets == {}


class TestRuntimeBaseDir:
    """Fabric resolves both reads and writes against the agent root, so a root the
    runtime cannot write to breaks session startup rather than config loading."""

    @staticmethod
    def _unwritable_root(tmp_path: Path, name: str = "mounted") -> Path:
        mounted = tmp_path / name
        (mounted / "skills").mkdir(parents=True)
        (mounted / "agent.yaml").write_text("config_format: nemo-agents-spec-v1\n")
        (mounted / "skills" / "SKILL.md").write_text(f"# {name}\n")
        # Mirrors a ConfigMap subPath parent: readable and executable, not writable.
        mounted.chmod(0o555)
        return mounted

    def test_a_writable_agent_root_is_used_as_is(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.fabric.environment import resolve_runtime_base_dir

        config = tmp_path / "agent.yaml"
        config.write_text("config_format: nemo-agents-spec-v1\n")

        base_dir = resolve_runtime_base_dir(config)

        assert base_dir.path == tmp_path
        assert base_dir.staged is False

    def test_an_unwritable_agent_root_is_staged_somewhere_writable(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.fabric.environment import resolve_runtime_base_dir

        mounted = self._unwritable_root(tmp_path)
        try:
            base_dir = resolve_runtime_base_dir(mounted / "agent.yaml")

            assert base_dir.path != mounted
            assert base_dir.staged is True
            assert os.access(base_dir.path, os.W_OK)
            # Sibling artifacts must survive the move or skills.paths stops resolving.
            assert (base_dir.path / "skills" / "SKILL.md").read_text() == "# mounted\n"
            assert (base_dir.path / "agent.yaml").exists()
        finally:
            mounted.chmod(0o755)

    def test_each_lifecycle_stages_to_its_own_directory(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.fabric.environment import resolve_runtime_base_dir

        mounted = self._unwritable_root(tmp_path)
        try:
            first = resolve_runtime_base_dir(mounted / "agent.yaml")
            second = resolve_runtime_base_dir(mounted / "agent.yaml")

            assert first.path != second.path
        finally:
            mounted.chmod(0o755)

    def test_a_staged_root_never_inherits_an_earlier_pass(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.fabric.environment import (
            release_runtime_base_dir,
            resolve_runtime_base_dir,
        )

        mounted = self._unwritable_root(tmp_path)
        try:
            first = resolve_runtime_base_dir(mounted / "agent.yaml")
            (first.path / "skills" / "STALE.md").write_text("# dropped skill\n")
            release_runtime_base_dir(first)

            second = resolve_runtime_base_dir(mounted / "agent.yaml")

            assert not (second.path / "skills" / "STALE.md").exists()
            assert (second.path / "skills" / "SKILL.md").exists()
        finally:
            mounted.chmod(0o755)

    def test_two_agent_roots_stage_to_separate_directories(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.fabric.environment import resolve_runtime_base_dir

        roots = [self._unwritable_root(tmp_path, name) for name in ("first", "second")]
        try:
            first = resolve_runtime_base_dir(roots[0] / "agent.yaml")
            second = resolve_runtime_base_dir(roots[1] / "agent.yaml")

            assert first.path != second.path
            assert (first.path / "skills" / "SKILL.md").read_text() == "# first\n"
            assert (second.path / "skills" / "SKILL.md").read_text() == "# second\n"
        finally:
            for mounted in roots:
                mounted.chmod(0o755)

    def test_releasing_a_staged_root_removes_it(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.fabric.environment import (
            release_runtime_base_dir,
            resolve_runtime_base_dir,
        )

        mounted = self._unwritable_root(tmp_path)
        try:
            base_dir = resolve_runtime_base_dir(mounted / "agent.yaml")
            release_runtime_base_dir(base_dir)

            assert not base_dir.path.exists()
        finally:
            mounted.chmod(0o755)

    def test_releasing_an_in_place_root_leaves_the_agent_alone(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.fabric.environment import (
            release_runtime_base_dir,
            resolve_runtime_base_dir,
        )

        config = tmp_path / "agent.yaml"
        config.write_text("config_format: nemo-agents-spec-v1\n")

        base_dir = resolve_runtime_base_dir(config)
        release_runtime_base_dir(base_dir)

        assert config.exists()
        assert tmp_path.exists()
