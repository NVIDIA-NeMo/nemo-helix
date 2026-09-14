# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DeploymentsRunnerBackend compile + lifecycle helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from nemo_agents_plugin.config import AgentsConfig, DeploymentsRunnerConfig
from nemo_agents_plugin.entities import ComputeResources, DeploymentMode, Endpoint
from nemo_agents_plugin.fabric.gateway_credentials import PLATFORM_IGW_API_KEY_ENV, PLATFORM_IGW_API_KEY_PLACEHOLDER
from nemo_agents_plugin.runner.deployments_backend import (
    DeploymentsRunnerBackend,
    ReservedSecretEnvVarError,
    UnreachableGatewayURLError,
    build_container_resources,
    build_deployment_config,
    executor_for_mode,
    map_status,
    require_executor_matches_mode,
    resolve_agent_gateway_url,
    rewrite_config_base_urls,
    rewrite_fabric_config_base_urls,
)
from nemo_agents_plugin.runner.fabric_artifact_staging import FabricArtifactStagingError
from nemo_deployments_plugin.entities import ConfigFile, Deployment, DeploymentConfig
from nemo_deployments_plugin.types import Endpoint as PluginEndpoint
from nemo_platform_plugin.auth import AuthContext
from nemo_platform_plugin.entities.client import AsyncEntitiesClient
from nemo_platform_plugin.entity_client import NemoEntityNotFoundError


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("PENDING", "starting"),
        ("STARTING", "starting"),
        ("READY", "running"),
        ("FAILED", "failed"),
        ("LOST", "failed"),
        ("DELETING", "deleting"),
        ("SUCCEEDED", "failed"),
        ("UNKNOWN", "starting"),
    ],
)
def test_map_status(backend: str, expected: str) -> None:
    assert map_status(backend) == expected


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "0.0.0.0", "[::1]"])
def test_resolve_docker_rewrites_loopback_to_host(host: str) -> None:
    assert resolve_agent_gateway_url(f"http://{host}:8080", mode="docker") == "http://host.docker.internal:8080"


def test_resolve_docker_passes_through_non_loopback_host() -> None:
    assert resolve_agent_gateway_url("http://my-platform:8080", mode="docker") == "http://my-platform:8080"


def test_resolve_k8s_uses_internal_base_url_regardless_of_base_url() -> None:
    # k8s always returns internal_base_url, never the platform base_url.
    for base_url in (
        "http://localhost:8080",
        "http://nemo-platform-envoy.aire-dev.svc.cluster.local:8080",
        "http://some-other-host:8080",
    ):
        assert (
            resolve_agent_gateway_url(base_url, mode="k8s", internal_base_url="http://nmp-api:8080/")
            == "http://nmp-api:8080"
        )


def test_resolve_k8s_without_internal_base_url_raises() -> None:
    with pytest.raises(UnreachableGatewayURLError, match="internal API Service"):
        resolve_agent_gateway_url("http://localhost:8080", mode="k8s")


def test_resolve_rejects_subprocess_mode() -> None:
    with pytest.raises(ValueError, match="container deployment modes"):
        resolve_agent_gateway_url("http://localhost:8080", mode="subprocess")


def test_resolve_override_wins_verbatim_for_every_mode() -> None:
    assert (
        resolve_agent_gateway_url("http://127.0.0.1:8080", mode="docker", override="http://igw:8080/")
        == "http://igw:8080"
    )
    assert (
        resolve_agent_gateway_url(
            "http://localhost:8080", mode="k8s", override="http://igw:8080/", internal_base_url="http://ignored:8080"
        )
        == "http://igw:8080"
    )


def test_rewrite_config_base_urls_rebases_igw_host() -> None:
    config = {
        "llms": {
            "llm": {
                "_type": "openai",
                "base_url": "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
            }
        }
    }
    result = rewrite_config_base_urls(config, "http://nmp-api:8080")
    assert result["llms"]["llm"]["base_url"] == (
        "http://nmp-api:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    )
    # Original not mutated.
    assert "localhost" in config["llms"]["llm"]["base_url"]


def test_rewrite_config_base_urls_leaves_third_party_base_url() -> None:
    config = {"llms": {"llm": {"_type": "openai", "base_url": "https://api.openai.com/v1"}}}
    result = rewrite_config_base_urls(config, "http://nmp-api:8080")
    assert result["llms"]["llm"]["base_url"] == "https://api.openai.com/v1"


def test_rewrite_fabric_config_base_urls_rebases_igw_host() -> None:
    config = {
        "config_format": "nemo-agents-spec-v1",
        "models": {
            "default": {
                "provider": "openai",
                "model": "test-model",
                "base_url": "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
            }
        },
        "harnesses": {
            "main": {
                "model": {
                    "provider": "openai",
                    "model": "test-model",
                    "base_url": "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
                }
            },
            "legacy": {
                "model": {
                    "provider": "openai",
                    "model": "test-model",
                    "settings": {
                        "base_url": "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
                    },
                }
            },
        },
        "telemetry": {
            "atif": {
                "storage": [
                    {
                        "type": "http",
                        "endpoint": "http://localhost:8080/apis/intake/v2/workspaces/default/ingest/atif",
                    },
                    {"type": "http", "endpoint": "https://telemetry.example.com/atif"},
                ]
            }
        },
    }
    result = rewrite_fabric_config_base_urls(config, "http://nmp-api:8080")
    expected = "http://nmp-api:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    assert result["models"]["default"]["base_url"] == expected
    assert result["harnesses"]["main"]["model"]["base_url"] == expected
    assert result["harnesses"]["legacy"]["model"]["settings"]["base_url"] == expected
    assert result["telemetry"]["atif"]["storage"][0]["endpoint"] == (
        "http://nmp-api:8080/apis/intake/v2/workspaces/default/ingest/atif"
    )
    assert result["telemetry"]["atif"]["storage"][1]["endpoint"] == "https://telemetry.example.com/atif"
    assert "localhost" in config["models"]["default"]["base_url"]
    assert "localhost" in config["telemetry"]["atif"]["storage"][0]["endpoint"]


def test_rewrite_fabric_config_base_urls_leaves_third_party_base_url() -> None:
    config = {
        "models": {
            "default": {
                "provider": "openai",
                "model": "test-model",
                "base_url": "https://api.openai.com/v1",
            }
        }
    }
    result = rewrite_fabric_config_base_urls(config, "http://nmp-api:8080")
    assert result["models"]["default"]["base_url"] == "https://api.openai.com/v1"


def test_rewrite_fabric_config_base_urls_preserves_https_atif_endpoint() -> None:
    config = {
        "telemetry": {
            "atif": {
                "storage": [
                    {
                        "type": "http",
                        "endpoint": "https://localhost:8080/apis/intake/v2/workspaces/default/ingest/atif",
                        "header_env": {"Authorization": "ATIF_AUTHORIZATION"},
                    }
                ]
            }
        }
    }

    result = rewrite_fabric_config_base_urls(config, "http://nmp-api:8080")

    assert result["telemetry"]["atif"]["storage"][0] == {
        "type": "http",
        "endpoint": "https://nmp-api:8080/apis/intake/v2/workspaces/default/ingest/atif",
        "header_env": {"Authorization": "ATIF_AUTHORIZATION"},
    }


def test_executor_for_mode_prefers_mode_specific() -> None:
    cfg = DeploymentsRunnerConfig(
        default_executor="default-exec",
        docker_executor="docker-exec",
        k8s_executor="k8s-exec",
    )
    assert executor_for_mode(cfg, "docker") == "docker-exec"
    assert executor_for_mode(cfg, "k8s") == "k8s-exec"


def _executors(*pairs: tuple[str, str], default: str | None = None) -> Any:
    """A standalone DeploymentsConfig.

    Not ``DeploymentsConfig.get()``: that is a cached singleton, and assigning to
    it outlives the test that did so.
    """
    from nemo_deployments_plugin.config import DeploymentsConfig, ExecutorConfigEntry

    return DeploymentsConfig(
        executors=[ExecutorConfigEntry(name=n, backend=b) for n, b in pairs],
        default_executor=default,
    )


def test_k8s_mode_on_a_docker_executor_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_deployments_plugin.config import DeploymentsConfig

    cfg = _executors(("default-exec", "docker"))
    monkeypatch.setattr(DeploymentsConfig, "get", classmethod(lambda cls: cfg))

    with pytest.raises(ValueError, match="runs on 'docker'"):
        require_executor_matches_mode("default-exec", "k8s")


def test_k8s_mode_on_a_non_deployable_backend_omits_the_mode_suggestion(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_deployments_plugin.config import DeploymentsConfig

    # 'openshell' is a deployments-plugin backend but not a DeploymentMode, so
    # the error must not suggest deploying with a mode that can't validate.
    cfg = _executors(("default-exec", "openshell"))
    monkeypatch.setattr(DeploymentsConfig, "get", classmethod(lambda cls: cfg))

    with pytest.raises(ValueError, match="runs on 'openshell'") as exc_info:
        require_executor_matches_mode("default-exec", "k8s")
    assert "deploy with deployment_mode" not in str(exc_info.value)


def test_k8s_mode_accepts_a_k8s_capable_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_deployments_plugin.config import DeploymentsConfig

    # The naive check — "is k8s_executor set" — would reject this, but a default
    # executor backed by k8s honours the requested mode.
    cfg = _executors(("default-exec", "k8s"))
    monkeypatch.setattr(DeploymentsConfig, "get", classmethod(lambda cls: cfg))

    require_executor_matches_mode("default-exec", "k8s")


def test_k8s_mode_with_no_runner_executors_still_checks_the_plugin_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_deployments_plugin.config import DeploymentsConfig

    # executor_for_mode returns None here, but ExecutorRegistry.resolve falls back
    # to the plugin's own default — so None is not "nothing will run".
    cfg = _executors(("plugin-default", "docker"), default="plugin-default")
    monkeypatch.setattr(DeploymentsConfig, "get", classmethod(lambda cls: cfg))

    with pytest.raises(ValueError, match="runs on 'docker'"):
        require_executor_matches_mode(None, "k8s")


def test_no_executor_anywhere_is_left_to_the_deployments_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_deployments_plugin.config import DeploymentsConfig

    cfg = _executors()
    monkeypatch.setattr(DeploymentsConfig, "get", classmethod(lambda cls: cfg))

    # Nothing to run it on at all is ExecutorNotFoundError's message to give.
    require_executor_matches_mode(None, "k8s")


def test_docker_mode_on_a_docker_executor_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_deployments_plugin.config import DeploymentsConfig

    cfg = _executors(("docker-exec", "docker"))
    monkeypatch.setattr(DeploymentsConfig, "get", classmethod(lambda cls: cfg))

    require_executor_matches_mode("docker-exec", "docker")


def test_an_unknown_executor_is_left_to_the_deployments_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_deployments_plugin.config import DeploymentsConfig

    # Naming an executor that is not configured is the deployments plugin's error
    # to report; guessing here would turn its message into a worse one.
    cfg = _executors(("something-else", "docker"))
    monkeypatch.setattr(DeploymentsConfig, "get", classmethod(lambda cls: cfg))

    require_executor_matches_mode("not-configured", "k8s")


def test_config_mount_path_default_is_under_writable_workspace() -> None:
    assert DeploymentsRunnerConfig().config_mount_path == "/tmp/nemo/config.yaml"


def test_build_deployment_config_always_single_container() -> None:
    cfg = build_deployment_config(
        name="hello-dep",
        workspace="default",
        image="nat-runtime:latest",
        port=8000,
        agent_config={"llms": {"nim": {"_type": "nim"}}},
        platform_base_url="http://host.docker.internal:8080",
        config_mount_path="/tmp/nemo/config.yaml",
        mode="docker",
    )
    assert cfg.restart_policy == "Always"
    assert len(cfg.containers) == 1
    container = cfg.containers[0]
    assert container.image == "nat-runtime:latest"
    assert container.command == ["nat", "start", "fastapi"]
    assert not any(e.name == "NAT_CONFIG_YAML" for e in container.env)
    assert next(e.value for e in container.env if e.name == "NMP_BASE_URL") == "http://host.docker.internal:8080"
    assert container.readiness_probe is not None
    assert cfg.init_containers == []
    assert len(cfg.config_files) == 1
    assert cfg.config_files[0].path == "/tmp/nemo/config.yaml"
    loaded = yaml.safe_load(cfg.config_files[0].content)
    assert loaded["llms"]["nim"]["_type"] == "nim"


def test_build_container_resources_none_is_empty() -> None:
    resources = build_container_resources(None, mode="k8s")
    assert resources.limits == {}
    assert resources.requests == {}


def test_build_container_resources_k8s_passes_both() -> None:
    compute = ComputeResources(limits={"cpu": "2", "nvidia.com/gpu": "1"}, requests={"cpu": "1"})
    resources = build_container_resources(compute, mode="k8s")
    assert resources.limits == {"cpu": "2", "nvidia.com/gpu": "1"}
    assert resources.requests == {"cpu": "1"}


def test_build_container_resources_docker_consolidates_to_limits() -> None:
    compute = ComputeResources(limits={"cpu": "2"}, requests={"cpu": "1", "memory": "1Gi"})
    resources = build_container_resources(compute, mode="docker")
    # Docker has no scheduling requests: requests fold into limits, limits win on collision.
    assert resources.limits == {"cpu": "2", "memory": "1Gi"}
    assert resources.requests == {}


def test_build_deployment_config_applies_k8s_resources() -> None:
    cfg = build_deployment_config(
        name="hello-dep",
        workspace="default",
        image="nat-runtime:latest",
        port=8000,
        agent_config={},
        platform_base_url="http://nmp-api:8080",
        config_mount_path="/workspace/config.yaml",
        mode="k8s",
        resources=ComputeResources(limits={"cpu": "2"}, requests={"cpu": "1"}),
    )
    container = cfg.containers[0]
    assert container.resources.limits == {"cpu": "2"}
    assert container.resources.requests == {"cpu": "1"}


def test_build_deployment_config_docker_resources_limits_only() -> None:
    cfg = build_deployment_config(
        name="hello-dep",
        workspace="default",
        image="nat-runtime:latest",
        port=8000,
        agent_config={},
        platform_base_url="http://host.docker.internal:8080",
        config_mount_path="/tmp/nemo/config.yaml",
        mode="docker",
        resources=ComputeResources(limits={"cpu": "2"}, requests={"memory": "1Gi"}),
    )
    container = cfg.containers[0]
    assert container.resources.limits == {"cpu": "2", "memory": "1Gi"}
    assert container.resources.requests == {}


def test_build_deployment_config_no_resources_is_empty() -> None:
    cfg = build_deployment_config(
        name="hello-dep",
        workspace="default",
        image="nat-runtime:latest",
        port=8000,
        agent_config={},
        platform_base_url="http://nmp-api:8080",
        config_mount_path="/workspace/config.yaml",
        mode="k8s",
    )
    assert cfg.containers[0].resources.limits == {}
    assert cfg.containers[0].resources.requests == {}


def test_build_deployment_config_emits_secret_ref_env_vars() -> None:
    cfg = build_deployment_config(
        name="hello-dep",
        workspace="default",
        image="nat-runtime:latest",
        port=8000,
        agent_config={},
        platform_base_url="http://nmp-api:8080",
        config_mount_path="/workspace/config.yaml",
        mode="k8s",
        secrets={"APP_TOKEN": "default/app-token", "OTHER": "prod/other-secret"},
    )
    by_name = {e.name: e for e in cfg.containers[0].env}
    # Secret-backed env vars carry a secret_ref, never a plaintext value.
    assert by_name["APP_TOKEN"].value is None
    assert by_name["APP_TOKEN"].secret_ref is not None
    assert by_name["APP_TOKEN"].secret_ref.workspace == "default"
    assert by_name["APP_TOKEN"].secret_ref.name == "app-token"
    # A workspace-qualified reference keeps its explicit workspace.
    assert by_name["OTHER"].secret_ref is not None
    assert by_name["OTHER"].secret_ref.workspace == "prod"
    assert by_name["OTHER"].secret_ref.name == "other-secret"
    # The plaintext secret value never appears in the rendered config.
    assert "app-token" not in yaml.safe_dump([e.model_dump() for e in cfg.containers[0].env if e.value])


def test_build_deployment_config_no_secrets_adds_no_secret_env() -> None:
    cfg = build_deployment_config(
        name="hello-dep",
        workspace="default",
        image="nat-runtime:latest",
        port=8000,
        agent_config={},
        platform_base_url="http://nmp-api:8080",
        config_mount_path="/workspace/config.yaml",
        mode="k8s",
    )
    assert all(e.secret_ref is None for e in cfg.containers[0].env)


def test_build_deployment_config_adds_workload_identity_when_requested() -> None:
    with patch(
        "nemo_agents_plugin.runner.deployments_backend.get_workload_identity_token_audience",
        return_value="agent-audience",
    ):
        cfg = build_deployment_config(
            name="hello-dep",
            workspace="default",
            image="nat-runtime:latest",
            port=8000,
            agent_config={},
            platform_base_url="http://nmp-api:8080",
            config_mount_path="/workspace/config.yaml",
            mode="k8s",
            workload_identity_enabled=True,
        )

    assert cfg.workload_identity is not None
    assert cfg.workload_identity.enabled is True
    assert cfg.workload_identity.workload_kind == "agent_deployment"
    assert cfg.workload_identity.workload_id == "hello-dep"
    assert cfg.workload_identity.token_audience == "agent-audience"


@pytest.mark.parametrize("mode", ["docker", "k8s"])
@pytest.mark.parametrize(
    "reserved_name",
    ["NMP_WORKSPACE", "NMP_AGENT_NAME", "NMP_BASE_URL", "PYTHONPATH", "AGENT_CONFIG_PATH", "NAT_CONFIG_PATH"],
)
def test_build_deployment_config_rejects_secret_name_colliding_with_reserved(
    reserved_name: str, mode: DeploymentMode
) -> None:
    # A secret env var whose name collides with a platform-generated container
    # env var is rejected up front (behavior would otherwise differ by substrate).
    with pytest.raises(ReservedSecretEnvVarError, match=reserved_name):
        build_deployment_config(
            name="hello-dep",
            workspace="default",
            image="nat-runtime:latest",
            port=8000,
            agent_config={},
            platform_base_url="http://nmp-api:8080",
            config_mount_path="/workspace/config.yaml",
            mode=mode,
            secrets={reserved_name: "default/some-secret"},
        )


def test_build_deployment_config_rejects_port_secret_when_using_image_entrypoint() -> None:
    with pytest.raises(ReservedSecretEnvVarError, match="PORT"):
        build_deployment_config(
            name="hello-dep",
            workspace="default",
            image="custom-agent:latest",
            port=8000,
            agent_config={},
            platform_base_url="http://nmp-api:8080",
            config_mount_path="/workspace/config.yaml",
            mode="docker",
            secrets={"PORT": "default/some-secret"},
            use_image_entrypoint=True,
        )


def test_build_deployment_config_k8s_uses_nat_entrypoint() -> None:
    cfg = build_deployment_config(
        name="hello-dep",
        workspace="default",
        image="nat-runtime:latest",
        port=8000,
        agent_config={},
        platform_base_url="http://nmp-api:8080",
        config_mount_path="/workspace/config.yaml",
        mode="k8s",
    )
    assert cfg.containers[0].command == ["nat", "start", "fastapi"]
    assert "--host" in cfg.containers[0].args and "0.0.0.0" in cfg.containers[0].args
    assert not any(e.name == "NAT_CONFIG_YAML" for e in cfg.containers[0].env)


def test_build_deployment_config_k8s_option_b_when_image_set() -> None:
    cfg = build_deployment_config(
        name="hello-dep",
        workspace="default",
        image="nat-runtime:latest",
        port=8000,
        agent_config={},
        platform_base_url="http://nmp-api:8080",
        config_mount_path="/workspace/config.yaml",
        mode="k8s",
        plugin_wheels_init_image="busybox:1.36",
    )
    assert len(cfg.init_containers) == 1
    assert cfg.init_containers[0].name == "plugin-wheels"
    assert any(e.name == "PYTHONPATH" for e in cfg.containers[0].env)


def test_build_deployment_config_docker_never_emits_init_containers() -> None:
    cfg = build_deployment_config(
        name="hello-dep",
        workspace="default",
        image="nat-runtime:latest",
        port=8000,
        agent_config={},
        platform_base_url="http://host.docker.internal:8080",
        config_mount_path="/workspace/config.yaml",
        mode="docker",
        plugin_wheels_init_image="busybox:1.36",
    )
    assert cfg.init_containers == []


_FABRIC_AGENT_CONFIG = {
    "config_format": "nemo-agents-spec-v1",
    "name": "fabric-agent",
    "default_harness": "main",
    "harnesses": {
        "main": {
            "provider": "codex",
            "model": {
                "provider": "openai",
                "model": "test-model",
                "base_url": "http://platform/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
                "settings": {},
            },
        }
    },
}


def test_build_deployment_config_fabric_docker_uses_fabric_server() -> None:
    cfg = build_deployment_config(
        name="fabric-dep",
        workspace="default",
        image="fabric-runtime:latest",
        port=8000,
        agent_config=_FABRIC_AGENT_CONFIG,
        platform_base_url="http://host.docker.internal:8080",
        config_mount_path="/tmp/nemo/config.yaml",
        mode="docker",
    )
    container = cfg.containers[0]
    assert container.command == ["python"]
    assert container.args[0] == "-m"
    assert container.args[1] == "nemo_agents_plugin.fabric.server"
    assert "--agent-config" in container.args
    assert "/tmp/nemo/agent.yaml" in container.args
    assert "--host" in container.args and "0.0.0.0" in container.args
    assert not any(e.name == "AGENT_CONFIG_YAML" for e in container.env)
    assert not any(e.name == "NAT_CONFIG_YAML" for e in container.env)
    assert any(e.name == "AGENT_CONFIG_PATH" and e.value == "/tmp/nemo/agent.yaml" for e in container.env)
    assert next(e.value for e in container.env if e.name == "NMP_BASE_URL") == "http://host.docker.internal:8080"
    assert next(e.value for e in container.env if e.name == PLATFORM_IGW_API_KEY_ENV) == (
        PLATFORM_IGW_API_KEY_PLACEHOLDER
    )
    assert cfg.config_files[0].path == "/tmp/nemo/agent.yaml"
    assert container.readiness_probe is not None
    assert container.readiness_probe.http_get is not None
    assert container.readiness_probe.http_get.path == "/health"


def test_build_deployment_config_fabric_can_preserve_image_entrypoint() -> None:
    cfg = build_deployment_config(
        name="fabric-dep",
        workspace="default",
        image="custom-fabric-runtime:latest",
        port=8000,
        agent_config=_FABRIC_AGENT_CONFIG,
        platform_base_url="http://host.docker.internal:8080",
        config_mount_path="/tmp/nemo/config.yaml",
        mode="docker",
        use_image_entrypoint=True,
    )
    container = cfg.containers[0]
    assert container.command == []
    assert container.args == []
    assert any(e.name == "AGENT_CONFIG_PATH" and e.value == "/tmp/nemo/agent.yaml" for e in container.env)
    assert any(e.name == "PORT" and e.value == "8000" for e in container.env)
    assert cfg.config_files[0].path == "/tmp/nemo/agent.yaml"
    assert "nemo_agents_plugin.fabric.server" not in " ".join(container.command + container.args)
    assert container.readiness_probe is not None
    assert container.readiness_probe.http_get is not None
    assert container.readiness_probe.http_get.path == "/health"


def test_build_deployment_config_fabric_k8s_uses_fabric_entrypoint() -> None:
    cfg = build_deployment_config(
        name="fabric-dep",
        workspace="default",
        image="fabric-runtime:latest",
        port=8000,
        agent_config=_FABRIC_AGENT_CONFIG,
        platform_base_url="http://host.docker.internal:8080",
        config_mount_path="/workspace/config.yaml",
        mode="k8s",
    )
    container = cfg.containers[0]
    assert container.command == ["python"]
    assert container.args[0] == "-m"
    assert container.args[1] == "nemo_agents_plugin.fabric.server"
    assert "--agent-config" in container.args
    assert "/workspace/agent.yaml" in container.args
    assert "--host" in container.args and "0.0.0.0" in container.args
    assert not any(e.name == "NAT_CONFIG_YAML" for e in container.env)
    assert next(e.value for e in container.env if e.name == PLATFORM_IGW_API_KEY_ENV) == (
        PLATFORM_IGW_API_KEY_PLACEHOLDER
    )
    assert cfg.config_files[0].path == "/workspace/agent.yaml"


def test_build_deployment_config_fabric_direct_endpoint_has_no_placeholder() -> None:
    config = {
        **_FABRIC_AGENT_CONFIG,
        "harnesses": {
            "main": {
                "provider": "codex",
                "model": {
                    "provider": "openai",
                    "model": "gpt-test",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "OPENAI_API_KEY",
                },
            }
        },
    }

    cfg = build_deployment_config(
        name="fabric-dep",
        workspace="default",
        image="fabric-runtime:latest",
        port=8000,
        agent_config=config,
        platform_base_url="http://host.docker.internal:8080",
        config_mount_path="/workspace/config.yaml",
        mode="docker",
    )

    assert not any(e.name in {PLATFORM_IGW_API_KEY_ENV, "OPENAI_API_KEY"} for e in cfg.containers[0].env)


def test_build_deployment_config_fabric_docker_mounts_multiple_config_files() -> None:
    staged_files = [
        ConfigFile(path="/tmp/nemo/agent.yaml", content="name: fabric-agent\n"),
        ConfigFile(path="/tmp/nemo/skills/review/SKILL.md", content="# Review\n"),
        ConfigFile(path="/tmp/nemo/prompts/system.md", content="You are helpful.\n"),
    ]
    cfg = build_deployment_config(
        name="fabric-dep",
        workspace="default",
        image="fabric-runtime:latest",
        port=8000,
        agent_config=_FABRIC_AGENT_CONFIG,
        platform_base_url="http://host.docker.internal:8080",
        config_mount_path="/tmp/nemo/config.yaml",
        mode="docker",
        config_files=staged_files,
    )
    container = cfg.containers[0]
    assert container.command == ["python"]
    assert not any(e.name == "STAGED_CONFIG_FILES_B64_JSON" for e in container.env)
    assert len(cfg.config_files) == 3
    assert {item.path for item in cfg.config_files} == {
        "/tmp/nemo/agent.yaml",
        "/tmp/nemo/skills/review/SKILL.md",
        "/tmp/nemo/prompts/system.md",
    }


def test_build_deployment_config_fabric_k8s_mounts_multiple_config_files() -> None:
    staged_files = [
        ConfigFile(path="/workspace/agent.yaml", content="name: fabric-agent\n"),
        ConfigFile(path="/workspace/skills/review/SKILL.md", content="# Review\n"),
    ]
    cfg = build_deployment_config(
        name="fabric-dep",
        workspace="default",
        image="fabric-runtime:latest",
        port=8000,
        agent_config=_FABRIC_AGENT_CONFIG,
        platform_base_url="http://nmp-api:8080",
        config_mount_path="/workspace/config.yaml",
        mode="k8s",
        config_files=staged_files,
    )
    container = cfg.containers[0]
    assert container.command == ["python"]
    assert not any(e.name == "STAGED_CONFIG_FILES_B64_JSON" for e in container.env)
    assert len(cfg.config_files) == 2


def _backend(**deployments_kwargs: Any) -> DeploymentsRunnerBackend:
    agents = AgentsConfig.model_validate({"deployments": DeploymentsRunnerConfig(**deployments_kwargs)})
    return DeploymentsRunnerBackend(agents)


def test_entity_client_adapts_sdk_to_typed_entities_client() -> None:
    backend = _backend()
    sdk = MagicMock()
    typed_client = MagicMock()
    entity_client = MagicMock()

    with (
        patch(
            "nemo_agents_plugin.runner.deployments_backend.get_async_platform_sdk",
            return_value=sdk,
        ),
        patch(
            "nemo_agents_plugin.runner.deployments_backend.client_from_platform",
            return_value=typed_client,
        ) as mock_adapter,
        patch(
            "nemo_agents_plugin.runner.deployments_backend.NemoEntitiesClient",
            return_value=entity_client,
        ) as mock_entity_client,
    ):
        result = backend._entity_client()

    mock_adapter.assert_called_once_with(sdk, AsyncEntitiesClient)
    mock_entity_client.assert_called_once_with(typed_client)
    assert result is entity_client


@pytest.mark.asyncio
async def test_create_deployment_writes_config_and_deployment() -> None:
    backend = _backend(default_image="nat:latest", default_executor="local-docker")
    entities = AsyncMock()
    backend._entities = entities

    with patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://127.0.0.1:8080"):
        info = await backend.create_deployment(
            workspace="default",
            name="hello-dep",
            config={"workflow": {"_type": "react_agent"}},
            port=0,
            deployment_mode="docker",
        )

    assert info.status == "starting"
    assert info.endpoint == ""
    assert entities.create.await_count == 2
    created_config = entities.create.await_args_list[0].args[0]
    created_dep = entities.create.await_args_list[1].args[0]
    assert isinstance(created_config, DeploymentConfig)
    assert isinstance(created_dep, Deployment)
    assert created_config.name == "hello-dep"
    assert created_dep.deployment_config == "hello-dep"
    assert created_dep.executor == "local-docker"
    assert created_dep.desired_state == "READY"


@pytest.mark.asyncio
async def test_create_deployment_docker_rewrites_loopback_base_url() -> None:
    backend = _backend(default_image="nat:latest", default_executor="local-docker")
    entities = AsyncMock()
    backend._entities = entities
    config = {
        "llms": {
            "llm": {
                "_type": "openai",
                "base_url": "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
            }
        }
    }
    with patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://localhost:8080"):
        info = await backend.create_deployment(
            workspace="default", name="hello-dep", config=config, port=0, deployment_mode="docker"
        )
    assert info.status == "starting"
    created_config = entities.create.await_args_list[0].args[0]
    baked = yaml.safe_load(created_config.config_files[0].content)
    assert baked["llms"]["llm"]["base_url"] == (
        "http://host.docker.internal:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    )
    assert next(e.value for e in created_config.containers[0].env if e.name == "NMP_BASE_URL") == (
        "http://host.docker.internal:8080"
    )


@pytest.mark.asyncio
async def test_create_deployment_k8s_rewrites_loopback_to_internal() -> None:
    backend = _backend(
        default_image="nat:latest",
        default_executor="k8s",
        k8s_internal_base_url="http://nmp-api:8080",
    )
    entities = AsyncMock()
    backend._entities = entities
    config = {
        "llms": {
            "llm": {
                "_type": "openai",
                "base_url": "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
            }
        }
    }
    with patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://localhost:8080"):
        info = await backend.create_deployment(
            workspace="default", name="hello-dep", config=config, port=0, deployment_mode="k8s"
        )
    assert info.status == "starting"
    created_config = entities.create.await_args_list[0].args[0]
    baked = yaml.safe_load(created_config.config_files[0].content)
    assert baked["llms"]["llm"]["base_url"] == (
        "http://nmp-api:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    )
    assert next(e.value for e in created_config.containers[0].env if e.name == "NMP_BASE_URL") == "http://nmp-api:8080"


@pytest.mark.asyncio
async def test_create_deployment_k8s_without_internal_url_fails() -> None:
    backend = _backend(default_image="nat:latest", default_executor="k8s")
    entities = AsyncMock()
    backend._entities = entities
    with (
        patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://localhost:8080"),
        patch("nemo_agents_plugin.runner.deployments_backend.get_internal_base_url", return_value=None),
    ):
        info = await backend.create_deployment(
            workspace="default", name="hello-dep", config={}, port=0, deployment_mode="k8s"
        )
    assert info.status == "failed"
    assert "internal api service" in info.error.lower()
    entities.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_deployment_k8s_auth_on_requests_auth_proxy_sidecar() -> None:
    # Agents layer only sets the DeploymentConfig flags + points the agent at the
    # proxy port; the deployments plugin compiles the actual sidecar container.
    backend = _backend(
        default_image="nmp-api:latest",
        default_executor="k8s",
        k8s_internal_base_url="http://nmp-api:8080",
    )
    entities = AsyncMock()
    backend._entities = entities
    config = {
        "llms": {
            "llm": {
                "_type": "openai",
                "base_url": "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
            }
        }
    }
    with (
        patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://localhost:8080"),
        patch("nemo_agents_plugin.runner.deployments_backend.platform_auth_enabled", return_value=True),
        patch("nemo_agents_plugin.runner.deployments_backend.auth_proxy_port", return_value=8090),
    ):
        info = await backend.create_deployment(
            workspace="default",
            name="hello-dep",
            config=config,
            port=0,
            deployment_mode="k8s",
            created_by="user:alice",
        )
    assert info.status == "starting"
    created_config = entities.create.await_args_list[0].args[0]

    # The DeploymentConfig requests the sidecar with the agents identity; the
    # agents layer does not build the container itself.
    assert created_config.auth_proxy_sidecar is True
    assert created_config.auth_proxy_sidecar_identity == "agents"
    # The creator principal is delegated via on-behalf-of so the running agent's
    # platform access is scoped to what the creator can reach.
    assert created_config.auth_proxy_sidecar_on_behalf_of == "user:alice"
    assert [c.name for c in created_config.containers] == ["agent"]
    assert [c.name for c in created_config.init_containers] == []

    # Agent's inference base_url points at the loopback proxy port.
    baked = yaml.safe_load(created_config.config_files[0].content)
    assert baked["llms"]["llm"]["base_url"] == (
        "http://127.0.0.1:8090/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    )


@pytest.mark.asyncio
async def test_create_deployment_k8s_auth_on_without_creator_omits_on_behalf_of() -> None:
    # When auth is on but the deployment has no known creator, the sidecar still
    # stamps the service principal but cannot delegate — access is unscoped.
    backend = _backend(
        default_image="nmp-api:latest",
        default_executor="k8s",
        k8s_internal_base_url="http://nmp-api:8080",
    )
    entities = AsyncMock()
    backend._entities = entities
    with (
        patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://localhost:8080"),
        patch("nemo_agents_plugin.runner.deployments_backend.platform_auth_enabled", return_value=True),
        patch("nemo_agents_plugin.runner.deployments_backend.auth_proxy_port", return_value=8090),
    ):
        info = await backend.create_deployment(
            workspace="default", name="hello-dep", config={}, port=0, deployment_mode="k8s"
        )
    assert info.status == "starting"
    created_config = entities.create.await_args_list[0].args[0]
    assert created_config.auth_proxy_sidecar is True
    assert created_config.auth_proxy_sidecar_identity == "agents"
    assert created_config.auth_proxy_sidecar_on_behalf_of is None


@pytest.mark.asyncio
async def test_create_deployment_k8s_auth_off_no_sidecar() -> None:
    backend = _backend(
        default_image="nmp-api:latest",
        default_executor="k8s",
        k8s_internal_base_url="http://nmp-api:8080",
    )
    entities = AsyncMock()
    backend._entities = entities
    config = {
        "llms": {
            "llm": {
                "_type": "openai",
                "base_url": "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
            }
        }
    }
    with (
        patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://localhost:8080"),
        patch("nemo_agents_plugin.runner.deployments_backend.platform_auth_enabled", return_value=False),
    ):
        info = await backend.create_deployment(
            workspace="default", name="hello-dep", config=config, port=0, deployment_mode="k8s"
        )
    assert info.status == "starting"
    created_config = entities.create.await_args_list[0].args[0]
    assert created_config.auth_proxy_sidecar is False
    assert created_config.auth_proxy_sidecar_identity is None
    baked = yaml.safe_load(created_config.config_files[0].content)
    # Auth off: agent talks directly to the internal Service DNS (PR #899 behavior).
    assert baked["llms"]["llm"]["base_url"] == (
        "http://nmp-api:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    )


@pytest.mark.asyncio
async def test_create_deployment_fabric_docker_rewrites_model_base_url() -> None:
    backend = _backend(default_image="fabric:latest", default_executor="local-docker")
    entities = AsyncMock()
    backend._entities = entities
    config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "main",
        "harnesses": {
            "main": {
                "provider": "codex",
                "model": {
                    "provider": "openai",
                    "model": "test-model",
                    "base_url": "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
                },
            }
        },
    }
    with patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://localhost:8080"):
        info = await backend.create_deployment(
            workspace="default", name="fabric-dep", config=config, port=0, deployment_mode="docker"
        )
    assert info.status == "starting"
    created_config = entities.create.await_args_list[0].args[0]
    baked = yaml.safe_load(created_config.config_files[0].content)
    assert baked["harnesses"]["main"]["model"]["base_url"] == (
        "http://host.docker.internal:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    )
    assert created_config.labels["nemo.agents/runtime"] == "fabric"
    assert created_config.containers[0].command == ["python"]
    assert "nemo_agents_plugin.fabric.server" in created_config.containers[0].args


@pytest.mark.asyncio
async def test_create_deployment_fabric_can_preserve_image_entrypoint() -> None:
    backend = _backend(default_image="fabric:latest", default_executor="local-docker")
    entities = AsyncMock()
    backend._entities = entities
    config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "main",
        "harnesses": {
            "main": {
                "provider": "codex",
                "model": {
                    "provider": "openai",
                    "model": "test-model",
                    "base_url": "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
                },
            }
        },
    }
    with patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://localhost:8080"):
        info = await backend.create_deployment(
            workspace="default",
            name="fabric-dep",
            config=config,
            port=0,
            deployment_mode="docker",
            image="hand-built-agent:latest",
            use_image_entrypoint=True,
        )
    assert info.status == "starting"
    created_config = entities.create.await_args_list[0].args[0]
    container = created_config.containers[0]
    assert container.image == "hand-built-agent:latest"
    assert container.command == []
    assert container.args == []
    assert any(e.name == "PORT" and e.value == "8000" for e in container.env)


@pytest.mark.asyncio
async def test_create_deployment_fabric_k8s_rewrites_model_base_url() -> None:
    backend = _backend(
        default_image="fabric:latest",
        default_executor="k8s",
        k8s_internal_base_url="http://nmp-api:8080",
    )
    entities = AsyncMock()
    backend._entities = entities
    config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "main",
        "harnesses": {
            "main": {
                "provider": "codex",
                "model": {
                    "provider": "openai",
                    "model": "test-model",
                    "base_url": "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
                },
            }
        },
    }
    with patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://localhost:8080"):
        info = await backend.create_deployment(
            workspace="default", name="fabric-dep", config=config, port=0, deployment_mode="k8s"
        )
    assert info.status == "starting"
    created_config = entities.create.await_args_list[0].args[0]
    baked = yaml.safe_load(created_config.config_files[0].content)
    assert baked["harnesses"]["main"]["model"]["base_url"] == (
        "http://nmp-api:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    )
    assert created_config.containers[0].command == ["python"]


@pytest.mark.asyncio
async def test_create_deployment_fabric_k8s_auth_on_rewrites_to_auth_proxy() -> None:
    backend = _backend(
        default_image="fabric:latest",
        default_executor="k8s",
        k8s_internal_base_url="http://nmp-api:8080",
    )
    entities = AsyncMock()
    backend._entities = entities
    config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "main",
        "harnesses": {
            "main": {
                "provider": "codex",
                "model": {
                    "provider": "openai",
                    "model": "test-model",
                    "base_url": "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
                },
            }
        },
    }
    with (
        patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://localhost:8080"),
        patch("nemo_agents_plugin.runner.deployments_backend.platform_auth_enabled", return_value=True),
        patch("nemo_agents_plugin.runner.deployments_backend.auth_proxy_port", return_value=8090),
    ):
        info = await backend.create_deployment(
            workspace="default", name="fabric-dep", config=config, port=0, deployment_mode="k8s"
        )
    assert info.status == "starting"
    created_config = entities.create.await_args_list[0].args[0]
    assert created_config.auth_proxy_sidecar is True
    baked = yaml.safe_load(created_config.config_files[0].content)
    assert baked["harnesses"]["main"]["model"]["base_url"] == (
        "http://127.0.0.1:8090/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    )


@pytest.mark.asyncio
async def test_deploying_one_config_twice_does_not_carry_the_first_workspace_over() -> None:
    """Telemetry wiring must not mutate the caller's config.

    ``create_deployment`` receives the deployment entity's own config dict, and
    the wiring keeps an ATIF storage endpoint it finds already present -- so a
    mutated dict would send the second deployment's trajectories to the first
    deployment's workspace.
    """
    backend = _backend(
        default_image="fabric:latest", default_executor="k8s", k8s_internal_base_url="http://nmp-api:8080"
    )
    backend._entities = AsyncMock()
    config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "main",
        # A config the telemetry guard can actually plan: it validates as
        # spec-v1 and names a harness whose adapter advertises Relay ATIF.
        "harnesses": {"main": {"kind": "hermes"}},
        "models": {"default": {"provider": "platform", "model": "default/m"}},
        "environment": {"provider": "local"},
    }

    with patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://localhost:8080"):
        await backend.create_deployment(
            workspace="workspace-a", name="dep-a", config=config, port=0, deployment_mode="k8s"
        )
        assert "telemetry" not in config, "the caller's config was mutated"
        await backend.create_deployment(
            workspace="workspace-b", name="dep-b", config=config, port=0, deployment_mode="k8s"
        )

    # Each deployment creates a DeploymentConfig and a Deployment; only the
    # former carries the baked agent config.
    configs = [
        call.args[0] for call in backend._entities.create.await_args_list if hasattr(call.args[0], "config_files")
    ]
    baked = yaml.safe_load(configs[1].config_files[0].content)
    endpoint = baked["telemetry"]["atif"]["storage"][0]["endpoint"]
    assert "workspace-b" in endpoint and "workspace-a" not in endpoint


@pytest.mark.asyncio
async def test_create_deployment_missing_image_fails() -> None:
    backend = _backend(default_image="")
    info = await backend.create_deployment(
        workspace="default",
        name="hello-dep",
        config={},
        port=0,
        deployment_mode="docker",
    )
    assert info.status == "failed"
    assert "image" in info.error.lower()


@pytest.mark.asyncio
async def test_create_deployment_cleans_config_on_deployment_failure() -> None:
    backend = _backend(default_image="nat:latest")
    entities = AsyncMock()
    entities.create = AsyncMock(side_effect=[None, RuntimeError("boom")])
    entities.delete = AsyncMock()
    backend._entities = entities

    with (
        patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://127.0.0.1:8080"),
        pytest.raises(RuntimeError, match="boom"),
    ):
        await backend.create_deployment(
            workspace="default",
            name="hello-dep",
            config={},
            port=0,
            deployment_mode="docker",
        )

    entities.delete.assert_awaited_once()
    delete_call = entities.delete.await_args
    assert delete_call is not None
    assert delete_call.args[0] is DeploymentConfig
    assert delete_call.kwargs["expected_db_version"] == 1


@pytest.mark.asyncio
async def test_get_deployment_status_projects_endpoints() -> None:
    backend = _backend()
    entities = AsyncMock()
    entities.get = AsyncMock(
        return_value=Deployment(
            name="hello-dep",
            workspace="default",
            deployment_config="hello-dep",
            status="READY",
            endpoints=[PluginEndpoint(name="http", url="http://127.0.0.1:32768", protocol="http")],
        )
    )
    backend._entities = entities

    info = await backend.get_deployment_status("default", "hello-dep")
    assert info is not None
    assert info.status == "running"
    assert info.endpoints == [Endpoint(name="http", url="http://127.0.0.1:32768", protocol="http")]
    assert info.endpoint == "http://127.0.0.1:32768"


@pytest.mark.asyncio
async def test_delete_waits_for_deployment_gone_before_config_delete() -> None:
    backend = _backend()
    entities = AsyncMock()
    deployment = Deployment(
        name="hello-dep",
        workspace="default",
        deployment_config="hello-dep",
        status="READY",
    )
    deployment_config = DeploymentConfig(name="hello-dep", workspace="default")
    # First get returns the deployment; the wait-loop get raises NotFound; final get
    # fetches the config version used for the conditional delete.
    entities.get = AsyncMock(side_effect=[deployment, NemoEntityNotFoundError("gone"), deployment_config])
    entities.update = AsyncMock()
    entities.delete = AsyncMock()
    backend._entities = entities

    with patch("nemo_agents_plugin.runner.deployments_backend.asyncio.sleep", new_callable=AsyncMock):
        cleaned = await backend.delete_deployment("default", "hello-dep")

    assert cleaned
    entities.update.assert_awaited_once()
    assert deployment.status == "DELETING"
    entities.delete.assert_awaited_once()
    delete_call = entities.delete.await_args
    assert delete_call is not None
    assert delete_call.args[0] is DeploymentConfig
    assert delete_call.kwargs["expected_db_version"] == 1


@pytest.mark.asyncio
async def test_delete_returns_false_when_deployment_still_present() -> None:
    backend = _backend()
    entities = AsyncMock()
    deployment = Deployment(
        name="hello-dep",
        workspace="default",
        deployment_config="hello-dep",
        status="READY",
    )
    entities.get = AsyncMock(return_value=deployment)
    entities.update = AsyncMock()
    entities.delete = AsyncMock()
    backend._entities = entities

    with patch("nemo_agents_plugin.runner.deployments_backend.asyncio.sleep", new_callable=AsyncMock):
        with patch("nemo_agents_plugin.runner.deployments_backend.time.monotonic", side_effect=[0.0, 0.0, 6.0]):
            cleaned = await backend.delete_deployment("default", "hello-dep")

    assert not cleaned
    entities.delete.assert_not_called()


def test_agent_deployment_defaults_are_subprocess() -> None:
    from nemo_agents_plugin.entities import AgentDeployment

    dep = AgentDeployment(name="d", workspace="default", agent="a")
    assert dep.deployment_mode == "subprocess"
    assert dep.endpoints == []
    assert dep.image == ""
    assert dep.plugin_deployment == ""


@pytest.mark.asyncio
async def test_create_deployment_fabric_docker_stages_fileset_artifacts() -> None:
    backend = _backend(default_image="fabric:latest", default_executor="local-docker")
    entities = AsyncMock()
    backend._entities = entities
    config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "main",
        "skills": {"paths": ["skills/review"]},
        "harnesses": {"main": {"kind": "codex", "settings": {}}},
    }
    staged_files = [
        ConfigFile(path="/tmp/nemo/agent.yaml", content=yaml.safe_dump(config, sort_keys=False)),
        ConfigFile(path="/tmp/nemo/skills/review/SKILL.md", content="# Review\n"),
    ]
    sdk = MagicMock()
    files_client = object()

    with (
        patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://localhost:8080"),
        patch(
            "nemo_agents_plugin.runner.deployments_backend.get_async_platform_sdk",
            return_value=sdk,
        ),
        patch("nemo_agents_plugin.runner.deployments_backend.client_from_platform", return_value=files_client),
        patch(
            "nemo_agents_plugin.runner.deployments_backend.stage_fabric_ethos_config_files",
            new_callable=AsyncMock,
            return_value=staged_files,
        ) as mock_stage,
    ):
        info = await backend.create_deployment(
            workspace="default",
            name="fabric-dep",
            config=config,
            port=0,
            deployment_mode="docker",
            agent="fabric-agent",
        )

    assert info.status == "starting"
    mock_stage.assert_awaited_once()
    stage_args = mock_stage.await_args
    assert stage_args is not None
    assert stage_args.kwargs["files_client"] is files_client
    created_config = entities.create.await_args_list[0].args[0]
    assert len(created_config.config_files) == 2
    assert not any(e.name == "STAGED_CONFIG_FILES_B64_JSON" for e in created_config.containers[0].env)


@pytest.mark.asyncio
async def test_create_deployment_fabric_k8s_stages_fileset_artifacts() -> None:
    backend = _backend(
        default_image="fabric:latest",
        default_executor="k8s",
        k8s_internal_base_url="http://nmp-api:8080",
    )
    entities = AsyncMock()
    backend._entities = entities
    config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "main",
        "skills": {"paths": ["skills/review"]},
        "harnesses": {"main": {"kind": "codex", "settings": {}}},
    }
    staged_files = [
        ConfigFile(path="/tmp/nemo/agent.yaml", content=yaml.safe_dump(config, sort_keys=False)),
        ConfigFile(path="/tmp/nemo/skills/review/SKILL.md", content="# Review\n"),
    ]
    sdk = MagicMock()
    files_client = object()

    with (
        patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://localhost:8080"),
        patch(
            "nemo_agents_plugin.runner.deployments_backend.get_async_platform_sdk",
            return_value=sdk,
        ),
        patch("nemo_agents_plugin.runner.deployments_backend.client_from_platform", return_value=files_client),
        patch(
            "nemo_agents_plugin.runner.deployments_backend.stage_fabric_ethos_config_files",
            new_callable=AsyncMock,
            return_value=staged_files,
        ),
    ):
        info = await backend.create_deployment(
            workspace="default",
            name="fabric-dep",
            config=config,
            port=0,
            deployment_mode="k8s",
            agent="fabric-agent",
        )

    assert info.status == "starting"
    created_config = entities.create.await_args_list[0].args[0]
    assert len(created_config.config_files) == 2
    assert created_config.containers[0].command == ["python"]


@pytest.mark.asyncio
async def test_create_deployment_fabric_staging_error_fails_before_entity_create() -> None:
    backend = _backend(default_image="fabric:latest", default_executor="local-docker")
    entities = AsyncMock()
    backend._entities = entities
    config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "main",
        "skills": {"paths": ["skills/review"]},
        "harnesses": {"main": {"kind": "codex", "settings": {}}},
    }
    sdk = MagicMock()
    files_client = object()

    with (
        patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://localhost:8080"),
        patch(
            "nemo_agents_plugin.runner.deployments_backend.get_async_platform_sdk",
            return_value=sdk,
        ),
        patch("nemo_agents_plugin.runner.deployments_backend.client_from_platform", return_value=files_client),
        patch(
            "nemo_agents_plugin.runner.deployments_backend.stage_fabric_ethos_config_files",
            new_callable=AsyncMock,
            side_effect=FabricArtifactStagingError("missing skills/review"),
        ),
    ):
        info = await backend.create_deployment(
            workspace="default",
            name="fabric-dep",
            config=config,
            port=0,
            deployment_mode="docker",
            agent="fabric-agent",
        )

    assert info.status == "failed"
    assert "skills/review" in info.error
    entities.create.assert_not_called()


@pytest.mark.asyncio
async def test_create_deployment_enables_workload_identity_with_auth_context() -> None:
    backend = _backend(default_image="nat:latest", default_executor="local-docker")
    entities = AsyncMock()
    backend._entities = entities
    auth_context = AuthContext(principal_id="user:alice", principal_groups=["research"])

    with (
        patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://127.0.0.1:8080"),
        patch(
            "nemo_agents_plugin.runner.deployments_backend.is_workload_identity_token_exchange_enabled",
            return_value=True,
        ),
        patch(
            "nemo_agents_plugin.runner.deployments_backend.get_workload_identity_token_audience",
            return_value="agent-audience",
        ),
    ):
        info = await backend.create_deployment(
            workspace="default",
            name="hello-dep",
            config={"workflow": {"_type": "react_agent"}},
            port=0,
            deployment_mode="docker",
            auth_context=auth_context,
        )

    assert info.status == "starting"
    created_config = entities.create.await_args_list[0].args[0]
    created_dep = entities.create.await_args_list[1].args[0]
    assert created_config.workload_identity is not None
    assert created_config.workload_identity.enabled is True
    assert created_config.workload_identity.workload_id == "hello-dep"
    assert created_config.workload_identity.token_audience == "agent-audience"
    assert created_dep.auth_context == auth_context
