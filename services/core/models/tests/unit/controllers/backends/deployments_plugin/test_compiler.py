# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml
from nemo_deployments_plugin.entities import SecretRef
from nemo_platform_plugin.auth import AuthContext
from nemo_platform_plugin.models.types import K8sNIMOperatorConfig
from nmp.common.config import Runtime
from nmp.core.models.app import ModelWeightsType
from nmp.core.models.controllers.backends.common import DeploymentConfigView
from nmp.core.models.controllers.backends.deployments_plugin.compiler import _busybox_image, compile_model_deployment
from nmp.core.models.controllers.backends.deployments_plugin.config import DeploymentsPluginConfig
from nmp.core.models.controllers.backends.deployments_plugin.nim_compiler import pod_security_context_for_engine
from nmp.core.models.controllers.backends.deployments_plugin.resolve import ResolvedPluginDeployment
from nmp.core.models.controllers.backends.vllm_compiler import MODEL_STORE_PATH


def _resolved(engine: str, *, lora: bool = False, runtime: Runtime = Runtime.KUBERNETES) -> ResolvedPluginDeployment:
    return ResolvedPluginDeployment(
        deployment=SimpleNamespace(name="my-dep", workspace="default", entity_version=1, auth_context=None),
        config=SimpleNamespace(engine=engine),
        model_entity=None,
        view=DeploymentConfigView(model_namespace="org", model_name="model", lora_enabled=lora),
        weights_type=ModelWeightsType.FILES_SERVICE,
        model_namespace="org",
        model_name="model",
        model_revision=None,
        files_hf_url="http://files/hf",
        huggingface_model_puller="puller:latest",
        runtime=runtime,
    )


def test_vllm_weighted_chain_has_on_failure_puller_and_always_server() -> None:
    compiled = compile_model_deployment(_resolved("vllm"), DeploymentsPluginConfig())
    assert compiled.volume is not None
    assert compiled.puller_config is not None and compiled.puller_config.restart_policy == "OnFailure"
    assert compiled.server_config.restart_policy == "Always"
    assert compiled.puller_prerequisite is True
    assert compiled.server_config.containers[0].volume_mounts[0].read_only is True


def test_docker_weights_volume_requests_init_chmod() -> None:
    # Docker named volumes are root-owned with no fs_group, so the weights volume
    # must be made writable (chmod) for the non-root puller. The compiler requests
    # this on the volume's docker backend config (docker analogue of k8s fsGroup).
    config = DeploymentsPluginConfig()
    compiled = compile_model_deployment(_resolved("vllm", runtime=Runtime.DOCKER), config)
    assert compiled.volume is not None
    docker_cfg = compiled.volume.backend_config.docker
    assert docker_cfg is not None
    assert docker_cfg.init_chmod == "0777"
    assert docker_cfg.init_image == _busybox_image(config)


def test_docker_weighted_server_fixes_pulled_weight_permissions() -> None:
    # The volume init chmod runs on an empty volume, so it cannot reach the 0600
    # HF tree manifest the puller writes later. A server init container fixes the
    # pulled contents before the non-root serving image checksums them.
    config = DeploymentsPluginConfig()
    compiled = compile_model_deployment(_resolved("nim", runtime=Runtime.DOCKER), config)
    init = compiled.server_config.init_containers[0]
    assert init.name == "weights-permissions-init"
    assert init.image == _busybox_image(config)
    assert init.command == ["sh", "-c", "chmod -R a+rX /model-store"]
    # The server mounts the weights read-only; the fixup needs write access.
    assert [(mount.name, mount.read_only) for mount in init.volume_mounts] == [(compiled.names.volume, False)]


@pytest.mark.parametrize("engine", ["nim", "vllm"])
def test_k8s_weighted_server_has_no_weights_permissions_init(engine: str) -> None:
    # A pod securityContext pins one uid for both the puller Job and the server, so
    # the manifest is already owned by the serving uid. Chmod-ing there would also
    # fail, since the mount root is owned by the fs_group rather than run_as_user.
    compiled = compile_model_deployment(_resolved(engine, runtime=Runtime.KUBERNETES), DeploymentsPluginConfig())
    assert "weights-permissions-init" not in [init.name for init in compiled.server_config.init_containers]


def test_k8s_generic_engine_fixes_pulled_weight_permissions() -> None:
    # The generic engine defaults to the image's own user, so no pod securityContext
    # is emitted and the root puller's 0600 manifest is unreadable by a non-root
    # serving image -- the same split docker has.
    config = DeploymentsPluginConfig()
    resolved = _resolved("generic", runtime=Runtime.KUBERNETES)
    resolved.view.image_name = "custom/server"
    assert pod_security_context_for_engine("generic", resolved.view, config) is None
    compiled = compile_model_deployment(resolved, config)
    assert compiled.server_config.init_containers[0].name == "weights-permissions-init"


def test_k8s_generic_engine_with_run_as_user_skips_permissions_init() -> None:
    # An explicit run_as_user restores the shared-uid guarantee.
    config = DeploymentsPluginConfig()
    resolved = _resolved("generic", runtime=Runtime.KUBERNETES)
    resolved.view.image_name = "custom/server"
    resolved.view.run_as_user = 1000
    compiled = compile_model_deployment(resolved, config)
    assert "weights-permissions-init" not in [init.name for init in compiled.server_config.init_containers]


def test_k8s_weights_volume_has_no_docker_init_chmod() -> None:
    # On k8s, pod securityContext/fs_group handles volume ownership, so no docker
    # init-chmod is emitted.
    compiled = compile_model_deployment(_resolved("vllm", runtime=Runtime.KUBERNETES), DeploymentsPluginConfig())
    assert compiled.volume is not None
    assert compiled.volume.backend_config.docker is None


def test_vllm_server_command_image_args_and_gpu() -> None:
    resolved = _resolved("vllm")
    resolved = ResolvedPluginDeployment(
        deployment=resolved.deployment,
        config=resolved.config,
        model_entity=resolved.model_entity,
        view=DeploymentConfigView(model_namespace="org", model_name="model", gpu=1),
        weights_type=resolved.weights_type,
        model_namespace=resolved.model_namespace,
        model_name=resolved.model_name,
        model_revision=resolved.model_revision,
        files_hf_url=resolved.files_hf_url,
        huggingface_model_puller=resolved.huggingface_model_puller,
        runtime=resolved.runtime,
    )
    compiled = compile_model_deployment(resolved, DeploymentsPluginConfig())
    server = compiled.server_config.containers[0]
    assert server.command == ["vllm", "serve"]
    assert server.args[0] == MODEL_STORE_PATH
    assert server.image == "docker.io/vllm/vllm-openai:v0.22.1"
    assert server.resources is not None
    assert server.resources.limits["nvidia.com/gpu"] == "1"
    assert compiled.puller_config is not None
    puller = compiled.puller_config.containers[0]
    puller_env = {item.name: item.value for item in puller.env}
    assert puller_env["HF_ENDPOINT"] == resolved.files_hf_url
    assert puller_env["HF_TOKEN"] == "service:models"
    assert puller.command == ["hf"]
    assert puller.args == ["download", "org/model", "--local-dir", "/model-store"]
    assert "nvidia.com/gpu" not in puller.resources.limits


def test_nim_gpu_resources_without_override() -> None:
    resolved = _resolved("nim")
    resolved = ResolvedPluginDeployment(
        deployment=resolved.deployment,
        config=resolved.config,
        model_entity=resolved.model_entity,
        view=DeploymentConfigView(
            model_namespace="org",
            model_name="model",
            gpu=1,
            image_name="nvcr.io/nim/meta/llama-3.1-8b-instruct",
            image_tag="1.8.5",
        ),
        weights_type=ModelWeightsType.BAKED_CONTAINER,
        model_namespace=None,
        model_name=None,
        model_revision=None,
        files_hf_url=resolved.files_hf_url,
        huggingface_model_puller=resolved.huggingface_model_puller,
        runtime=Runtime.KUBERNETES,
    )
    compiled = compile_model_deployment(resolved, DeploymentsPluginConfig())
    server = compiled.server_config.containers[0]
    assert server.resources is not None
    assert server.resources.limits["nvidia.com/gpu"] == "1"


def test_nim_weighted_chain_sets_model_path_env() -> None:
    secret_ref = SecretRef(workspace="system", name="ngc-api-key")
    with patch(
        "nmp.core.models.controllers.backends.deployments_plugin.compiler.platform_ngc_secret_ref",
        return_value=secret_ref,
    ):
        compiled = compile_model_deployment(_resolved("nim"), DeploymentsPluginConfig())
    assert compiled.volume is not None
    assert compiled.puller_config is not None
    env_items = compiled.server_config.containers[0].env
    env = {item.name: item.value for item in env_items}
    assert env["NIM_MODEL_NAME"] == "/model-store"
    assert env["NIM_MODEL_PATH"] == "/model-store"
    assert env["NIM_SERVED_MODEL_NAME"] == "org/model"
    assert env["NIM_FT_MODEL"] == "/model-store"
    assert env["NIM_CUSTOM_MODEL"] == "/model-store"
    ngc_env = next(item for item in env_items if item.name == "NGC_API_KEY")
    assert ngc_env.value is None
    assert ngc_env.secret_ref == secret_ref
    assert "resolved-secret" not in compiled.server_config.model_dump_json()


def test_nim_explicit_ngc_env_is_not_persisted_as_plaintext() -> None:
    resolved = _resolved("nim")
    resolved.view.additional_envs = {"NGC_API_KEY": "explicit-value"}
    with patch(
        "nmp.core.models.controllers.backends.deployments_plugin.compiler.platform_ngc_secret_ref",
        return_value=SecretRef(workspace="system", name="ngc-api-key"),
    ):
        compiled = compile_model_deployment(resolved, DeploymentsPluginConfig())

    ngc_env = next(item for item in compiled.server_config.containers[0].env if item.name == "NGC_API_KEY")
    assert ngc_env.value is None
    assert ngc_env.secret_ref == SecretRef(workspace="system", name="ngc-api-key")
    assert "explicit-value" not in compiled.server_config.model_dump_json(by_alias=True)


def test_nim_explicit_ngc_env_fails_without_platform_secret_ref() -> None:
    resolved = _resolved("nim")
    resolved.view.additional_envs = {"NGC_API_KEY": "explicit-value"}
    with (
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.compiler.platform_ngc_secret_ref",
            return_value=None,
        ),
        pytest.raises(ValueError, match="platform.ngc_api_key_secret"),
    ):
        compile_model_deployment(resolved, DeploymentsPluginConfig())


def test_nim_multi_llm_weighted_omits_ft_model_env() -> None:
    resolved = _resolved("nim")
    resolved = ResolvedPluginDeployment(
        deployment=resolved.deployment,
        config=resolved.config,
        model_entity=resolved.model_entity,
        view=DeploymentConfigView(
            model_namespace="nvidia",
            model_name="Llama-3.1-Nemotron-Nano-4B-v1.1",
            image_name="nvcr.io/nim/nvidia/llm-nim",
        ),
        weights_type=resolved.weights_type,
        model_namespace="nvidia",
        model_name="Llama-3.1-Nemotron-Nano-4B-v1.1",
        model_revision=resolved.model_revision,
        files_hf_url=resolved.files_hf_url,
        huggingface_model_puller=resolved.huggingface_model_puller,
        runtime=resolved.runtime,
    )
    compiled = compile_model_deployment(resolved, DeploymentsPluginConfig())
    env = {item.name: item.value for item in compiled.server_config.containers[0].env}
    assert env["NIM_MODEL_NAME"] == "/model-store"
    assert "NIM_FT_MODEL" not in env
    assert "NIM_CUSTOM_MODEL" not in env
    assert env["NIM_SERVED_MODEL_NAME"] == "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"


def test_nim_tool_call_plugin_adds_init_containers_and_env() -> None:
    resolved = _resolved("nim")
    resolved = ResolvedPluginDeployment(
        deployment=resolved.deployment,
        config=resolved.config,
        model_entity=resolved.model_entity,
        view=DeploymentConfigView(
            model_namespace="org",
            model_name="model",
            tool_call_config=SimpleNamespace(
                tool_call_plugin="test-ws/my-plugin-fileset",
                tool_call_parser=None,
                auto_tool_choice=None,
            ),
        ),
        weights_type=resolved.weights_type,
        model_namespace=resolved.model_namespace,
        model_name=resolved.model_name,
        model_revision=resolved.model_revision,
        files_hf_url=resolved.files_hf_url,
        huggingface_model_puller="nvcr.io/nvidia/model-puller:latest",
        runtime=resolved.runtime,
    )
    compiled = compile_model_deployment(resolved, DeploymentsPluginConfig())
    assert len(compiled.server_config.init_containers) == 3
    pull = compiled.server_config.init_containers[1]
    assert pull.command == ["download", "test-ws/my-plugin-fileset", "--local-dir", "/scratch/plugin"]
    pull_env = {item.name: item.value for item in pull.env}
    assert pull_env["HF_TOKEN"] == "service:models"
    assert pull_env["HF_ENDPOINT"] == resolved.files_hf_url
    server_env = {item.name: item.value for item in compiled.server_config.containers[0].env}
    assert server_env["NIM_TOOL_PARSER_PLUGIN"] == "/model-store/plugin/plugin.py"


def test_nim_k8s_nim_operator_config_maps_tolerations_and_resources() -> None:
    resolved = _resolved("nim")
    resolved = ResolvedPluginDeployment(
        deployment=resolved.deployment,
        config=resolved.config,
        model_entity=resolved.model_entity,
        view=DeploymentConfigView(
            model_namespace="org",
            model_name="model",
            k8s_nim_operator_config=K8sNIMOperatorConfig(
                tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
                node_selector={"node-type": "gpu-node"},
                resources={"requests": {"cpu": "4"}, "limits": {"memory": "32Gi"}},
                startup_probe_grace_seconds=600,
            ),
        ),
        weights_type=resolved.weights_type,
        model_namespace=resolved.model_namespace,
        model_name=resolved.model_name,
        model_revision=resolved.model_revision,
        files_hf_url=resolved.files_hf_url,
        huggingface_model_puller=resolved.huggingface_model_puller,
        runtime=Runtime.KUBERNETES,
    )
    compiled = compile_model_deployment(resolved, DeploymentsPluginConfig())
    k8s = compiled.server_config.backend_config.k8s
    assert k8s is not None
    assert len(k8s.tolerations) == 1
    assert k8s.tolerations[0].key == "nvidia.com/gpu"
    assert k8s.affinity is not None
    server = compiled.server_config.containers[0]
    assert server.resources.requests["cpu"] == "4"
    assert server.resources.limits["memory"] == "32Gi"
    assert server.readiness_probe is not None
    assert server.readiness_probe.failure_threshold == 60
    assert k8s.security_context is not None
    assert k8s.security_context.run_as_user == 1000
    assert k8s.security_context.run_as_group == 2000


def test_nim_override_config_wins_over_k8s_nim_operator_config() -> None:
    resolved = _resolved("nim")
    resolved = ResolvedPluginDeployment(
        deployment=resolved.deployment,
        config=resolved.config,
        model_entity=resolved.model_entity,
        view=DeploymentConfigView(
            model_namespace="org",
            model_name="model",
            k8s_nim_operator_config=K8sNIMOperatorConfig(
                node_selector={"zone": "a"},
                resources={"requests": {"cpu": "2"}},
            ),
            override_config={
                "nodeSelector": {"zone": "override-zone"},
                "resources": {"requests": {"cpu": "16"}},
            },
        ),
        weights_type=ModelWeightsType.BAKED_CONTAINER,
        model_namespace=None,
        model_name=None,
        model_revision=None,
        files_hf_url=resolved.files_hf_url,
        huggingface_model_puller=resolved.huggingface_model_puller,
        runtime=Runtime.KUBERNETES,
    )
    compiled = compile_model_deployment(resolved, DeploymentsPluginConfig())
    server = compiled.server_config.containers[0]
    assert server.resources.requests["cpu"] == "16"
    k8s = compiled.server_config.backend_config.k8s
    assert k8s is not None
    assert k8s.affinity is not None


def test_generic_weightless_is_server_only() -> None:
    resolved = _resolved("generic")
    resolved = ResolvedPluginDeployment(
        deployment=resolved.deployment,
        config=resolved.config,
        model_entity=None,
        view=DeploymentConfigView(
            image_name="custom/image",
            image_tag="1",
            additional_args=["python3", "-m", "http.server", "8000"],
        ),
        weights_type=ModelWeightsType.BAKED_CONTAINER,
        model_namespace=None,
        model_name=None,
        model_revision=None,
        files_hf_url=resolved.files_hf_url,
        huggingface_model_puller=resolved.huggingface_model_puller,
        runtime=resolved.runtime,
    )
    compiled = compile_model_deployment(resolved, DeploymentsPluginConfig())
    assert compiled.volume is None
    assert compiled.puller_config is None
    assert compiled.puller_prerequisite is False
    assert compiled.server_config.containers[0].image == "custom/image:1"
    assert compiled.server_config.containers[0].args == ["python3", "-m", "http.server", "8000"]
    assert compiled.server_config.containers[0].command == []
    k8s = compiled.server_config.backend_config.k8s
    assert k8s is None


def test_lora_uses_native_sidecar_on_k8s_and_container_on_docker() -> None:
    config = DeploymentsPluginConfig()
    platform = MagicMock()
    platform.base_url = "http://platform.example:8080"
    with (
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.compiler.get_qualified_image",
            return_value="registry/nmp-api:tag",
        ),
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.compiler.get_platform_config",
            return_value=platform,
        ),
    ):
        k8s = compile_model_deployment(_resolved("vllm", lora=True), config)
        docker = compile_model_deployment(_resolved("vllm", lora=True, runtime=Runtime.DOCKER), config)

    assert k8s.scratch_volume is not None
    init = k8s.server_config.init_containers[0]
    assert init.name == "lora-cache-init"
    assert init.image == "docker.io/library/busybox:latest"
    sidecar = k8s.server_config.init_containers[-1]
    assert sidecar.restart_policy == "Always"
    assert sidecar.image == "registry/nmp-api:tag"
    assert sidecar.command == ["python", "-m", "nmp.core.models.sidecars.adapters.main"]
    env = {item.name: item.value for item in sidecar.env}
    assert env["NIM_PEFT_SOURCE"] == "/scratch/loras"
    # Sidecar writes under $XDG_STATE_HOME, $XDG_DATA_HOME (via nmp_user_data_dir), and
    # $XDG_CONFIG_HOME. All three must land on the writable scratch volume, not the image
    # $HOME (unwritable under the pod's vLLM uid). Invoking the adapters module directly
    # (not `nemo services run`) avoids the platform runner's home-dir writes that caused
    # PermissionError / READY→PENDING regressions (NVBug 6573168). See _lora_sidecar.
    assert env["XDG_STATE_HOME"] == "/scratch/.local"
    assert env["XDG_DATA_HOME"] == "/scratch/.local"
    assert env["XDG_CONFIG_HOME"] == "/scratch/.local"
    assert env["VLLM_LORA_BASE_MODEL_OVERRIDE"] == "/model-store"
    assert env["NMP_BASE_URL"] == "http://platform.example:8080"
    assert env["VLLM_ENDPOINT"] == "http://127.0.0.1:8000"
    assert len(docker.server_config.containers) == 2
    # The docker sidecar's command becomes docker's entrypoint (replacing the
    # image ENTRYPOINT), so it must be a complete executable list, never a bare
    # `-m` fragment (which runc would look up as the executable -> `exec: "-m"`).
    docker_sidecar = docker.server_config.containers[1]
    assert docker_sidecar.name == "lora-adapters"
    assert docker_sidecar.command == ["python", "-m", "nmp.core.models.sidecars.adapters.main"]
    assert docker_sidecar.command[0] != "-m"


def test_docker_lora_sidecar_command_is_a_valid_executable_under_override() -> None:
    """A LoRA sidecar command overridden in config must still be a valid executable.

    The docker backend maps ``Container.command`` onto docker's ``entrypoint``,
    replacing the image ENTRYPOINT, so a bare ``["-m", "<module>"]`` fragment
    makes runc treat ``-m`` as the executable and fail with ``exec: "-m":
    executable file not found in $PATH``. This guards the config override shape
    (mirroring the local.yaml deployments-plugin section) so that regression is
    caught in unit tests rather than only on a live docker deployment.
    """
    config = DeploymentsPluginConfig(
        lora_sidecar_image_name="nmp-customizer-tasks",
        lora_sidecar_command=["python", "-m", "nmp.core.models.sidecars.adapters.main"],
    )
    platform = MagicMock()
    platform.base_url = "http://platform.example:8080"
    with (
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.compiler.get_qualified_image",
            return_value="registry/nmp-customizer-tasks:tag",
        ),
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.compiler.get_platform_config",
            return_value=platform,
        ),
    ):
        docker = compile_model_deployment(_resolved("vllm", lora=True, runtime=Runtime.DOCKER), config)

    sidecar = docker.server_config.containers[1]
    assert sidecar.name == "lora-adapters"
    # command[0] is what the docker backend uses as the entrypoint executable; it
    # must be a real program, not an interpreter flag.
    assert sidecar.command
    assert sidecar.command[0] != "-m"
    assert sidecar.command == ["python", "-m", "nmp.core.models.sidecars.adapters.main"]


def _repo_root() -> Path:
    """Walk up from this test file to the repo root (the dir holding ``packages/``)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "packages" / "nmp_platform" / "config" / "local.yaml").is_file():
            return parent
    raise AssertionError("could not locate repo root from test file")


@pytest.mark.parametrize(
    "config_rel_path",
    [
        "packages/nmp_platform/config/local.yaml",
        "packages/nmp_platform_runner/src/nmp/platform_runner/config/local.yaml",
    ],
)
def test_shipped_local_yaml_lora_sidecar_command_is_a_valid_executable(config_rel_path: str) -> None:
    """The shipped local.yaml overrides must not reintroduce the bare ``-m`` bug.

    The docker backend maps ``lora_sidecar_command`` onto docker's ``entrypoint``
    (replacing the image ENTRYPOINT), so ``command[0]`` is the executable. A bare
    ``["-m", "<module>"]`` fragment makes runc fail with ``exec: "-m"``. This test
    reads the ACTUAL shipped config files (not a hardcoded config object), so a
    future revert of either ``local.yaml`` back to a relative ``-m`` fragment fails
    here rather than only on a live docker deployment.
    """
    doc = yaml.safe_load((_repo_root() / config_rel_path).read_text())
    plugin = doc["models"]["controller"]["backends"]["deployments_plugin"]
    command = plugin["lora_sidecar_command"]
    assert command, f"{config_rel_path}: lora_sidecar_command must not be empty"
    assert command[0] != "-m", (
        f"{config_rel_path}: lora_sidecar_command[0] is '-m'; the docker backend maps command -> "
        "entrypoint, so command[0] must be a real executable (e.g. 'python'), not an interpreter flag"
    )
    assert not command[0].startswith("-"), (
        f"{config_rel_path}: lora_sidecar_command[0] '{command[0]}' looks like a flag, not an executable"
    )


def test_lora_sidecar_rewrites_loopback_nmp_base_url_for_docker() -> None:
    """Docker LoRA sidecars must not keep host loopback as NMP_BASE_URL (jobs/auth-proxy pattern)."""
    config = DeploymentsPluginConfig()
    platform = MagicMock()
    platform.base_url = "http://127.0.0.1:8080"
    platform.loopback_address = None
    with (
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.compiler.get_qualified_image",
            return_value="registry/nmp-api:tag",
        ),
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.compiler.get_platform_config",
            return_value=platform,
        ),
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.resolve.determine_loopback_override",
            return_value=None,
        ),
    ):
        k8s = compile_model_deployment(_resolved("vllm", lora=True), config)
        docker = compile_model_deployment(_resolved("vllm", lora=True, runtime=Runtime.DOCKER), config)

    k8s_env = {item.name: item.value for item in k8s.server_config.init_containers[-1].env}
    docker_env = {item.name: item.value for item in docker.server_config.containers[1].env}
    assert k8s_env["NMP_BASE_URL"] == "http://127.0.0.1:8080"
    assert docker_env["NMP_BASE_URL"] == "http://host.docker.internal:8080"

    platform.loopback_address = "172.16.83.1"
    with (
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.compiler.get_qualified_image",
            return_value="registry/nmp-api:tag",
        ),
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.compiler.get_platform_config",
            return_value=platform,
        ),
    ):
        docker_bridge = compile_model_deployment(_resolved("vllm", lora=True, runtime=Runtime.DOCKER), config)
    bridge_env = {item.name: item.value for item in docker_bridge.server_config.containers[1].env}
    assert bridge_env["NMP_BASE_URL"] == "http://172.16.83.1:8080"


def test_compile_model_deployment_adds_workload_identity_when_auth_context_present() -> None:
    resolved = _resolved("vllm")
    resolved.deployment.auth_context = AuthContext(principal_id="user:alice", principal_groups=["research"])
    resolved.deployment.entity_version = 3

    with (
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.compiler.is_workload_identity_token_exchange_enabled",
            return_value=True,
        ),
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.compiler.get_workload_identity_token_audience",
            return_value="model-audience",
        ),
    ):
        compiled = compile_model_deployment(resolved, DeploymentsPluginConfig())

    assert compiled.puller_config is not None
    assert compiled.puller_config.workload_identity is not None
    assert compiled.server_config.workload_identity is not None
    assert compiled.puller_config.workload_identity.workload_kind == "model_deployment"
    assert compiled.puller_config.workload_identity.workload_id == "my-dep/v3"
    assert compiled.puller_config.workload_identity.token_audience == "model-audience"
    assert compiled.server_config.workload_identity.workload_id == "my-dep/v3"
    assert compiled.server_config.workload_identity.token_audience == "model-audience"
