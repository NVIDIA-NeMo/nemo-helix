# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile ModelDeployments into deployments-plugin entity specifications.

Parity ports from ``k8s_nim_operator/nimservice_compiler.py`` — please review
mapping of ``k8s_nim_operator_config`` → plugin ``K8sDeploymentConfig``.
"""

from dataclasses import dataclass

from nemo_deployments_plugin.entities import (
    Container,
    ContainerPort,
    DeploymentBackendConfig,
    DeploymentConfig,
    DockerVolumeConfig,
    EnvVar,
    HTTPGetAction,
    K8sVolumeConfig,
    Probe,
    Volume,
    VolumeBackendConfig,
    VolumeMount,
    WorkloadIdentitySpec,
)
from nemo_deployments_plugin.secrets import platform_ngc_secret_ref
from nemo_platform_plugin.config import get_platform_config
from nemo_platform_plugin.jobs.image import get_qualified_image
from nmp.common.auth import get_workload_identity_token_audience, is_workload_identity_token_exchange_enabled
from nmp.common.config import Runtime
from nmp.core.models.app import ModelWeightsType
from nmp.core.models.app.constants import MODEL_MANAGED_BY_LABEL, MODEL_MANAGED_BY_MODELS_CONTROLLER
from nmp.core.models.controllers.backends.deployments_plugin.config import DeploymentsPluginConfig
from nmp.core.models.controllers.backends.deployments_plugin.naming import EntityNames, entity_names
from nmp.core.models.controllers.backends.deployments_plugin.nim_compiler import (
    apply_container_resources,
    apply_k8s_nim_operator_container_overrides,
    apply_nim_override_config,
    build_k8s_deployment_backend_config,
    compile_nim_server_env,
    tool_call_plugin_init_containers,
    tool_call_plugin_install_path,
)
from nmp.core.models.controllers.backends.deployments_plugin.resolve import (
    ResolvedPluginDeployment,
    rewrite_loopback_for_docker_container,
)
from nmp.core.models.controllers.backends.engine import (
    ENGINE_GENERIC,
    ENGINE_NIM,
    ENGINE_VLLM,
    config_engine,
    resolve_health_path,
)
from nmp.core.models.controllers.backends.generic_compiler import (
    compile_generic_args,
    compile_generic_env_vars,
    resolve_generic_image,
)
from nmp.core.models.controllers.backends.vllm_compiler import (
    MODEL_STORE_PATH,
    VLLM_SERVER_PORT,
    compile_vllm_args,
    compile_vllm_env_vars,
    resolve_vllm_image,
)

_WEIGHTS_MOUNT = "/model-store"
_SCRATCH_MOUNT = "/scratch"
_LORA_MOUNT = "/scratch/loras"
# The adapters sidecar writes under the XDG base dirs: $XDG_STATE_HOME (default
# ~/.local/state), $XDG_DATA_HOME (default ~/.local/share, via nmp_user_data_dir()),
# and $XDG_CONFIG_HOME (default ~/.config). The pod runs it as the vLLM uid (2000),
# which does not own the nmp-api image's $HOME (/home/nvs, uid 1000), so those defaults
# are unwritable and the sidecar crash-loops. Redirect all three to the writable
# scratch volume, outside the /scratch/loras subtree the adapters controller GCs.
_LORA_SIDECAR_XDG_HOME = f"{_SCRATCH_MOUNT}/.local"
_SCRATCH_VOLUME_SIZE = "1Gi"


@dataclass(frozen=True)
class CompiledModelDeployment:
    """The plugin entities and dependency metadata for one model deployment."""

    names: EntityNames
    volume: Volume | None
    scratch_volume: Volume | None
    puller_config: DeploymentConfig | None
    server_config: DeploymentConfig
    puller_prerequisite: bool


def _labels(resolved: ResolvedPluginDeployment, engine: str, role: str) -> dict[str, str]:
    return {
        MODEL_MANAGED_BY_LABEL: MODEL_MANAGED_BY_MODELS_CONTROLLER,
        "nmp.nvidia.com/deployment-workspace": resolved.deployment.workspace,
        "nmp.nvidia.com/deployment-name": resolved.deployment.name,
        "nmp.nvidia.com/models-role": role,
        "nmp.nvidia.com/engine": engine,
    }


def _image(name: str, tag: str) -> str:
    return name if "@" in name or name.endswith(f":{tag}") else f"{name}:{tag}"


def _busybox_image(config: DeploymentsPluginConfig) -> str:
    return _image(config.busybox_image, config.busybox_image_tag)


def _weights_permissions_init_container(config: DeploymentsPluginConfig, volume_name: str) -> Container:
    """chmod the model store so a non-root server can read the puller's 0600 HF cache."""
    return Container(
        name="weights-permissions-init",
        image=_busybox_image(config),
        command=["sh", "-c", f"chmod -R a+rX {_WEIGHTS_MOUNT}"],
        volumeMounts=[VolumeMount(name=volume_name, mountPath=_WEIGHTS_MOUNT)],
    )


def _weighted(resolved: ResolvedPluginDeployment, engine: str) -> bool:
    is_files = resolved.weights_type == ModelWeightsType.FILES_SERVICE
    if engine == ENGINE_VLLM:
        return is_files or bool(resolved.model_namespace and resolved.model_name)
    if engine == ENGINE_NIM:
        return is_files
    return is_files or bool(resolved.model_entity and resolved.model_entity.fileset)


def _model_deployment_workload_id(resolved: ResolvedPluginDeployment) -> str:
    return f"{resolved.deployment.name}/v{resolved.deployment.entity_version}"


def _workload_identity_for_model(resolved: ResolvedPluginDeployment) -> WorkloadIdentitySpec | None:
    if resolved.deployment.auth_context is None:
        return None
    if not is_workload_identity_token_exchange_enabled():
        return None
    return WorkloadIdentitySpec(
        enabled=True,
        workloadKind="model_deployment",
        workloadId=_model_deployment_workload_id(resolved),
        tokenAudience=get_workload_identity_token_audience(),
    )


def _env(values: dict[str, str]) -> list[EnvVar]:
    return [EnvVar(name=name, value=value) for name, value in values.items()]


def _server_env(engine: str, values: dict[str, str]) -> list[EnvVar]:
    """Compile server env while replacing raw NGC credentials with a secret reference."""
    if engine != ENGINE_NIM:
        return _env(values)

    has_ngc_api_key = "NGC_API_KEY" in values
    env = _env({name: value for name, value in values.items() if name != "NGC_API_KEY"})
    secret_ref = platform_ngc_secret_ref()
    if secret_ref is None:
        if has_ngc_api_key:
            raise ValueError("NIM NGC_API_KEY requires a valid platform.ngc_api_key_secret reference")
        return env
    env.append(EnvVar(name="NGC_API_KEY", secretRef=secret_ref))
    return env


def _apply_gpu_resources(container: Container, gpu: int) -> None:
    """Map executor_config.gpu to nvidia.com/gpu requests and limits."""
    if gpu < 1:
        return
    quantity = str(gpu)
    apply_container_resources(
        container,
        {
            "requests": {"nvidia.com/gpu": quantity},
            "limits": {"nvidia.com/gpu": quantity},
        },
    )


def _container_reachable_platform_base_url(*, runtime: Runtime) -> str:
    """Rewrite ``platform.base_url`` so a Docker container can reach the host API.

    Mirrors jobs (``get_job_runtime_shared_envvars`` / ``_replace_loopback_address``)
    and the deployments auth-proxy (``_upstream_base_url``): when the configured
    base URL is a loopback host, substitute ``platform.loopback_address``, then
    ``determine_loopback_override()``, then ``host.docker.internal`` (agents docker
    path). Non-loopback URLs and non-Docker runtimes are returned unchanged.
    """
    platform = get_platform_config()
    base_url = platform.base_url.rstrip("/")
    return rewrite_loopback_for_docker_container(
        base_url,
        runtime=runtime,
        loopback_address=platform.loopback_address,
    )


def _lora_sidecar(
    resolved: ResolvedPluginDeployment,
    *,
    engine: str,
    config: DeploymentsPluginConfig,
    names: EntityNames,
    weighted: bool,
) -> Container:
    """Build the adapters sidecar with the same env contract as existing backends.

    ``NMP_BASE_URL`` must reach the host platform API from inside the container
    netns. Loopback ``platform.base_url`` values are rewritten for Docker the same
    way jobs rewrite shared env URLs.
    """
    entity_workspace = resolved.model_entity.workspace if resolved.model_entity else resolved.deployment.workspace
    entity_name = resolved.model_entity.name if resolved.model_entity else resolved.deployment.name
    sidecar_env = {
        "NIM_PEFT_SOURCE": _LORA_MOUNT,
        "NIM_PEFT_REFRESH_INTERVAL": str(config.peft_refresh_interval),
        "NMP_MODEL_ENTITY_WORKSPACE": entity_workspace,
        "NMP_MODEL_ENTITY_NAME": entity_name,
        "NMP_BASE_URL": _container_reachable_platform_base_url(runtime=resolved.runtime),
        "XDG_STATE_HOME": _LORA_SIDECAR_XDG_HOME,
        "XDG_DATA_HOME": _LORA_SIDECAR_XDG_HOME,
        "XDG_CONFIG_HOME": _LORA_SIDECAR_XDG_HOME,
    }
    if engine == ENGINE_VLLM:
        sidecar_env["VLLM_LORA_BASE_MODEL_OVERRIDE"] = MODEL_STORE_PATH
        # Native sidecar / pod-local network: talk to the sibling vLLM server.
        sidecar_env["VLLM_ENDPOINT"] = f"http://127.0.0.1:{VLLM_SERVER_PORT}"
    mounts = [VolumeMount(name=names.scratch, mountPath=_SCRATCH_MOUNT)]
    if weighted:
        mounts.append(VolumeMount(name=names.volume, mountPath=_WEIGHTS_MOUNT, readOnly=True))
    return Container(
        name="lora-adapters",
        image=get_qualified_image(config.lora_sidecar_image_name),
        command=config.lora_sidecar_command,
        args=config.lora_sidecar_args,
        env=_env(sidecar_env),
        volumeMounts=mounts,
        restartPolicy="Always",
    )


def compile_model_deployment(
    resolved: ResolvedPluginDeployment,
    config: DeploymentsPluginConfig,
) -> CompiledModelDeployment:
    """Compile a model deployment into deployments-plugin entity specifications.

    Chooses the engine compiler path (nim / vllm / generic), optionally emits a
    weighted puller chain, and maps ``k8s_nim_operator_config`` onto plugin k8s
    backend settings when the platform runtime is Kubernetes.
    """
    engine = config_engine(resolved.config)
    if engine not in {ENGINE_NIM, ENGINE_VLLM, ENGINE_GENERIC}:
        raise ValueError(f"Unsupported engine {engine!r}.")
    names = entity_names(resolved.deployment.name)
    weighted = _weighted(resolved, engine)
    lora_enabled = resolved.view.lora_enabled and engine != ENGINE_GENERIC
    k8s_backend = (
        build_k8s_deployment_backend_config(engine, resolved.view, config)
        if resolved.runtime == Runtime.KUBERNETES
        else None
    )
    backend_config = k8s_backend if k8s_backend is not None and k8s_backend.k8s is not None else None
    workload_identity = _workload_identity_for_model(resolved)
    volume = None
    scratch_volume = None
    puller_config = None
    if weighted:
        # On docker, a freshly created named volume is root-owned (0755) and there
        # is no fs_group equivalent, so the puller image's default non-root user
        # (e.g. `nvs`/uid 1000) cannot create its HF cache under --local-dir
        # (/model-store/.cache). Ask the docker backend to make the volume
        # world-writable on create (busybox `chmod 0777`), the docker analogue of
        # k8s fs_group. No-op on k8s, where pod securityContext/fs_group handles it.
        docker_volume_config = (
            DockerVolumeConfig(initChmod="0777", initImage=_busybox_image(config))
            if resolved.runtime == Runtime.DOCKER
            else None
        )
        volume = Volume(
            name=names.volume,
            workspace=resolved.deployment.workspace,
            size=resolved.view.disk_size or config.default_pvc_size,
            backendConfig=VolumeBackendConfig(
                docker=docker_volume_config,
                k8s=K8sVolumeConfig(storageClass=config.default_storage_class),
            ),
        )
        # `hf download --local-dir` still writes its refs/commit-hash bookkeeping (and
        # incomplete-download staging) into the hub cache, which defaults to
        # $HOME/.cache/huggingface — ephemeral container storage, not the PVC. Point
        # HF_HOME at the mounted PVC so that bookkeeping lands beside the weights;
        # otherwise a large checkpoint overflows ephemeral storage (ENOSPC). Mirrors
        # how other HF consumers scope the cache (unsloth/rl training tasks).
        puller_env = {
            "HF_ENDPOINT": resolved.files_hf_url,
            "HF_TOKEN": "service:models",
            "HF_HOME": _WEIGHTS_MOUNT,
        }
        puller_args = ["download", f"{resolved.model_namespace}/{resolved.model_name}", "--local-dir", _WEIGHTS_MOUNT]
        if resolved.model_revision:
            puller_args.extend(["--revision", resolved.model_revision])
        puller = Container(
            name="weight-puller",
            image=resolved.huggingface_model_puller,
            command=["hf"],
            args=puller_args,
            env=_env(puller_env),
            volumeMounts=[VolumeMount(name=names.volume, mountPath=_WEIGHTS_MOUNT)],
        )
        puller_config = DeploymentConfig(
            name=names.puller,
            workspace=resolved.deployment.workspace,
            containers=[puller],
            labels=_labels(resolved, engine, "puller"),
            restartPolicy="OnFailure",
            backoffLimit=config.max_restart_count,
            backendConfig=backend_config or DeploymentBackendConfig(),
            workloadIdentity=workload_identity,
        )

    tool_call_inits = tool_call_plugin_init_containers(
        resolved,
        config,
        names_volume=names.volume,
        names_scratch=names.scratch,
        weighted=weighted,
    )
    needs_scratch = lora_enabled or tool_call_inits is not None
    tool_call_plugin_path = tool_call_plugin_install_path(weighted=weighted) if tool_call_inits is not None else None

    if engine == ENGINE_VLLM:
        image_name, image_tag = resolve_vllm_image(
            resolved.view, config.default_vllm_image, config.default_vllm_image_tag
        )
        args = compile_vllm_args(resolved.view, resolved.model_entity)
        env = compile_vllm_env_vars(resolved.view)
    elif engine == ENGINE_NIM:
        image_name, image_tag = (
            resolved.view.image_name or config.default_nimservice_image,
            resolved.view.image_tag or config.default_nimservice_image_tag,
        )
        args = list(resolved.view.additional_args or [])
        env = compile_nim_server_env(
            resolved,
            config,
            weighted=weighted,
            tool_call_plugin_path=tool_call_plugin_path,
        )
        if lora_enabled:
            env["NIM_PEFT_SOURCE"] = _LORA_MOUNT
            env["NIM_PEFT_REFRESH_INTERVAL"] = str(config.peft_refresh_interval)
    else:
        image_name, image_tag = resolve_generic_image(resolved.view)
        args = compile_generic_args(resolved.view)
        env = compile_generic_env_vars(resolved.view)

    mounts: list[VolumeMount] = []
    if weighted:
        mounts.append(VolumeMount(name=names.volume, mountPath=_WEIGHTS_MOUNT, readOnly=True))
    init_containers: list[Container] = []
    # Skip when k8s pins a shared uid; otherwise the root puller's 0600 HF cache is unreadable.
    security_context = backend_config.k8s.security_context if backend_config and backend_config.k8s else None
    if weighted and (security_context is None or security_context.run_as_user is None):
        init_containers.append(_weights_permissions_init_container(config, names.volume))
    init_containers.extend(tool_call_inits or [])
    server_config_containers: list[Container]
    readiness_probe = Probe(httpGet=HTTPGetAction(path=resolve_health_path(engine, resolved.view), port=8000))
    vllm_command = ["vllm", "serve"] if engine == ENGINE_VLLM else None
    if needs_scratch:
        scratch_volume = Volume(
            name=names.scratch,
            workspace=resolved.deployment.workspace,
            size=_SCRATCH_VOLUME_SIZE,
            backendConfig=VolumeBackendConfig(k8s=K8sVolumeConfig(storageClass=config.default_storage_class)),
        )
        mounts.append(VolumeMount(name=names.scratch, mountPath=_SCRATCH_MOUNT))
    if lora_enabled:
        # Ensure the LoRA cache dir exists before the server/sidecar start.
        init_containers.append(
            Container(
                name="lora-cache-init",
                image=_busybox_image(config),
                command=["sh", "-c", f"mkdir -p {_LORA_MOUNT} && chmod -R 777 {_LORA_MOUNT}"],
                volumeMounts=[VolumeMount(name=names.scratch, mountPath=_SCRATCH_MOUNT)],
            )
        )
        lora = _lora_sidecar(resolved, engine=engine, config=config, names=names, weighted=weighted)
        server = Container(
            name="server",
            image=_image(image_name, image_tag),
            command=vllm_command or [],
            args=args,
            env=_server_env(engine, env),
            ports=[ContainerPort(name="http", containerPort=8000)],
            volumeMounts=mounts,
            readinessProbe=readiness_probe,
        )
        _apply_gpu_resources(server, resolved.view.gpu)
        if engine == ENGINE_NIM:
            apply_k8s_nim_operator_container_overrides(server, readiness_probe, resolved.view)
        if resolved.runtime == Runtime.DOCKER:
            # Docker emits the sidecar as a second container in the server
            # DeploymentConfig (rather than a k8s-style native/init sidecar).
            # The docker backend starts it sharing the server's network
            # namespace and volumes; its command/args must be a valid
            # executable (see lora_sidecar_command in config).
            server_config_containers = [server, lora]
        else:
            server_config_containers = [server]
            init_containers.append(lora)
    else:
        server = Container(
            name="server",
            image=_image(image_name, image_tag),
            command=vllm_command or [],
            args=args,
            env=_server_env(engine, env),
            ports=[ContainerPort(name="http", containerPort=8000)],
            volumeMounts=mounts,
            readinessProbe=readiness_probe,
        )
        _apply_gpu_resources(server, resolved.view.gpu)
        if engine == ENGINE_NIM:
            apply_k8s_nim_operator_container_overrides(server, readiness_probe, resolved.view)
        server_config_containers = [server]

    server_config = DeploymentConfig(
        name=names.server,
        workspace=resolved.deployment.workspace,
        containers=server_config_containers,
        initContainers=init_containers,
        labels=_labels(resolved, engine, "server"),
        restartPolicy="Always",
        backendConfig=backend_config or DeploymentBackendConfig(),
        workloadIdentity=workload_identity,
    )
    if engine == ENGINE_NIM:
        apply_nim_override_config(
            server,
            server_config,
            resolved.view,
            engine=engine,
            runtime=resolved.runtime,
        )
    return CompiledModelDeployment(
        names=names,
        volume=volume,
        scratch_volume=scratch_volume,
        puller_config=puller_config,
        server_config=server_config,
        puller_prerequisite=puller_config is not None,
    )
