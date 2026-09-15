# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIM-specific compilation helpers for the deployments-plugin models backend.

Parity ports from ``k8s_nim_operator/nimservice_compiler.py`` — please review
mapping of ``k8s_nim_operator_config`` → plugin ``K8sDeploymentConfig``.
"""

from __future__ import annotations

import math
from typing import Any

from nemo_deployments_plugin.entities import (
    Affinity,
    Container,
    DeploymentBackendConfig,
    DeploymentConfig,
    EnvVar,
    ExecAction,
    HTTPGetAction,
    K8sDeploymentConfig,
    PodSecurityContext,
    Probe,
    ResourceRequirements,
    Toleration,
    VolumeMount,
)
from nemo_platform_plugin.models.types import K8sNIMOperatorConfig, ModelEntity
from nmp.common.config import Runtime
from nmp.core.models.app import is_multi_llm_image, parse_model_name_revision
from nmp.core.models.controllers.backends.common import DeploymentConfigView
from nmp.core.models.controllers.backends.deployments_plugin.config import DeploymentsPluginConfig
from nmp.core.models.controllers.backends.deployments_plugin.resolve import ResolvedPluginDeployment
from nmp.core.models.controllers.backends.engine import ENGINE_GENERIC, ENGINE_NIM, ENGINE_VLLM

_WEIGHTS_MOUNT = "/model-store"
_SCRATCH_MOUNT = "/scratch"
_TOOL_CALL_PLUGIN_PATH = "/model-store/plugin/plugin.py"
_TOOL_CALL_PLUGIN_SCRATCH_PATH = "/scratch/plugin/plugin.py"
_TOOL_CALL_PLUGIN_SCRATCH_DIR = "/scratch/plugin"
_TOOL_CALL_PLUGIN_FINALIZE_SCRIPT = """set -euo pipefail
py_files="$(find "{scratch_dir}" -type f -name '*.py' || true)"
count="$(printf '%s\\n' "$py_files" | sed '/^$/d' | wc -l | tr -d ' ')"
if [ "$count" -eq 0 ]; then
  echo "tool_call_plugin fileset contains no .py files"
  exit 1
fi
if [ "$count" -ne 1 ]; then
  echo "tool_call_plugin fileset must contain exactly one .py file, found $count"
  printf '%s\\n' "$py_files"
  exit 1
fi
plugin_file="$(printf '%s\\n' "$py_files" | sed '/^$/d' | sed -n '1p')"
if [ "$plugin_file" != "{plugin_path}" ]; then
  mv "$plugin_file" "{plugin_path}"
fi
"""

_SUPPORTED_NIM_OVERRIDE_CONFIG_KEYS = frozenset(
    {
        "image",
        "command",
        "args",
        "resources",
        "env",
        "readinessProbe",
        "livenessProbe",
        "startupProbe",
        "nodeSelector",
        "tolerations",
        "userID",
        "groupID",
        "labels",
        "initContainers",
        "sidecarContainers",
    }
)


def _plugin_fileset(view: DeploymentConfigView, model_entity: ModelEntity | None) -> str | None:
    if view.tool_call_config and view.tool_call_config.tool_call_plugin:
        return view.tool_call_config.tool_call_plugin
    if (
        model_entity
        and model_entity.spec
        and model_entity.spec.tool_call_config
        and model_entity.spec.tool_call_config.tool_call_plugin
    ):
        return model_entity.spec.tool_call_config.tool_call_plugin
    return None


def _puller_image_parts(image: str) -> tuple[str, str] | None:
    last_slash_idx = image.rfind("/")
    last_colon_idx = image.rfind(":")
    if last_colon_idx <= last_slash_idx:
        return None
    return image[:last_colon_idx], image[last_colon_idx + 1 :]


def tool_call_plugin_install_path(*, weighted: bool) -> str:
    """Return the in-container path where the tool-call plugin is installed."""
    return _TOOL_CALL_PLUGIN_PATH if weighted else _TOOL_CALL_PLUGIN_SCRATCH_PATH


def tool_call_plugin_init_containers(
    resolved: ResolvedPluginDeployment,
    config: DeploymentsPluginConfig,
    *,
    names_volume: str,
    names_scratch: str,
    weighted: bool,
) -> list[Container] | None:
    """Build init containers that fetch and install a tool-call plugin fileset."""
    plugin_fileset = _plugin_fileset(resolved.view, resolved.model_entity)
    if plugin_fileset is None:
        return None
    if not resolved.huggingface_model_puller:
        return None
    puller_parts = _puller_image_parts(resolved.huggingface_model_puller)
    if puller_parts is None:
        return None
    puller_repo, puller_tag = puller_parts
    scratch_mount = VolumeMount(name=names_scratch, mountPath=_SCRATCH_MOUNT)
    weights_mount = VolumeMount(name=names_volume, mountPath=_WEIGHTS_MOUNT)
    plugin_path = tool_call_plugin_install_path(weighted=weighted)
    busybox = _image(config.busybox_image, config.busybox_image_tag)
    prepare_mounts = [scratch_mount]
    finalize_mounts = [scratch_mount]
    if weighted:
        prepare_mounts.append(weights_mount)
        finalize_mounts.append(weights_mount)
        prepare_script = (
            "set -e; mkdir -p /model-store/plugin /scratch/plugin; "
            f"rm -f {_TOOL_CALL_PLUGIN_PATH}; rm -rf /scratch/plugin/*"
        )
    else:
        prepare_script = f"set -e; mkdir -p /scratch/plugin; rm -f {plugin_path}; rm -rf /scratch/plugin/*"
    return [
        Container(
            name="tool-call-plugin-prepare",
            image=busybox,
            command=["sh", "-c", prepare_script],
            volumeMounts=prepare_mounts,
        ),
        Container(
            name="tool-call-plugin-pull",
            image=f"{puller_repo}:{puller_tag}",
            command=["download", plugin_fileset, "--local-dir", _TOOL_CALL_PLUGIN_SCRATCH_DIR],
            env=[
                EnvVar(name="HF_ENDPOINT", value=resolved.files_hf_url),
                EnvVar(name="HF_TOKEN", value="service:models"),
            ],
            volumeMounts=[scratch_mount],
        ),
        Container(
            name="tool-call-plugin-finalize",
            image=busybox,
            command=[
                "sh",
                "-c",
                _TOOL_CALL_PLUGIN_FINALIZE_SCRIPT.format(
                    scratch_dir=_TOOL_CALL_PLUGIN_SCRATCH_DIR,
                    plugin_path=plugin_path,
                ),
            ],
            volumeMounts=finalize_mounts,
        ),
    ]


def compile_nim_server_env(
    resolved: ResolvedPluginDeployment,
    config: DeploymentsPluginConfig,
    *,
    weighted: bool,
    tool_call_plugin_path: str | None,
) -> dict[str, str]:
    """Compile NIM server environment variables."""
    view = resolved.view
    env: dict[str, str] = dict(view.additional_envs or {})

    model_fqdn: str | None = None
    if view.model_name:
        parsed_namespace, parsed_name, parsed_revision = parse_model_name_revision(
            model_namespace=view.model_namespace,
            model_name=view.model_name,
            model_revision=view.model_revision,
        )
        if parsed_namespace and parsed_name:
            model_fqdn = f"{parsed_namespace}/{parsed_name}"
        elif parsed_name:
            model_fqdn = parsed_name
        if model_fqdn and parsed_revision:
            model_fqdn += f"@{parsed_revision}"

    if model_fqdn:
        env["NIM_SERVED_MODEL_NAME"] = model_fqdn

    if weighted:
        env["NIM_MODEL_NAME"] = _WEIGHTS_MOUNT
        env["NIM_MODEL_PATH"] = _WEIGHTS_MOUNT
        effective_image = view.image_name or config.default_nimservice_image
        if not is_multi_llm_image(effective_image):
            env["NIM_FT_MODEL"] = _WEIGHTS_MOUNT
            env["NIM_CUSTOM_MODEL"] = _WEIGHTS_MOUNT
    elif resolved.model_name:
        served = (
            f"{resolved.model_namespace}/{resolved.model_name}" if resolved.model_namespace else resolved.model_name
        )
        env.setdefault("NIM_SERVED_MODEL_NAME", served)

    model_entity = resolved.model_entity
    if model_entity:
        env["NMP_MODEL_ENTITY_WORKSPACE"] = model_entity.workspace
        env["NMP_MODEL_ENTITY_NAME"] = model_entity.name
        if model_entity.trust_remote_code:
            env["NIM_FORCE_TRUST_REMOTE_CODE"] = "1"
            env["NIM_TRUST_CUSTOM_CODE"] = "1"
        if model_entity.spec:
            if model_entity.spec.chat_template:
                env["NIM_CHAT_TEMPLATE"] = model_entity.spec.chat_template
            tool_cfg = model_entity.spec.tool_call_config
            if tool_cfg:
                if tool_cfg.tool_call_parser:
                    env["NIM_TOOL_CALL_PARSER"] = tool_cfg.tool_call_parser
                if tool_call_plugin_path:
                    env["NIM_TOOL_PARSER_PLUGIN"] = tool_call_plugin_path
                if tool_cfg.auto_tool_choice is not None:
                    env["NIM_ENABLE_AUTO_TOOL_CHOICE"] = "1" if tool_cfg.auto_tool_choice else "0"

    if view.chat_template:
        env["NIM_CHAT_TEMPLATE"] = view.chat_template

    deploy_tool_cfg = view.tool_call_config
    if deploy_tool_cfg:
        if deploy_tool_cfg.tool_call_parser:
            env["NIM_TOOL_CALL_PARSER"] = deploy_tool_cfg.tool_call_parser
        if deploy_tool_cfg.tool_call_plugin and tool_call_plugin_path:
            env["NIM_TOOL_PARSER_PLUGIN"] = tool_call_plugin_path
        if deploy_tool_cfg.auto_tool_choice is not None:
            env["NIM_ENABLE_AUTO_TOOL_CHOICE"] = "1" if deploy_tool_cfg.auto_tool_choice else "0"

    return env


def _image(name: str, tag: str) -> str:
    return name if "@" in name or name.endswith(f":{tag}") else f"{name}:{tag}"


def _k8s_config_dict(k8s_config: K8sNIMOperatorConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(k8s_config, K8sNIMOperatorConfig):
        return k8s_config.model_dump(exclude_none=True)
    return {key: value for key, value in k8s_config.items() if value is not None}


def _tolerations_from_config(raw: list[dict[str, Any]]) -> list[Toleration]:
    tolerations: list[Toleration] = []
    for item in raw:
        if isinstance(item, dict):
            tolerations.append(Toleration(**{key: value for key, value in item.items() if value is not None}))
    return tolerations


def _affinity_from_node_selector(node_selector: dict[str, str]) -> Affinity:
    return Affinity.model_validate(
        {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {"key": key, "operator": "In", "values": [value]}
                                for key, value in node_selector.items()
                            ]
                        }
                    ]
                }
            }
        }
    )


def k8s_backend_config_from_nim_operator(view: DeploymentConfigView) -> K8sDeploymentConfig | None:
    """Map per-deployment ``k8s_nim_operator_config`` onto plugin ``K8sDeploymentConfig``."""
    if view.k8s_nim_operator_config is None:
        return None
    config_dict = _k8s_config_dict(view.k8s_nim_operator_config)
    if not config_dict:
        return None

    k8s_kwargs: dict[str, Any] = {}
    tolerations = config_dict.get("tolerations")
    if isinstance(tolerations, list):
        parsed = _tolerations_from_config(tolerations)
        if parsed:
            k8s_kwargs["tolerations"] = parsed
    node_selector = config_dict.get("node_selector")
    if isinstance(node_selector, dict) and node_selector:
        k8s_kwargs["affinity"] = _affinity_from_node_selector(node_selector)
    if not k8s_kwargs:
        return None
    return K8sDeploymentConfig(**k8s_kwargs)


def pod_security_context_for_engine(
    engine: str,
    view: DeploymentConfigView,
    config: DeploymentsPluginConfig,
) -> PodSecurityContext | None:
    """Map backend default uid/gid onto pod securityContext for k8s workloads."""
    if engine == ENGINE_VLLM:
        user_id = view.run_as_user if view.run_as_user is not None else config.default_vllm_user_id
        group_id = view.run_as_group if view.run_as_group is not None else config.default_vllm_group_id
    elif engine == ENGINE_GENERIC:
        # Generic containers use the image's own user unless run_as_* is set explicitly.
        user_id = view.run_as_user
        group_id = view.run_as_group
    else:
        user_id = view.run_as_user if view.run_as_user is not None else config.default_user_id
        group_id = view.run_as_group if view.run_as_group is not None else config.default_group_id
    if user_id is None and group_id is None:
        return None
    return PodSecurityContext.model_validate(
        {
            "runAsUser": user_id,
            "runAsGroup": group_id,
            "fsGroup": group_id,
        }
    )


def _default_tolerations(config: DeploymentsPluginConfig) -> list[Toleration]:
    """Parse platform-default tolerations from config into plugin Toleration models."""
    return _tolerations_from_config([item for item in config.default_tolerations if isinstance(item, dict)])


def build_k8s_deployment_backend_config(
    engine: str,
    view: DeploymentConfigView,
    config: DeploymentsPluginConfig,
) -> DeploymentBackendConfig:
    """Merge operator overrides, engine security defaults, and platform defaults into backend_config.k8s.

    Platform defaults (``default_pod_annotations``, ``default_node_selector``,
    ``default_tolerations``, ``default_affinity``, ``default_topology_spread_constraints``)
    apply to every engine (nim/vllm/generic). Annotations merge key-wise with a per-entity
    value winning over the platform default for the same key; per-entity
    tolerations/node_selector/affinity/topology-spread (nim operator config) win wholesale
    over the platform default when present. This helper only runs on the Kubernetes
    runtime, so the defaults are k8s-only by construction.
    """
    if engine == ENGINE_NIM:
        k8s = k8s_backend_config_from_nim_operator(view) or K8sDeploymentConfig()
    else:
        k8s = K8sDeploymentConfig()
    security_context = pod_security_context_for_engine(engine, view, config)
    if security_context is not None:
        k8s.security_context = security_context

    # Platform-default pod annotations (all engines), key-wise merge: platform
    # default first, per-entity annotation for the same key wins.
    if config.default_pod_annotations:
        merged_annotations = {**config.default_pod_annotations, **k8s.pod_annotations}
        k8s.pod_annotations = merged_annotations

    # Platform-default node selector (all engines); a per-entity node selector
    # (mapped onto affinity by the nim operator path) wins when present.
    if config.default_node_selector and k8s.affinity is None:
        k8s.node_selector = {**config.default_node_selector, **k8s.node_selector}

    # Platform-default tolerations (all engines); per-entity tolerations win wholesale.
    if config.default_tolerations and not k8s.tolerations:
        k8s.tolerations = _default_tolerations(config)

    # Platform-default affinity (all engines); a per-entity affinity (including the
    # node-selector-derived affinity from the nim operator path) wins wholesale.
    if config.default_affinity and k8s.affinity is None:
        k8s.affinity = Affinity.model_validate(config.default_affinity)

    # Platform-default topology spread constraints (all engines); per-entity
    # constraints win wholesale.
    if config.default_topology_spread_constraints and not k8s.topology_spread_constraints:
        k8s.topology_spread_constraints = [dict(item) for item in config.default_topology_spread_constraints]

    if any(
        (
            k8s.tolerations,
            k8s.affinity,
            k8s.security_context,
            k8s.namespace,
            k8s.service_account,
            k8s.node_selector,
            k8s.pod_annotations,
            k8s.topology_spread_constraints,
        )
    ):
        return DeploymentBackendConfig(k8s=k8s)
    return DeploymentBackendConfig()


def startup_probe_failure_threshold(view: DeploymentConfigView, *, period_seconds: int = 10) -> int | None:
    """Derive readiness failure threshold from ``startup_probe_grace_seconds``."""
    if view.k8s_nim_operator_config is None:
        return None
    config_dict = _k8s_config_dict(view.k8s_nim_operator_config)
    grace_seconds = config_dict.get("startup_probe_grace_seconds")
    if grace_seconds is None:
        return None
    return max(1, math.ceil(int(grace_seconds) / period_seconds))


def apply_container_resources(container: Container, resources: dict[str, Any]) -> None:
    """Apply k8s resource requirements to a plugin container."""
    raw_requests = resources.get("requests")
    raw_limits = resources.get("limits")
    requests = raw_requests if isinstance(raw_requests, dict) else {}
    limits = raw_limits if isinstance(raw_limits, dict) else {}
    if not requests and not limits:
        return
    existing = container.resources
    merged_requests = dict(existing.requests) if existing and existing.requests else {}
    merged_limits = dict(existing.limits) if existing and existing.limits else {}
    merged_requests.update({str(key): str(value) for key, value in requests.items()})
    merged_limits.update({str(key): str(value) for key, value in limits.items()})
    container.resources = ResourceRequirements(
        requests=merged_requests,
        limits=merged_limits,
    )


def apply_k8s_nim_operator_container_overrides(
    container: Container,
    probe: Probe,
    view: DeploymentConfigView,
) -> None:
    """Apply per-deployment operator config fields that target the server container."""
    if view.k8s_nim_operator_config is None:
        return
    config_dict = _k8s_config_dict(view.k8s_nim_operator_config)
    resources = config_dict.get("resources")
    if isinstance(resources, dict):
        apply_container_resources(container, resources)
    failure_threshold = startup_probe_failure_threshold(view, period_seconds=probe.period_seconds)
    if failure_threshold is not None:
        probe.failure_threshold = failure_threshold


def _resource_string(value: Any) -> str:
    if isinstance(value, dict) and "root" in value:
        return str(value["root"])
    return str(value)


def _normalize_resources(resources: dict[str, Any]) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    for section in ("requests", "limits"):
        raw = resources.get(section)
        if isinstance(raw, dict):
            normalized[section] = {str(key): _resource_string(val) for key, val in raw.items()}
    return normalized


def _merge_env_override(container: Container, env_list: list[Any]) -> None:
    by_name = {item.name: index for index, item in enumerate(container.env)}
    for item in env_list:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        value = item.get("value")
        if value is None:
            continue
        env_var = EnvVar(name=str(name), value=str(value))
        if name in by_name:
            container.env[by_name[name]] = env_var
        else:
            container.env.append(env_var)
            by_name[name] = len(container.env) - 1


def _probe_from_k8s_dict(data: dict[str, Any]) -> Probe | None:
    if not isinstance(data, dict):
        return None
    kwargs: dict[str, Any] = {}
    for src, dest in (
        ("initialDelaySeconds", "initial_delay_seconds"),
        ("periodSeconds", "period_seconds"),
        ("timeoutSeconds", "timeout_seconds"),
        ("failureThreshold", "failure_threshold"),
    ):
        if src in data and data[src] is not None:
            kwargs[dest] = int(data[src])
    http_get = data.get("httpGet")
    if isinstance(http_get, dict):
        kwargs["http_get"] = HTTPGetAction(
            path=http_get.get("path", "/"),
            port=http_get.get("port", 8000),
            scheme=http_get.get("scheme", "HTTP"),
        )
    exec_action = data.get("exec")
    if isinstance(exec_action, dict) and exec_action.get("command"):
        kwargs["exec_action"] = ExecAction(command=[str(item) for item in exec_action["command"]])
    if not kwargs:
        return None
    return Probe(**kwargs)


def _probe_from_nim_service_probe(probe_wrapper: dict[str, Any]) -> Probe | None:
    if probe_wrapper.get("enabled") is False:
        return None
    inner = probe_wrapper.get("probe")
    if isinstance(inner, dict):
        return _probe_from_k8s_dict(inner)
    return _probe_from_k8s_dict(probe_wrapper)


def _container_from_spec_entry(spec_container: dict[str, Any], *, native_sidecar: bool) -> Container | None:
    name = spec_container.get("name")
    image_info = spec_container.get("image")
    if not name or not isinstance(image_info, dict):
        return None
    repository = image_info.get("repository")
    if not repository:
        return None
    tag = image_info.get("tag") or "latest"
    env: list[EnvVar] = []
    for item in spec_container.get("env") or []:
        if isinstance(item, dict) and item.get("name") and item.get("value") is not None:
            env.append(EnvVar(name=str(item["name"]), value=str(item["value"])))
    return Container(
        name=str(name),
        image=_image(str(repository), str(tag)),
        command=[str(item) for item in spec_container.get("command") or []],
        args=[str(item) for item in spec_container.get("args") or []],
        env=env,
        restartPolicy="Always" if native_sidecar else None,
    )


def _ensure_k8s_backend(server_config: DeploymentConfig) -> K8sDeploymentConfig:
    backend = server_config.backend_config
    if backend.k8s is None:
        backend.k8s = K8sDeploymentConfig()
    return backend.k8s


def _validate_nim_override_config_keys(override: dict[str, Any]) -> None:
    """Reject override_config keys that the deployments_plugin NIM compiler does not apply."""
    unsupported = sorted(set(override) - _SUPPORTED_NIM_OVERRIDE_CONFIG_KEYS)
    if unsupported:
        supported = ", ".join(sorted(_SUPPORTED_NIM_OVERRIDE_CONFIG_KEYS))
        raise ValueError(
            f"override_config contains unsupported keys: {', '.join(unsupported)}. Supported keys: {supported}."
        )


def apply_nim_override_config(
    container: Container,
    server_config: DeploymentConfig,
    view: DeploymentConfigView,
    *,
    engine: str,
    runtime: Runtime,
) -> None:
    """Apply ``override_config`` NIMService Spec fragments onto plugin entities.

    Precedence: generated defaults < ``k8s_nim_operator_config`` < ``override_config``.
    Only honored for NIM engine deployments on the Kubernetes runtime.
    """
    if engine != ENGINE_NIM or runtime != Runtime.KUBERNETES:
        return
    override = view.override_config
    if not override:
        return

    _validate_nim_override_config_keys(override)

    image_info = override.get("image")
    if isinstance(image_info, dict):
        repository = image_info.get("repository")
        if repository:
            container.image = _image(str(repository), str(image_info.get("tag") or "latest"))

    if "command" in override:
        container.command = [str(item) for item in override["command"]]
    if "args" in override:
        container.args = [str(item) for item in override["args"]]

    resources = override.get("resources")
    if isinstance(resources, dict):
        apply_container_resources(container, _normalize_resources(resources))

    env_list = override.get("env")
    if isinstance(env_list, list):
        _merge_env_override(container, env_list)

    readiness = override.get("readinessProbe")
    if isinstance(readiness, dict):
        container.readiness_probe = _probe_from_nim_service_probe(readiness)

    liveness = override.get("livenessProbe")
    if isinstance(liveness, dict):
        container.liveness_probe = _probe_from_nim_service_probe(liveness)

    startup = override.get("startupProbe")
    if isinstance(startup, dict):
        probe = container.readiness_probe or Probe(httpGet=HTTPGetAction(path="/v1/health/ready", port=8000))
        startup_probe = _probe_from_nim_service_probe(startup)
        if startup_probe is not None and startup_probe.failure_threshold is not None:
            probe.failure_threshold = startup_probe.failure_threshold
        container.readiness_probe = probe

    k8s = _ensure_k8s_backend(server_config)
    node_selector = override.get("nodeSelector")
    if isinstance(node_selector, dict) and node_selector:
        k8s.affinity = _affinity_from_node_selector({str(k): str(v) for k, v in node_selector.items()})

    tolerations = override.get("tolerations")
    if isinstance(tolerations, list):
        parsed = _tolerations_from_config([item for item in tolerations if isinstance(item, dict)])
        if parsed:
            k8s.tolerations = parsed

    user_id = override.get("userID")
    group_id = override.get("groupID")
    if user_id is not None or group_id is not None:
        security_context = k8s.security_context or PodSecurityContext()
        if user_id is not None:
            security_context.run_as_user = int(user_id)
        if group_id is not None:
            security_context.run_as_group = int(group_id)
            security_context.fs_group = int(group_id)
        k8s.security_context = security_context

    labels = override.get("labels")
    if isinstance(labels, dict):
        server_config.labels.update({str(key): str(value) for key, value in labels.items()})

    init_containers = override.get("initContainers")
    if isinstance(init_containers, list):
        for entry in init_containers:
            if isinstance(entry, dict):
                parsed = _container_from_spec_entry(entry, native_sidecar=False)
                if parsed is not None:
                    server_config.init_containers.append(parsed)

    sidecar_containers = override.get("sidecarContainers")
    if isinstance(sidecar_containers, list):
        for entry in sidecar_containers:
            if isinstance(entry, dict):
                parsed = _container_from_spec_entry(entry, native_sidecar=True)
                if parsed is not None:
                    server_config.init_containers.append(parsed)

    server_config.backend_config = DeploymentBackendConfig(k8s=k8s)
