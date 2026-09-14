# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The sandbox_k8s backend and its live wiring.

This module holds everything specific to the agent-sandbox / Harbor
*sandbox-k8s* path: the concrete :class:`SandboxK8sBackend` (which implements
the generic :class:`~scaled_evals.dispatch.runtime_backend.RuntimeBackend`
contract) and the live submitter that drives it against a real cluster. The
"one backend, many targets" model — the backend is constant; a *target*
(namespace, kube context, image, pull secret, TLS verification) is pure
configuration.

Dispatch renders a per-evaluation Harbor config (``job_name`` = the evaluation
id, so the Harbor job dir is ``jobs/sandbox-k8s/ev_...``) and fires ``harbor run``
against the agent-sandbox ``sandbox_k8s.harbor:K8sSandboxEnvironment`` adapter
for the configured target. Harbor's adapter turns that into a Sandbox CR on the
cluster.

Task-environment image vs agent: in a sandbox-k8s Harbor config the
``environment.kwargs.image`` (the ``${TASK_IMAGE}`` placeholder) is the
*task-environment* image — the sandbox the trial runs in. The *agent* is
selected separately under ``agents:`` by ``name``/``model_name`` and is not a
distinct image slot. ``spec.image_ref`` — the task revision image finalize
built and pushed — is the task-environment image, so dispatch binds it to that
``image:`` slot (it supplies/overrides ``${TASK_IMAGE}``), closing the
publish→build→run loop. In the claude-code-baked-into-the-image variant the
sandbox image and the agent image coincide; image_ref still governs the single
``image:`` slot, and we never synthesize a separate agent image.

Disabled by default (``SANDBOX_K8S_ENABLED=false``): :func:`build_backend`
returns a :class:`SandboxK8sBackend` with no submitter, so ``launch`` raises and
the run is marked failed, and unit tests stay cluster-free by injecting a fake
backend. Enable it and point the settings at an agent-sandbox harness target
env file for compose dispatch.

Scope: fire-and-forget ``harbor run`` against a target config, with
``spec.image_ref`` honored as the task-environment image (above) and
``spec.harbor_config`` applied as non-secret Harbor profile overrides.

Task tree: for finalized task images, the Harbor task *definition*
(``task.toml`` / ``tests/`` / ``solution/`` / ``instruction.md``) is sourced
per-eval from the task revision's uploaded pack — ``spec.tarball_object_key``,
the same object the BuildKit build downloads — extracted and staged into the
per-eval ``/work`` mount, with the rendered config's ``tasks: - path:`` bound
to the staged tree. When a revision has no uploaded task tree or no finalized
``image_ref``, dispatch falls back to the task tree baked beside the global
``SANDBOX_K8S_CONFIG_PATH`` (``<config>.parent.parent/task``) in the
harbor-runner image, so already-baked/static-image tasks keep working.
"""

from __future__ import annotations

import fcntl
import filecmp
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import uuid
from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from scaled_evals.api.redaction import redact_secret_text
from scaled_evals.api.schemas.common import validate_scoped_egress_config
from scaled_evals.api.settings import settings
from scaled_evals.dispatch.credentials import merged_env_file
from scaled_evals.dispatch.inference_headers import (
    INFERENCE_HEADERS_ENV,
    headers_from_json,
    inference_header_runner_env,
    with_default_anthropic_custom_headers,
    with_default_inference_priority,
)
from scaled_evals.dispatch.kubectl import execute_kubectl
from scaled_evals.dispatch.paths import (
    container_harness_host_path,
    resolve_host_env_file,
    resolve_host_path,
    setting_evaluation_dir,
)
from scaled_evals.dispatch.process import spawn_detached_process
from scaled_evals.dispatch.runtime_backend import (
    CallableRuntimeBackend,
    LaunchHandle,
    LaunchSpec,
    ResultSummary,
    RuntimeBackendCapabilities,
    RuntimeBackendRegistration,
    RuntimeStatus,
)
from scaled_evals.models.resource_usage import ResourceUsageSample

LOG = logging.getLogger(__name__)
_HARBOR_DEFAULT_ENVIRONMENT_BUILD_TIMEOUT_SECONDS = 600.0
_RESOURCE_SAMPLE_COMMAND_TIMEOUT_SECONDS = 5.0


class SandboxK8sBackend(CallableRuntimeBackend):
    """Backend for the agent-sandbox / Harbor *sandbox-k8s* path.

    ``launch`` delegates to the injected ``submitter``, which runs the
    evaluation on the target cluster. With no submitter — the default for the
    control plane today — ``launch`` raises and the dispatcher marks the run
    failed; the live submitter is the integration point (see
    :func:`make_sandbox_k8s_submitter`).
    """

    name = "sandbox_k8s"

    def __init__(
        self,
        *,
        submitter: Callable[[LaunchSpec], LaunchHandle] | None = None,
        terminator: Callable[[LaunchHandle], None] | None = None,
        status_reader: Callable[[LaunchHandle], RuntimeStatus] | None = None,
        resource_sampler: Callable[[LaunchHandle], list[ResourceUsageSample]] | None = None,
    ) -> None:
        super().__init__(
            name=self.name,
            submitter=submitter,
            terminator=terminator,
            status_reader=status_reader,
            resource_sampler=resource_sampler,
            summarizer=summarize_harbor_result,
            launch_unavailable=(
                "sandbox_k8s live submission is not wired; inject a submitter "
                "(integration point with the sandbox-k8s Harbor adapter)"
            ),
        )


# TODO(harbor-lib): explore driving Harbor via its Python API instead of the
# CLI. Harbor is a Python framework — we could `import harbor` (or instantiate
# sandbox_k8s.harbor:K8sSandboxEnvironment directly) for structured
# start/status/stop. ``.status()`` is wired today by polling ``result.json``
# off disk (see :func:`make_sandbox_k8s_status_reader`); the Python API would
# give a richer in-flight signal and let us wire ``.teardown()`` too, instead of
# the file-poll + fire-and-forget launch shape. The programmatic interface is
# the better long-term shape. Trade-off / why not yet: it pulls Harbor's full
# dependency tree (k8s client, cloud SDKs) into our process and gives up process
# isolation + the HARBOR_DIR/CLI decoupling we have now. Best attempted in the
# out-of-process worker (see worker.py), and only once it clearly beats the
# working CLI path — don't gut a working integration to chase it.
#
# Host fallback when Harbor has no checked-in venv (``uv run harbor run``).
HARBOR_RUN = ("uv", "run", "--no-sync", "harbor", "run")

_VAR_RE = re.compile(r"\$\{(\w+)\}")
_JOB_NAME_RE = re.compile(r"(?m)^job_name:.*$")
_SESSION_ID_RE = re.compile(r"(?m)^session_id:.*$")
_N_ATTEMPTS_RE = re.compile(r"(?m)^n_attempts:.*$")
_N_CONCURRENT_TRIALS_RE = re.compile(r"(?m)^n_concurrent_trials:.*$")
# Matches the first ``- path:`` entry of the ``tasks:`` block so dispatch can
# rebind it onto a staged task tree.
# Tolerates blank/comment lines between ``tasks:`` and the first entry.
_TASKS_PATH_RE = re.compile(
    r"(?ms)^(?P<lead>tasks:[ \t]*\r?\n(?:[ \t]*(?:#[^\r\n]*)?\r?\n)*?[ \t]+-[ \t]+path:[ \t]+)"
    r"(?P<val>\S[^\r\n]*)"
)
_PROFILE_ENV_KEYS = ("env", "environment", "vars")
_PROFILE_TEMPLATE_KEYS = ("config", "harbor_config", "template", "harbor_template")
_PROFILE_METADATA_KEYS = ("dataset_only", "dataset_image_mode")
# The file that marks the root of a Harbor task tree inside an uploaded pack.
_TASK_TREE_MARKER = "task.toml"
_EVALUATION_LABEL = "scaled-evals.nvidia.com/evaluation-id"
_BENCHMARK_RUN_LABEL = "scaled-evals.nvidia.com/benchmark-run-id"
_INITIAL_USER_TURNS_ENV = "SCALED_EVALS_INITIAL_USER_TURNS_JSON"
_ROOT_AUTHORIZATION_KEY = "_scaled_evals_root_authorized"
_WRITABLE_ROOT_AUTHORIZATION_KEY = "_scaled_evals_writable_root_authorized"

# (argv, cwd, log_path) -> None. Injected in tests so nothing is spawned.
Runner = Callable[[list[str], Path, Path], None]
# Reads a launched run's state from the cluster/filesystem and normalizes it to
# a RuntimeStatus. Injected in tests so status reads need no real result.json.
StatusReader = Callable[[LaunchHandle], RuntimeStatus]


def _should_stage_uploaded_task_tree(spec: LaunchSpec) -> bool:
    """Only finalized uploaded-image revisions use the uploaded task tree at dispatch."""
    return bool(spec.image_ref and spec.tarball_object_key)


def _dataset_image_map(spec: LaunchSpec) -> dict[str, str]:
    return {str(item["source_image"]): str(item["runtime_image"]) for item in spec.harbor_dataset_image_imports}


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a ``targets/<target>.env`` file (KEY=VALUE, ``#`` comments) into a dict."""
    env: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        try:
            parts = shlex.split(value, comments=False, posix=True)
        except ValueError:
            parts = []
        env[key.strip()] = " ".join(parts) if parts else value.strip('"').strip("'")
    return env


def _staged_task_name(spec: LaunchSpec) -> str:
    # Harbor derives local task names from the directory basename.
    name = spec.task_slug or "task"
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name):
        raise ValueError("task slug is not a safe directory name")
    return name


def render_harbor_config(
    config_text: str,
    env: Mapping[str, str],
    *,
    job_name: str,
    n_attempts: int = 1,
    n_concurrent_trials: int = 1,
    image_ref: str | None = None,
    profile_config: Mapping[str, Any] | None = None,
    task_path: str | None = None,
    network_policy: str = "unrestricted",
    network_policy_config: Mapping[str, Any] | None = None,
    allow_insecure_tls: bool = False,
    root_authorized: bool = False,
    writable_root_authorized: bool = False,
    agent_bundle: Mapping[str, Any] | None = None,
    benchmark_run_id: str | None = None,
    dataset_image_map: Mapping[str, str] | None = None,
) -> str:
    """Interpolate ``${VAR}`` from ``env`` and set ``job_name`` to the evaluation id.

    Mirrors run.sh's envsubst step (Harbor's ``--env-file`` does NOT interpolate
    the config YAML). Raises ``KeyError`` on an unresolved ``${VAR}`` rather than
    blanking it, so a misconfigured target fails loudly. The ``job_name`` becomes
    the evaluation id, placing the run under ``<jobs_dir>/ev_...``. Trial attempts
    and concurrency are bound to the explicit evaluation request values.

    When ``image_ref`` is set (the published task revision image), it is the
    authoritative task-environment image: it supplies/overrides ``${TASK_IMAGE}``
    during substitution and then ``environment.kwargs.image`` is bound to it, so
    the trial runs in the freshly built task image rather than the target's
    static default — even if the main image is hardcoded or omits the placeholder.
    Profile-owned sidecar images remain independent unless they explicitly use
    ``${TASK_IMAGE}``. Falsy ``image_ref`` leaves target/profile behavior untouched.

    When ``task_path`` is set (the per-eval task tree dispatch staged from the
    task upload), it is the authoritative Harbor task tree: it
    supplies/overrides the ``${TASK_PATH}`` placeholder during substitution and
    then the rendered ``tasks:`` block's first ``- path:`` is bound to it, so the
    run uses the uploaded task tree rather than one baked into the runner image.
    Falsy ``task_path`` leaves the baked/global task path untouched.
    """
    profile = profile_config or {}
    dataset_only = _is_dataset_only_harbor_profile(profile)
    profile_text, profile_env = _normalize_harbor_profile_config(profile)
    template = (
        _dataset_only_harbor_template(
            config_text,
            profile_text,
            dataset_image_map=dataset_image_map,
        )
        if dataset_only
        else profile_text or config_text
    )
    sub_env = {**env, **profile_env}
    if image_ref and not dataset_only:
        sub_env["TASK_IMAGE"] = image_ref
    if task_path:
        # Resolve the task-path placeholder to the staged tree so configs that
        # reference a var (not just a hardcoded path) interpolate cleanly.
        sub_env["TASK_PATH"] = task_path

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in sub_env:
            raise KeyError(f"unresolved ${{{key}}} in harbor config (set it in the target env)")
        return sub_env[key]

    rendered = _VAR_RE.sub(_sub, template)
    rendered, replaced = _JOB_NAME_RE.subn(f"job_name: {job_name}", rendered, count=1)
    if replaced == 0:
        rendered = f"job_name: {job_name}\n{rendered}"
    rendered, replaced = _SESSION_ID_RE.subn(f"session_id: {job_name}", rendered, count=1)
    if replaced == 0:
        rendered = f"session_id: {job_name}\n{rendered}"
    trial_settings = (
        (_N_ATTEMPTS_RE, "n_attempts", n_attempts),
        (_N_CONCURRENT_TRIALS_RE, "n_concurrent_trials", n_concurrent_trials),
    )
    for pattern, key, value in trial_settings:
        rendered, replaced = pattern.subn(f"{key}: {value}", rendered, count=1)
        if replaced == 0:
            rendered = f"{key}: {value}\n{rendered}"
    if image_ref and not dataset_only:
        rendered = _bind_task_image(rendered, image_ref)
    if task_path:
        rendered = _TASKS_PATH_RE.sub(lambda m: f"{m.group('lead')}{task_path}", rendered, count=1)
    rendered = _bind_evaluation_labels(
        rendered,
        job_name,
        benchmark_run_id=benchmark_run_id,
    )
    rendered = _bind_network_policy(rendered, network_policy, network_policy_config or {})
    if agent_bundle:
        rendered = _bind_agent_bundle(rendered, agent_bundle)
    rendered = _bind_sandbox_startup_timeout(
        rendered,
        env.get("SANDBOX_STARTUP_TIMEOUT_SECONDS"),
    )
    rendered = _bind_sandbox_scheduling(
        rendered,
        tolerations_json=env.get("SANDBOX_TOLERATIONS_JSON"),
        node_selector_json=env.get("SANDBOX_NODE_SELECTOR_JSON"),
    )
    rendered = _bind_root_authorization(rendered, root_authorized)
    rendered = _bind_writable_root_authorization(rendered, writable_root_authorized)
    rendered = _bind_default_inference_priority(rendered)
    return _enforce_sandbox_tls_policy(rendered, allow_insecure_tls=allow_insecure_tls)


def _dataset_only_harbor_template(
    base_text: str,
    profile_text: str | None,
    *,
    dataset_image_map: Mapping[str, str] | None = None,
) -> str:
    """Merge a dataset selector into the target's K8s Harbor environment.

    A Harbor dataset provides task paths and task-local images, but not the
    deployment's K8s connection settings.  Keep those settings from the target
    template while removing its anchor task and image, then let the adapter
    select each resolved task's declared image.
    """
    try:
        base = yaml.safe_load(base_text) or {}
        profile = yaml.safe_load(profile_text or "") or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid Harbor config: {exc}") from exc
    if not isinstance(base, dict) or not isinstance(profile, dict):
        raise ValueError("Harbor config must be an object")

    merged = _deep_merge_harbor_config(base, profile)
    merged.pop("tasks", None)
    environment = merged.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("dataset-only Harbor profiles require a target environment object")
    kwargs = environment.setdefault("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("dataset-only Harbor environment.kwargs must be an object")
    kwargs.pop("image", None)
    kwargs["_scaled_evals_prefer_task_image"] = True
    if dataset_image_map:
        kwargs["_scaled_evals_task_image_map"] = dict(dataset_image_map)
    return yaml.safe_dump(merged, sort_keys=False)


def _deep_merge_harbor_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_harbor_config(existing, value)
        else:
            merged[key] = value
    return merged


def _bind_default_inference_priority(config_text: str) -> str:
    """Force the batch header in supported Harbor agent configurations."""
    config = yaml.safe_load(config_text) or {}
    if not isinstance(config, dict):
        raise ValueError("rendered Harbor config must be an object")
    agents = config.get("agents")
    if not isinstance(agents, list):
        return config_text

    for agent in agents:
        if not isinstance(agent, dict):
            continue
        identity = " ".join(str(agent.get(key) or "").lower() for key in ("name", "import_path"))
        agent_env = agent.setdefault("env", {})
        if not isinstance(agent_env, dict):
            raise ValueError("rendered Harbor config agent.env must be an object")
        existing_contract = headers_from_json(agent_env.get(INFERENCE_HEADERS_ENV))
        agent_env[INFERENCE_HEADERS_ENV] = json.dumps(
            with_default_inference_priority(existing_contract),
            separators=(",", ":"),
            sort_keys=True,
        )
        if "codex" in identity:
            existing = headers_from_json(agent_env.get("CODEX_GATEWAY_HTTP_HEADERS_JSON"))
            agent_env.update(inference_header_runner_env(existing))
        if "claude" in identity:
            agent_env["ANTHROPIC_CUSTOM_HEADERS"] = with_default_anthropic_custom_headers(
                agent_env.get("ANTHROPIC_CUSTOM_HEADERS")
            )
        if "terminus-2" in identity or "terminus_2" in identity:
            kwargs = agent.setdefault("kwargs", {})
            if not isinstance(kwargs, dict):
                raise ValueError("rendered Harbor config agent.kwargs must be an object")
            call_kwargs = kwargs.setdefault("llm_call_kwargs", {})
            if not isinstance(call_kwargs, dict):
                raise ValueError("Terminus llm_call_kwargs must be an object")
            extra_headers = call_kwargs.get("extra_headers", {})
            if not isinstance(extra_headers, Mapping):
                raise ValueError("Terminus llm_call_kwargs.extra_headers must be an object")
            call_kwargs["extra_headers"] = with_default_inference_priority(extra_headers)
        if "opencode" in identity:
            _bind_opencode_inference_headers(agent)
        if "openclaw" in identity:
            _bind_openclaw_inference_headers(agent)
        if "openhands" in identity:
            existing = headers_from_json(agent_env.get("LLM_EXTRA_HEADERS"))
            agent_env["LLM_EXTRA_HEADERS"] = json.dumps(
                with_default_inference_priority(existing),
                separators=(",", ":"),
                sort_keys=True,
            )

    return yaml.safe_dump(config, sort_keys=False)


def _selected_agent_provider(agent: Mapping[str, Any]) -> str | None:
    model_name = agent.get("model_name")
    if not isinstance(model_name, str) or "/" not in model_name:
        return None
    return model_name.split("/", 1)[0]


def _agent_kwargs(agent: dict[str, Any]) -> dict[str, Any]:
    kwargs = agent.setdefault("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("rendered Harbor config agent.kwargs must be an object")
    return kwargs


def _mapping_child(parent: dict[str, Any], key: str, *, label: str) -> dict[str, Any]:
    child = parent.setdefault(key, {})
    if not isinstance(child, dict):
        raise ValueError(f"{label} must be an object")
    return child


def _bind_opencode_inference_headers(agent: dict[str, Any]) -> None:
    provider = _selected_agent_provider(agent)
    if provider is None:
        return
    config = _mapping_child(_agent_kwargs(agent), "opencode_config", label="opencode_config")
    providers = _mapping_child(config, "provider", label="opencode_config.provider")
    provider_config = _mapping_child(providers, provider, label=f"opencode_config.provider.{provider}")
    options = _mapping_child(provider_config, "options", label=f"opencode_config.provider.{provider}.options")
    headers = options.get("headers", {})
    if not isinstance(headers, Mapping):
        raise ValueError(f"opencode_config.provider.{provider}.options.headers must be an object")
    options["headers"] = with_default_inference_priority(headers)


def _bind_openclaw_inference_headers(agent: dict[str, Any]) -> None:
    provider = _selected_agent_provider(agent)
    if provider is None:
        return
    config = _mapping_child(_agent_kwargs(agent), "openclaw_config", label="openclaw_config")
    models = _mapping_child(config, "models", label="openclaw_config.models")
    providers = _mapping_child(models, "providers", label="openclaw_config.models.providers")
    provider_config = _mapping_child(providers, provider, label=f"openclaw_config.models.providers.{provider}")
    headers = provider_config.get("headers", {})
    if not isinstance(headers, Mapping):
        raise ValueError(f"openclaw_config.models.providers.{provider}.headers must be an object")
    provider_config["headers"] = with_default_inference_priority(headers)


def _bind_agent_bundle(config_text: str, bundle: Mapping[str, Any]) -> str:
    """Attach one registered immutable bundle without changing the task image."""
    config = yaml.safe_load(config_text) or {}
    if not isinstance(config, dict):
        raise ValueError("rendered Harbor config must be an object")
    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("rendered Harbor config has no environment object")
    kwargs = environment.setdefault("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("rendered Harbor config environment.kwargs must be an object")

    agent_name = str(bundle.get("agent_name") or "")
    version = str(bundle.get("agent_version") or "")
    image_ref = str(bundle.get("image_ref") or "")
    image_digest = str(bundle.get("image_digest") or "")
    entrypoint = str(bundle.get("entrypoint") or "")
    source_lock = str(bundle.get("source_lock_digest") or "")
    fingerprint = str(bundle.get("fingerprint") or "")
    if (
        not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", agent_name)
        or not re.fullmatch(r"[^\s@]+:[^/\s@]+", image_ref)
        or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image_digest)
        or image_ref.rsplit(":", 1)[0] != image_digest.split("@", 1)[0]
    ):
        raise ValueError("invalid agent bundle identity or immutable image digest")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", entrypoint) or ".." in entrypoint.split("/"):
        raise ValueError("invalid agent bundle entrypoint")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", source_lock) or not re.fullmatch(r"sha256:[0-9a-f]{64}", fingerprint):
        raise ValueError("invalid agent bundle lock identity")

    volumes = kwargs.setdefault("volumes", [])
    mounts = kwargs.setdefault("volume_mounts", [])
    sidecars = kwargs.setdefault("sidecars", [])
    if not all(isinstance(value, list) for value in (volumes, mounts, sidecars)):
        raise ValueError("agent bundle requires list-valued volumes, volume_mounts, and sidecars")
    if any(item.get("name") == "agent-bundle" for item in sidecars if isinstance(item, dict)):
        raise ValueError("Harbor profile already defines reserved sidecar 'agent-bundle'")

    installed_volume = next(
        (item for item in volumes if isinstance(item, dict) and item.get("name") == "harbor-installed-agent"),
        None,
    )
    if installed_volume is None:
        volumes.append({"name": "harbor-installed-agent", "empty_dir": {}})
    elif "empty_dir" not in installed_volume:
        raise ValueError("harbor-installed-agent must be an empty_dir volume")

    installed_mount = next(
        (item for item in mounts if isinstance(item, dict) and item.get("mount_path") == "/installed-agent"),
        None,
    )
    if installed_mount is None:
        mounts.append(
            {
                "name": "harbor-installed-agent",
                "mount_path": "/installed-agent",
                "read_only": True,
            }
        )
    elif installed_mount.get("name") != "harbor-installed-agent":
        raise ValueError("/installed-agent is already mounted from a conflicting volume")
    else:
        installed_mount["read_only"] = True

    # Both the installer path and the identity env prefix belong to the bundle
    # image's layout, so they come from configuration rather than being fixed here.
    installer = settings.agent_bundle_installer_path
    prefix = settings.agent_bundle_env_prefix
    sidecars.append(
        {
            "name": "agent-bundle",
            "image": image_ref,
            "image_pull_policy": "Always",
            "command": ["/bin/sh", "-c"],
            "args": [
                "rm -f /installed-agent/.ready /installed-agent/.ready.tmp; "
                f"{installer} /installed-agent && "
                "printf ready > /installed-agent/.ready.tmp && "
                "mv /installed-agent/.ready.tmp /installed-agent/.ready && "
                "exec sleep infinity"
            ],
            "env": {
                f"{prefix}NAME": agent_name,
                f"{prefix}VERSION": version,
                f"{prefix}PLATFORM": str(bundle.get("platform") or ""),
                f"{prefix}RUNTIME_ABI": str(bundle.get("runtime_abi") or ""),
                f"{prefix}BUNDLE_LAYOUT_VERSION": str(bundle.get("bundle_layout_version") or ""),
                f"{prefix}BUILDER_PROFILE": str(bundle.get("builder_profile") or ""),
                f"{prefix}SOURCE_LOCK_DIGEST": source_lock,
                f"{prefix}BUNDLE_FINGERPRINT": fingerprint,
                "HOME": "/tmp",
                "XDG_CACHE_HOME": "/tmp/.cache",
                "XDG_CONFIG_HOME": "/tmp/.config",
            },
            "volume_mounts": [
                {"name": "harbor-installed-agent", "mount_path": "/installed-agent"},
                {"name": "tmp", "mount_path": "/tmp"},
            ],
        }
    )

    env = environment.setdefault("env", {})
    if not isinstance(env, dict):
        raise ValueError("rendered Harbor config environment.env must be an object")
    base_path = str(env.get("PATH") or "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    executable_dir = f"/installed-agent/{entrypoint.rsplit('/', 1)[0]}"
    if executable_dir not in base_path.split(":"):
        env["PATH"] = f"{executable_dir}:{base_path}"

    executable = f"/installed-agent/{entrypoint}"
    wait = (
        "for i in $(seq 1 120); do "
        f"test -f /installed-agent/.ready && test -x {executable} && break; sleep 1; done; "
        "test -f /installed-agent/.ready; "
        f"test -x {executable}; "
        f"{executable} --version"
    )
    existing_setup = str(kwargs.get("setup_command") or "").strip()
    kwargs["setup_command"] = f"{wait}; {existing_setup}" if existing_setup else wait

    agents = config.get("agents")
    if isinstance(agents, list):
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            agent_env = agent.setdefault("env", {})
            if not isinstance(agent_env, dict):
                raise ValueError("rendered Harbor config agent.env must be an object")
            agent_env["SCALED_EVALS_AGENT_BUNDLE_ATTACHED"] = "true"
            agent_env["SCALED_EVALS_INSTALLED_AGENT_ATTACHED"] = "true"

    return yaml.safe_dump(config, sort_keys=False)


def _bind_root_authorization(config_text: str, authorized: bool) -> str:
    """Overwrite the private runner authorization after all profile merging.

    A user-controlled Harbor profile may request ``run_as_user: 0`` but cannot
    authorize it. The patched adapter consumes this internal value and the
    sandbox-k8s SDK still fails closed unless it is true.
    """
    config = yaml.safe_load(config_text) or {}
    if not isinstance(config, dict):
        raise ValueError("rendered Harbor config must be an object")
    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("rendered Harbor config has no environment object")
    kwargs = environment.setdefault("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("rendered Harbor config environment.kwargs must be an object")
    kwargs[_ROOT_AUTHORIZATION_KEY] = authorized
    return yaml.safe_dump(config, sort_keys=False)


def _bind_writable_root_authorization(config_text: str, authorized: bool) -> str:
    """Bind deployment-owned writable-root authorization after profile merging."""
    config = yaml.safe_load(config_text) or {}
    if not isinstance(config, dict):
        raise ValueError("rendered Harbor config must be an object")
    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("rendered Harbor config has no environment object")
    kwargs = environment.setdefault("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("rendered Harbor config environment.kwargs must be an object")
    kwargs[_WRITABLE_ROOT_AUTHORIZATION_KEY] = authorized
    return yaml.safe_dump(config, sort_keys=False)


def _writable_root_requested(config_text: str) -> bool:
    config = yaml.safe_load(config_text) or {}
    environment = config.get("environment") if isinstance(config, dict) else None
    kwargs = environment.get("kwargs") if isinstance(environment, dict) else None
    return isinstance(kwargs, dict) and kwargs.get("read_only_root_filesystem") is False


def _bind_sandbox_startup_timeout(config_text: str, timeout_seconds: str | None) -> str:
    """Make the operator target's Sandbox readiness budget authoritative."""
    if timeout_seconds is None or not timeout_seconds.strip():
        return config_text
    try:
        timeout = int(timeout_seconds)
    except ValueError as exc:
        raise ValueError("SANDBOX_STARTUP_TIMEOUT_SECONDS must be a positive integer") from exc
    if timeout <= 0:
        raise ValueError("SANDBOX_STARTUP_TIMEOUT_SECONDS must be a positive integer")

    config = yaml.safe_load(config_text) or {}
    if not isinstance(config, dict):
        raise ValueError("rendered Harbor config must be an object")
    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("rendered Harbor config has no environment object")
    kwargs = environment.setdefault("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("rendered Harbor config environment.kwargs must be an object")
    kwargs["timeout"] = timeout

    multiplier = timeout / _HARBOR_DEFAULT_ENVIRONMENT_BUILD_TIMEOUT_SECONDS
    existing_multiplier = config.get("environment_build_timeout_multiplier")
    if existing_multiplier is not None:
        try:
            multiplier = max(multiplier, float(existing_multiplier))
        except (TypeError, ValueError) as exc:
            raise ValueError("rendered Harbor config environment_build_timeout_multiplier must be numeric") from exc
    if multiplier > 1.0 or existing_multiplier is not None:
        config["environment_build_timeout_multiplier"] = multiplier
    return yaml.safe_dump(config, sort_keys=False)


def _bind_sandbox_scheduling(
    config_text: str,
    *,
    tolerations_json: str | None,
    node_selector_json: str | None,
) -> str:
    """Apply deployment-owned scheduling constraints to direct Sandbox pods."""
    tolerations = _load_optional_json_env(tolerations_json, "SANDBOX_TOLERATIONS_JSON", list)
    node_selector = _load_optional_json_env(
        node_selector_json,
        "SANDBOX_NODE_SELECTOR_JSON",
        dict,
    )
    if not tolerations and not node_selector:
        return config_text

    if tolerations is not None and not all(isinstance(item, dict) for item in tolerations):
        raise ValueError("SANDBOX_TOLERATIONS_JSON must contain a list of objects")
    if node_selector is not None and not all(
        isinstance(key, str) and isinstance(value, str) for key, value in node_selector.items()
    ):
        raise ValueError("SANDBOX_NODE_SELECTOR_JSON must contain string keys and values")

    config = yaml.safe_load(config_text) or {}
    if not isinstance(config, dict):
        raise ValueError("rendered Harbor config must be an object")
    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("rendered Harbor config has no environment object")
    kwargs = environment.setdefault("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("rendered Harbor config environment.kwargs must be an object")

    if tolerations:
        kwargs["tolerations"] = tolerations
    if node_selector:
        kwargs["node_selector"] = node_selector
    return yaml.safe_dump(config, sort_keys=False)


def _load_optional_json_env(
    value: str | None,
    name: str,
    expected_type: type,
) -> Any | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if not isinstance(parsed, expected_type):
        raise ValueError(f"{name} must be a JSON {expected_type.__name__}")
    return parsed


def _root_authorized_for_spec(spec: LaunchSpec) -> bool:
    """Return whether operators approved root for this immutable task image."""
    if not settings.sandbox_k8s_allow_root:
        return False
    values = [str(item.get("image_digest") or "") for item in spec.harbor_dataset_image_imports] or [
        spec.image_digest or ""
    ]
    digests = []
    for value in values:
        digest_match = re.search(
            r"(?:^|@)(sha256:[0-9a-f]{64})$",
            value.strip().lower(),
        )
        if digest_match is None:
            return False
        digests.append(digest_match.group(1))
    if not digests:
        return False
    if settings.sandbox_k8s_root_allow_all_images:
        return True
    allowed = {
        item_match.group(1)
        for item in settings.sandbox_k8s_root_allowed_image_digests.split(",")
        if (
            item_match := re.search(
                r"(?:^|@)(sha256:[0-9a-f]{64})$",
                item.strip().lower(),
            )
        )
    }
    return set(digests) <= allowed


def _task_image_ref_for_sandbox(spec: LaunchSpec) -> str:
    """Select the image-reference shape accepted by the target registry policy.

    Hosted deployments explicitly select tag mode because signature-enforcing
    admission admits the signed tag but rejects equivalent digest forms.
    Generic deployments default to immutable digest substitution. In both modes
    the recorded digest remains mandatory provenance and dispatch verifies
    mutable tags before launch.
    """
    image_ref = spec.image_ref.strip()
    if settings.sandbox_k8s_task_image_reference_mode == "tag":
        last_slash = image_ref.rfind("/")
        last_colon = image_ref.rfind(":")
        if "@" in image_ref or last_colon <= last_slash:
            raise ValueError(
                "sandbox target requires the signed tag-form task image reference; "
                "digest-only and tag-plus-digest references are not admitted"
            )
        return image_ref
    digest = (spec.image_digest or "").strip()
    if re.fullmatch(r"[^\s]+@sha256:[0-9a-fA-F]{64}", digest):
        return digest
    if re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
        tagged = image_ref.rsplit("@", 1)[0]
        slash = tagged.rfind("/")
        colon = tagged.rfind(":")
        repository = tagged[:colon] if colon > slash else tagged
        return f"{repository}@{digest.lower()}"
    return image_ref


def _bind_task_image(config_text: str, image_ref: str) -> str:
    """Bind only the primary task image, preserving profile-owned sidecars."""
    import yaml

    config = yaml.safe_load(config_text) or {}
    if not isinstance(config, dict):
        raise ValueError("rendered Harbor config must be an object")
    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("rendered Harbor config has no environment object")
    kwargs = environment.setdefault("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("rendered Harbor config environment.kwargs must be an object")
    kwargs["image"] = image_ref
    return yaml.safe_dump(config, sort_keys=False)


def _bind_network_policy(
    config_text: str,
    network_policy: str,
    network_policy_config: Mapping[str, Any],
) -> str:
    """Make the evaluation's direct-egress policy authoritative.

    Switchyard may add a separate, evaluation-scoped grant to its own proxy.
    Kubernetes combines matching policies additively, so this function owns
    only direct sandbox egress and never infers a Switchyard book mode.
    """
    config = yaml.safe_load(config_text) or {}
    if not isinstance(config, dict):
        raise ValueError("rendered Harbor config must be an object")
    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("rendered Harbor config has no environment object")
    kwargs = environment.setdefault("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("rendered Harbor config environment.kwargs must be an object")

    if network_policy == "unrestricted":
        egress: list[dict[str, Any]] = [{}]
    elif network_policy == "default_deny":
        egress = [_cluster_dns_egress_rule()]
    elif network_policy == "scoped_egress":
        validate_scoped_egress_config(dict(network_policy_config))
        egress = [_cluster_dns_egress_rule(), *network_policy_config["egress"]]
    else:
        raise ValueError(f"unsupported sandbox network policy: {network_policy!r}")

    kwargs["k8s_network_policy"] = {
        "policyTypes": ["Egress"],
        "egress": egress,
    }
    kwargs["skip_network_policy_check"] = False
    return yaml.safe_dump(config, sort_keys=False)


def _cluster_dns_egress_rule() -> dict[str, Any]:
    """Allow only the standard CoreDNS or OpenShift DNS pods on port 53."""
    return {
        "to": [
            {
                "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "kube-system"}},
                "podSelector": {"matchLabels": {"k8s-app": "kube-dns"}},
            },
            {
                "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "openshift-dns"}},
                "podSelector": {"matchLabels": {"dns.operator.openshift.io/daemonset-dns": "default"}},
            },
        ],
        "ports": [
            {"protocol": "UDP", "port": 53},
            {"protocol": "TCP", "port": 53},
        ],
    }


def _enforce_sandbox_tls_policy(config_text: str, *, allow_insecure_tls: bool) -> str:
    """Reject final rendered configs that disable Kubernetes API TLS checks."""
    try:
        config = yaml.safe_load(config_text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid rendered Harbor config: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("rendered Harbor config must be an object")
    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("rendered Harbor config has no environment object")
    kwargs = environment.get("kwargs")
    if not isinstance(kwargs, dict):
        raise ValueError("rendered Harbor config environment.kwargs must be an object")
    value = kwargs.get("verify_ssl", True)
    if isinstance(value, bool):
        verify_ssl = value
    elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        verify_ssl = value.strip().lower() == "true"
    else:
        raise ValueError("rendered Harbor config verify_ssl must be true or false")
    if verify_ssl:
        return config_text
    if not allow_insecure_tls:
        raise ValueError(
            "sandbox Kubernetes TLS verification cannot be disabled; hosted dispatch requires verify_ssl=true"
        )
    LOG.warning(
        "INSECURE LOCAL MODE: sandbox Kubernetes API TLS verification is disabled; "
        "do not use SANDBOX_K8S_ALLOW_INSECURE_TLS outside local development"
    )
    return config_text


def _bind_evaluation_labels(
    config_text: str,
    evaluation_id: str,
    *,
    benchmark_run_id: str | None = None,
) -> str:
    """Label every Sandbox spawned by this Harbor run with its evaluation id.

    Harbor chooses a distinct, random ``session_id`` for each trial, so the
    resulting Sandbox names cannot be predicted from the evaluation id.  The
    reserved label is stable across trials and retries and gives cancellation a
    complete, ownership-safe selector for Sandbox, Pod, and NetworkPolicy cleanup.
    """
    try:
        config = yaml.safe_load(config_text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid rendered Harbor config: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("rendered Harbor config must be an object")
    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("rendered Harbor config has no environment object")
    kwargs = environment.setdefault("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("rendered Harbor config environment.kwargs must be an object")
    labels = kwargs.setdefault("labels", {})
    if not isinstance(labels, dict):
        raise ValueError("rendered Harbor config environment.kwargs.labels must be an object")
    labels[_EVALUATION_LABEL] = evaluation_id
    if benchmark_run_id is not None:
        labels[_BENCHMARK_RUN_LABEL] = benchmark_run_id
    return yaml.safe_dump(config, sort_keys=False)


def _normalize_harbor_profile_config(
    profile_config: Mapping[str, Any],
) -> tuple[str | None, dict[str, str]]:
    """Return optional Harbor template text and env overrides from a profile.

    Harbor profiles carry non-secret runtime settings. They may provide env-style
    substitutions under ``env``/``environment``/``vars`` and, for explicit smoke
    or target selection, a full config template under ``config``/``harbor_config``.
    Secret material stays in credentials and is merged separately.
    """
    profile_text: str | None = None
    for key in _PROFILE_TEMPLATE_KEYS:
        value = profile_config.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"harbor profile '{key}' must be a string")
        profile_text = value

    structured_template = profile_text is None and _is_structured_harbor_profile_config(profile_config)
    profile_env: dict[str, str] = {}
    env_keys = ("env", "vars") if structured_template else _PROFILE_ENV_KEYS
    for key in env_keys:
        value = profile_config.get(key)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"harbor profile '{key}' must be an object")
        profile_env.update({str(env_key): str(env_value) for env_key, env_value in value.items()})

    if structured_template:
        # API-created profiles may store the Harbor YAML parsed as a JSON object
        # instead of wrapping the raw YAML under ``harbor_config``. Treat that
        # shape as an equivalent full template.
        profile_text = yaml.safe_dump(
            {key: value for key, value in profile_config.items() if key not in _PROFILE_METADATA_KEYS},
            sort_keys=False,
        )

    top_level_env = {
        key: value
        for key, value in profile_config.items()
        if isinstance(key, str) and key.isupper() and key not in _PROFILE_ENV_KEYS and key not in _PROFILE_TEMPLATE_KEYS
    }
    profile_env.update({key: str(value) for key, value in top_level_env.items()})
    return profile_text, profile_env


def _is_dataset_only_harbor_profile(profile_config: Mapping[str, Any]) -> bool:
    """Return whether a profile delegates task resolution to Harbor's dataset registry."""
    value = profile_config.get("dataset_only", False)
    if not isinstance(value, bool):
        raise ValueError("harbor profile 'dataset_only' must be a boolean")
    if not value:
        return False
    template, _ = _normalize_harbor_profile_config(profile_config)
    try:
        config = yaml.safe_load(template or "")
    except yaml.YAMLError as exc:
        raise ValueError("dataset-only Harbor profile must contain valid YAML") from exc
    if not isinstance(config, dict) or not isinstance(config.get("datasets"), list) or not config["datasets"]:
        raise ValueError("dataset-only Harbor profile requires a non-empty datasets list")
    if config.get("tasks"):
        raise ValueError("dataset-only Harbor profile must not define tasks")
    return True


def _is_structured_harbor_profile_config(profile_config: Mapping[str, Any]) -> bool:
    if profile_config.get("dataset_only") is True and isinstance(profile_config.get("datasets"), list):
        return True
    environment = profile_config.get("environment")
    if not isinstance(environment, Mapping):
        return False
    if any(key in profile_config for key in ("agents", "tasks", "job_name", "jobs_dir", "n_attempts", "retry")):
        return True
    return any(key in environment for key in ("import_path", "kwargs", "delete", "env"))


def _resolve_task_path(env: dict[str, str], env_file: Path) -> None:
    """Make a relative ``TASK_PATH`` absolute, like run.sh.

    run.sh resolves ``TASK_PATH`` relative to the harness root (where run.sh
    lives). The target env file sits at ``<harness>/targets/<target>.env``, so
    the harness root is the env file's grandparent.
    """
    task_path = env.get("TASK_PATH")
    if task_path and not os.path.isabs(task_path):
        harness_root = env_file.parent.parent
        env["TASK_PATH"] = str((harness_root / task_path).resolve())


def _augment_harbor_env(
    env: dict[str, str],
    env_file: Path,
    config_path: Path,
    *,
    staged_task_path: str | None = None,
) -> None:
    """Fill paths Harbor configs expect for compose dispatch and harness debug.

    ``staged_task_path`` (the per-eval task tree dispatch extracted from the
    task upload) takes precedence over the task tree baked beside the global
    config at ``<config>.parent.parent/task``. With no staged tree we fall back
    to the baked path, so tasks already in the runner image keep working.
    """
    _resolve_task_path(env, env_file)
    if staged_task_path is not None:
        # A container/host path that need not exist on the worker filesystem, so
        # set it directly rather than guarding on is_dir().
        env["TASK_PATH"] = staged_task_path
    else:
        task_dir = config_path.resolve().parent.parent / "task"
        if task_dir.is_dir():
            env["TASK_PATH"] = str(task_dir)
    env.setdefault("HOME", "/root")


def _extract_pack(tarball_path: Path, dest: Path) -> None:
    """Extract a bounded gzip pack into ``dest``, rejecting unsafe paths."""

    with tarfile.open(tarball_path, "r:gz") as tar:
        members = tar.getmembers()
        if len(members) > settings.task_pack_max_members:
            raise ValueError(
                f"task pack exceeds configured member limit ({len(members)} > {settings.task_pack_max_members})"
            )
        extracted_size = sum(member.size for member in members)
        if extracted_size > settings.task_pack_max_extracted_size_bytes:
            raise ValueError(
                "task pack exceeds configured extracted-size limit "
                f"({extracted_size} bytes > "
                f"{settings.task_pack_max_extracted_size_bytes} bytes)"
            )
        tar.extractall(dest, members=members, filter="data")


def _find_task_tree(root: Path) -> Path | None:
    """Return the shallowest directory under ``root`` that holds a Harbor task.

    A Harbor task tree is the directory containing ``task.toml`` (alongside
    ``tests/``/``solution/``/``instruction.md``). Uploaded packs may place it at
    the root or under a subdir, so we search rather than assume a layout; the
    shallowest match wins. ``None`` when the pack carries no task tree (e.g. a
    bare Dockerfile-only build context) so callers fall back to the baked path.
    """
    candidates = sorted(root.rglob(_TASK_TREE_MARKER), key=lambda p: (len(p.relative_to(root).parts), str(p)))
    for marker in candidates:
        if marker.is_file():
            return marker.parent
    return None


def _task_trees_match(source: Path, dest: Path) -> bool:
    """Return whether two staged task trees have identical entries and bytes."""
    source_entries = {path.relative_to(source) for path in source.rglob("*")}
    dest_entries = {path.relative_to(dest) for path in dest.rglob("*")}
    if source_entries != dest_entries:
        return False
    for relative in source_entries:
        source_path = source / relative
        dest_path = dest / relative
        if source_path.is_dir() != dest_path.is_dir():
            return False
        if source_path.is_file() and not filecmp.cmp(source_path, dest_path, shallow=False):
            return False
    return True


_COMPOSE_FILENAMES = ("docker-compose.yaml", "docker-compose.yml")


def _restore_compose_definition(pack_root: Path, task_tree: Path) -> None:
    """Put a pack-root compose file back where Harbor looks for it.

    Packing flattens ``environment/`` to the pack root so the root doubles as the
    image build context, which leaves the staged task tree without the
    ``environment/docker-compose.yaml`` a multi-service task declares upstream.
    Harbor reads that path to decide whether a task's compose services can exist,
    and refuses any artifact entry or verifier collect hook naming a service when
    it finds nothing -- so the file has to travel with the tree.

    This only restores a declaration. Nothing on this runtime builds or runs
    compose: the services themselves come from the profile's ``sidecars`` list.
    A pack with no compose file is left alone, so single-container tasks see no
    change at all.
    """
    environment_dir = task_tree / "environment"
    for filename in _COMPOSE_FILENAMES:
        source = pack_root / filename
        if not source.is_file() or (environment_dir / filename).exists():
            continue
        environment_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, environment_dir / filename)


def _stage_task_tree(tarball_object_key: str, dest: Path) -> Path | None:
    """Download the task revision pack and stage its Harbor task tree at ``dest``.

    Reuses the object-store download + tar-extract pattern from the build path
    (api/build/buildkit.py) to source the task tree from the *same* uploaded
    tarball the image was built from, per-eval. Returns ``dest`` when a task tree
    was staged, or ``None`` when the pack carries none -- a valid pack with no
    ``task.toml`` (e.g. a Dockerfile-only context) -- so the caller can fall back
    to the task baked beside the global config.

    Failing to *fetch* the pack object is a different matter: the revision
    recorded a ``tarball_object_key`` it expects to exist, so a missing or
    unreachable object is a data-integrity error, not a "no task tree" signal.
    Falling back to the baked path here would silently score the agent against
    the wrong task, so we raise an actionable error instead. (This is the
    object-store eviction we hit when RustFS resets across compose restarts:
    re-upload + finalize the task to repopulate the object.) Extract and copy
    failures are also fatal: falling back after a partial local stage can launch
    Harbor against the wrong or incomplete task.
    """
    from botocore.exceptions import BotoCoreError, ClientError

    from scaled_evals.api import s3

    try:
        with tempfile.TemporaryDirectory(prefix="se-task-") as tmp:
            tmp_path = Path(tmp)
            tarball_path = tmp_path / "tarball.tar.gz"
            extracted = tmp_path / "extracted"
            extracted.mkdir()
            try:
                s3.download_object(tarball_object_key, str(tarball_path))
            except (BotoCoreError, ClientError) as exc:
                raise RuntimeError(
                    f"could not fetch task pack object {tarball_object_key!r} from "
                    f"the object store: {exc}. The revision's artifact may be missing "
                    f"(re-upload + finalize the task) or the store unreachable."
                ) from exc
            _extract_pack(tarball_path, extracted)
            task_src = _find_task_tree(extracted)
            if task_src is None:
                return None
            dest.parent.mkdir(parents=True, exist_ok=True)
            staging = dest.with_name(f".{dest.name}.staging-{uuid.uuid4().hex}")
            lock_path = dest.with_name(f".{dest.name}.lock")
            try:
                shutil.copytree(task_src, staging)
                _restore_compose_definition(extracted, staging)
                with lock_path.open("a+") as lock_file:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                    if dest.exists():
                        if _task_trees_match(staging, dest):
                            return dest
                        if dest.is_dir() and not dest.is_symlink():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    os.replace(staging, dest)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
    except (OSError, tarfile.TarError, ValueError) as exc:
        raise RuntimeError(f"could not stage task tree from {tarball_object_key!r}: {exc}") from exc
    return dest


def _inject_extra_skills(task_tree: Path, object_keys: list[str]) -> list[dict[str, Any]]:
    """Download extra skill files into the staged task tree's skills/ dir.

    Each object key must point to a single file; the filename is taken from the
    last path component of the key. Files land in a per-key subdirectory so that
    a skill delivered as a bare SKILL.md becomes skills/<name>/SKILL.md where
    <name> is derived from the key. Best-effort: failures are logged and skipped
    so a bad key doesn't abort the whole launch.
    """
    from scaled_evals.api import s3

    # Harbor sets environment_dir = task_tree/environment/ (it appends /environment
    # to the task path). Candidate 0 in _upload_environment_skills is
    # environment_dir/skills = task_tree/environment/skills/, so that is where
    # skills must be staged regardless of the tarball's flattened layout.
    skills_dir = task_tree / "environment" / "skills"
    materials: list[dict[str, Any]] = []
    for key in object_keys:
        filename = Path(key).name
        # For SKILL.md files the parent directory in the key path carries the
        # intended skill name (set by the uploader); fall back to the file stem.
        if filename.lower() == "skill.md":
            parent = Path(key).parent.name
            skill_name = parent if parent and parent != "." else Path(key).stem
        else:
            skill_name = filename
        dest_dir = skills_dir / skill_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        try:
            s3.download_object(key, str(dest))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("could not inject extra skill %s: %s", key, exc)
            materials.append({"object_key": key, "status": "download_failed"})
            continue
        body = dest.read_bytes()
        materials.append(
            {
                "object_key": key,
                "staged_path": dest.relative_to(task_tree).as_posix(),
                "sha256": f"sha256:{hashlib.sha256(body).hexdigest()}",
                "size_bytes": len(body),
                "status": "staged",
            }
        )
    return materials


def _save_extra_skill_materials_artifact(
    materials: list[dict[str, Any]],
    evaluation_id: str,
    *,
    harbor_dir: Path | None = None,
) -> None:
    if not materials:
        return
    root = harbor_dir or Path(settings.harbor_dir).expanduser()
    artifact_dir = root / settings.sandbox_k8s_jobs_dir / evaluation_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "scaled-evals-extra-skill-materials.json"
    path.write_text(json.dumps({"materials": materials}, indent=2, sort_keys=True) + "\n")


def _patch_instruction(task_tree: Path, prefix: str | None, postfix: str | None) -> None:
    """Prepend/append text to instruction.md in the staged task tree.

    Called after _inject_extra_skills so skills are already staged. No-op when
    both prefix and postfix are None/empty.
    """
    if not prefix and not postfix:
        return
    instr = task_tree / "instruction.md"
    if not instr.exists():
        return
    text = instr.read_text()
    instr.write_text((prefix or "") + text + (postfix or ""))


def _format_timeout_sec(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _rewrite_agent_timeout_sec(text: str, value: float) -> str:
    """Set ``[agent].timeout_sec`` in a task.toml body without a TOML writer."""
    formatted = _format_timeout_sec(value)
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_agent = False
    found = False
    for line in lines:
        stripped = line.split("#", 1)[0].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_agent and not found:
                out.append(f"timeout_sec = {formatted}\n")
                found = True
            in_agent = stripped == "[agent]"
            out.append(line)
            continue
        if in_agent and stripped.split("=", 1)[0].strip() == "timeout_sec":
            out.append(f"timeout_sec = {formatted}\n")
            found = True
            continue
        out.append(line)
    if in_agent and not found:
        out.append(f"timeout_sec = {formatted}\n")
        found = True
    if not found:
        if out and not out[-1].endswith("\n"):
            out.append("\n")
        if out and out[-1].strip():
            out.append("\n")
        out.append("[agent]\n")
        out.append(f"timeout_sec = {formatted}\n")
    return "".join(out)


def apply_agent_timeout_floor(task_tree: Path | None, floor_sec: int) -> dict[str, float | None]:
    """Raise staged ``task.toml`` ``[agent].timeout_sec`` to ``max(original, floor)``.

    Mutates only the staged tree and returns original/effective for callers.
    Fails the launch rather than running with the unmodified benchmark budget
    when there is no staged tree, or no ``task.toml`` inside it, to apply to.
    """
    path = None if task_tree is None else task_tree / "task.toml"
    if path is None or not path.is_file():
        raise RuntimeError(
            f"benchmark variant agent_timeout_floor_sec={floor_sec} cannot be applied: "
            "this evaluation has no staged task.toml to raise "
            "(the benchmark task must be an uploaded task pack)"
        )
    text = path.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    agent = data.get("agent")
    original: float | None = None
    if isinstance(agent, dict) and agent.get("timeout_sec") is not None:
        original = float(agent["timeout_sec"])
    effective = float(floor_sec) if original is None else max(original, float(floor_sec))
    if original is not None and effective == original:
        return {"original": original, "effective": effective}
    path.write_text(_rewrite_agent_timeout_sec(text, effective), encoding="utf-8")
    try:
        written = tomllib.loads(path.read_text(encoding="utf-8")).get("agent") or {}
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(f"benchmark variant agent_timeout_floor_sec={floor_sec} corrupted {path}: {exc}") from exc
    if float(written.get("timeout_sec") or 0) != effective:
        raise RuntimeError(
            f"benchmark variant agent_timeout_floor_sec={floor_sec} was not applied: "
            f"{path} still reports timeout_sec={written.get('timeout_sec')!r}"
        )
    return {"original": original, "effective": effective}


def _save_instruction_artifact(task_tree: Path, evaluation_id: str, *, harbor_dir: Path | None = None) -> None:
    """Copy the final instruction.md (post-patch) into the job artifact directory."""
    instr = task_tree / "instruction.md"
    if not instr.exists():
        return
    root = harbor_dir or Path(settings.harbor_dir).expanduser()
    artifact_dir = root / settings.sandbox_k8s_jobs_dir / evaluation_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(instr, artifact_dir / "instruction.md")


def _harbor_run_argv(harbor_dir: Path) -> tuple[str, ...]:
    """Prefer ``<harbor_dir>/.venv/bin/harbor`` (compose mount) over host ``uv run``."""
    venv_harbor = harbor_dir / ".venv" / "bin" / "harbor"
    if venv_harbor.is_file():
        return (str(venv_harbor), "run")
    return HARBOR_RUN


def _spawn_detached(argv: list[str], cwd: Path, log_path: Path) -> None:
    """Fire-and-forget ``harbor run``, detached, logging to ``log_path``."""
    pid_path = log_path.with_suffix(f"{log_path.suffix}.pid")
    exit_path = log_path.with_suffix(f"{log_path.suffix}.exit.json")
    token = uuid.uuid4().hex
    exit_path.unlink(missing_ok=True)
    with log_path.open("w") as log:
        proc = spawn_detached_process(
            [
                sys.executable,
                "-m",
                "scaled_evals.dispatch.detached_runner",
                "run",
                str(pid_path),
                str(exit_path),
                token,
                *argv,
            ],
            cwd=cwd,
            log=log,
        )
    start_ticks = _process_start_ticks(proc.pid)
    pid_path.write_text(json.dumps({"pid": proc.pid, "start_ticks": start_ticks, "token": token}))


def _process_start_ticks(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except (OSError, IndexError):
        return None


def _runner_identity_alive(pid_path: Path) -> bool:
    try:
        identity = json.loads(pid_path.read_text())
        pid = int(identity["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    expected_start = identity.get("start_ticks")
    return expected_start is not None and _process_start_ticks(pid) == expected_start


def _terminate_process_group(pid: int) -> None:
    """Terminate a detached Harbor process group and wait for it to exit."""
    try:
        process_group = os.getpgid(pid)
    except ProcessLookupError:
        return
    if process_group != pid:
        raise RuntimeError(
            f"refusing to terminate pid {pid}: expected detached process group {pid}, found {process_group}"
        )

    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    os.killpg(pid, signal.SIGKILL)


def make_sandbox_k8s_process_terminator() -> Callable[[LaunchHandle], None]:
    """Build a terminator for host-launched detached ``harbor run`` processes."""

    def terminate(handle: LaunchHandle) -> None:
        failures: list[str] = []
        pid_path_value = handle.raw.get("pid_file")
        if isinstance(pid_path_value, str) and pid_path_value:
            pid_path = Path(pid_path_value)
            claimed_pid_path = pid_path.with_name(f"{pid_path.name}.terminating")
            try:
                pid_path.rename(claimed_pid_path)
            except FileNotFoundError:
                pass
            else:
                try:
                    identity = json.loads(claimed_pid_path.read_text())
                    pid = int(identity["pid"])
                    expected_start = identity.get("start_ticks")
                    if expected_start is None or _process_start_ticks(pid) != expected_start:
                        raise RuntimeError(f"refusing to terminate reused runner pid {pid}")
                    _terminate_process_group(pid)
                except (OSError, ValueError, RuntimeError) as exc:
                    failures.append(f"harbor runner termination failed: {exc}")
                finally:
                    claimed_pid_path.unlink(missing_ok=True)
        try:
            _cleanup_sandbox_k8s_resources(handle)
        except RuntimeError as exc:
            failures.append(str(exc))
        if failures:
            raise RuntimeError("; ".join(failures))

    return terminate


def _resolve_host_path(path: str) -> Path:
    """Resolve a path to an absolute HOST path for docker bind mounts.

    Relative paths are anchored to SCALED_EVALS_HOST_DIR (the host checkout root)
    so they work correctly when resolved from inside the dispatch-worker container.
    """
    return resolve_host_path(path, host_root=settings.scaled_evals_host_dir)


def _resolve_host_env_file(host_env_file: str | None, container_env_file: Path) -> Path:
    """Resolve target env file to a host path for ``docker run -v``."""
    return resolve_host_env_file(
        container_env_file,
        explicit_host_env_file=host_env_file,
        host_root=settings.scaled_evals_host_dir,
    )


def _container_harness_host_path(container_path: Path) -> Path:
    """Map ``/harness/...`` paths inside the API container to host checkout paths."""
    return container_harness_host_path(container_path, host_root=settings.scaled_evals_host_dir)


def _harbor_container_name(evaluation_id: str) -> str:
    return f"harbor-{evaluation_id}"


def make_sandbox_k8s_docker_terminator() -> Callable[[LaunchHandle], None]:
    """Build an idempotent terminator for per-evaluation harbor-runner containers."""

    def terminate(handle: LaunchHandle) -> None:
        from docker.errors import NotFound

        import docker

        client = None
        failures: list[str] = []
        try:
            client = docker.from_env()
            container_id = handle.raw.get("container_id")
            lookup = container_id if isinstance(container_id, str) and container_id else None
            try:
                container = client.containers.get(lookup or _harbor_container_name(handle.external_id))
            except NotFound:
                container = None
            if container is not None:
                container.reload()
                state = container.attrs.get("State") or {}
                if state.get("Running"):
                    container.stop(timeout=10)
                try:
                    container.remove(force=True)
                except NotFound:
                    pass
                except Exception as exc:
                    # ``stop`` may trigger Harbor's configured auto-removal just
                    # before our explicit remove. Docker reports that harmless
                    # race as 409 until its asynchronous removal completes.
                    response = getattr(exc, "response", None)
                    status_code = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
                    if (
                        status_code != 409
                        or "removal" not in str(exc).lower()
                        or ("in progress" not in str(exc).lower())
                    ):
                        raise
        except Exception as exc:  # noqa: BLE001 — still attempt cluster cleanup
            failures.append(f"harbor runner container cleanup failed: {exc}")
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001 — report after cluster cleanup
                    failures.append(f"docker client close failed: {exc}")
        try:
            _cleanup_sandbox_k8s_resources(handle)
        except RuntimeError as exc:
            failures.append(str(exc))
        if failures:
            raise RuntimeError("; ".join(failures))

    return terminate


def _cleanup_sandbox_k8s_resources(handle: LaunchHandle) -> None:
    """Delete and verify per-evaluation Sandbox resources.

    Harbor normally deletes these when ``harbor run`` exits cleanly. Cancellation
    kills the runner, so delete the evaluation's Sandbox resources directly.
    Missing resources are successful idempotent cleanup; API/RBAC failures are
    raised so the cancelled evaluation records a teardown failure.
    """
    cleanup = handle.raw.get("cleanup")
    if isinstance(cleanup, Mapping):
        cleanup_metadata = cleanup
    else:
        config_path_value = handle.raw.get("config")
        if not isinstance(config_path_value, str) or not config_path_value:
            raise RuntimeError("sandbox cleanup requires persisted metadata or a rendered config")
        config_path = Path(config_path_value)
        try:
            cleanup_metadata = _sandbox_cleanup_metadata(config_path.read_text(), handle.external_id)
        except OSError as exc:
            raise RuntimeError(f"could not read sandbox cleanup config {config_path}: {exc}") from exc

    namespace = str(cleanup_metadata.get("namespace") or "default")
    context = cleanup_metadata.get("context")
    kubeconfig_path = cleanup_metadata.get("kubeconfig_path")
    sandbox_name = str(cleanup_metadata.get("sandbox_name") or _sanitize_k8s_name(f"hbr-{handle.external_id[:50]}"))
    evaluation_selector = cleanup_metadata.get("selector")
    kubectl = shutil.which("kubectl") or shutil.which("oc")
    if kubectl is None:
        raise RuntimeError("sandbox cleanup requires kubectl or oc")

    base = [kubectl]
    if kubeconfig_path:
        base.extend(["--kubeconfig", str(kubeconfig_path)])
    if context:
        base.extend(["--context", str(context)])
    if cleanup_metadata.get("verify_ssl") is False:
        base.append("--insecure-skip-tls-verify=true")
    base.extend(["-n", namespace])

    template_name = cleanup_metadata.get("template_name")
    primary_kind = "sandboxclaim" if template_name else "sandbox"
    failures: list[str] = []
    primary_target = (
        [primary_kind, "-l", str(evaluation_selector)] if evaluation_selector else [primary_kind, sandbox_name]
    )
    sandbox_names = [sandbox_name]
    if not template_name and evaluation_selector:
        try:
            listed = _run_kubectl([*base, "get", "sandbox", "-l", str(evaluation_selector), "-o", "name"])
            discovered = [line.rsplit("/", 1)[-1] for line in listed.splitlines() if line.strip()]
            if discovered:
                sandbox_names = discovered
        except RuntimeError as exc:
            # Still attempt the primary selector delete and the deterministic
            # fallback name. Report the discovery failure so cleanup is retried.
            failures.append(str(exc))
    try:
        _run_kubectl(
            [
                *base,
                "delete",
                *primary_target,
                "--ignore-not-found=true",
                "--wait=true",
                "--timeout=60s",
            ]
        )
    except RuntimeError as exc:
        failures.append(str(exc))

    # Claim mode returns warm-pool capacity through the claim controller. Do not
    # directly delete its adopted Sandbox/Pod. Direct CRD mode owns both the pod
    # and the optional NetworkPolicy, including old-UID orphans left by a failed
    # controller reconciliation. The evaluation label is shared with Switchyard,
    # so it must never be used for dependent deletion. Discover the Sandbox CR
    # names first, then use sandbox-k8s's ownership label on Pods and policies.
    dependent_selector = f"sandbox-k8s/sandbox in ({','.join(sandbox_names)})"
    if not template_name:
        try:
            _run_kubectl(
                [
                    *base,
                    "delete",
                    "pod,networkpolicy",
                    "-l",
                    dependent_selector,
                    "--ignore-not-found=true",
                    "--wait=true",
                    "--timeout=60s",
                ]
            )
        except RuntimeError as exc:
            failures.append(str(exc))

    try:
        primary_get = (
            [primary_kind, "-l", str(evaluation_selector)]
            if evaluation_selector
            else [primary_kind, sandbox_name, "--ignore-not-found=true"]
        )
        remaining = _run_kubectl([*base, "get", *primary_get, "-o", "name"]).strip()
        if remaining:
            failures.append(f"sandbox cleanup left resource behind: {remaining}")
    except RuntimeError as exc:
        failures.append(str(exc))
    if not template_name:
        try:
            remaining = _run_kubectl(
                [
                    *base,
                    "get",
                    "pod,networkpolicy",
                    "-l",
                    dependent_selector,
                    "-o",
                    "name",
                ]
            ).strip()
            if remaining:
                failures.append(f"sandbox cleanup left dependents behind: {remaining}")
        except RuntimeError as exc:
            failures.append(str(exc))
    if failures:
        raise RuntimeError("; ".join(failures))


def _sandbox_cleanup_metadata(config_text: str, evaluation_id: str) -> dict[str, str | bool | None]:
    import yaml

    try:
        config = yaml.safe_load(config_text) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"invalid rendered sandbox config: {exc}") from exc
    if not isinstance(config, Mapping):
        raise RuntimeError("rendered sandbox config must be an object")
    env = config.get("environment")
    kwargs = env.get("kwargs") if isinstance(env, Mapping) else None
    if not isinstance(kwargs, Mapping):
        raise RuntimeError("rendered sandbox config has no environment.kwargs")
    labels = kwargs.get("labels")
    evaluation_label = labels.get(_EVALUATION_LABEL) if isinstance(labels, Mapping) else None
    verify_ssl_value = kwargs.get("verify_ssl", True)
    verify_ssl = not (
        verify_ssl_value is False or (isinstance(verify_ssl_value, str) and verify_ssl_value.strip().lower() == "false")
    )
    return {
        "sandbox_name": _sanitize_k8s_name(f"hbr-{evaluation_id[:50]}"),
        "selector": (f"{_EVALUATION_LABEL}={evaluation_id}" if evaluation_label == evaluation_id else None),
        "namespace": str(kwargs.get("namespace") or "default"),
        "context": str(kwargs["context"]) if kwargs.get("context") else None,
        "kubeconfig_path": (str(kwargs["kubeconfig_path"]) if kwargs.get("kubeconfig_path") else None),
        "template_name": str(kwargs["template_name"]) if kwargs.get("template_name") else None,
        "verify_ssl": verify_ssl,
    }


def _run_kubectl(argv: list[str]) -> str:
    result = execute_kubectl(argv, runner=subprocess.run)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown kubectl error").strip()
        raise RuntimeError(f"sandbox cleanup command failed ({' '.join(argv)}): {detail}")
    return result.stdout or ""


def _parse_cpu_cores(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    factors = {"n": Decimal("1e-9"), "u": Decimal("1e-6"), "m": Decimal("1e-3")}
    suffix = text[-1]
    try:
        number = Decimal(text[:-1]) * factors[suffix] if suffix in factors else Decimal(text)
    except InvalidOperation:
        return None
    return float(number)


def _parse_memory_bytes(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    factors = {
        "Ki": 1024,
        "Mi": 1024**2,
        "Gi": 1024**3,
        "Ti": 1024**4,
        "K": 1000,
        "M": 1000**2,
        "G": 1000**3,
        "T": 1000**4,
    }
    for suffix, factor in factors.items():
        if text.endswith(suffix):
            try:
                return int(Decimal(text[: -len(suffix)]) * factor)
            except InvalidOperation:
                return None
    try:
        return int(Decimal(text))
    except InvalidOperation:
        return None


def _sandbox_kubectl_base(handle: LaunchHandle) -> tuple[list[str], str] | None:
    cleanup = handle.raw.get("cleanup")
    if not isinstance(cleanup, Mapping) or not cleanup.get("selector"):
        return None
    kubectl = shutil.which("kubectl") or shutil.which("oc")
    if kubectl is None:
        return None
    base = [kubectl]
    if cleanup.get("kubeconfig_path"):
        base.extend(["--kubeconfig", str(cleanup["kubeconfig_path"])])
    if cleanup.get("context"):
        base.extend(["--context", str(cleanup["context"])])
    if cleanup.get("verify_ssl") is False:
        base.append("--insecure-skip-tls-verify=true")
    base.extend(["-n", str(cleanup.get("namespace") or "default")])
    return base, str(cleanup["selector"])


def sample_sandbox_k8s_resources(handle: LaunchHandle) -> list[ResourceUsageSample]:
    """Best-effort aggregate of evaluation-labeled sandbox Pod resources."""

    target = _sandbox_kubectl_base(handle)
    if target is None:
        return []
    base, selector = target
    pods_result = execute_kubectl(
        [*base, "get", "pods", "-l", selector, "-o", "json"],
        runner=subprocess.run,
        timeout_seconds=_RESOURCE_SAMPLE_COMMAND_TIMEOUT_SECONDS,
    )
    if pods_result.returncode != 0:
        detail = redact_secret_text((pods_result.stderr or pods_result.stdout or "").strip())
        return [
            ResourceUsageSample(
                source="kubernetes_metrics_api",
                collection_status="metrics_unavailable",
                collection_error=detail[:2000] or "pod specification query failed",
            )
        ]
    try:
        pods = json.loads(pods_result.stdout).get("items") or []
    except (AttributeError, json.JSONDecodeError):
        return [
            ResourceUsageSample(
                source="kubernetes_metrics_api",
                collection_status="metrics_unavailable",
                collection_error="pod specification query returned invalid JSON",
            )
        ]
    if not pods:
        return [
            ResourceUsageSample(
                source="kubernetes_metrics_api",
                collection_status="metrics_unavailable",
                collection_error="no matching sandbox pods were found",
            )
        ]

    cpu_request = cpu_limit = gpu_request = 0.0
    memory_request = memory_limit = 0
    has_cpu_request = has_cpu_limit = has_gpu_request = False
    has_memory_request = has_memory_limit = False
    for pod in pods:
        for container in (pod.get("spec") or {}).get("containers") or []:
            resources = container.get("resources") or {}
            requests = resources.get("requests") or {}
            limits = resources.get("limits") or {}
            if (value := _parse_cpu_cores(requests.get("cpu"))) is not None:
                cpu_request += value
                has_cpu_request = True
            if (value := _parse_cpu_cores(limits.get("cpu"))) is not None:
                cpu_limit += value
                has_cpu_limit = True
            if (value := _parse_memory_bytes(requests.get("memory"))) is not None:
                memory_request += value
                has_memory_request = True
            if (value := _parse_memory_bytes(limits.get("memory"))) is not None:
                memory_limit += value
                has_memory_limit = True
            gpu_value = requests.get("nvidia.com/gpu", limits.get("nvidia.com/gpu"))
            if gpu_value is not None:
                try:
                    gpu_request += float(gpu_value)
                    has_gpu_request = True
                except (TypeError, ValueError):
                    pass

    metrics_result = execute_kubectl(
        [*base, "top", "pod", "-l", selector, "--containers", "--no-headers"],
        runner=subprocess.run,
        timeout_seconds=_RESOURCE_SAMPLE_COMMAND_TIMEOUT_SECONDS,
    )
    cpu_usage = 0.0
    memory_usage = 0
    has_cpu_usage = has_memory_usage = False
    if metrics_result.returncode == 0:
        for line in metrics_result.stdout.splitlines():
            fields = line.split()
            if len(fields) < 4:
                continue
            if (value := _parse_cpu_cores(fields[-2])) is not None:
                cpu_usage += value
                has_cpu_usage = True
            if (value := _parse_memory_bytes(fields[-1])) is not None:
                memory_usage += value
                has_memory_usage = True

    if metrics_result.returncode != 0:
        metrics_error = (
            redact_secret_text((metrics_result.stderr or metrics_result.stdout or "").strip())[:2000]
            or "Kubernetes Metrics API query failed"
        )
    elif not has_cpu_usage and not has_memory_usage:
        metrics_error = "Kubernetes Metrics API returned no container metrics"
    else:
        metrics_error = None

    return [
        ResourceUsageSample(
            source="kubernetes_metrics_api",
            collection_status="sampled" if metrics_error is None else "metrics_unavailable",
            collection_error=metrics_error,
            cpu_usage_cores=cpu_usage if has_cpu_usage else None,
            memory_usage_bytes=memory_usage if has_memory_usage else None,
            cpu_request_cores=cpu_request if has_cpu_request else None,
            cpu_limit_cores=cpu_limit if has_cpu_limit else None,
            memory_request_bytes=memory_request if has_memory_request else None,
            memory_limit_bytes=memory_limit if has_memory_limit else None,
            gpu_request=gpu_request if has_gpu_request else None,
        )
    ]


def _sanitize_k8s_name(value: str) -> str:
    name = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    return (name or "sandbox")[:63].rstrip("-")


def launch_harbor_runner_container(
    *,
    image: str,
    harbor_dir: str,
    evaluation_id: str,
    harbor_jobs_dir: Path,
    kube_config_dir: Path,
    work_volume: str,
    rendered_config_name: str,
    env_file_host: Path,
    env_file_in_container: str = "/run/sandbox.env",
) -> str:
    """Start a detached harbor-runner container; return the container id."""
    import docker

    client = docker.from_env()
    jobs_mount = harbor_jobs_dir / "jobs"
    jobs_mount.mkdir(parents=True, exist_ok=True)
    container = client.containers.run(
        image=image,
        entrypoint=[f"{harbor_dir}/.venv/bin/harbor"],
        command=[
            "run",
            "-c",
            f"/work/{rendered_config_name}",
            "--env-file",
            env_file_in_container,
            "-y",
        ],
        working_dir=harbor_dir,
        detach=True,
        name=_harbor_container_name(evaluation_id),
        volumes={
            str(jobs_mount): {"bind": f"{harbor_dir}/jobs", "mode": "rw"},
            str(kube_config_dir): {"bind": "/root/.kube", "mode": "ro"},
            work_volume: {"bind": "/work", "mode": "ro"},
            str(env_file_host): {"bind": "/run/sandbox.env", "mode": "ro"},
        },
        environment={"HOME": "/root"},
        remove=False,
    )
    return container.id


def make_sandbox_k8s_docker_submitter(
    *,
    image: str,
    harbor_dir: str,
    harbor_jobs_dir: str,
    kube_config_dir: str,
    config_path: str,
    env_file: str,
    work_dir: str,
    work_volume: str,
    host_env_file: str | None = None,
    allow_insecure_tls: bool = False,
    runner: Runner | None = None,
) -> Callable[[LaunchSpec], LaunchHandle]:
    """Build a submitter that launches one harbor-runner container per evaluation."""
    harbor = Path(harbor_dir).expanduser()
    cfg = Path(config_path).expanduser()
    envf = Path(env_file).expanduser()
    work = Path(work_dir).expanduser().resolve()
    harbor_host = _resolve_host_path(harbor_jobs_dir)
    kube_host = Path(kube_config_dir).expanduser().resolve()
    envf_host = _resolve_host_env_file(host_env_file, envf)

    def submit(spec: LaunchSpec) -> LaunchHandle:
        selected_harbor = Path(spec.harbor_dir or harbor).expanduser()
        selected_image = spec.runner_image_ref or image
        env = {**os.environ, **load_env_file(envf)}
        env.update(spec.credential_env)
        LOG.info(
            "dispatch %s: credential_env keys=%s",
            spec.evaluation_id,
            sorted(spec.credential_env.keys()),
        )
        if env.pop("SANDBOX_OC_TOKEN", None):
            LOG.warning(
                "Docker sandbox dispatch ignores the evaluation OpenShift credential and "
                "uses the operator kubeconfig mounted at /root/.kube"
            )
        # Stage the uploaded task tree into the per-eval /work mount so the run
        # uses the task's own task definition. The work dir is mounted into
        # the harbor-runner at /work, so the staged tree is addressed there.
        staged_task_path: str | None = None
        staged_task_dir = work / spec.evaluation_id / _staged_task_name(spec)
        tarball_object_key = spec.tarball_object_key
        if (
            not _is_dataset_only_harbor_profile(spec.harbor_config)
            and _should_stage_uploaded_task_tree(spec)
            and tarball_object_key is not None
            and _stage_task_tree(tarball_object_key, staged_task_dir)
        ):
            staged_task_path = f"/work/{spec.evaluation_id}/{staged_task_dir.name}"
            if spec.extra_skill_object_keys:
                materials = _inject_extra_skills(staged_task_dir, spec.extra_skill_object_keys)
                _save_extra_skill_materials_artifact(
                    materials,
                    spec.evaluation_id,
                    harbor_dir=selected_harbor,
                )
            _patch_instruction(staged_task_dir, spec.instruction_prefix, spec.instruction_postfix)
            _save_instruction_artifact(staged_task_dir, spec.evaluation_id, harbor_dir=selected_harbor)
        agent_timeout_apply: dict[str, float | None] | None = None
        if spec.agent_timeout_floor_sec is not None:
            agent_timeout_apply = apply_agent_timeout_floor(
                staged_task_dir if staged_task_path else None, spec.agent_timeout_floor_sec
            )
        _augment_harbor_env(env, envf, cfg, staged_task_path=staged_task_path)
        root_authorized = _root_authorized_for_spec(spec)
        writable_root_authorized = root_authorized and settings.sandbox_k8s_allow_writable_root
        if root_authorized:
            LOG.warning(
                "dispatch %s: operator-authorized root sandbox for task image digest %s",
                spec.evaluation_id,
                spec.image_digest,
            )
        task_image_ref = _task_image_ref_for_sandbox(spec)
        rendered = render_harbor_config(
            cfg.read_text(),
            env,
            job_name=spec.evaluation_id,
            n_attempts=spec.n_attempts,
            n_concurrent_trials=spec.parallelism,
            image_ref=task_image_ref,
            profile_config=spec.harbor_config,
            task_path=staged_task_path,
            network_policy=spec.network_policy,
            network_policy_config=spec.network_policy_config,
            allow_insecure_tls=allow_insecure_tls,
            root_authorized=root_authorized,
            writable_root_authorized=writable_root_authorized,
            agent_bundle=spec.agent_bundle,
            benchmark_run_id=spec.benchmark_run_id,
            dataset_image_map=_dataset_image_map(spec),
        )
        writable_root_requested = _writable_root_requested(rendered)
        if writable_root_requested:
            LOG.warning(
                "dispatch %s: writable-root task container requested for task image digest %s (operator_authorized=%s)",
                spec.evaluation_id,
                spec.image_digest,
                writable_root_authorized,
            )
        rendered_name = f"{spec.evaluation_id}.yaml"
        rendered_path = work / rendered_name
        rendered_path.write_text(rendered)
        cleanup_metadata = _sandbox_cleanup_metadata(rendered, spec.evaluation_id)
        log_path = work / spec.evaluation_id / "harbor.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_agent_env = {key: value for key, value in spec.credential_env.items() if key != "SANDBOX_OC_TOKEN"}
        if spec.initial_user_turns:
            runtime_agent_env[_INITIAL_USER_TURNS_ENV] = json.dumps(spec.initial_user_turns, separators=(",", ":"))
        runtime_envf = merged_env_file(
            source_env_file=envf,
            output_env_file=work / spec.evaluation_id / "sandbox.env",
            credential_env=runtime_agent_env,
        )
        env_file_in_container = "/run/sandbox.env"
        if runtime_envf != envf:
            env_file_in_container = f"/work/{spec.evaluation_id}/sandbox.env"
        container_id_holder: dict[str, str] = {}

        def _live_runner(_argv: list[str], _cwd: Path, log: Path) -> None:
            container_id = launch_harbor_runner_container(
                image=selected_image,
                harbor_dir=str(selected_harbor),
                evaluation_id=spec.evaluation_id,
                harbor_jobs_dir=harbor_host,
                kube_config_dir=kube_host,
                work_volume=work_volume,
                rendered_config_name=rendered_name,
                env_file_host=envf_host,
                env_file_in_container=env_file_in_container,
            )
            container_id_holder["container_id"] = container_id
            log.write_text(
                f"launched harbor-runner container {container_id}\n"
                f"image: {selected_image}\n"
                f"harbor: {selected_harbor}\n"
                f"config: /work/{rendered_name}\n"
                f"harbor jobs: {harbor_host / 'jobs'}\n"
                f"logs: docker logs harbor-{spec.evaluation_id}\n"
            )

        (runner or _live_runner)([], selected_harbor, log_path)
        return LaunchHandle(
            backend=SandboxK8sBackend.name,
            external_id=spec.evaluation_id,
            raw={
                "config": str(rendered_path),
                "log": str(log_path),
                "container_id": container_id_holder.get("container_id"),
                "harbor_runner_image": selected_image,
                "harbor_dir": str(selected_harbor),
                "agent_timeout_apply": agent_timeout_apply,
                "docker": True,
                "cleanup": cleanup_metadata,
                "root_authorized": root_authorized,
                "writable_root_requested": writable_root_requested,
                "writable_root_authorized": writable_root_authorized,
                "task_image_ref": task_image_ref,
                "harbor_dataset_image_imports": spec.harbor_dataset_image_imports,
            },
        )

    return submit


def _harbor_run_finished(result: Mapping[str, Any]) -> bool:
    """Whether a Harbor ``result.json`` represents a *completed* run.

    Harbor writes ``result.json`` incrementally — it exists (with trials still
    ``pending``/``running`` and ``finished_at`` null) while the run is in flight,
    and is also left behind verbatim if the run dies early. So "the file exists"
    does not mean "the run finished". The run is done only once ``finished_at``
    is set, or — defensively, in case that field is ever absent — once every
    trial is accounted for (nothing pending or running).
    """
    if result.get("finished_at"):
        return True
    total = result.get("n_total_trials") or 0
    if not total:
        return False
    stats = result.get("stats") or {}
    accounted = (
        (stats.get("n_completed_trials") or 0)
        + (stats.get("n_errored_trials") or 0)
        + (stats.get("n_cancelled_trials") or 0)
    )
    pending = stats.get("n_pending_trials") or 0
    running = stats.get("n_running_trials") or 0
    return pending == 0 and running == 0 and accounted >= total


def _harbor_exception_counts(stats: Mapping[str, Any]) -> dict[str, int]:
    """Aggregate per-eval ``exception_stats`` into ``exception_name -> count``.

    Harbor records the exceptions that errored a trial under each eval's
    ``exception_stats`` (the retry ``include_exceptions`` — e.g.
    ``EnvironmentStartTimeoutError``, ``SandboxExecutionError`` — surface here
    once retries are exhausted). Tolerate the shapes it may take (a name->count
    mapping, a name->list mapping, or a bare list of names); this is best-effort
    detail for ``status_detail``, so an odd shape degrades to a count rather than
    raising.
    """
    counts: dict[str, int] = {}
    evals = stats.get("evals") or {}
    if not isinstance(evals, Mapping):
        return counts
    for ev in evals.values():
        if not isinstance(ev, Mapping):
            continue
        exc = ev.get("exception_stats")
        if isinstance(exc, Mapping):
            for name, value in exc.items():
                if isinstance(value, bool):
                    inc = 1
                elif isinstance(value, int):
                    inc = value
                elif isinstance(value, list | tuple):
                    inc = len(value)
                else:
                    inc = 1
                counts[str(name)] = counts.get(str(name), 0) + inc
        elif isinstance(exc, list | tuple):
            for name in exc:
                counts[str(name)] = counts.get(str(name), 0) + 1
    return counts


def _harbor_failed_solve_count(stats: Mapping[str, Any]) -> int | None:
    """Count scored zero/false trials when Harbor exposes reward buckets."""
    failed_ids: set[str] = set()
    saw_reward_stats = False
    evals = stats.get("evals") or {}
    if not isinstance(evals, Mapping):
        return None
    for ev in evals.values():
        if not isinstance(ev, Mapping):
            continue
        reward_stats = ev.get("reward_stats")
        by_reward = reward_stats.get("reward") if isinstance(reward_stats, Mapping) else None
        if not isinstance(by_reward, Mapping):
            continue
        saw_reward_stats = True
        for reward, trial_ids in by_reward.items():
            normalized = str(reward).strip().lower()
            try:
                is_failed = normalized == "false" or float(normalized) == 0.0
            except ValueError:
                is_failed = False
            if not is_failed:
                continue
            if isinstance(trial_ids, list | tuple | set):
                failed_ids.update(str(trial_id) for trial_id in trial_ids)
            elif trial_ids is not None:
                failed_ids.add(str(trial_ids))
    return len(failed_ids) if saw_reward_stats else None


def _harbor_error_summary(result: Mapping[str, Any]) -> str | None:
    """Summarize infra/environment errors in a *finished* Harbor result.

    A trial that *errored* is counted in ``stats.n_errored_trials`` and did NOT
    run to a scored verdict — a sandbox/agent/environment failure (split-DNS,
    blocked egress, an unreachable model endpoint, a sandbox that never came up).
    That is categorically different from a trial that ran and genuinely scored
    reward 0, which is a *completed* trial. Keying on errored trials (and the
    per-eval ``exception_stats``) is what keeps the two apart: this returns a
    detail string only when trials actually errored, so a legitimate reward-0 run
    is never mistaken for an infra failure. Returns ``None`` when nothing errored.
    """
    stats = result.get("stats") or {}
    if not isinstance(stats, Mapping):
        return None
    n_errored = stats.get("n_errored_trials") or 0
    exceptions = _harbor_exception_counts(stats)
    if not n_errored and not exceptions:
        return None
    total = result.get("n_total_trials")
    head = f"{n_errored}/{total} trials errored" if total else f"{n_errored} trials errored"
    if exceptions:
        names = ", ".join(f"{name} x{count}" if count > 1 else name for name, count in exceptions.items())
        if "SandboxCreationError" in exceptions:
            LOG.warning(
                "SandboxCreationError in trial results — K8s credentials are likely expired. "
                "Re-authenticate against the cluster (for example `oc login`, "
                "`gcloud container clusters get-credentials`, or `aws eks update-kubeconfig`) "
                "then run `make compose-up` to refresh the kubeconfig mount."
            )
        return f"{head}: {names}"
    return head


def _sandbox_oom_summary(job_dir: Path) -> str | None:
    """Return the terminated sandbox container's OOM detail from Harbor artifacts."""
    for status_path in job_dir.glob("*/artifacts/k8s/pod-status.json"):
        try:
            pod_status = json.loads(status_path.read_text())
        except (OSError, ValueError):
            continue
        container_statuses = pod_status.get("container_statuses") or pod_status.get("containerStatuses")
        if not isinstance(container_statuses, list):
            continue
        for container_status in container_statuses:
            if not isinstance(container_status, Mapping):
                continue
            state = container_status.get("state") or {}
            terminated = state.get("terminated") if isinstance(state, Mapping) else None
            if not isinstance(terminated, Mapping) or terminated.get("reason") != "OOMKilled":
                continue
            name = container_status.get("name") or "sandbox"
            exit_code = terminated.get("exit_code", terminated.get("exitCode"))
            suffix = f", exit {exit_code}" if exit_code is not None else ""
            return f"{name} container was OOMKilled{suffix}"
    return None


def make_sandbox_k8s_docker_status_reader(
    *,
    harbor_dir: str,
    jobs_dir: str,
    work_dir: str,
    artifact_root: str | None = None,
) -> StatusReader:
    """Poll harbor-runner container exit, then read Harbor ``result.json``."""
    file_reader = make_sandbox_k8s_status_reader(
        harbor_dir=harbor_dir,
        jobs_dir=jobs_dir,
        artifact_root=artifact_root,
    )
    work = Path(work_dir).expanduser().resolve()

    def read(handle: LaunchHandle) -> RuntimeStatus:
        from docker.errors import NotFound

        import docker

        client = docker.from_env()
        try:
            container = client.containers.get(_harbor_container_name(handle.external_id))
        except NotFound:
            return file_reader(handle)

        container.reload()
        state = container.attrs.get("State") or {}
        if state.get("Running"):
            return RuntimeStatus(phase="running", detail="harbor-runner container running")

        exit_code = state.get("ExitCode")
        result_path = _harbor_result_path(
            handle,
            harbor_dir=harbor_dir,
            jobs_dir=jobs_dir,
            artifact_root=artifact_root,
        )
        result: dict[str, Any] | None = None
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
            except (OSError, ValueError):
                result = None
        # Only defer to the result.json (success) once the container has gone AND
        # Harbor actually completed the run. A container that exited while the
        # result is still partial (trials pending, no finished_at) crashed — e.g.
        # a missing task tree or a bad image — and must read as failed, not as a
        # deceptive ``succeeded`` with a null reward.
        if result is not None and _harbor_run_finished(result):
            return file_reader(handle)

        log_path = work / handle.external_id / "harbor.log"
        tail = ""
        try:
            logs = container.logs(tail=80).decode("utf-8", errors="replace")
            tail = logs[-800:] if len(logs) > 800 else logs
        except Exception:  # noqa: BLE001
            if log_path.is_file():
                text = log_path.read_text(errors="replace")
                tail = text[-800:] if len(text) > 800 else text
        # Name the path we looked at: if the run actually produced a result, a
        # missing file here is a jobs-dir mount/anchor mismatch (the runner wrote
        # somewhere the worker doesn't read) rather than a crashed run.
        missing = "" if result is not None else f" (no result.json at {result_path})"
        detail = f"harbor-runner exited {exit_code} without a completed result{missing}"
        if tail:
            detail = f"{detail}: {tail}"
        return RuntimeStatus(phase="failed", detail=detail, raw=result or {})

    return read


# In-cluster defaults for a user-token kubeconfig: the pod reaches the API at the
# default service, validated by the projected SA CA. The user's OAuth bearer
# token is accepted there regardless of where the client runs.
_IN_CLUSTER_API = "https://kubernetes.default.svc"
_IN_CLUSTER_CA = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
_USER_CONTEXT = "user"
_NAMESPACE_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")


def _validated_user_kubeconfig_fields(
    *, token: str, server: str, ca_path: str, namespace: str, verify_ssl: bool
) -> tuple[str, str, str, str]:
    if not token or token != token.strip() or len(token) > 16384 or _CONTROL_CHAR_RE.search(token):
        raise ValueError("OpenShift credential token is empty, too long, padded, or contains control characters")
    if any(character.isspace() for character in token):
        raise ValueError("OpenShift credential token must not contain whitespace")
    if namespace != namespace.strip() or _CONTROL_CHAR_RE.search(namespace):
        raise ValueError("sandbox namespace must be a valid Kubernetes DNS label")
    if not _NAMESPACE_RE.fullmatch(namespace):
        raise ValueError("sandbox namespace must be a valid Kubernetes DNS label")
    if server != server.strip() or _CONTROL_CHAR_RE.search(server):
        raise ValueError("sandbox API server is not a valid URL")
    try:
        parsed = urlsplit(server)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("sandbox API server is not a valid URL") from exc
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("sandbox API server must be an HTTPS origin without credentials, path, query, or fragment")
    if verify_ssl and (
        not ca_path
        or ca_path != ca_path.strip()
        or _CONTROL_CHAR_RE.search(ca_path)
        or not Path(ca_path).is_absolute()
        or ".." in Path(ca_path).parts
    ):
        raise ValueError("sandbox CA path must be an absolute path without traversal")
    return token, server, ca_path, namespace


def _write_user_kubeconfig(
    dest: Path,
    *,
    token: str,
    server: str,
    ca_path: str,
    namespace: str,
    verify_ssl: bool,
    allow_insecure_tls: bool = False,
) -> Path:
    """Write a per-evaluation kubeconfig that authenticates as the token's user.

    The token is a user's OpenShift bearer token (provider ``openshift``); running
    ``harbor run`` against this kubeconfig makes every cluster action (incl. the
    per-sandbox NetworkPolicy that direct/CRD mode creates) happen as that user,
    so a user can run custom-image Harbor tasks with their own rights — no
    service-account NetworkPolicy grant. File is 0600; it holds the token, so it
    stays in the per-eval work dir and is never passed to the agent env-file.
    """
    if not verify_ssl and not allow_insecure_tls:
        raise ValueError("sandbox Kubernetes TLS verification cannot be disabled for a user kubeconfig")
    token, server, ca_path, namespace = _validated_user_kubeconfig_fields(
        token=token,
        server=server,
        ca_path=ca_path,
        namespace=namespace,
        verify_ssl=verify_ssl,
    )
    cluster = {"server": server}
    if verify_ssl:
        cluster["certificate-authority"] = ca_path
    else:
        cluster["insecure-skip-tls-verify"] = True
        LOG.warning("INSECURE LOCAL MODE: generated user kubeconfig disables Kubernetes API TLS verification")
    config = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [{"name": _USER_CONTEXT, "cluster": cluster}],
        "contexts": [
            {
                "name": _USER_CONTEXT,
                "context": {
                    "cluster": _USER_CONTEXT,
                    "namespace": namespace,
                    "user": _USER_CONTEXT,
                },
            }
        ],
        "current-context": _USER_CONTEXT,
        "users": [{"name": _USER_CONTEXT, "user": {"token": token}}],
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(config, sort_keys=False))
    dest.chmod(0o600)
    return dest


def _bind_user_kubeconfig(rendered: str, kubeconfig: Path) -> str:
    """Point the rendered Harbor config's kubeconfig_path/context at the user kubeconfig."""
    try:
        config = yaml.safe_load(rendered) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid rendered Harbor config: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("rendered Harbor config must be an object")
    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("rendered Harbor config has no environment object")
    kwargs = environment.get("kwargs")
    if not isinstance(kwargs, dict):
        raise ValueError("rendered Harbor config environment.kwargs must be an object")
    kwargs["kubeconfig_path"] = str(kubeconfig)
    kwargs["context"] = _USER_CONTEXT
    return yaml.safe_dump(config, sort_keys=False)


def make_sandbox_k8s_submitter(
    *,
    harbor_dir: str,
    config_path: str,
    env_file: str,
    work_dir: str = "/tmp",
    allow_insecure_tls: bool = False,
    runner: Runner | None = None,
) -> Callable[[LaunchSpec], LaunchHandle]:
    """Build the live submitter that :meth:`SandboxK8sBackend.launch` calls.

    Renders the Harbor config for the evaluation and starts ``harbor run``
    against the target. ``runner`` is injected in tests so no subprocess is
    spawned.
    """
    runner = runner or _spawn_detached
    harbor = Path(harbor_dir).expanduser()
    cfg = Path(config_path).expanduser()
    envf = Path(env_file).expanduser()
    work = Path(work_dir)

    def submit(spec: LaunchSpec) -> LaunchHandle:
        selected_harbor = Path(spec.harbor_dir or harbor).expanduser()
        env = {**os.environ, **load_env_file(envf)}
        env.update(spec.credential_env)
        LOG.info(
            "dispatch %s: credential_env keys=%s",
            spec.evaluation_id,
            sorted(spec.credential_env.keys()),
        )
        # Host path: harbor runs in-process, so the staged tree is referenced by
        # its real filesystem path rather than a /work container path.
        staged_task_dir = work / spec.evaluation_id / _staged_task_name(spec)
        staged_task_path: str | None = None
        tarball_object_key = spec.tarball_object_key
        if (
            not _is_dataset_only_harbor_profile(spec.harbor_config)
            and _should_stage_uploaded_task_tree(spec)
            and tarball_object_key is not None
            and _stage_task_tree(tarball_object_key, staged_task_dir)
        ):
            staged_task_path = str(staged_task_dir)
            if spec.extra_skill_object_keys:
                materials = _inject_extra_skills(staged_task_dir, spec.extra_skill_object_keys)
                _save_extra_skill_materials_artifact(
                    materials,
                    spec.evaluation_id,
                    harbor_dir=selected_harbor,
                )
            _patch_instruction(staged_task_dir, spec.instruction_prefix, spec.instruction_postfix)
            _save_instruction_artifact(staged_task_dir, spec.evaluation_id, harbor_dir=selected_harbor)
        agent_timeout_apply: dict[str, float | None] | None = None
        if spec.agent_timeout_floor_sec is not None:
            agent_timeout_apply = apply_agent_timeout_floor(
                staged_task_dir if staged_task_path else None, spec.agent_timeout_floor_sec
            )
        _augment_harbor_env(env, envf, cfg, staged_task_path=staged_task_path)
        # A user-supplied OpenShift token (credential provider 'openshift') runs
        # this eval as that user: synthesize a per-eval kubeconfig from the token
        # and bind the Harbor config to it, instead of the pod ServiceAccount.
        oc_token = env.pop("SANDBOX_OC_TOKEN", None)
        user_kubeconfig: Path | None = None
        if oc_token:
            verify_ssl = env.get("VERIFY_SSL", "true").strip().lower() != "false"
            user_kubeconfig = _write_user_kubeconfig(
                work / spec.evaluation_id / "user-kube" / "config",
                token=oc_token,
                server=env.get("SANDBOX_API_SERVER", _IN_CLUSTER_API),
                ca_path=env.get("SANDBOX_CA_PATH", _IN_CLUSTER_CA),
                namespace=env.get("SANDBOX_NAMESPACE", "default"),
                verify_ssl=verify_ssl,
                allow_insecure_tls=allow_insecure_tls,
            )
            env["SANDBOX_CONTEXT"] = _USER_CONTEXT
        root_authorized = _root_authorized_for_spec(spec)
        writable_root_authorized = root_authorized and settings.sandbox_k8s_allow_writable_root
        if root_authorized:
            LOG.warning(
                "dispatch %s: operator-authorized root sandbox for task image digest %s",
                spec.evaluation_id,
                spec.image_digest,
            )
        task_image_ref = _task_image_ref_for_sandbox(spec)
        rendered = render_harbor_config(
            cfg.read_text(),
            env,
            job_name=spec.evaluation_id,
            n_attempts=spec.n_attempts,
            n_concurrent_trials=spec.parallelism,
            image_ref=task_image_ref,
            profile_config=spec.harbor_config,
            task_path=staged_task_path,
            network_policy=spec.network_policy,
            network_policy_config=spec.network_policy_config,
            allow_insecure_tls=allow_insecure_tls,
            root_authorized=root_authorized,
            writable_root_authorized=writable_root_authorized,
            agent_bundle=spec.agent_bundle,
            benchmark_run_id=spec.benchmark_run_id,
            dataset_image_map=_dataset_image_map(spec),
        )
        writable_root_requested = _writable_root_requested(rendered)
        if writable_root_requested:
            LOG.warning(
                "dispatch %s: writable-root task container requested for task image digest %s (operator_authorized=%s)",
                spec.evaluation_id,
                spec.image_digest,
                writable_root_authorized,
            )
        if user_kubeconfig is not None:
            rendered = _bind_user_kubeconfig(rendered, user_kubeconfig)
        rendered_path = work / f"{spec.evaluation_id}.yaml"
        rendered_path.write_text(rendered)
        cleanup_metadata = _sandbox_cleanup_metadata(rendered, spec.evaluation_id)
        run_work_dir = work / spec.evaluation_id
        run_work_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_work_dir / "harbor.log"
        runtime_agent_env = {k: v for k, v in spec.credential_env.items() if k != "SANDBOX_OC_TOKEN"}
        if spec.initial_user_turns:
            runtime_agent_env[_INITIAL_USER_TURNS_ENV] = json.dumps(spec.initial_user_turns, separators=(",", ":"))
        runtime_envf = merged_env_file(
            source_env_file=envf,
            output_env_file=work / f"{spec.evaluation_id}.env",
            # Strip the OpenShift token: it built the kubeconfig above and must
            # never reach the agent env-file / sandbox pod.
            credential_env=runtime_agent_env,
        )
        argv = [
            *_harbor_run_argv(selected_harbor),
            "-c",
            str(rendered_path),
            "--env-file",
            str(runtime_envf),
            "-y",
        ]
        runner(argv, selected_harbor, log_path)
        return LaunchHandle(
            backend=SandboxK8sBackend.name,
            external_id=spec.evaluation_id,  # = Harbor job_name → <jobs_dir>/<id>
            raw={
                "config": str(rendered_path),
                "log": str(log_path),
                "pid_file": str(log_path.with_suffix(f"{log_path.suffix}.pid")),
                "exit_file": str(log_path.with_suffix(f"{log_path.suffix}.exit.json")),
                "argv": argv,
                "harbor_dir": str(selected_harbor),
                "agent_timeout_apply": agent_timeout_apply,
                "cleanup": cleanup_metadata,
                "root_authorized": root_authorized,
                "writable_root_requested": writable_root_requested,
                "writable_root_authorized": writable_root_authorized,
                "task_image_ref": task_image_ref,
                "harbor_dataset_image_imports": spec.harbor_dataset_image_imports,
            },
        )

    return submit


def _harbor_result_path(
    handle: LaunchHandle,
    *,
    harbor_dir: str,
    jobs_dir: str,
    artifact_root: str | None = None,
) -> Path:
    if artifact_root:
        return Path(artifact_root).expanduser() / handle.external_id / "result.json"
    root = Path(str(handle.raw.get("harbor_dir") or harbor_dir)).expanduser()
    return root / jobs_dir / handle.external_id / "result.json"


def make_sandbox_k8s_status_reader(
    *,
    harbor_dir: str,
    jobs_dir: str,
    artifact_root: str | None = None,
) -> StatusReader:
    """Build the status reader that :meth:`SandboxK8sBackend.status` calls.

    A Harbor run writes ``<harbor_dir>/<jobs_dir>/<job_name>/result.json``
    (``job_name`` = the evaluation id = ``handle.external_id``). The file is
    written *incrementally* — it exists with trials still pending/running while
    the run is in flight — so its mere presence is not "finished". The run is
    only terminal once :func:`_harbor_run_finished` is true (``finished_at``
    set); until then it is still ``running``. A finished run with errored trials
    (see :func:`_harbor_error_summary`) is ``failed`` — an infra/environment
    error must not masquerade as a ``succeeded`` run scored from whatever trials
    happened to complete; otherwise it is ``succeeded``. The parsed
    framework-typed envelope rides along in ``RuntimeStatus.raw`` for the worker
    to summarize and persist.
    This is the cluster-side read — the integration seam — so it is injected as a
    fake in unit tests; nothing here touches a cluster.
    """

    def read(handle: LaunchHandle) -> RuntimeStatus:
        result_path = _harbor_result_path(
            handle,
            harbor_dir=harbor_dir,
            jobs_dir=jobs_dir,
            artifact_root=artifact_root,
        )
        result: dict[str, Any] | None = None
        result_error: str | None = None
        if result_path.exists():
            try:
                loaded = json.loads(result_path.read_text())
                if isinstance(loaded, dict):
                    result = loaded
                else:
                    result_error = "result.json is not an object"
            except (OSError, ValueError) as exc:
                result_error = f"invalid result.json: {exc}"
        if result is None or not _harbor_run_finished(result):
            terminal = _host_runner_terminal_status(
                handle,
                result=result,
                result_error=result_error,
                result_path=result_path,
            )
            if terminal is not None:
                return terminal
            if result is None:
                return RuntimeStatus(phase="running", detail="awaiting result.json")
            return RuntimeStatus(phase="running", detail="harbor run in progress", raw=result)
        oom_summary = _sandbox_oom_summary(result_path.parent)
        if oom_summary:
            return RuntimeStatus(
                phase="failed",
                detail=f"harbor sandbox failed ({oom_summary})",
                raw=result,
            )
        # The run finished, but if any trial errored (a sandbox/agent/environment
        # failure that never produced a scored verdict — split-DNS, blocked
        # egress, an unreachable model) report it as failed, not as a deceptive
        # succeeded with a reward derived from the trials that did complete. A
        # genuinely-scored reward 0 is a *completed* trial and is untouched here.
        error_summary = _harbor_error_summary(result)
        if error_summary:
            return RuntimeStatus(
                phase="failed",
                detail=f"harbor run finished with errored trials ({error_summary})",
                raw=result,
            )
        return RuntimeStatus(phase="succeeded", raw=result)

    return read


def _host_runner_terminal_status(
    handle: LaunchHandle,
    *,
    result: dict[str, Any] | None,
    result_error: str | None,
    result_path: Path,
) -> RuntimeStatus | None:
    """Translate durable detached-runner exit metadata into a failed status."""
    exit_value = handle.raw.get("exit_file")
    pid_value = handle.raw.get("pid_file")
    if not isinstance(exit_value, str) or not exit_value:
        return None
    exit_path = Path(exit_value)
    if exit_path.exists():
        try:
            terminal = json.loads(exit_path.read_text())
            exit_code = int(terminal["exit_code"])
            if not terminal.get("token") or not terminal.get("finished_at"):
                raise ValueError("missing terminal identity")
            reason = f"harbor runner exited {exit_code} without a completed result"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reason = f"harbor runner terminal metadata is invalid: {exc}"
        return RuntimeStatus(
            phase="failed",
            detail=_runner_failure_detail(reason, handle, result_path, result_error),
            raw=result or {},
        )
    if isinstance(pid_value, str) and pid_value and not _runner_identity_alive(Path(pid_value)):
        return RuntimeStatus(
            phase="failed",
            failure_code="runner_disappeared",
            detail=_runner_failure_detail(
                "harbor runner disappeared without terminal metadata",
                handle,
                result_path,
                result_error,
            ),
            raw=result or {},
        )
    return None


def _runner_failure_detail(
    reason: str,
    handle: LaunchHandle,
    result_path: Path,
    result_error: str | None,
) -> str:
    suffix = f" ({result_error})" if result_error else f" (no completed result at {result_path})"
    detail = reason + suffix
    log_value = handle.raw.get("log")
    if isinstance(log_value, str):
        try:
            tail = Path(log_value).read_text(errors="replace")[-800:]
        except OSError:
            tail = ""
        if tail:
            detail += ": " + redact_secret_text(tail)
    return detail


def summarize_harbor_result(result: Mapping[str, Any]) -> ResultSummary:
    """Reduce a Harbor framework-typed result to the generic :class:`ResultSummary`.

    Pulls the headline reward from each eval's primary aggregate metric,
    trial counts, and ``finished_at`` out of the Harbor envelope — see
    the agent-sandbox harness oracle result fixture for the shape:
    top-level ``n_total_trials``/``finished_at`` and
    ``stats.{n_completed_trials,n_errored_trials,evals.<name>.metrics[0]}``.
    Harbor's default mean uses ``mean`` for a single reward key and preserves
    the key name for multi-key rewards, so the conventional ``reward`` key takes
    precedence within that primary metric. Later entries are separate reducers
    and must not override the primary score.
    Tolerant of a partial/odd envelope: anything missing becomes ``None`` rather
    than raising, so a finished-but-unusual run still persists with whatever
    summary could be read (the full envelope is always kept verbatim in
    ``result``).
    """
    stats = result.get("stats") or {}
    evals = stats.get("evals") or {}
    n_completed = stats.get("n_completed_trials")
    n_errored = stats.get("n_errored_trials")
    rewards: list[int | float] = []
    for ev in evals.values():
        if not isinstance(ev, Mapping):
            continue
        metrics = ev.get("metrics")
        if not isinstance(metrics, list) or not metrics or not isinstance(metrics[0], Mapping):
            continue
        primary = metrics[0]
        value = primary.get("reward")
        if isinstance(value, bool) or not isinstance(value, int | float):
            value = primary.get("mean")
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        rewards.append(value)
    return ResultSummary(
        reward=sum(rewards) / len(rewards) if rewards else None,
        n_trials=result.get("n_total_trials"),
        n_completed=n_completed,
        n_errored=n_errored,
        n_failed_solve=_harbor_failed_solve_count(stats),
        exception_counts=_harbor_exception_counts(stats),
        finished_at=result.get("finished_at"),
    )


def build_backend() -> SandboxK8sBackend:
    """Construct the sandbox_k8s backend per settings.

    Default (``SANDBOX_K8S_ENABLED=false``): no submitter — ``launch`` raises,
    the run is marked failed, and nothing touches a cluster. Enabled: the live
    submitter above. Enabled-but-unconfigured raises at construction so the
    misconfig surfaces immediately rather than mid-dispatch.
    """
    if not settings.sandbox_k8s_enabled:
        return SandboxK8sBackend()
    if not (settings.sandbox_k8s_config_path and settings.sandbox_k8s_env_file):
        raise RuntimeError(
            "SANDBOX_K8S_ENABLED is set but SANDBOX_K8S_CONFIG_PATH / "
            "SANDBOX_K8S_ENV_FILE are not — point them at a checked-out "
            "agent-sandbox harness (configs/ + targets/<target>.env)."
        )
    if settings.harbor_runner_image:
        kube_host = settings.kube_config_dir_host or settings.kube_config_dir
        if not (settings.harbor_jobs_dir and kube_host):
            raise RuntimeError(
                "HARBOR_RUNNER_IMAGE is set but HARBOR_JOBS_DIR / KUBE_CONFIG_DIR_HOST "
                "are not — required for harbor-runner docker bind mounts."
            )
        # The docker submitter resolves bind-mount sources to *host* paths via
        # _resolve_host_path: a relative path is anchored to SCALED_EVALS_HOST_DIR.
        # Without that anchor a relative HARBOR_JOBS_DIR silently resolves against
        # the worker's own cwd (e.g. /app/logs inside the container), so the
        # harbor-runner writes result.json to a host directory the worker never
        # mounts back — and a finished, scored run is then reported as *failed*.
        # Fail loudly at startup instead of shipping that silent data loss.
        if not Path(settings.harbor_jobs_dir).expanduser().is_absolute() and (not settings.scaled_evals_host_dir):
            raise RuntimeError(
                f"HARBOR_RUNNER_IMAGE is set with a relative HARBOR_JOBS_DIR "
                f"({settings.harbor_jobs_dir!r}) but SCALED_EVALS_HOST_DIR is unset. "
                "The dispatch worker runs in a container, so a relative jobs dir "
                "resolves against the worker's cwd rather than the host: the "
                "harbor-runner writes result.json where the worker can't read it "
                "back, and a finished run is reported as failed. Set "
                "SCALED_EVALS_HOST_DIR to the host checkout path (compose defaults "
                "it to the invocation dir), or make HARBOR_JOBS_DIR absolute."
            )
        submitter = make_sandbox_k8s_docker_submitter(
            image=settings.harbor_runner_image,
            harbor_dir=settings.harbor_dir,
            harbor_jobs_dir=settings.harbor_jobs_dir,
            kube_config_dir=kube_host,
            config_path=settings.sandbox_k8s_config_path,
            env_file=settings.sandbox_k8s_env_file,
            work_dir=settings.sandbox_k8s_work_dir,
            work_volume=settings.sandbox_k8s_docker_volume,
            host_env_file=settings.sandbox_k8s_host_env_file,
            allow_insecure_tls=settings.sandbox_k8s_allow_insecure_tls,
        )
        status_reader = make_sandbox_k8s_docker_status_reader(
            harbor_dir=settings.harbor_dir,
            jobs_dir=settings.sandbox_k8s_jobs_dir,
            work_dir=settings.sandbox_k8s_work_dir,
            artifact_root=settings.sandbox_k8s_artifact_root,
        )
        terminator = make_sandbox_k8s_docker_terminator()
    else:
        submitter = make_sandbox_k8s_submitter(
            harbor_dir=settings.harbor_dir,
            config_path=settings.sandbox_k8s_config_path,
            env_file=settings.sandbox_k8s_env_file,
            work_dir=settings.sandbox_k8s_work_dir,
            allow_insecure_tls=settings.sandbox_k8s_allow_insecure_tls,
        )
        status_reader = make_sandbox_k8s_status_reader(
            harbor_dir=settings.harbor_dir,
            jobs_dir=settings.sandbox_k8s_jobs_dir,
            artifact_root=settings.sandbox_k8s_artifact_root,
        )
        terminator = make_sandbox_k8s_process_terminator()
    return SandboxK8sBackend(
        submitter=submitter,
        status_reader=status_reader,
        terminator=terminator,
        resource_sampler=sample_sandbox_k8s_resources,
    )


def _artifact_root(evaluation_id: str) -> Path:
    if settings.sandbox_k8s_artifact_root:
        return Path(settings.sandbox_k8s_artifact_root).expanduser() / evaluation_id
    return Path(settings.harbor_dir).expanduser() / settings.sandbox_k8s_jobs_dir / evaluation_id


def _legacy_log_paths(evaluation_id: str) -> tuple[Path, ...]:
    work_dir = Path(str(settings.sandbox_k8s_work_dir)).expanduser()
    return (work_dir / f"{evaluation_id}.log",)


def register_runtime_backends(registry: Any) -> None:
    """Register the sandbox-k8s backend with a RuntimeBackendRegistry-like object."""
    registry.register(
        RuntimeBackendRegistration(
            name=SandboxK8sBackend.name,
            factory=build_backend,
            description="Harbor on agent-sandbox / sandbox-k8s.",
            capabilities=RuntimeBackendCapabilities(
                artifact_root=_artifact_root,
                dispatch_work_dir=setting_evaluation_dir(settings, "sandbox_k8s_work_dir"),
                dispatch_log_name="harbor.log",
                runner_container_prefix="harbor",
                extra_log_paths=_legacy_log_paths,
                supported_network_policies=("unrestricted", "default_deny", "scoped_egress"),
            ),
        )
    )
