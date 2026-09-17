# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a deployed NeMo Platform agent into an Agent Hardener manifest.

This is the centerpiece that makes ``nemo agent-hardener init`` trivial: instead of pointing
agent-hardener at a NAT project and answering discovery prompts, the user names an agent already
deployed in NeMo Platform and we derive the manifest from the agent registry.

Design constraints (both deliberate):

- **No ``agent_hardener`` import.** agent-hardener runs from its own venv, driven by subprocess: its
  garak-based attacker pulls a dependency closure (``litellm``/``torch``) that conflicts with the
  platform's, so it stays out of our lockfile. We therefore build the manifest *dict* matching
  agent-hardener's ``AgentManifest``/``AgentSpec`` schema and let ``agent-hardener run`` validate it. The
  schema authority is ``agent_hardener/manifest.py`` (``AgentSpec`` fields: name, project_dir,
  workflow, port, secrets, secrets_file, egress).
- **Read the agent over HTTP.** We fetch it via the platform SDK (``client.agents.get`` /
  ``client.agents.deployments.list``), which returns plain dicts, so resolution needs no
  ``nemo_agents_plugin`` entity classes. The ~10-line IGW injection below is a local copy of
  ``nemo_agents_plugin.utils.inject_gateway_url``. That copy originally existed to avoid a
  cross-plugin dependency; that rationale is obsolete (``nemo-agents-plugin`` is now a declared
  dependency and ``api/v2/runs.py`` imports ``Agent`` from it), so the copy is free to drift from
  upstream.

Models resolve through the Inference Gateway (the platform standard): the victim workflow's
OpenAI/NIM LLMs get the IGW ``base_url`` injected, so no raw model keys are needed.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

logger = logging.getLogger(__name__)

# Scaffold dir (relative to the manifest location) holding the materialized workflow + project.
SCAFFOLD_ROOT = ".agent-hardener-agents"
WORKFLOW_FILENAME = "workflow.yaml"
# NAT LLM _types whose base_url should point at the Inference Gateway.
_IGW_LLM_TYPES = frozenset({"openai", "nim"})


class AgentResolutionError(Exception):
    """Raised when an agent reference cannot be resolved into a usable manifest."""


@dataclass
class ResolvedManifest:
    """Result of resolving an agent reference into an Agent Hardener manifest."""

    manifest: dict[str, Any]
    workflow_path: Path
    project_dir: Path
    workspace: str
    agent_name: str
    port: int
    secrets: list[str]
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable without a live platform)
# --------------------------------------------------------------------------- #
def parse_agent_ref(ref: str, default_workspace: str) -> tuple[str, str]:
    """Split an agent reference into ``(workspace, name)``.

    Accepts ``"name"`` or ``"workspace/name"``. A URL (anything containing ``"://"``) is
    rejected — ``init --agent`` targets a platform-managed agent, not an arbitrary endpoint.
    """
    if "://" in ref:
        raise AgentResolutionError(f"--agent expects a deployed agent name or workspace/name, not a URL: {ref!r}")
    ref = ref.strip().strip("/")
    if not ref:
        raise AgentResolutionError("agent reference is empty")
    if "/" in ref:
        workspace, name = ref.split("/", 1)
        return workspace or default_workspace, name
    return default_workspace, ref


def inject_gateway_url(config: dict[str, Any], workspace: str, base_url: str) -> dict[str, Any]:
    """Deep-copy *config* and point OpenAI/NIM LLMs at the Inference Gateway.

    A local copy of ``nemo_agents_plugin.utils.inject_gateway_url`` (see the module docstring — the
    copy predates the plugin taking a dependency on ``nemo-agents-plugin``). Uses ``setdefault`` so
    explicit values in the config are preserved: the victim's own model is the target under test and
    is never rewritten here — changing it belongs in the project's workflow, not in the war-game.
    """
    base = base_url.rstrip("/")
    gateway_url = f"{base}/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1"
    config = copy.deepcopy(config)
    for llm_cfg in config.get("llms", {}).values():
        if isinstance(llm_cfg, dict) and llm_cfg.get("_type") in _IGW_LLM_TYPES:
            llm_cfg.setdefault("base_url", gateway_url)
            llm_cfg.setdefault("api_key", "not-used")
    return config


def strip_platform_telemetry(config: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy *config* without ``general.telemetry`` — the victim sandbox cannot honour it.

    Deployed agents carry a ``nemo_files`` tracing exporter that the platform injects at deploy time
    (``nemo_agents_plugin.utils.inject_nemo_files_telemetry``); it is registered by
    ``nemo-agents-plugin`` itself, via a ``nat.plugins`` entry point. The victim runs from the minimal
    project :func:`scaffold_project` writes — ``nvidia-nat[langchain]`` only — so NAT cannot resolve the
    ``nemo_files`` tag and exits on config validation before serving, surfacing as a health-check
    timeout minutes into the run.

    Installing ``nemo-agents-plugin`` into the sandbox to fix that would drag the platform into a
    container the war-game deliberately isolates, so the telemetry is dropped instead: it targets the
    platform's file service, which the victim can neither reach nor should write to. The war-game has
    its own event stream for observability.

    Only applies to the agent-source path, where this module authors the project. A project-source
    manifest installs the user's own ``pyproject.toml``, so declaring the exporter there keeps working.
    """
    config = copy.deepcopy(config)
    general = config.get("general")
    if isinstance(general, dict) and "telemetry" in general:
        general.pop("telemetry", None)
        if not general:
            config.pop("general", None)
    return config


def strip_gateway_url(config: dict[str, Any]) -> dict[str, Any]:
    """Reverse :func:`inject_gateway_url`: drop the Inference-Gateway ``base_url``/``api_key`` we injected.

    The hardened workflow agent-hardener hands back is gateway-bound (that is how the sandboxed victim reached
    the IGW). Before writing it onto the stored agent config we undo that binding, so the agent stays
    deployment-neutral — its next deploy re-injects the gateway. Only the values we add are removed:
    an IGW ``base_url`` and the ``"not-used"`` placeholder ``api_key``; anything the author set stays.
    """
    config = copy.deepcopy(config)
    for llm_cfg in config.get("llms", {}).values():
        if not isinstance(llm_cfg, dict) or llm_cfg.get("_type") not in _IGW_LLM_TYPES:
            continue
        base_url = llm_cfg.get("base_url")
        if isinstance(base_url, str) and "/apis/inference-gateway/" in base_url:
            llm_cfg.pop("base_url", None)
        if llm_cfg.get("api_key") == "not-used":
            llm_cfg.pop("api_key", None)
    return config


def detect_custom_components(agent_config: dict[str, Any]) -> list[str]:
    """Return ``_type`` values that look like custom (non-packaged) NAT components.

    Heuristic: a ``_type`` containing a dot or colon (e.g. ``my_pkg.tools:search``) signals a
    user-defined component whose source is not in the agent's stored config — so the OpenShell
    generic victim build needs the real project. Returns an empty list for config-only agents.
    """
    custom: list[str] = []
    entries: list[Any] = []
    # functions/tools are mappings of named component dicts; workflow is a single component dict.
    for section in ("functions", "tools"):
        block = agent_config.get(section)
        if isinstance(block, dict):
            entries.extend(block.values())
    workflow = agent_config.get("workflow")
    if isinstance(workflow, dict):
        entries.append(workflow)
    for entry in entries:
        if isinstance(entry, dict):
            type_name = entry.get("_type", "")
            if isinstance(type_name, str) and ("." in type_name or ":" in type_name):
                custom.append(type_name)
    return sorted(set(custom))


def derive_secret_names(agent_config: dict[str, Any], extra: list[str] | None = None) -> list[str]:
    """Collect env-var secret names the victim build needs (non-model creds).

    Scans the config for ``${ENV_VAR}`` references and obvious ``*_token`` / ``*_api_key`` keys.
    Model credentials are intentionally excluded — those resolve through the IGW. When the scan finds
    nothing at all, falls back to ``["INFERENCE_API_KEY"]``; note this is a fallback, not an addition —
    a config that declares its own secrets returns only those.
    """
    found: set[str] = set(extra or [])

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    found.add(value[2:-1])
                if isinstance(key, str) and key.lower().endswith(("_token", "_api_key")) and isinstance(value, str):
                    # value like "GITHUB_TOKEN" or "${GITHUB_TOKEN}"
                    candidate = value.strip("${}")
                    if candidate.isupper():
                        found.add(candidate)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(agent_config)
    names = sorted(found) or ["INFERENCE_API_KEY"]
    return names


def gateway_backend(base_url: str) -> dict[str, Any] | None:
    """Route-only backend so the sandboxed victim can reach a *local* Inference Gateway.

    agent-hardener then rewrites ``localhost:<port>`` -> ``host.docker.internal:<port>`` and opens the
    egress route. Remote gateways are reachable directly (via egress discovery), so skip them.
    """
    parts = urlsplit(base_url)
    if parts.hostname not in ("localhost", "127.0.0.1"):
        return None
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return {"name": "nemo-inference-gateway", "ports": [port]}


def build_manifest_dict(
    *,
    agent_name: str,
    project_dir: str,
    workflow: str,
    port: int,
    secrets: list[str],
    secrets_file: str = ".env",
    egress: list[str] | None = None,
    backends: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the ``agent-hardener.yaml`` mapping (mirrors ``agent_hardener.cli.build_manifest``)."""
    agent: dict[str, Any] = {
        "name": agent_name,
        "project_dir": project_dir,
        "workflow": workflow,
        "port": port,
        "secrets": secrets,
        "secrets_file": secrets_file,
    }
    if egress:
        agent["egress"] = egress
    return {"agent": agent, "backends": backends or []}


# --------------------------------------------------------------------------- #
# Filesystem materialization
# --------------------------------------------------------------------------- #
def materialize_workflow(workflow_config: dict[str, Any], project_path: Path) -> Path:
    """Write the (IGW-injected) NAT workflow config to ``project_path/workflow.yaml``."""
    project_path.mkdir(parents=True, exist_ok=True)
    workflow_path = project_path / WORKFLOW_FILENAME
    workflow_path.write_text(yaml.safe_dump(workflow_config, sort_keys=False), encoding="utf-8")
    return workflow_path


def scaffold_project(project_path: Path, agent_name: str) -> None:
    """Write a minimal installable NAT project for a config-only agent.

    OpenShell's generic victim build needs an installable project (``uv pip install`` then
    ``nat serve``). For agents that reference only packaged NAT components, a tiny pyproject
    depending on ``nvidia-nat`` is enough to serve the materialized workflow.
    """
    project_path.mkdir(parents=True, exist_ok=True)
    pyproject = project_path / "pyproject.toml"
    if not pyproject.exists():
        # A config-only victim has no Python package, so hatchling's default file selection fails
        # ("no directory matches the project name"). Ship just the workflow via an explicit
        # only-include so `uv pip install .` builds inside the sandbox.
        pyproject.write_text(
            "[project]\n"
            f'name = "agent-hardener-victim-{agent_name}"\n'
            'version = "0.0.0"\n'
            'requires-python = ">=3.11,<3.13"\n'
            # Pin to nvidia-nat 1.7.x: the guardrails defender writes `_type: pre_tool_verifier`
            # middlewares, which live in ``nat.middleware.defense`` — present in 1.7.x but removed
            # in 1.8.0. On 1.8.0 the hardened victim fails config validation ("middleware type
            # pre_tool_verifier not found") and never serves, so the replay/benign validators all
            # get "Server disconnected". Revisit when agent-hardener targets the 1.8+ defense API.
            'dependencies = ["nvidia-nat[langchain]>=1.7.0,<1.8"]\n'
            "\n[build-system]\n"
            'requires = ["hatchling"]\n'
            'build-backend = "hatchling.build"\n'
            "\n[tool.hatch.build.targets.wheel]\n"
            f'only-include = ["{WORKFLOW_FILENAME}"]\n',
            encoding="utf-8",
        )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _fetch_agent_config(sdk: Any, workspace: str, name: str) -> dict[str, Any]:
    """Fetch the agent's stored NAT workflow config, raising a clean error if unusable."""
    try:
        agent = sdk.agents.get(name, workspace=workspace)
    except Exception as exc:  # any SDK/transport failure → one clean, actionable error
        raise AgentResolutionError(
            f"agent {workspace}/{name!r} not found. Deploy it first (nemo agents create + nemo agents deploy)."
        ) from exc
    agent_config = agent.get("config") or {}
    if not agent_config:
        raise AgentResolutionError(f"agent {workspace}/{name!r} has an empty config; nothing to build a victim from.")
    return agent_config


def _resolve_victim_port(sdk: Any, workspace: str, name: str) -> tuple[int, list[str]]:
    """Return the running deployment's port (else agent-hardener's default 8000) plus any warnings."""
    try:
        resp = sdk.agents.deployments.list(workspace=workspace)
    except Exception:  # transport error → fall back to the default port, but surface why (not a silent miss)
        logger.warning(
            "could not list deployments for %s/%s; defaulting victim port to 8000", workspace, name, exc_info=True
        )
        return 8000, [f"could not reach the deployments API for {workspace}/{name!r}; defaulting victim port to 8000."]
    # The deployments API returns {"data": [...], "pagination": {...}}; normalize to the list of
    # deployment dicts (tolerating a bare list too, for robustness).
    deployments = resp.get("data", []) if isinstance(resp, dict) else (resp or [])
    running = [
        d for d in deployments if isinstance(d, dict) and d.get("agent") == name and d.get("status") == "running"
    ]
    if running and running[0].get("port"):
        return int(running[0]["port"]), []
    return 8000, [f"no running deployment for {workspace}/{name!r}; defaulting victim port to 8000."]


def inspect_agent(ref: str, *, sdk: Any, default_workspace: str) -> tuple[str, int, list[str], list[str]]:
    """Derive the create-form defaults for a deployed agent without materializing anything.

    Returns ``(qualified_ref, port, secrets, warnings)``: the victim port from the running deployment
    (else agent-hardener's default) and the secret names scanned from the stored config. Cheap read-only
    counterpart to :func:`resolve_agent_to_manifest`, used to pre-fill (and let the operator override)
    the port/secret fields before creating the manifest.
    """
    workspace, name = parse_agent_ref(ref, default_workspace)
    agent_config = _fetch_agent_config(sdk, workspace, name)
    port, warnings = _resolve_victim_port(sdk, workspace, name)
    secrets = derive_secret_names(agent_config)
    return f"{workspace}/{name}", port, secrets, warnings


def resolve_agent_to_manifest(
    ref: str,
    *,
    sdk: Any,
    base_url: str,
    default_workspace: str,
    manifest_dir: Path,
    project_dir: str | None = None,
    egress: list[str] | None = None,
    port: int | None = None,
    secrets: list[str] | None = None,
) -> ResolvedManifest:
    """Resolve a deployed-agent reference into a ready Agent Hardener manifest.

    ``sdk`` is a ``nemo_platform.NeMoPlatform`` client. ``manifest_dir`` is where
    ``agent-hardener.yaml`` will be written (paths in the manifest are relative to it).

    Pipeline: parse ref → fetch ``Agent`` config → resolve the victim port from a running
    deployment → IGW-inject the workflow → materialize it under a scaffold/project dir → build
    the manifest dict. For custom-code agents, ``project_dir`` must be supplied (the stored config
    lacks the component source); config-only agents get a generated scaffold.

    ``egress`` allow-lists external hosts the victim may reach (needed for config-only agents,
    whose tool hosts live in packaged code and so aren't found by egress discovery). ``port`` and
    ``secrets`` override the auto-derived victim port / secret names; leave them unset to derive.
    """
    workspace, name = parse_agent_ref(ref, default_workspace)
    agent_config = _fetch_agent_config(sdk, workspace, name)
    resolved_port, warnings = _resolve_victim_port(sdk, workspace, name)
    port = port or resolved_port

    # Custom-code detection gates whether we can scaffold a project automatically.
    custom = detect_custom_components(agent_config)
    if custom and project_dir is None:
        raise AgentResolutionError(
            f"agent {workspace}/{name!r} references custom components {custom} whose source is not "
            "in the stored config. Re-run with --project-dir pointing at the agent's NAT project."
        )

    injected = inject_gateway_url(agent_config, workspace, base_url)

    if project_dir is not None:
        # The user's project supplies its own dependencies, so any telemetry it declares is theirs
        # to satisfy — leave the config alone.
        project_path = Path(project_dir)
        rel_project = project_dir
    else:
        # We author the project here, and deliberately keep it to `nvidia-nat[langchain]`, so drop
        # the platform-only telemetry the config would otherwise ask that install to resolve.
        injected = strip_platform_telemetry(injected)
        rel_project = str(Path(SCAFFOLD_ROOT) / name)
        project_path = manifest_dir / rel_project
        scaffold_project(project_path, name)

    workflow_path = materialize_workflow(injected, project_path)
    secrets = secrets or derive_secret_names(agent_config)

    gw_backend = gateway_backend(base_url)
    manifest = build_manifest_dict(
        agent_name=name,
        project_dir=rel_project,
        workflow=WORKFLOW_FILENAME,
        port=port,
        secrets=secrets,
        egress=egress,
        backends=[gw_backend] if gw_backend else [],
    )

    return ResolvedManifest(
        manifest=manifest,
        workflow_path=workflow_path,
        project_dir=project_path,
        workspace=workspace,
        agent_name=name,
        port=port,
        secrets=secrets,
        warnings=warnings,
    )
