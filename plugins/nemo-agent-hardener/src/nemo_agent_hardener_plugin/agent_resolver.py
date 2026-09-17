# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a registered NeMo Platform agent into a runnable Agent Hardener victim.

This is the only intake path. The user names an agent already registered in the platform and the
plugin hands Agent Hardener a directory containing a runnable agent — a config plus the Dockerfile that
serves it — rather than a query Agent Hardener has to re-evaluate.

Two constraints shape it:

- **All ``nemo-agents-spec-v1`` knowledge lives here.** Agent Hardener's contract is "a directory with
  an agent in it", so it never learns the platform's config format. It also cannot: reading the spec
  means importing ``nemo-agents-plugin``, which pins six ``nvidia-nat-*`` distributions that
  agent-hardener exists to be free of.
- **No ``agent_hardener`` import.** agent-hardener runs from its own venv, driven by subprocess: its
  garak-based attacker pulls a dependency closure (``litellm``/``torch``) that conflicts with the
  platform's. We build the manifest *dict* matching agent-hardener's ``AgentSpec`` schema and let
  ``agent-hardener run`` validate it. The schema authority is ``agent_hardener/manifest.py``.

The image is not hand-rolled. ``render_fabric_dockerfile(..., sandbox_runtime="openshell")`` is the
platform's own Fabric packaging pipeline, and the ``openshell`` sandbox profile already bakes in
exactly what Agent Hardener's sandbox requires — a non-root ``sandbox`` user, ``iproute2`` and
``nftables`` — so the agent under test is packaged the same way a deployed one is.
"""

from __future__ import annotations

import copy
import logging
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

logger = logging.getLogger(__name__)

# Scaffold dir (relative to the manifest location) holding the materialized agent package.
SCAFFOLD_ROOT = ".agent-hardener-agents"
AGENT_CONFIG_FILENAME = "agent.yaml"
DOCKERFILE_FILENAME = "Dockerfile"

#: The sandbox Agent Hardener runs victims in. Selects the image profile that bakes in the non-root
#: ``sandbox`` user, ``iproute2`` and ``nftables`` the OpenShell supervisor needs.
SANDBOX_RUNTIME = "openshell"

#: Where the Fabric image puts its venv and the agent config (``container/template.py``). Agent Hardener
#: needs both literally: ``openshell sandbox exec`` does not propagate the image's ENV, so the start
#: command cannot rely on ``PATH`` or ``AGENT_CONFIG_PATH``.
IMAGE_VENV = "/workspace/.venv"
IMAGE_AGENT_CONFIG = f"/workspace/{AGENT_CONFIG_FILENAME}"

#: Processes allowed to egress. Scoped to the image's venv rather than left open, because the
#: sandbox policy uses these globs to decide which binaries may reach the network at all.
VICTIM_BINARIES = (f"{IMAGE_VENV}/bin/**",)


class AgentResolutionError(Exception):
    """Raised when an agent reference cannot be resolved into a usable manifest."""


@dataclass
class ResolvedManifest:
    """Result of resolving an agent reference into an Agent Hardener manifest."""

    manifest: dict[str, Any]
    agent_config_path: Path
    project_dir: Path
    workspace: str
    agent_name: str
    port: int
    secrets: list[str]
    #: The operator's ``--egress`` plus what the agent's own config declares. Reported rather than
    #: recomputed, so the summary cannot claim "no egress" while the manifest allow-lists a host.
    egress: list[str] = field(default_factory=list)
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
    """Bind the agent's models to the Inference Gateway, so the victim needs no raw model keys.

    Delegates to the platform's own implementation rather than keeping a copy. The copy this
    replaces existed to avoid a cross-plugin dependency, a rationale that is long obsolete —
    ``nemo-agents-plugin`` is a declared dependency — and it had started to drift: it still rewrote
    NAT ``llms`` entries, which a ``nemo-agents-spec-v1`` agent does not have.
    """
    from nemo_agents_plugin.utils import inject_fabric_gateway_url  # noqa: PLC0415

    return inject_fabric_gateway_url(config, workspace, base_url)


def strip_gateway_url(config: dict[str, Any]) -> dict[str, Any]:
    """Reverse :func:`inject_gateway_url` before writing a config back to the agent registry.

    The config Agent Hardener hardened is gateway-bound — that is how the sandboxed victim reached the
    IGW. Undoing the binding keeps the stored agent deployment-neutral, so its next deploy re-injects
    whatever gateway that environment has. Only the values we add are removed; anything the author
    set stays.
    """
    config = copy.deepcopy(config)
    models = config.get("models")
    for model_cfg in models.values() if isinstance(models, dict) else ():
        if not isinstance(model_cfg, dict):
            continue
        base_url = model_cfg.get("base_url")
        if isinstance(base_url, str) and "/apis/inference-gateway/" in base_url:
            model_cfg.pop("base_url", None)
        if model_cfg.get("api_key") == "not-used":
            model_cfg.pop("api_key", None)
    return config


#: Harnesses whose tool calls a guardrail can actually refuse. Both run Relay as a Python library in
#: the agent's own process, so the plugin can register into it.
GUARDABLE_HARNESSES = frozenset({"deepagents", "hermes"})

#: The rest run Relay as the compiled ``nemo-relay`` gateway in a separate process, which has five
#: built-in kinds and no way to load a Python one. Offering it ours is fatal rather than ignored:
#: ``plugin activation failed: ... is not registered``, and the gateway never starts.
_GATEWAY_HARNESSES = frozenset({"claude", "codex"})


def agent_harness(agent_config: dict[str, Any]) -> str | None:
    """The harness this agent runs under, from ``default_harness``."""
    harness = agent_config.get("default_harness")
    return str(harness) if harness else None


def require_guardable_harness(agent_config: dict[str, Any], ref: str) -> str | None:
    """Reject an agent Agent Hardener cannot harden, before anything is built.

    Checked here rather than in Agent Hardener because this is the last point where the harness is still
    known: ``build_manifest_dict`` emits a plain BYO victim, and by then the agent is indistinguishable
    from a hand-built image. Failing at ``init`` costs a message; failing later costs a docker build
    and a run that dies at gateway boot.
    """
    harness = agent_harness(agent_config)
    if harness in _GATEWAY_HARNESSES:
        raise AgentResolutionError(
            f"agent {ref!r} uses the {harness!r} harness, which runs NeMo Relay as a separate gateway "
            "process that cannot load Agent Hardener's guardrail plugin — a war-game against it would fail "
            f"at startup. Supported harnesses: {', '.join(sorted(GUARDABLE_HARNESSES))}."
        )
    return harness


def detect_custom_components(agent_config: dict[str, Any]) -> list[str]:
    """Local paths the agent's config references, which the packaged image must carry.

    A ``nemo-agents-spec-v1`` agent is config-only, but it may point at files beside it — skills
    directories most commonly. Those are relative to the config, so a scaffold that copies only the
    config produces an image whose agent starts and then cannot find its own skills.

    (Under NAT this looked for a dotted ``_type``, meaning "a component whose code is not in the
    config". The Fabric equivalent is not a type name but a path.)
    """
    skills = agent_config.get("skills")
    paths = skills.get("paths") if isinstance(skills, dict) else None
    return sorted({str(path) for path in paths if isinstance(path, str)}) if isinstance(paths, list) else []


def derive_secret_names(agent_config: dict[str, Any], extra: list[str] | None = None) -> list[str]:
    """Collect the env-var names the victim needs at run time.

    A ``nemo-agents-spec-v1`` agent names its credentials rather than embedding them —
    ``models.*.api_key_env``, and env passed to MCP servers — so this reads the declarations instead
    of pattern-matching values, and no secret is ever copied into the manifest.

    Falls back to ``INFERENCE_API_KEY`` when the config declares nothing; note fallback, not
    addition — a config that declares its own returns only those.
    """
    found: set[str] = set(extra or [])

    models = agent_config.get("models")
    for model_cfg in models.values() if isinstance(models, dict) else ():
        if isinstance(model_cfg, dict) and isinstance(model_cfg.get("api_key_env"), str):
            found.add(model_cfg["api_key_env"])

    servers = agent_config.get("mcp", {}).get("servers") if isinstance(agent_config.get("mcp"), dict) else None
    for server in servers.values() if isinstance(servers, dict) else ():
        env = server.get("env") if isinstance(server, dict) else None
        for value in env.values() if isinstance(env, dict) else ():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                found.add(value[2:-1])

    return sorted(found) or ["INFERENCE_API_KEY"]


def derive_agent_env(agent_config: dict[str, Any]) -> dict[str, str]:
    """The agent's declared non-secret environment, for the manifest.

    Fabric applies ``environment.env`` itself at runtime, so the victim gets these either way.
    Agent Hardener needs them for a different reason: ``agent_env`` is what egress discovery reads
    to learn which hosts the agent talks to. Without it, an agent whose env names a backend URL
    receives the variable and is then refused the connection by its own sandbox policy.

    Values that name a secret (``${VAR}``) are dropped rather than copied: they resolve to nothing
    useful here, and the manifest is written to disk and rendered in reports.
    """
    environment = agent_config.get("environment")
    declared = environment.get("env") if isinstance(environment, dict) else None
    if not isinstance(declared, dict):
        return {}
    return {
        str(key): value
        for key, value in declared.items()
        if isinstance(value, str) and not (value.startswith("${") and value.endswith("}"))
    }


#: MCP transports whose ``url`` is a network address. ``stdio`` names a local executable instead —
#: allow-listing it would put a filesystem path in a network policy.
#: Transports that reach the network, in the adapter's normalized spelling. The Fabric adapter
#: lowercases and maps ``-`` to ``_`` before validating, so both ``streamable-http`` and
#: ``streamable_http`` are legal in a config and must be recognised here.
_NETWORK_MCP_TRANSPORTS = frozenset({"sse", "streamable_http", "http", "websocket"})


def _normalized_transport(server: dict[str, Any]) -> str:
    """The server's transport, spelled the way the adapter spells it."""
    return str(server.get("transport", "stdio")).strip().lower().replace("-", "_")


def derive_egress(agent_config: dict[str, Any]) -> list[str]:
    """Hosts the agent was configured to call, for the sandbox's egress allow-list.

    Egress discovery scans the *code* an agent ships, which finds nothing for a config-only agent:
    its MCP servers and model endpoints are declarations, not calls. So an agent that names a
    network MCP server or a remote model is admitted, started, and then refused the connection by
    its own policy — a default-deny sandbox failing an agent that was configured correctly.

    ``stdio`` servers are skipped: their ``url`` is a local interpreter path, and there is nothing to
    reach over the network.
    """
    found: list[str] = []

    servers = agent_config.get("mcp", {}).get("servers") if isinstance(agent_config.get("mcp"), dict) else None
    for server in servers.values() if isinstance(servers, dict) else ():
        if not isinstance(server, dict) or _normalized_transport(server) not in _NETWORK_MCP_TRANSPORTS:
            continue
        url = server.get("url")
        if isinstance(url, str) and "://" in url:
            found.extend(_egress_host_entries(url))

    models = agent_config.get("models")
    for model_cfg in models.values() if isinstance(models, dict) else ():
        base_url = model_cfg.get("base_url") if isinstance(model_cfg, dict) else None
        if isinstance(base_url, str) and "://" in base_url:
            found.extend(_egress_host_entries(base_url))

    return sorted(dict.fromkeys(found))


def _egress_host_entries(url: str) -> list[str]:
    """A `host[:port]` egress entry for *url* — the manifest's egress contract, not a full URL."""
    parts = urlsplit(url)
    if not parts.hostname:
        return []
    port = parts.port or {"http": 80, "https": 443}.get(parts.scheme.lower())
    return [f"{parts.hostname}:{port}" if port else parts.hostname]


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
    port: int,
    secrets: list[str],
    secrets_file: str = ".env",
    egress: list[str] | None = None,
    backends: list[dict[str, Any]] | None = None,
    relay_artifacts: str | None = None,
    harness: str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the ``agent-hardener.yaml`` mapping (mirrors ``agent_hardener.manifest.build_manifest``).

    ``start_command`` spells out the interpreter and config path in full because
    ``openshell sandbox exec`` does not propagate the image's ENV: neither ``PATH`` nor
    ``AGENT_CONFIG_PATH`` is visible to it, so relying on either would start nothing.
    """
    agent: dict[str, Any] = {
        "name": agent_name,
        "project_dir": project_dir,
        # Carried so Agent Hardener can stage Hermes' extra wiring and name what it could not enforce.
        # Not a victim *kind* — the launch shape is identical for every harness.
        "harness": harness,
        "dockerfile": DOCKERFILE_FILENAME,
        "start_command": (
            f"{IMAGE_VENV}/bin/python -m nemo_agents_plugin.fabric.server "
            f"--agent-config {IMAGE_AGENT_CONFIG} --host 0.0.0.0 --port {port}"
        ),
        "binaries": list(VICTIM_BINARIES),
        "port": port,
        "secrets": secrets,
        "secrets_file": secrets_file,
    }
    if env:
        agent["env"] = env
    if relay_artifacts:
        agent["relay_artifacts"] = relay_artifacts
    if egress:
        agent["egress"] = egress
    return {"agent": agent, "backends": backends or []}


# --------------------------------------------------------------------------- #
# Filesystem materialization
# --------------------------------------------------------------------------- #
def download_agent_bundle(sdk: Any, agent_name: str, workspace: str, destination: Path) -> bool:
    """Copy the whole directory the author registered into *destination*.

    Registration uploads everything beside ``agent.yaml`` — an MCP server, skills, whatever the image
    COPYs — so taking only the config and the Dockerfile produces a build context that is missing the
    files the Dockerfile references. That surfaces as a build failure late, from Docker, naming a file
    the author did register:

        COPY failed: file not found in build context: stat ledger_mcp.py: file does not exist

    ``agent.yaml`` is deliberately not copied: the caller writes the gateway-injected config over it.

    Returns whether a bundle was found; ``False`` is the ordinary case for a config-only agent.
    """
    from nemo_agent_hardener_plugin.filesets import download_fileset  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="agent-hardener-ethos-") as directory:
        source = Path(directory)
        try:
            download_fileset(sdk, f"{workspace}/{agent_name}-ethos", source)
        except Exception:
            logger.info("agent %s/%s has no readable ethos fileset", workspace, agent_name)
            return False
        destination.mkdir(parents=True, exist_ok=True)
        copied = False
        for item in source.iterdir():
            if item.name == AGENT_CONFIG_FILENAME:
                continue
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copyfile(item, target)
            copied = True
        return copied


def shipped_dockerfile(sdk: Any, agent_name: str, workspace: str) -> str | None:
    """The Dockerfile the author registered beside ``agent.yaml``, if there is one.

    Registration uploads the whole directory holding ``agent.yaml`` into ``{agent}-ethos``
    (``nemo_agents_plugin.cli._upload_ethos_fileset``), so an author who ships a Dockerfile has
    already put it on the platform — it is simply never read back.

    Preferring it matters beyond convenience. A rendered Dockerfile pins the packaging machine's own
    ``nemo-platform`` version and a fixed ``nemo-relay``, so an agent needing a different Relay has
    no way to ask for one, and a platform installed from a git checkout pins a version no index
    serves. The author's own file has neither problem, and it is the image they actually ship.

    Returns ``None`` when the agent shipped none, which is the common case; the caller renders then.
    A failure to *read* an existing fileset is logged as a warning rather than swallowed: silently
    rendering would quietly ignore the author's file and reintroduce both pins.
    """
    from nemo_agent_hardener_plugin.filesets import download_fileset  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="agent-hardener-ethos-") as directory:
        dest = Path(directory)
        try:
            download_fileset(sdk, f"{workspace}/{agent_name}-ethos", dest)
        except Exception:
            logger.info("agent %s/%s has no readable ethos fileset; rendering a Dockerfile", workspace, agent_name)
            return None
        candidate = dest / DOCKERFILE_FILENAME
        if not candidate.is_file():
            return None
        logger.info("using the Dockerfile shipped with agent %s/%s", workspace, agent_name)
        return candidate.read_text(encoding="utf-8") or None


def materialize_agent_package(
    agent_config: dict[str, Any],
    project_path: Path,
    *,
    dockerfile_override: str | None = None,
    bundle: Callable[[Path], bool] | None = None,
) -> Path:
    """Write the agent package Agent Hardener runs: the registered bundle, plus the resolved config.

    ``bundle`` stages the author's other files (an MCP server, skills) into the build context; a
    Dockerfile that COPYs them fails without it. ``dockerfile_override`` is the author's own file when
    they registered one. Otherwise the Dockerfile is rendered by the platform's own Fabric packaging
    pipeline with the ``openshell`` sandbox profile applied, so the agent under test is built the same
    way a deployed one is — rather than by a second, Agent-Hardener-specific recipe that could drift.
    """
    from nemo_agents_plugin.container.template import render_fabric_dockerfile  # noqa: PLC0415

    project_path.mkdir(parents=True, exist_ok=True)
    if bundle is not None:
        bundle(project_path)
    config_path = project_path / AGENT_CONFIG_FILENAME
    config_path.write_text(yaml.safe_dump(agent_config, sort_keys=False), encoding="utf-8")

    dockerfile = dockerfile_override or render_fabric_dockerfile(config_path, sandbox_runtime=SANDBOX_RUNTIME)
    (project_path / DOCKERFILE_FILENAME).write_text(dockerfile, encoding="utf-8")
    return config_path


#: Where a sandboxed victim must write its Relay telemetry. Agent Hardener reads this exact path out of
#: the sandbox (``agent_hardener.relay_plugin.victim.SANDBOX_ARTIFACTS_DIR``); an agent that points
#: ``telemetry.output_dir`` anywhere else produces a stream nothing collects.
SANDBOX_TELEMETRY_DIR = "/home/sandbox/.agent-hardener/relay"


def check_relay_artifacts_dir(agent_config: dict[str, Any]) -> str | None:
    """Return a complaint if the agent writes Relay telemetry where Agent Hardener will not look.

    Deliberately *not* forwarded to the manifest as ``relay_artifacts``: that field names a
    **host** directory for the fetched copy, while ``telemetry.output_dir`` names a path inside the
    container. Passing the container path across made the host try to write ``/home/sandbox``, which
    on macOS is an autofs mount and fails with ``[Errno 45] Operation not supported`` — and would
    silently create a stray real directory on Linux.
    """
    telemetry = agent_config.get("telemetry")
    output_dir = telemetry.get("output_dir") if isinstance(telemetry, dict) else None
    if output_dir and str(output_dir) != SANDBOX_TELEMETRY_DIR:
        return (
            f"telemetry.output_dir is {output_dir!r}, but Agent Hardener reads the victim's ATOF stream "
            f"from {SANDBOX_TELEMETRY_DIR!r}. Point it there or the run will report an "
            "uninstrumented victim."
        )
    return None


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


def inspect_agent(ref: str, *, sdk: Any, default_workspace: str) -> tuple[str, int, list[str], list[str], list[str]]:
    """Derive the create-form defaults for a deployed agent without materializing anything.

    Returns ``(qualified_ref, port, secrets, egress, warnings)``: the victim port from the running
    deployment (else agent-hardener's default), the secret names scanned from the stored config, and the
    hosts that config names. Cheap read-only counterpart to :func:`resolve_agent_to_manifest`, used
    to pre-fill (and let the operator override) those fields before creating the manifest.

    Egress is included even though ``resolve_agent_to_manifest`` derives it again on submit, because
    a blank box does not read as "derived for you" — it reads as "no egress", which invites an
    operator to type the hosts in by hand. That is worse than leaving it empty: an explicit value
    wins over the derived one, so the manifest stops tracking the agent's own config from then on.
    """
    workspace, name = parse_agent_ref(ref, default_workspace)
    agent_config = _fetch_agent_config(sdk, workspace, name)
    # Reject here too, not only in resolve_agent_to_manifest: this is what the Studio create form
    # calls, so an unguardable agent is refused before an operator fills anything in.
    require_guardable_harness(agent_config, f"{workspace}/{name}")
    port, warnings = _resolve_victim_port(sdk, workspace, name)
    secrets = derive_secret_names(agent_config)
    return f"{workspace}/{name}", port, secrets, derive_egress(agent_config), warnings


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
    harness = require_guardable_harness(agent_config, f"{workspace}/{name}")
    resolved_port, warnings = _resolve_victim_port(sdk, workspace, name)
    port = port or resolved_port

    telemetry_complaint = check_relay_artifacts_dir(agent_config)
    if telemetry_complaint:
        warnings.append(telemetry_complaint)

    # Skills and other local artifacts live beside the config, so a scaffold that copies only the
    # config yields an agent that starts and then cannot find them.
    referenced = detect_custom_components(agent_config)
    if referenced and project_dir is None:
        raise AgentResolutionError(
            f"agent {workspace}/{name!r} references local paths {referenced} that are not in the "
            "stored config. Re-run with --project-dir pointing at the directory holding them."
        )

    injected = inject_gateway_url(agent_config, workspace, base_url)

    if project_dir is not None:
        project_path = Path(project_dir)
        rel_project = project_dir
    else:
        rel_project = str(Path(SCAFFOLD_ROOT) / name)
        project_path = manifest_dir / rel_project

    # The author's own Dockerfile wins when they registered one: it is the image they ship, and it
    # carries neither the rendered file's `nemo-platform==<packaging machine's version>` pin nor its
    # fixed nemo-relay, so it can be built from a source checkout and can choose its own Relay.
    config_path = materialize_agent_package(
        injected,
        project_path,
        dockerfile_override=shipped_dockerfile(sdk, name, workspace),
        bundle=lambda destination: download_agent_bundle(sdk, name, workspace, destination),
    )
    secrets = secrets or derive_secret_names(agent_config)

    # The operator's --egress first: an explicit answer should win over a derived one.
    egress = [*(egress or []), *derive_egress(agent_config)]
    gw_backend = gateway_backend(base_url)
    manifest = build_manifest_dict(
        agent_name=name,
        project_dir=rel_project,
        port=port,
        secrets=secrets,
        egress=egress,
        backends=[gw_backend] if gw_backend else [],
        harness=harness,
        env=derive_agent_env(agent_config),
    )

    return ResolvedManifest(
        manifest=manifest,
        agent_config_path=config_path,
        project_dir=project_path,
        workspace=workspace,
        agent_name=name,
        port=port,
        secrets=secrets,
        egress=egress,
        warnings=warnings,
    )
