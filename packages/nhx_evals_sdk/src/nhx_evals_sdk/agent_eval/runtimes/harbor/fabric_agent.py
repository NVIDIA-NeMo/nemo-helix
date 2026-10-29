# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Harbor agent that runs one complete NeMo Fabric agent config inside the task container.

Point :class:`HarborRuntimeConfig.agent_import_path` (or the platform's ``HarborRunnerTarget``) at
``nhx_evals_sdk.agent_eval.runtimes.harbor.fabric_agent:NemoFabricAgent`` and hand it the agent
through ``agent_kwargs["fabric_config"]`` -- a Fabric ``agent.yaml`` as a JSON-shaped mapping, the same
document :class:`~nhx_evals_sdk.agent_eval.runtimes.fabric.runtime.FabricAgentRuntime` runs on the
host. The config is used **verbatim**: harness, models, instructions, skills, MCP servers, tools, and
telemetry all come from it, and so does the agent's identity (``metadata.name``, the Relay
``agent_name``). Harbor decides only where a trial lives -- the workspace and artifact paths inside
the container -- the request/response schemas its runner protocol needs, the settings a harness needs
to run unattended in a task container, and the placeholder key a gateway-routed model's client insists
on. A config handed to Harbor must therefore hold no credential-named value at all, placeholder
included; Harbor persists ``agent_kwargs`` unredacted, and :class:`HarborRuntimeConfig` refuses one.

The flat ``fabric_*`` keywords the upstream ``nemo_fabric.integrations.harbor:FabricAgent`` builds a
config from (``fabric_adapter_id``, ``fabric_system_instruction``, ``fabric_max_turns``, ...) are
refused rather than merged; the install/run keywords (``fabric_package``, ``fabric_config_bundle``,
``fabric_workspace``, ``fabric_python``, ...) are unchanged.

Files the config refers to by relative path (``skills.paths``) resolve against ``fabric_config_target``
when a ``fabric_config_bundle`` is uploaded, exactly as upstream: stage the agent's files into a
directory, pass it as the bundle, and keep the paths relative. Custom adapters shipped in the bundle
are found only if the config lists their directory under ``discovery.local_paths``, and the host needs
their descriptor too (``fabric_adapter_descriptor``) to confirm the skills, MCP servers and telemetry
the config asks of them before a trial starts.
"""

from __future__ import annotations

import ipaddress
import json
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from nemo_fabric import (
    DiscoveryConfig,
    EnvironmentConfig,
    FabricConfig,
    FabricConfigError,
    RelayAtifConfig,
    RelayAtofConfig,
    RelayAtofFileSinkConfig,
    RelayObservabilityConfig,
    RelayOpenTelemetryConfig,
    RelayOpenTelemetryEndpointConfig,
    inspect_adapter,
)
from nemo_fabric.integrations.harbor.fabric_agent import HARBOR_ARTIFACT_ROOT, FabricAgent, ensure_success
from nhx_evals_sdk.agent_eval.runtimes.fabric import otlp_receiver, tcp_forwarder
from nhx_evals_sdk.agent_eval.runtimes.fabric.image import fabric_version
from nhx_evals_sdk.agent_eval.runtimes.fabric.otlp_writer import TRACES_SUBDIR, fold_exports, otlp_endpoint_fields
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from nemo_fabric.integrations.harbor.fabric_agent import FabricAgentOptions
except ImportError as error:
    # Upstream defines the options only beside its real FabricAgent; without the Harbor extra it
    # exports a placeholder agent and nothing else. Keep this module importable (the examples import
    # NVIDIA_MODEL_BASE_URL from it) and surface the missing extra where the agent is first used.
    _HARBOR_IMPORT_ERROR: ImportError | None = error
    FabricAgentOptions = BaseModel
else:
    _HARBOR_IMPORT_ERROR = None

_HARBOR_EXTRA_HINT = (
    "NemoFabricAgent requires the Harbor optional dependency: install nemo-fabric[harbor] "
    "(the SDK's `harbor` extra) on Python 3.12 or later"
)


@dataclass(frozen=True)
class DetachedHelper:
    name: str
    command: str
    ready: str
    """A file the helper writes once it is serving."""


def detached_start_command(helpers: list[DetachedHelper], log_dir: str, pid_dir: str = "/tmp") -> str:
    """A shell command that starts each helper detached and waits until every one is serving.

    Each ``nohup … &`` is a statement of its own, so ``$!`` is the helper's own process, which the pid file then
    names, and the shell keeps none of the helper's output open.
    """
    statements = []
    for helper in helpers:
        log = shlex.quote(f"{log_dir}/{helper.name}.log")
        statements += [
            f"rm -f {shlex.quote(helper.ready)}",
            f"nohup {helper.command} > {log} 2>&1 < /dev/null &",
            f"echo $! > {shlex.quote(f'{pid_dir}/nemo-fabric-relay-{helper.name}.pid')}",
        ]
    waits = " && ".join(f"[ -f {shlex.quote(helper.ready)} ]" for helper in helpers)
    statements.append(f"for _ in $(seq 1 50); do {waits} && exit 0; sleep 0.1; done; exit 1")
    return "\n".join(statements)


def _plain_http_upstream(base_url: str) -> tuple[str, int] | None:
    """The host and port of a plain-HTTP, non-loopback URL; ``None`` for any other."""
    url = urlsplit(base_url)
    if url.scheme != "http" or url.hostname is None or _is_loopback(url.hostname):
        return None
    return url.hostname, url.port or 80


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


def _require_harbor() -> None:
    if _HARBOR_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(_HARBOR_EXTRA_HINT) from _HARBOR_IMPORT_ERROR


#: build.nvidia.com's OpenAI-compatible endpoint, for configs that name an ``nvidia`` model directly.
NVIDIA_MODEL_BASE_URL = "https://integrate.api.nvidia.com/v1"

#: Upstream ``FabricAgent`` keywords that say how to install and run the agent; everything else it
#: accepts describes the agent and is refused here in favour of ``fabric_config``.
INSTALL_RUN_KWARGS: frozenset[str] = frozenset(
    {
        "fabric_config_base_dir",
        "fabric_config_bundle",
        "fabric_config_target",
        "fabric_discovery_paths",
        "fabric_adapter_descriptor",
        "fabric_workspace",
        "fabric_python",
        "fabric_package",
        "fabric_venv_path",
        "fabric_install_command",
        "fabric_cwd",
        "fabric_timeout_sec",
    }
)
#: Upstream ``FabricAgent`` keywords that describe the agent rather than how to install or run it.
FLAT_CONFIG_KWARGS: frozenset[str] = frozenset(FabricAgentOptions.model_fields) - INSTALL_RUN_KWARGS

HARBOR_INPUT_SCHEMA = "text"
HARBOR_OUTPUT_SCHEMA = "message"

#: Claude Code refuses ``bypassPermissions`` as root unless it is told it is sandboxed.
_CLAUDE_ADAPTER_ID = "nvidia.fabric.claude"
_CLAUDE_UNATTENDED_ENV = {"IS_SANDBOX": "1"}
_CLAUDE_UNATTENDED_SETTINGS = {"permission_mode": "bypassPermissions"}

#: The platform's Inference Gateway authenticates the job, not the request, but a harness's client
#: still refuses an empty key, so a gateway-routed model is handed a placeholder under its
#: ``api_key_env``. Must agree with ``nemo_agents_plugin.fabric.gateway_credentials``.
PLATFORM_GATEWAY_PATH_MARKER = "/apis/inference-gateway/"
PLATFORM_GATEWAY_API_KEY_ENV = "NEMO_AGENTS_IGW_API_KEY"
PLATFORM_GATEWAY_API_KEY_PLACEHOLDER = "not-used"

#: Where Relay's OTel export goes: a receiver beside the harness writes each export under the agent's
#: logs, which Harbor brings back with the trial, and they are folded into ``traces/relay.jsonl`` there.
CONTAINER_LOGS_DIR = "/logs/agent"
RELAY_TRACES_DIR = f"{CONTAINER_LOGS_DIR}/{TRACES_SUBDIR}"
RELAY_OTLP_PORT = 14318
_RECEIVER_PATH = "/tmp/nemo-fabric-otlp-receiver.py"
_RECEIVER_PID_PATH = "/tmp/nemo-fabric-otlp-receiver.pid"
#: Relay sends model credentials only over HTTPS or to loopback, so a plain-HTTP gateway is reached through a
#: forwarder on loopback, one port per gateway host from this one up.
RELAY_GATEWAY_FORWARD_PORT = 14320
_FORWARDER_PATH = "/tmp/nemo-fabric-tcp-forwarder.py"
_RELAY_PID_GLOB = "/tmp/nemo-fabric-relay-*.pid"

#: Relay's Codex and Claude adapters launch the ``nemo-relay`` CLI, which the ``nemo-relay`` wheel
#: doesn't ship; setup installs the release matching the installed bindings unless one is on PATH.
#: Downloading it needs ``github.com`` and ``release-assets.githubusercontent.com`` during setup.
RELAY_CLI_RELEASES_URL = "https://github.com/NVIDIA/NeMo-Relay/releases/download"
RELAY_CLI_DIR = "/tmp/nemo-relay/bin"
#: Waits between attempts to download the CLI or its checksum; a setup sandbox's egress can fail transiently.
RELAY_CLI_DOWNLOAD_RETRY_DELAYS_SEC = (2.0, 5.0, 10.0)
_RELAY_CLI_INSTALL = """
import glob, hashlib, importlib.metadata, os, platform, shutil, subprocess, sys, time, urllib.error, urllib.request
if shutil.which("nemo-relay"):
    sys.exit(0)
delays = [float(delay) for delay in sys.argv[3].split(",") if delay]
def fetch(url, timeout):
    for attempt, delay in enumerate([*delays, None], start=1):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read()
        except urllib.error.HTTPError as error:
            if error.code < 500 or delay is None:
                sys.exit(f"could not download {url}: {error} (attempts: {attempt})")
        except OSError as error:
            if delay is None:
                sys.exit(f"could not download {url}: {error} (attempts: {attempt})")
        time.sleep(delay)
try:
    version = importlib.metadata.version("nemo-relay")
except importlib.metadata.PackageNotFoundError:
    sys.exit("Relay telemetry needs the nemo-relay Python package: install nemo-fabric[relay]")
arch = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}.get(platform.machine().lower())
if arch is None:
    sys.exit(f"no nemo-relay CLI release for {platform.machine()}")
libc = "musl" if glob.glob("/lib/ld-musl-*") else "gnu"
asset = f"nemo-relay-cli-{arch}-unknown-linux-{libc}-{version}"
url = f"{sys.argv[1]}/{version}/{asset}"
binary = fetch(url, 120)
expected = fetch(url + ".sha256", 60).decode().split()[0]
if hashlib.sha256(binary).hexdigest() != expected:
    sys.exit(f"{asset} does not match its published sha256")
os.makedirs(sys.argv[2], exist_ok=True)
target = os.path.join(sys.argv[2], "nemo-relay")
with open(target, "wb") as handle:
    handle.write(binary)
os.chmod(target, 0o755)
check = subprocess.run([target, "--version"], capture_output=True, text=True)
if check.returncode != 0:
    os.remove(target)
    sys.exit(f"{asset} does not run in this image: {(check.stderr or check.stdout).strip()[:500]}")
"""

#: An exact ``nemo-fabric`` pin inside ``fabric_package``, with or without extras.
_FABRIC_DISTRIBUTION_PIN = re.compile(
    r"nemo[-_]fabric(?:\[[^\]]*\])?\s*==\s*(?P<version>[^\s,;]+)(?:\s*[,;].*)?\Z", re.IGNORECASE | re.DOTALL
)


def _parse_fabric_config(fabric_config: Mapping[str, Any]) -> FabricConfig:
    config = FabricConfig.from_mapping(fabric_config)
    if config.harness is None or not config.harness.adapter_id.strip():
        raise ValueError("fabric_config must select a harness: set `harness.adapter_id`")
    return config


def _host_adapter_descriptor(adapter_id: str, descriptor_path: Path | None) -> Mapping[str, Any] | None:
    """The adapter descriptor the host can read, as upstream ``FabricAgent`` resolves it at construction."""
    if descriptor_path is not None:
        return json.loads(descriptor_path.read_text(encoding="utf-8"))
    try:
        from nemo_fabric_adapter_catalog import get_adapter_descriptor
    except ModuleNotFoundError as error:
        if error.name != "nemo_fabric_adapter_catalog":
            raise
        return None
    try:
        return get_adapter_descriptor(adapter_id)
    except KeyError:
        return None


def _require_adapter_inspectable(config: FabricConfig, descriptor_path: Path | None) -> None:
    """Run the host-side inspection upstream runs in ``__init__``, so it fails preflight rather than a trial.

    Fabric refuses to run a config that asks an adapter for skills, MCP servers or telemetry unless the
    host can read that adapter's descriptor. Harbor does not catch construction errors per trial, so
    without this the whole job aborts with Fabric's wording and no pointer at what is missing.
    """
    assert config.harness is not None  # checked by _parse_fabric_config
    try:
        inspect_adapter(config, _host_adapter_descriptor(config.harness.adapter_id, descriptor_path))
    except FabricConfigError as error:
        raise ValueError(
            f"fabric_config: {error}. A catalog adapter needs the `nemo-fabric-adapter-catalog` package on the "
            "host (installed by `nemo-fabric[harbor]`); an adapter shipped in the bundle needs its descriptor "
            "passed as `fabric_adapter_descriptor`."
        ) from error


def _require_fabric_package_matches_host(fabric_package: str | None) -> None:
    """Refuse a ``fabric_package`` that pins ``nemo-fabric`` to a release other than the host's.

    Fabric's runner protocol is versioned by payload shape: the task-side runner rejects a payload the
    host serialises with fields it does not know. An exact pin that disagrees with the host is caught
    here; a range is left to pip.
    """
    if not fabric_package:
        return
    for requirement in shlex.split(fabric_package):
        pin = _FABRIC_DISTRIBUTION_PIN.fullmatch(requirement)
        if pin is None:
            continue
        host = fabric_version()
        if pin.group("version") != host:
            raise ValueError(
                f"fabric_package pins nemo-fabric to {pin.group('version')} but the host runs {host}: the "
                "task-side Fabric runner must match the host's release, so pin the same version"
            )


def _flat_config_kwargs_error(rejected: list[str]) -> ValueError:
    return ValueError(
        f"{', '.join(rejected)}: NemoFabricAgent runs the `fabric_config` it is given; describe the "
        "agent there instead of through FabricAgent's flat keywords"
    )


def _refuse_flat_config_kwargs(kwargs: Mapping[str, Any]) -> None:
    rejected = sorted(FLAT_CONFIG_KWARGS & kwargs.keys())
    if rejected:
        raise _flat_config_kwargs_error(rejected)


def _refuse_non_default_flat_config_kwargs(kwargs: Mapping[str, Any]) -> None:
    """The keys-present check is too strict for ``parse_options``: upstream's constructor forwards every
    flat keyword at its default, so only a value that differs from the default is a description."""
    rejected = sorted(
        name
        for name in FLAT_CONFIG_KWARGS & kwargs.keys()
        if name != "fabric_adapter_id" and kwargs[name] != FabricAgentOptions.model_fields[name].default
    )
    if rejected:
        raise _flat_config_kwargs_error(rejected)


class NemoFabricAgentOptions(FabricAgentOptions):  # ty: ignore[unsupported-base]
    """Harbor-facing options: a complete ``fabric_config`` stands in for the flat agent keywords."""

    fabric_config: dict[str, Any] = Field(description="Fabric agent config to run verbatim.")
    fabric_default_max_turns: int | None = Field(
        default=None, gt=0, description="Maximum agent turns when fabric_config sets no runtime.max_turns."
    )
    fabric_adapter_id: str = Field(default="", description="Derived from fabric_config.harness.adapter_id.")

    @field_validator("fabric_config")
    @classmethod
    def validate_fabric_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        _parse_fabric_config(value)
        return value

    @model_validator(mode="after")
    def derive_adapter_id(self) -> NemoFabricAgentOptions:
        harness = _parse_fabric_config(self.fabric_config).harness
        assert harness is not None  # checked by validate_fabric_config
        if "fabric_adapter_id" in self.model_fields_set and self.fabric_adapter_id != harness.adapter_id:
            raise _flat_config_kwargs_error(["fabric_adapter_id"])
        self.fabric_adapter_id = harness.adapter_id
        return self


class NemoFabricAgent(FabricAgent):  # ty: ignore[unsupported-base]
    """``FabricAgent`` that runs a complete ``fabric_config`` instead of building one from keywords.

    ``fabric_default_max_turns`` fills ``runtime.max_turns`` only when the config leaves it unset; an
    explicit ``null`` keeps Fabric's unbounded harness.
    """

    options_model = NemoFabricAgentOptions
    options: NemoFabricAgentOptions

    def __init__(
        self,
        logs_dir: Any,
        *args: Any,
        fabric_config: Mapping[str, Any],
        fabric_default_max_turns: int | None = None,
        **kwargs: Any,
    ) -> None:
        _require_harbor()
        _refuse_flat_config_kwargs(kwargs)
        config = _parse_fabric_config(fabric_config)
        assert config.harness is not None  # checked by _parse_fabric_config
        self.fabric_config = config
        self.fabric_default_max_turns = fabric_default_max_turns
        super().__init__(
            logs_dir,
            *args,
            fabric_adapter_id=config.harness.adapter_id,
            fabric_config=dict(fabric_config),
            fabric_default_max_turns=fabric_default_max_turns,
            **kwargs,
        )

    @classmethod
    def parse_options(cls, kwargs: dict[str, Any] | None = None, env: Mapping[str, str] | None = None) -> Any:
        """Refuse the flat agent keywords here too, so ``harbor config validate`` agrees with preflight.

        ``fabric_adapter_id`` is exempt because the constructor hands upstream the one derived from the
        config; the options model refuses any other value.
        """
        _require_harbor()
        _refuse_non_default_flat_config_kwargs(kwargs or {})
        return super().parse_options(kwargs, env)

    @classmethod
    def options_schema(cls) -> dict[str, Any]:
        """The schema Harbor shows users, without the keywords this agent refuses."""
        _require_harbor()
        schema = super().options_schema()
        schema["properties"] = {k: v for k, v in schema.get("properties", {}).items() if k not in FLAT_CONFIG_KWARGS}
        if "required" in schema:
            schema["required"] = [name for name in schema["required"] if name not in FLAT_CONFIG_KWARGS]
        return schema

    @classmethod
    def preflight(cls, kwargs: dict[str, Any] | None = None, env: Mapping[str, str] | None = None) -> None:
        options = cls.parse_options(kwargs, env)
        assert isinstance(options, NemoFabricAgentOptions)  # options_model
        _require_fabric_package_matches_host(options.fabric_package)
        _require_adapter_inspectable(_parse_fabric_config(options.fabric_config), options.fabric_adapter_descriptor)

    @staticmethod
    def name() -> str:
        return "nemo-fabric"

    async def setup(self, environment: Any) -> None:
        await super().setup(environment)
        await self.prepare_relay(environment)

    async def prepare_relay(self, environment: Any) -> None:
        """Install what capturing Relay's traces needs in the container; nothing when the config has no Relay."""
        if self.fabric_config.relay is None:
            return
        for module, target in ((otlp_receiver, _RECEIVER_PATH), (tcp_forwarder, _FORWARDER_PATH)):
            script = Path(self.logs_dir) / Path(target).name
            script.write_text(Path(module.__file__).read_text(encoding="utf-8"), encoding="utf-8")
            await environment.upload_file(script, target)
        result = await environment.exec(
            f"{shlex.quote(self._runner_python)} -c {shlex.quote(_RELAY_CLI_INSTALL)} "
            f"{shlex.quote(RELAY_CLI_RELEASES_URL)} {shlex.quote(RELAY_CLI_DIR)} "
            f"{','.join(str(delay) for delay in RELAY_CLI_DOWNLOAD_RETRY_DELAYS_SEC)}",
            env=self._install_env,
            timeout_sec=self.fabric_timeout_sec,
        )
        ensure_success("Installing the nemo-relay CLI failed", result)

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        if self.fabric_config.relay is None:
            await super().run(instruction, environment, context)
            return
        python = shlex.quote(self._runner_python)
        helpers = [
            DetachedHelper(
                name="otlp-receiver",
                command=f"{python} {shlex.quote(_RECEIVER_PATH)} {shlex.quote(RELAY_TRACES_DIR)} {RELAY_OTLP_PORT}",
                ready=f"{RELAY_TRACES_DIR}/{otlp_receiver.READY_FILENAME}",
            )
        ]
        for (host, port), local_port in self._gateway_forwards().items():
            ready = f"/tmp/nemo-fabric-forwarder-{local_port}.ready"
            helpers.append(
                DetachedHelper(
                    name=f"forwarder-{local_port}",
                    command=(
                        f"{python} {shlex.quote(_FORWARDER_PATH)} {local_port} {shlex.quote(host)} {port} "
                        f"{shlex.quote(ready)}"
                    ),
                    ready=ready,
                )
            )
        result = await environment.exec(
            f"mkdir -p {shlex.quote(RELAY_TRACES_DIR)} || exit 1\n"
            + detached_start_command(helpers, CONTAINER_LOGS_DIR),
            timeout_sec=30,
        )
        ensure_success("The helpers Relay needs beside the harness did not start", result)
        try:
            await super().run(instruction, environment, context)
        finally:
            await environment.exec(
                f"for pid in {_RELAY_PID_GLOB}; do kill $(cat $pid) 2>/dev/null; rm -f $pid; done; true", timeout_sec=30
            )

    def populate_context_post_run(self, context: Any) -> None:
        super().populate_context_post_run(context)
        if self.fabric_config.relay is not None:
            fold_exports(Path(self.logs_dir) / TRACES_SUBDIR)

    def _gateway_forwards(self) -> dict[tuple[str, int], int]:
        """The loopback port forwarding to each plain-HTTP gateway host a Relay-run model calls."""
        if self.fabric_config.relay is None:
            return {}
        upstreams = set()
        for model in self.fabric_config.models.values():
            if not model.base_url or PLATFORM_GATEWAY_PATH_MARKER not in model.base_url:
                continue
            if (upstream := _plain_http_upstream(model.base_url)) is not None:
                upstreams.add(upstream)
        return {upstream: RELAY_GATEWAY_FORWARD_PORT + index for index, upstream in enumerate(sorted(upstreams))}

    @property
    def _runner_path_prefix(self) -> str | None:
        prefix = super()._runner_path_prefix
        if self.fabric_config.relay is None:
            return prefix
        return RELAY_CLI_DIR if prefix is None else f"{RELAY_CLI_DIR}:{prefix}"

    def _build_config(self) -> FabricConfig:
        """The supplied config, with only the per-trial mechanics Harbor owns written over it."""
        config = self.fabric_config.model_copy(deep=True)
        artifact_root = f"{HARBOR_ARTIFACT_ROOT}/{config.metadata.name}"

        runtime = config.runtime
        runtime.input_schema = runtime.input_schema or HARBOR_INPUT_SCHEMA
        runtime.output_schema = runtime.output_schema or HARBOR_OUTPUT_SCHEMA
        runtime.artifacts = artifact_root
        if "max_turns" not in runtime.model_fields_set and self.fabric_default_max_turns is not None:
            runtime.max_turns = self.fabric_default_max_turns

        if self.fabric_discovery_paths:
            discovery = config.discovery or DiscoveryConfig()
            discovery.local_paths = list(dict.fromkeys([*discovery.local_paths, *self.fabric_discovery_paths]))
            config.discovery = discovery

        # Harbor's task container is Fabric's local provider, whatever the config says.
        environment = config.environment or EnvironmentConfig(provider="local")
        environment.provider = "local"
        environment.workspace = self.fabric_workspace
        environment.artifacts = artifact_root
        config.environment = environment

        assert config.harness is not None  # checked at construction
        if config.harness.adapter_id == _CLAUDE_ADAPTER_ID:
            for key, value in _CLAUDE_UNATTENDED_SETTINGS.items():
                config.harness.settings.setdefault(key, value)
            for key, value in _CLAUDE_UNATTENDED_ENV.items():
                environment.env.setdefault(key, value)
        forwards = self._gateway_forwards()
        for model in config.models.values():
            if model.base_url and PLATFORM_GATEWAY_PATH_MARKER in model.base_url:
                model.api_key_env = model.api_key_env or PLATFORM_GATEWAY_API_KEY_ENV
                # A key Harbor hands the agent (a trial's own gateway key) must not be shadowed by the placeholder.
                if model.api_key_env not in self._extra_env:
                    environment.env.setdefault(model.api_key_env, PLATFORM_GATEWAY_API_KEY_PLACEHOLDER)
                upstream = _plain_http_upstream(model.base_url)
                if upstream is not None and (local_port := forwards.get(upstream)) is not None:
                    model.base_url = urlunsplit(urlsplit(model.base_url)._replace(netloc=f"127.0.0.1:{local_port}"))

        if self.model_name:
            default = config.models.get("default")
            if default is None:
                raise ValueError(
                    "agent_model_name was given but fabric_config declares no `models.default` to apply it to"
                )
            default.model = self.model_name

        # Relay output must sit under the artifact root so Harbor collects the trajectory with the trial.
        if config.relay is not None:
            relay_output = f"{artifact_root}/relay"
            config.relay.output_dir = relay_output
            observability = config.relay.observability or RelayObservabilityConfig()
            config.relay.observability = observability
            receiver = RelayOpenTelemetryEndpointConfig(
                **otlp_endpoint_fields(
                    endpoint=f"http://127.0.0.1:{RELAY_OTLP_PORT}{otlp_receiver.TRACES_PATH}",
                    service_name=config.metadata.name,
                )
            )
            existing = observability.opentelemetry
            observability.opentelemetry = RelayOpenTelemetryConfig(
                enabled=True,
                endpoints=[*(existing.endpoints if existing is not None and existing.enabled else []), receiver],
            )
            if isinstance(observability.atif, RelayAtifConfig):
                observability.atif.output_directory = relay_output
            if isinstance(observability.atof, RelayAtofConfig):
                for sink in observability.atof.sinks or ():
                    if isinstance(sink, RelayAtofFileSinkConfig):
                        sink.output_directory = relay_output

        for server in self.mcp_servers:
            if server.transport == "stdio":
                config.add_mcp_server(
                    server.name,
                    transport="stdio",
                    url=str(server.command),
                    args=list(server.args),
                    exposure="harness_native",
                )
            else:
                config.add_mcp_server(
                    server.name,
                    transport=server.transport,
                    url=str(server.url),
                    exposure="harness_native",
                )
        return config
