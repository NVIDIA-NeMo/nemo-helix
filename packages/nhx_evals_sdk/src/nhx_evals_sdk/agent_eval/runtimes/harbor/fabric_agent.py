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

import json
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nemo_fabric import (
    DiscoveryConfig,
    EnvironmentConfig,
    FabricConfig,
    FabricConfigError,
    RelayAtofFileSinkConfig,
    inspect_adapter,
)
from nemo_fabric.integrations.harbor.fabric_agent import HARBOR_ARTIFACT_ROOT, FabricAgent
from nhx_evals_sdk.agent_eval.runtimes.fabric.image import fabric_version
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

#: An exact ``nemo-fabric`` pin inside ``fabric_package``, with or without extras.
_FABRIC_DISTRIBUTION_PIN = re.compile(r"nemo[-_]fabric(?:\[[^\]]*\])?\s*==\s*(?P<version>[^\s,;]+)\Z", re.IGNORECASE)


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
        for model in config.models.values():
            if model.base_url and PLATFORM_GATEWAY_PATH_MARKER in model.base_url:
                model.api_key_env = model.api_key_env or PLATFORM_GATEWAY_API_KEY_ENV
                environment.env.setdefault(model.api_key_env, PLATFORM_GATEWAY_API_KEY_PLACEHOLDER)

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
            observability = config.relay.observability
            if observability is not None:
                if observability.atif is not None:
                    observability.atif.output_directory = relay_output
                if observability.atof is not None:
                    for sink in observability.atof.sinks:
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
