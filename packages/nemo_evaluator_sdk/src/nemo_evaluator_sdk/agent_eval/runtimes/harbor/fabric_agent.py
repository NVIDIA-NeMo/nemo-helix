# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Harbor agent that runs one complete NeMo Fabric agent config inside the task container.

Point :class:`HarborRuntimeConfig.agent_import_path` (or the platform's ``HarborRunnerTarget``) at
``nemo_evaluator_sdk.agent_eval.runtimes.harbor.fabric_agent:NemoFabricAgent`` and hand it the agent
through ``agent_kwargs["fabric_config"]`` -- a Fabric ``agent.yaml`` as a JSON-shaped mapping, the same
document :class:`~nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime.FabricAgentRuntime` runs on the
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
are found only if the config lists their directory under ``discovery.local_paths``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nemo_fabric import EnvironmentConfig, FabricConfig, RelayAtofFileSinkConfig
from nemo_fabric.integrations.harbor.fabric_agent import HARBOR_ARTIFACT_ROOT, FabricAgent, harbor_harness_defaults

#: build.nvidia.com's OpenAI-compatible endpoint, for configs that name an ``nvidia`` model directly.
NVIDIA_MODEL_BASE_URL = "https://integrate.api.nvidia.com/v1"

#: Upstream ``FabricAgent`` keywords that describe the agent rather than how to install or run it.
FLAT_CONFIG_KWARGS: frozenset[str] = frozenset(
    {
        "fabric_adapter_id",
        "fabric_harness_settings",
        "fabric_model_base_url",
        "fabric_model_api_key_env",
        "fabric_system_instruction",
        "fabric_max_turns",
        "fabric_runtime_timeout_seconds",
        "fabric_environment_env",
        "fabric_blocked_tools",
        "fabric_enabled_tools",
        "fabric_telemetry",
    }
)

HARBOR_INPUT_SCHEMA = "text"
HARBOR_OUTPUT_SCHEMA = "message"

#: Claude Code refuses ``bypassPermissions`` as root unless it is told it is sandboxed.
_CLAUDE_ADAPTER_ID = "nvidia.fabric.claude"
_CLAUDE_UNATTENDED_ENV = {"IS_SANDBOX": "1"}

#: The platform's Inference Gateway authenticates the job, not the request, but a harness's client
#: still refuses an empty key, so a gateway-routed model is handed a placeholder under its
#: ``api_key_env``. Must agree with ``nemo_agents_plugin.fabric.gateway_credentials``.
PLATFORM_GATEWAY_PATH_MARKER = "/apis/inference-gateway/"
PLATFORM_GATEWAY_API_KEY_ENV = "NEMO_AGENTS_IGW_API_KEY"
PLATFORM_GATEWAY_API_KEY_PLACEHOLDER = "not-used"


class NemoFabricAgent(FabricAgent):  # ty: ignore[unsupported-base]
    """``FabricAgent`` that runs a complete ``fabric_config`` instead of building one from keywords.

    ``fabric_default_max_turns`` fills ``runtime.max_turns`` only when the config leaves it unset; an
    explicit ``null`` keeps Fabric's unbounded harness.
    """

    def __init__(
        self,
        logs_dir: Any,
        *args: Any,
        fabric_config: Mapping[str, Any],
        fabric_default_max_turns: int | None = None,
        **kwargs: Any,
    ) -> None:
        rejected = sorted(FLAT_CONFIG_KWARGS & kwargs.keys())
        if rejected:
            raise ValueError(
                f"{', '.join(rejected)}: NemoFabricAgent runs the `fabric_config` it is given; describe the "
                "agent there instead of through FabricAgent's flat keywords"
            )
        config = FabricConfig.from_mapping(fabric_config)
        if config.harness is None or not config.harness.adapter_id.strip():
            raise ValueError("fabric_config must select a harness: set `harness.adapter_id`")
        self.fabric_config = config
        self.fabric_default_max_turns = fabric_default_max_turns
        super().__init__(logs_dir, *args, fabric_adapter_id=config.harness.adapter_id, **kwargs)

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

        # Harbor's task container is Fabric's local provider, whatever the config says.
        environment = config.environment or EnvironmentConfig(provider="local")
        environment.provider = "local"
        environment.workspace = self.fabric_workspace
        environment.artifacts = artifact_root
        config.environment = environment

        assert config.harness is not None  # checked at construction
        for key, value in harbor_harness_defaults(config.harness.adapter_id).items():
            config.harness.settings.setdefault(key, value)
        if config.harness.adapter_id == _CLAUDE_ADAPTER_ID:
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
        if self.skills_dir is not None:
            config.add_skill_path(self.skills_dir)
        return config
