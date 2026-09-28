# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Harbor agent that runs one complete NeMo Fabric agent config inside the task container.

Point :class:`HarborRuntimeConfig.agent_import_path` (or the platform's ``HarborRunnerTarget``) at
``nemo_evaluator_sdk.agent_eval.runtimes.harbor_fabric_agent:NemoFabricAgent`` and hand it the agent
through ``agent_kwargs["fabric_config"]`` -- a Fabric ``agent.yaml`` as a JSON-shaped mapping, the same
document :class:`~nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime.FabricAgentRuntime` runs on the
host. The config is used **verbatim**: harness, models, instructions, skills, MCP servers, tools, and
telemetry all come from it, and so does the agent's identity (``metadata.name``, the Relay
``agent_name``). Harbor decides only where a trial lives -- the workspace and artifact paths inside
the container -- and the request/response schemas its runner protocol needs.

That is the difference from the upstream ``nemo_fabric.integrations.harbor:FabricAgent`` this class
extends. Upstream *builds* a config from flat ``fabric_*`` keyword arguments (``fabric_adapter_id``,
``fabric_system_instruction``, ``fabric_max_turns``, ...) and stamps it ``harbor-<adapter>``, so
anything without a keyword -- skills, Fabric MCP servers, tool policy, telemetry sinks -- cannot be
expressed and the agent under test is anonymous to Intake. Those keywords are rejected here: build the
whole config upstream of the agent and pass it in. The install/run keywords (``fabric_package``,
``fabric_config_bundle``, ``fabric_workspace``, ``fabric_python``, ...) are unchanged.

Files the config refers to by relative path (``skills.paths``) resolve against ``fabric_config_target``
when a ``fabric_config_bundle`` is uploaded, exactly as upstream: stage the agent's files into a
directory, pass it as the bundle, and keep the paths relative.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nemo_fabric import EnvironmentConfig, FabricConfig, RelayAtofFileSinkConfig
from nemo_fabric.integrations.harbor.fabric_agent import HARBOR_ARTIFACT_ROOT, FabricAgent

#: build.nvidia.com's OpenAI-compatible endpoint, for configs that name an ``nvidia`` model directly.
NVIDIA_MODEL_BASE_URL = "https://integrate.api.nvidia.com/v1"

#: Upstream ``FabricAgent`` keywords that *describe the agent* rather than how to install or run it.
#: They compete with ``fabric_config`` for the same fields, so they are refused rather than merged.
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

#: Harbor's runner protocol: the instruction arrives as text, the answer leaves as a message.
HARBOR_INPUT_SCHEMA = "text"
HARBOR_OUTPUT_SCHEMA = "message"


class NemoFabricAgent(FabricAgent):  # ty: ignore[unsupported-base]
    """``FabricAgent`` that runs a complete ``fabric_config`` instead of building one from keywords.

    ``fabric_default_max_turns`` applies only when the config sets no ``runtime.max_turns``: Fabric
    leaves a harness unbounded by default, and an unbounded harness on a task it cannot finish runs
    until Harbor kills the agent phase -- which yields no ``RunResult`` at all. A config that names its
    own budget (including an explicit ``null``) is left alone.
    """

    def __init__(
        self,
        logs_dir: Any,
        *args: Any,
        fabric_config: Mapping[str, Any] | FabricConfig,
        fabric_default_max_turns: int | None = None,
        **kwargs: Any,
    ) -> None:
        rejected = sorted(FLAT_CONFIG_KWARGS & kwargs.keys())
        if rejected:
            raise ValueError(
                f"{', '.join(rejected)}: NemoFabricAgent runs the `fabric_config` it is given; describe the "
                "agent there instead of through FabricAgent's flat keywords"
            )
        config = fabric_config if isinstance(fabric_config, FabricConfig) else FabricConfig.from_mapping(fabric_config)
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

        # Runner protocol + artifact root. A config-supplied schema wins; most agent configs set none.
        runtime = config.runtime
        runtime.input_schema = runtime.input_schema or HARBOR_INPUT_SCHEMA
        runtime.output_schema = runtime.output_schema or HARBOR_OUTPUT_SCHEMA
        runtime.artifacts = artifact_root
        if runtime.max_turns is None and self.fabric_default_max_turns is not None:
            runtime.max_turns = self.fabric_default_max_turns

        # Where the trial lives. Harbor's task environment is always the local provider from Fabric's
        # point of view; the config's own ``env`` (gateway placeholders, forwarded platform URLs) stays.
        environment = config.environment or EnvironmentConfig(provider="local")
        environment.provider = "local"
        environment.workspace = self.fabric_workspace
        environment.artifacts = artifact_root
        config.environment = environment

        # Harbor's ``agent_model_name`` swaps the model *id* only; endpoint and credential wiring are the
        # config's. Absent a default model there is nothing to swap onto, and inventing one would drop
        # the ``base_url``/``api_key_env`` a caller expected.
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

        # Servers and skills Harbor itself provides for the task, on top of the agent's own.
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
