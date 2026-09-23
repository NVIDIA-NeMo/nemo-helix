# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment FileSet manifest schemas (native-v1, wheels-v1, adapter-wheels-v1)."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator


class EnvironmentFormat(StrEnum):
    NATIVE_V1 = "native-v1"
    WHEELS_V1 = "wheels-v1"
    ADAPTER_WHEELS_V1 = "adapter-wheels-v1"


class EnvironmentManifestMetadata(BaseModel):
    """Provenance metadata in nemo-environment.yaml. Descriptive only -- nothing here
    changes how the environment is installed or run.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255, description="Environment name.", examples=["ascii-tree"])
    description: str | None = Field(default=None, max_length=2048)
    hub_id: str | None = Field(
        default=None,
        description="Identifier of the upstream package this environment was built from.",
        examples=["primeintellect/ascii-tree"],
    )
    vf_env_id: str | None = Field(
        default=None,
        description="For verifiers-based environments, the id passed to verifiers.load_environment(). "
        "Dataset rows carry the same value in their vf_env_id field.",
        examples=["ascii-tree"],
    )
    adapter_agent: str | None = Field(
        default=None,
        description="Agent harness this package was built against, mirroring adapter.agent.",
        examples=["verifiers_agent"],
    )


class _ManifestBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_paths: list[str] = Field(
        min_length=1,
        description="Gym config YAMLs to load, as paths relative to the package root. Required by "
        "every format -- an environment with no config starts no servers -- but where they may "
        "live differs: native-v1 under responses_api_agents/, resources_servers/ or "
        "responses_api_models/, adapter-wheels-v1 under configs/, wheels-v1 anywhere.",
        examples=[["configs/verifiers_agent.yaml"]],
    )
    metadata: EnvironmentManifestMetadata

    @field_validator("config_paths")
    @classmethod
    def _relative_contained_paths(cls, paths: list[str]) -> list[str]:
        for p in paths:
            if not p or p.startswith("/") or p.startswith("\\") or ".." in p.split("/"):
                raise ValueError(f"config_paths entry must be relative and contained: {p!r}")
        return paths


class NativeV1Manifest(_ManifestBase):
    format: Literal[EnvironmentFormat.NATIVE_V1] = EnvironmentFormat.NATIVE_V1

    @field_validator("config_paths")
    @classmethod
    def _under_gym_dirs(cls, paths: list[str]) -> list[str]:
        allowed = ("responses_api_agents/", "resources_servers/", "responses_api_models/")
        for p in paths:
            if not p.startswith(allowed):
                raise ValueError(f"native-v1 config_paths must be under {allowed}: {p!r}")
        return paths


class WheelsV1Manifest(_ManifestBase):
    format: Literal[EnvironmentFormat.WHEELS_V1] = EnvironmentFormat.WHEELS_V1


class AdapterRef(BaseModel):
    """Agent harness an adapter-wheels-v1 package runs on, supplied by the training image."""

    model_config = ConfigDict(extra="forbid")

    agent: str = Field(
        min_length=1,
        description="Agent harness to run this environment on. Must be one of the harnesses built "
        "into the training image (IMAGE_ADAPTER_ALLOWLIST in "
        "nhx.rl.tasks.environment.allowlist); the package selects existing image code rather than "
        "shipping its own, so an unlisted value is rejected at validation.",
        examples=["verifiers_agent"],
    )
    agent_type: Literal["responses_api_agents"] = "responses_api_agents"
    image_config_root: str | None = Field(
        default=None,
        description="Override for the harness directory inside the image. Unset resolves through the allowlist.",
    )


class AdapterWheelsV1Manifest(_ManifestBase):
    format: Literal[EnvironmentFormat.ADAPTER_WHEELS_V1] = EnvironmentFormat.ADAPTER_WHEELS_V1
    adapter: AdapterRef

    @model_validator(mode="after")
    def _config_paths_under_configs(self) -> AdapterWheelsV1Manifest:
        for p in self.config_paths:
            if not p.startswith("configs/"):
                raise ValueError(f"adapter-wheels-v1 config_paths should live under configs/: {p!r}")
        return self


EnvironmentManifest = Annotated[
    NativeV1Manifest | WheelsV1Manifest | AdapterWheelsV1Manifest,
    Field(discriminator="format"),
]

ENVIRONMENT_MANIFEST_ADAPTER: TypeAdapter[EnvironmentManifest] = TypeAdapter(EnvironmentManifest)


class ModelServerRefSpec(BaseModel):
    """Reference to a NeMo Gym model server entry. During GRPO this points at the policy
    being trained, which Gym serves from the job's own vLLM engine.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["responses_api_models"] = "responses_api_models"
    name: str = "policy_model"


class VerifiersAgentInstanceConfig(BaseModel):
    """Fields under ``responses_api_agents.verifiers_agent`` in an environment's Hydra YAML.

    This is the config that loads a verifiers environment and points it at the model to
    roll out against.

    ``max_tokens`` here wins over the job's ``max_new_tokens``: the compiler stamps that
    onto every row as ``responses_create_params.max_output_tokens``, but the verifiers_agent
    server reads ``max_tokens`` from this config and drops the row value. The effective cap
    is ``min(max_tokens, max_seq_length - prompt_len)``, applied by vLLM's
    ``_clamp_max_tokens``. ``max_new_tokens`` only takes effect once the agent honours
    ``max_output_tokens``.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    entrypoint: str = "app.py"
    model_server: ModelServerRefSpec = Field(
        default_factory=ModelServerRefSpec,
        description="Model server the agent rolls out against; normally the policy model.",
    )
    model_name: str = ""
    vf_env_id: str = Field(
        description="Identifier passed to verifiers.load_environment().",
        examples=["ascii-tree"],
    )
    vf_env_args: dict = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to that environment loader.",
    )
    max_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 1.0
    # NeMo-Gym validates this against a closed set (math, coding, agent, knowledge,
    # instruction_following, long_context, safety, games, translation, e2e, rlhf, other) when
    # it parses the global config. An empty string or an off-list value fails that check while
    # the block still looks like a server, so Gym reports it as an "almost-server" and raises
    # AlmostServerError -- error_on_almost_servers defaults to true and nothing here turns it
    # off, so spin-up aborts rather than skipping the server.
    domain: str = "other"
    description: str = ""
    value: str = ""
    group_size: int = 1
    max_concurrent_generation: int = -1
    max_concurrent_scoring: int = -1


class AgentRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["responses_api_agents"] = "responses_api_agents"
    name: str


class GymDatasetRow(BaseModel):
    """Fields every NeMo-Gym rollout row carries, whatever agent runs it.

    NeMo-RL reads ``agent_ref.name`` to route the row and ``responses_create_params`` to
    apply sampling settings; everything else is agent-specific and passed through.
    """

    model_config = ConfigDict(extra="allow")

    responses_create_params: dict
    agent_ref: AgentRef
    task_idx: int | None = None
    answer: str = ""
    task: str = ""
    example_id: int | str = 0
    info: dict = Field(default_factory=dict)
    question: str | None = None


class GymVerifiersDatasetRow(GymDatasetRow):
    """A row targeting ``verifiers_agent``, which resolves its environment by id."""

    task_idx: int
    vf_env_id: str
