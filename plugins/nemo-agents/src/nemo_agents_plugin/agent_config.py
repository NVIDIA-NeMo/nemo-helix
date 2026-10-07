# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform-owned agent.yaml config models for NeMo Agents.

These models back Agent.config when config_format is nemo-agents-spec-v1.
First-class environment_spec, sandbox_spec, and harness_spec fields on Agent
are planned; until those shapes are finalized, this config keeps those inputs
together in the versioned Agent.config payload and can be migrated once that
contract lands.
"""

from __future__ import annotations

import posixpath
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

import yaml
from nemo_agents_plugin.entities import AGENT_CONFIG_FILENAME
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class AgentConfigLoadError(ValueError):
    """Raised when a Platform-owned agent.yaml cannot be loaded."""


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    top_p: float | None = Field(default=None, strict=True, ge=0, le=1)
    max_tokens: int | None = Field(default=None, strict=True, gt=0, le=(1 << 64) - 1)
    settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _normalize_integral_max_tokens(cls, value: Any) -> Any:
        # Match Fabric's handling of whole-number JSON/YAML floats.
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value


class HarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    model: ModelConfig | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "local"
    workspace: str = "./workspace"
    artifacts: str = "./artifacts"
    settings: dict[str, Any] = Field(default_factory=dict)
    # Fabric environment mirror fields. Additive with backward-compatible
    # defaults; populated when an AgentEnvironmentSpec is merged at deploy time and
    # forwarded into FabricConfig.environment by the translator.
    env: dict[str, str] = Field(default_factory=dict)
    control_location: str | None = None
    ownership: str | None = None
    connection: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_turns: int | None = Field(default=None, gt=0, le=(1 << 32) - 1)


class TelemetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Tri-state so a config can decline telemetry without being mistaken for one
    # that never mentioned it: unset lets the backend wire an export for this
    # deployment context, False opts out, True turns it on and still lets the
    # backend fill in whatever the config left out.
    enabled: bool | None = None
    provider: str | None = None
    output_dir: str | None = None
    project: str | None = None
    agent_name: str | None = None
    atif: dict[str, Any] | None = None
    atof: dict[str, Any] | None = None
    opentelemetry: dict[str, Any] | None = None


class InstructionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, pattern=r"\S")
    mode: Literal["replace"] = "replace"


class InstructionsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system: InstructionConfig | None = None


class SkillsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(default_factory=list)


class McpServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transport: str
    url: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    custom_headers: dict[str, str] = Field(default_factory=dict)
    exposure: Literal["harness_native", "fabric_managed"] = "harness_native"
    allowed_tools: list[str] | None = None
    blocked_tools: list[str] = Field(default_factory=list)


class McpConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    servers: dict[str, McpServerConfig] = Field(default_factory=dict)


class ToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blocked: list[str] = Field(default_factory=list)


class IncludeConfig(BaseModel):
    """A file or directory from elsewhere in the agent's fileset, copied into the agent root."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        min_length=1,
        description="Path relative to agent.yaml's directory; may climb with '..' but not out of the fileset.",
    )
    target: str = Field(min_length=1, description="Where the source lands, relative to the agent root.")

    @field_validator("source")
    @classmethod
    def _source_is_relative(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute():
            raise ValueError(f"include source must be relative to agent.yaml, got {value!r}")
        if posixpath.normpath(value) == ".":
            raise ValueError("include source must name something other than the agent directory itself")
        return value

    @field_validator("target")
    @classmethod
    def _target_stays_in_agent_root(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in ("..", ".") for part in path.parts):
            raise ValueError(f"include target must be a plain path inside the agent root, got {value!r}")
        if path.as_posix() == AGENT_CONFIG_FILENAME:
            raise ValueError(f"include target must not replace {AGENT_CONFIG_FILENAME}")
        return path.as_posix()


class AgentConfig(BaseModel):
    """Platform-owned agent.yaml config for nemo-agents-spec-v1."""

    model_config = ConfigDict(extra="forbid")

    config_format: Literal["nemo-agents-spec-v1"]
    name: str
    description: str = ""
    default_harness: str
    harnesses: dict[str, HarnessConfig]
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    prompts: dict[str, str] = Field(default_factory=dict)
    instructions: InstructionsConfig | None = None
    skills: SkillsConfig | None = None
    mcp: McpConfig | None = None
    tools: ToolsConfig | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    includes: list[IncludeConfig] = Field(
        default_factory=list,
        description="Files from elsewhere in the agent's fileset to copy into the agent root at deploy time.",
    )

    @model_validator(mode="after")
    def _validate_include_targets(self) -> Self:
        targets = sorted(PurePosixPath(include.target) for include in self.includes)
        for earlier, later in zip(targets, targets[1:], strict=False):
            if earlier == later or earlier in later.parents:
                raise ValueError(f"include targets {earlier.as_posix()!r} and {later.as_posix()!r} overlap")
        return self

    @model_validator(mode="after")
    def _validate_default_harness(self) -> Self:
        if self.default_harness not in self.harnesses:
            available = ", ".join(sorted(self.harnesses))
            raise ValueError(f"default_harness must reference one of harnesses: {available}")
        return self


def load_agent_config(path: str | Path) -> AgentConfig:
    """Load a Platform-owned agent.yaml file as an AgentConfig."""
    config_path = Path(path)

    try:
        raw_config = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise AgentConfigLoadError(f"Unable to read agent config {config_path}: {error}") from error
    except UnicodeDecodeError as error:
        raise AgentConfigLoadError(f"Agent config {config_path} is not valid UTF-8: {error}") from error

    try:
        data = yaml.safe_load(raw_config)
    except yaml.YAMLError as error:
        raise AgentConfigLoadError(f"YAML parse error in agent config {config_path}: {error}") from error

    if not isinstance(data, dict):
        raise AgentConfigLoadError(f"Agent config {config_path} root must be a YAML mapping.")

    try:
        return AgentConfig.model_validate(data)
    except ValidationError as error:
        raise AgentConfigLoadError(f"Invalid agent config {config_path}: {error}") from error


def load_agent_config_from_dir(agent_dir: str | Path) -> AgentConfig:
    """Load the canonical agent.yaml file from an agent directory."""
    return load_agent_config(Path(agent_dir) / AGENT_CONFIG_FILENAME)
