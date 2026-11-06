# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job specification types for the Jobs service.

``HelixJobSpec`` and its children describe *what* a job runs: an ordered
list of steps, each with an executor (:mod:`nemo_helix_plugin.jobs.providers`),
environment, and configuration.  This is the core of the job-creation request
body (``CreateHelixJobRequest.platform_spec``).

Pure pydantic — no server, docker, or kubernetes runtime dependencies.  The
Jobs service (``nhx.core.jobs.app.schemas``) re-exports from here so both the
server and the typed HTTP client share one definition.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional, Self

from nemo_helix_plugin.entity_naming import NAME_PATTERN, NAME_PATTERN_DESCRIPTION
from nemo_helix_plugin.jobs._mapping_access import JobSpecModel
from nemo_helix_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_helix_plugin.jobs.providers import Provider
from nemo_helix_plugin.openshell.types import GatewayProviderBinding
from pydantic import ConfigDict, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema


class HelixJobSecretEnvironmentVariableRef(JobSpecModel):
    """Reference to a secret to populate an environment variable for a job step."""

    name: str = Field(description="The name of the secret to reference")


class HelixJobSecret(JobSpecModel):
    """Inline secret material submitted with a job spec."""

    name: str = Field(description="The name of the secret")
    value: Optional[str] = Field(default=None, description="The secret value submitted with the job")
    ref_id: Optional[str] = Field(default=None, description="Reference id for the stored secret value")


class HelixJobEnvironmentVariable(JobSpecModel):
    """Environment variable for a job step"""

    name: str = Field(description="The environment variable name")
    value: Optional[str] = Field(default=None, description="The environment variable value")
    from_secret: Optional[HelixJobSecretEnvironmentVariableRef] = Field(
        default=None, description="Reference to a secret environment variable to populate the environment variable"
    )

    @model_validator(mode="after")
    def validate_self(self) -> Self:
        # Ensure one of value or from_secret is provided
        if self.value is None and self.from_secret is None:
            raise ValueError("Either value or from_secret must be provided for environment variables.")

        # Ensure only one of value or from_secret is provided
        if self.value is not None and self.from_secret is not None:
            raise ValueError("Only one of value or from_secret can be provided for environment variables.")

        return self


class StepLifecycle(JobSpecModel):
    """Controller-level lifecycle configuration for a job step.

    These settings control how the jobs controller manages the step,
    as opposed to ``config`` which is the task payload forwarded to
    the container.
    """

    staleness_timeout_seconds: int = Field(
        default=0,
        description="If every active task in the step goes this many seconds without an update, the step is terminated. "
        "A value of 0 disables staleness detection.",
    )
    pause_deadline_seconds: int = Field(
        default=0,
        ge=0,
        description="How long a pause may stay in progress before the workload is deleted and the step fails. "
        "A value of 0 means this step cannot be paused.",
    )
    ttl_seconds_active: int = Field(
        default=0,
        ge=0,
        description="How long the step may stay active before it is terminated, in place of the execution "
        "profile's `ttl_seconds_active`. A value of 0 uses the profile's value.",
    )


_K8S_LABEL_KEY_PATTERN = r"^([a-z0-9]([-a-z0-9.]*[a-z0-9])?/)?[A-Za-z0-9]([-A-Za-z0-9_.]{0,61}[A-Za-z0-9])?$"


class TcpPortRange(JobSpecModel):
    """Inclusive TCP span. ``high`` is the last port."""

    low: Annotated[int, Field(ge=1, le=65535)]
    high: Annotated[int, Field(ge=1, le=65535)]

    @model_validator(mode="after")
    def _high_follows_low(self) -> Self:
        if self.high < self.low:
            raise ValueError("tcp port range high must be greater than or equal to low")
        return self


class StepIngressRule(JobSpecModel):
    """Ingress into the step's pods from peer pods that carry this job's ID under ``job_id_label``."""

    job_id_label: str = Field(
        pattern=_K8S_LABEL_KEY_PATTERN,
        description="Label key on peer pods whose value must equal the job ID, e.g. 'nemo-rl-job-id'.",
    )
    tcp_ports: list[Annotated[int, Field(ge=1, le=65535)]] = Field(
        min_length=1, description="TCP ports on the step's pods that matching peers may reach."
    )
    # The controller expands each span into one NetworkPolicy port. Kept out of the HTTP schema.
    tcp_port_ranges: SkipJsonSchema[list[TcpPortRange]] = Field(default_factory=list)


class StepNetwork(JobSpecModel):
    """Network isolation for a step's pods.

    Enforced only when the execution profile enables step network policies. The step's pods then
    accept ingress from other pods of the same job plus the listed rules, and nothing else.
    """

    ingress: list[StepIngressRule] = Field(
        default_factory=list, description="Additional ingress allowed into the step's pods."
    )


class HelixJobStepSpec(JobSpecModel):
    """Specification for a single step in a platform job."""

    name: str = Field(
        description=f"The name of the step. Must be unique for all steps in a job. {NAME_PATTERN_DESCRIPTION}",
        pattern=NAME_PATTERN,
        examples=["preprocess", "train-model", "eval-step-v1"],
    )
    environment: Optional[list[HelixJobEnvironmentVariable]] = Field(
        default=None, description="Environment variables for the step"
    )
    executor: Provider = Field(description="The executor for the step")
    config: dict = Field(default_factory=dict, description="Configuration for the step")
    gateway_providers: dict[str, GatewayProviderBinding] = Field(default_factory=dict)
    gateway_read_filesets: list[str] = Field(
        default_factory=list,
        description="Input filesets readable by a provider-bearing sandbox; authorized at job creation.",
    )
    lifecycle: StepLifecycle = Field(
        default_factory=StepLifecycle, description="Lifecycle configuration settings for the step"
    )
    network: Optional[StepNetwork] = Field(
        default=None, description="Network isolation for the step's pods. Unset leaves ingress unrestricted."
    )

    @field_validator("environment", mode="before")
    @classmethod
    def _coerce_empty_mapping_environment(cls, value: Any) -> Any:
        if value == {}:
            return []
        return value

    @property
    def requires_persistent_storage(self) -> bool:
        """
        Determine if the step requires persistent storage.

        This is determined by checking if the step has an environment variable
        matching the value of PERSISTENT_JOB_STORAGE_PATH_ENVVAR.
        """
        for envvar in self.environment or []:
            if envvar.name == PERSISTENT_JOB_STORAGE_PATH_ENVVAR:
                return True
        return False

    model_config = ConfigDict(regex_engine="python-re")


class HelixJobSpec(JobSpecModel):
    """Specification for a platform job, containing steps and secrets."""

    steps: list[HelixJobStepSpec] = Field(description="List of steps to be executed in the job")
    secrets: Optional[list[HelixJobSecret]] = Field(default=None, description="Secrets referenced by the job")

    @model_validator(mode="after")
    def validate_steps(self) -> Self:
        # Ensure there is at least one step.
        if not self.steps:
            raise ValueError("At least one step is required in the job specification.")

        # Ensure that each step has a unique name.
        step_names = [step.name for step in self.steps]
        if len(step_names) != len(set(step_names)):
            raise ValueError("Each step must have a unique name.")
        return self


# String aliases for provider / profile / backend references.
ProviderRef = str
ProfileRef = str
BackendRef = str


class BaseExecutionProfile(JobSpecModel):
    """Execution configuration for a job.

    Base class for the concrete execution profiles in
    :mod:`nemo_helix_plugin.jobs.execution_profiles`.
    """

    provider: ProviderRef = Field(
        default="cpu",
        description="The compute provider for the executor, e.g., cpu, gpu",
    )
    profile: str = Field(
        default="default",
        description="The profile name for the executor, e.g., high_priority_a100, low_priority, etc.",
    )

    @property
    def supports_persistent_storage(self) -> bool:
        """Indicates if the execution profile supports persistent storage."""
        return False

    def __str__(self) -> str:
        return f"{self.profile}:{self.provider}"
