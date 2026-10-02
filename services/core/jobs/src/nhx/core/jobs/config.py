# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Jobs service."""

from typing import Self

from nhx.common.config import Runtime, create_service_config_class, get_platform_config, get_service_config
from nhx.core.jobs.app.profiles import ExecutionProfileT
from nhx.core.jobs.controllers.backends.config import (
    DefaultExecutionProfileConfig,
    get_default_executor_profiles_for_runtime,
    merge_executor_profiles,
)
from pydantic import Field, model_validator


class JobsServiceConfig(create_service_config_class("jobs")):  # type: ignore
    """
    Configuration for the Jobs Service.

    Environment variables use the NHX_JOBS_ prefix.
    """

    executors: list[ExecutionProfileT] = Field(
        default_factory=list, description="List of executor profiles for the Jobs service"
    )
    executor_defaults: DefaultExecutionProfileConfig = Field(
        default_factory=DefaultExecutionProfileConfig, description="Default executor profile configurations"
    )
    reconcile_interval_seconds: int = Field(default=2, description="Interval in seconds for the job reconciler to run")
    schedule_interval_seconds: int = Field(default=5, description="Interval in seconds for the job scheduler to run")
    enable_subprocess_executor: bool | None = Field(
        default=None,
        description=(
            "Register the subprocess/default execution profile. When unset, defaults to true for "
            "docker/none runtimes and false for kubernetes."
        ),
    )
    include_job_logs_in_diagnostics: bool = Field(
        default=False,
        description=(
            "Include raw job log lines in controller diagnostics snapshots. Disabled by default because "
            "job logs may contain secrets or PII. Enable only for local debugging or test environments."
        ),
    )

    def resolved_enable_subprocess_executor(self) -> bool:
        """Whether host subprocess execution is registered for default profiles."""
        if self.enable_subprocess_executor is not None:
            return self.enable_subprocess_executor
        return get_platform_config().runtime != Runtime.KUBERNETES

    @model_validator(mode="after")
    def validate_executors(self) -> Self:
        """
        Validates that executor profiles have unique (provider, profile) combinations.
        Raises a ValueError if duplicates are found.
        """
        seen_keys = set()
        for executor in self.executors:
            key = (executor.provider, executor.profile)
            if key in seen_keys:
                raise ValueError(
                    f"Duplicate executor profile found for provider '{executor.provider}' and profile '{executor.profile}'"
                )
            seen_keys.add(key)
        return self


# Module-level singleton instances
config = get_service_config(JobsServiceConfig)
platform_runtime = get_platform_config().runtime
# Capability-filtered at merge time (probe_docker). In local standalone the
# controller may further prune this list in place after backend registration.
# Split topologies (API vs controller pods) advertise this merge result from the
# API process — do not gate on controller-only process state (AIRCORE-971).
profiles = merge_executor_profiles(
    config.executors,
    get_default_executor_profiles_for_runtime(
        runtime=platform_runtime,
        defaults=config.executor_defaults,
        enable_subprocess_executor=config.resolved_enable_subprocess_executor(),
    ),
    runtime=platform_runtime,
)
