# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Literal

from nemo_helix_plugin.jobs.image import get_qualified_image
from nhx.common.config import Runtime, create_service_config_class, get_platform_config, get_service_config
from nhx.core.models.controllers.backends.deployments_plugin.config import DeploymentsPluginBackendConfigModel
from nhx.core.models.controllers.backends.registry import BackendConfig
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Parallelism estimation configuration (Pydantic models with defaults)
# -----------------------------------------------------------------------------


class ModelSizeThresholds(BaseModel):
    """Model size thresholds for categorizing models."""

    very_large: float = Field(default=300.0, description=">300B: Very large models")
    large: float = Field(default=100.0, description="100-300B: Large models")
    medium: float = Field(default=50.0, description="50-100B: Medium models")
    small_tp: float = Field(default=70.0, description="<70B: Small models for TP cost")
    small_moe: float = Field(default=40.0, description="<40B: Small MoE models")


class ParallelismMemoryConfig(BaseModel):
    """Memory pressure thresholds and penalties for parallelism estimation."""

    pressure_threshold: float = Field(default=0.60, description="Start penalizing above 60% memory usage")
    pressure_moderate: float = Field(default=0.50, description="Moderate memory pressure")
    pressure_low: float = Field(default=0.45, description="Low memory pressure")
    base_penalty: float = Field(default=1e9, description="Base penalty for exceeding threshold")
    scale_divisor: float = Field(default=0.1, description="Divisor for quadratic penalty: (excess / divisor) ** 2")
    pp_discount_max: float = Field(default=0.7, description="Maximum discount factor (70% off)")
    pp_discount_scale: float = Field(default=2.0, description="Scale factor for discount calculation")


class TensorParallelismConfig(BaseModel):
    """Tensor Parallelism (TP) cost configuration."""

    base_cost_very_large_model: float = Field(default=50.0, description="For models > 300B")
    base_cost_standard_model: float = Field(default=100.0, description="For models <= 300B")
    excessive_very_large: int = Field(default=8, description="TP=8 is standard for 340B+")
    excessive_standard: int = Field(default=4, description="TP=4 is standard for 70B and below")
    penalty_large_model: float = Field(default=1e8, description="Penalty for excessive TP on large models (>70B)")
    penalty_small_model: float = Field(default=3e8, description="Penalty for excessive TP on small models (<70B)")


class DataParallelismConfig(BaseModel):
    """Data Parallelism (DP) cost configuration."""

    bonus_minimal: float = Field(default=-1e4, description="total_parallelism >= 64")
    bonus_small: float = Field(default=-5e4, description="total_parallelism >= 32")
    bonus_very_strong: float = Field(default=-5e7, description="total_parallelism <= 2")
    bonus_strong: float = Field(default=-3e7, description="total_parallelism <= 4")
    bonus_moderate: float = Field(default=-1.5e7, description="total_parallelism <= 8")
    bonus_medium: float = Field(default=-2e7, description="total_parallelism > 8")
    bonus_small_moe: float = Field(default=-1e6, description="Smaller MoE models")
    total_parallelism_very_high: int = Field(default=64, description="Total parallelism threshold")
    total_parallelism_high: int = Field(default=32, description="Total parallelism threshold")
    total_parallelism_very_low: int = Field(default=2, description="Total parallelism threshold")
    total_parallelism_low: int = Field(default=4, description="Total parallelism threshold")
    total_parallelism_medium: int = Field(default=8, description="Total parallelism threshold")
    cp_bonus_multiplier: float = Field(default=2.0, description="Double DP bonus for pure DP+CP configurations")


class PipelineParallelismConfig(BaseModel):
    """Pipeline Parallelism (PP) cost configuration."""

    cost_moe: float = Field(default=5e5, description="MoE models (lower cost due to higher compute per stage)")
    cost_very_large_model: float = Field(default=5e6, description="param_count_b > 300")
    cost_large_model: float = Field(default=1e7, description="param_count_b > 100")
    cost_medium_model: float = Field(default=3e7, description="param_count_b > 50")
    cost_small_with_tp: float = Field(default=1e8, description="param_count_b <= 50 and tp > 1")
    cost_small_without_tp: float = Field(default=3e8, description="param_count_b <= 50 and tp == 1")


class ContextParallelismConfig(BaseModel):
    """Context Parallelism (CP) cost configuration."""

    bonus_optimal: float = Field(default=-3e8, description="Strong bonus for optimal CP")
    bonus_good: float = Field(default=-2e8, description="Bonus for CP=2 in medium sequences")
    penalty_suboptimal: float = Field(
        default=2e8, description="Penalty: not using enough CP or using CP when not needed"
    )
    penalty_too_much: float = Field(default=1e8, description="Penalty: too much CP")
    penalty_should_use: float = Field(default=3e8, description="Penalty: should use CP but using CP=1")
    seq_to_param_ratio_high: float = Field(default=1.0, description="Very long sequences")
    seq_to_param_ratio_medium: float = Field(default=0.3, description="Medium sequences")
    max_value: int = Field(default=8, description="Maximum CP value to consider")
    optimal_value: int = Field(default=2, description="CP=2 is optimal for medium sequences")
    param_memory_multiplier: float = Field(default=6.0, description="Multiplier for parameter memory calculation")
    param_layers_multiplier: float = Field(default=12.0, description="Multiplier for layers in parameter calculation")
    seq_memory_multiplier: float = Field(default=38.0, description="Multiplier for sequence memory calculation")
    seq_threshold_enable: int = Field(default=8192, description="Enable CP for sequences >= 8K")
    seq_threshold_cp4: int = Field(default=16384, description="Enable CP=4 for sequences >= 16K")
    seq_threshold_cp8: int = Field(default=32768, description="Enable CP=8 for sequences >= 32K")
    seq_threshold_cp16: int = Field(default=131072, description="Enable CP=16 for sequences >= 128K")


class ExpertParallelismConfig(BaseModel):
    """Expert Parallelism (EP) cost configuration."""

    penalty_non_moe: float = Field(default=1e9, description="Huge penalty for EP > 1 on non-MoE models")
    bonus_perfect: float = Field(default=-5e8, description="Perfect: 1 routed expert per GPU")
    bonus_very_efficient: float = Field(default=-4e8, description="Very efficient: <= 8 experts per GPU")
    bonus_good: float = Field(default=-3e8, description="Good: <= 32 experts per GPU")
    bonus_acceptable: float = Field(default=-2e8, description="Acceptable: <= 64 experts per GPU")
    bonus_high_count: float = Field(default=-1e8, description="High expert count per GPU (>64)")
    penalty_no_sharding: float = Field(default=8e8, description="Huge penalty: no EP on MoE")
    penalty_non_divisor: float = Field(default=3e8, description="Penalty for non-divisor EP")
    experts_per_gpu_very_efficient: int = Field(default=8, description="Experts per GPU threshold")
    experts_per_gpu_good: int = Field(default=32, description="Experts per GPU threshold")
    experts_per_gpu_acceptable: int = Field(default=64, description="Experts per GPU threshold")


class BalanceConfig(BaseModel):
    """Balance bonus configuration."""

    ratio_perfect: float = Field(default=1.0, description="TP == PP (perfect balance)")
    ratio_good: float = Field(default=2.0, description="max(TP, PP) / min(TP, PP) <= 2.0")
    bonus_perfect_very_large: float = Field(
        default=-5e8, description="Perfect balance for very large dense models (>300B)"
    )
    bonus_good_very_large: float = Field(default=-3e8, description="Good balance for very large dense models")
    bonus_perfect_large: float = Field(default=-4e8, description="Perfect balance for large models (>100B)")
    bonus_good_large: float = Field(default=-2e8, description="Good balance for large models")
    bonus_perfect_small: float = Field(default=-3e8, description="Perfect balance for smaller models")
    bonus_good_small: float = Field(default=-1e8, description="Good balance for smaller models")
    bonus_strong_moe: float = Field(default=-5e8, description="Very large MoE with tight memory")
    tp_squared_multiplier: float = Field(default=5e6, description="Quadratic TP penalty factor")
    pp_significant_threshold: int = Field(default=4, description="PP >= 4 is significant")
    ep_significant_threshold: int = Field(default=4, description="EP >= 4 is significant")


class ParallelismConfig(BaseModel):
    """
    Main configuration object for parallelism estimation heuristics.
    Loaded from the models service config (YAML/env) under the 'parallelism' key.
    """

    gpus_per_node_default: int = Field(default=8, description="Standard node configuration (e.g., DGX H100)")
    gpu_memory_gb_default: int = Field(default=80, description="Standard GPU memory in GB (e.g., H100 80GB)")
    model_size_thresholds: ModelSizeThresholds = Field(default_factory=ModelSizeThresholds)
    memory: ParallelismMemoryConfig = Field(default_factory=ParallelismMemoryConfig)
    tensor_parallelism: TensorParallelismConfig = Field(default_factory=TensorParallelismConfig)
    data_parallelism: DataParallelismConfig = Field(default_factory=DataParallelismConfig)
    pipeline_parallelism: PipelineParallelismConfig = Field(default_factory=PipelineParallelismConfig)
    context_parallelism: ContextParallelismConfig = Field(default_factory=ContextParallelismConfig)
    expert_parallelism: ExpertParallelismConfig = Field(default_factory=ExpertParallelismConfig)
    balance: BalanceConfig = Field(default_factory=BalanceConfig)


# -----------------------------------------------------------------------------
# Backend and controller configuration
# -----------------------------------------------------------------------------

BackendName = Literal["deployments_plugin"]

# Map backend names to their config model classes.
BACKEND_CONFIG_MODELS: dict[str, type[BackendConfig]] = {
    "deployments_plugin": DeploymentsPluginBackendConfigModel,
}


def get_default_backends_for_runtime(runtime: Runtime) -> dict[BackendName, BackendConfig]:
    """Returns default backend configurations based on the deployment runtime.

    Args:
        runtime: The runtime type (DOCKER or KUBERNETES)

    Returns:
        Dictionary of backend configurations appropriate for the runtime
    """
    logger.debug("Getting default backends for runtime: %s", runtime)
    backends: dict[BackendName, BackendConfig] = {}

    # Default backend for each runtime is enabled so that a minimal platform config
    # (e.g. only platform.runtime: "docker") works without requiring models.controller.backends
    if runtime == Runtime.DOCKER:
        backends["deployments_plugin"] = DeploymentsPluginBackendConfigModel(
            enabled=True,
            docker_executor="local-docker",
            default_executor="local-docker",
        )
    elif runtime == Runtime.KUBERNETES:
        backends["deployments_plugin"] = DeploymentsPluginBackendConfigModel(
            enabled=True,
            k8s_executor="local-k8s",
            default_executor="local-k8s",
        )
    elif runtime == Runtime.NONE:
        backends["deployments_plugin"] = DeploymentsPluginBackendConfigModel(enabled=True)
    if not backends:
        logger.warning(f"No default backends defined for runtime type: {runtime}")

    return backends


def _deep_merge_dicts(default: dict, custom: dict) -> dict:
    """
    Recursively merge two dictionaries, with custom values taking precedence.

    Args:
        default: Base dictionary with default values
        custom: Dictionary with custom values to override defaults

    Returns:
        Merged dictionary with custom values overriding defaults at all nesting levels
    """
    result = default.copy()

    for key, custom_value in custom.items():
        if key in result and isinstance(result[key], dict) and isinstance(custom_value, dict):
            # Recursively merge nested dictionaries
            result[key] = _deep_merge_dicts(result[key], custom_value)
        else:
            # Override with custom value
            result[key] = custom_value

    return result


def merge_backends(
    custom_backends: dict[BackendName, BackendConfig],
    default_backends: dict[BackendName, BackendConfig],
) -> dict[BackendName, BackendConfig]:
    """
    Merge custom backend configurations with default backends, giving precedence to custom backends.

    If a custom backend has the same name as a default backend, the custom backend will override the default.
    If the custom backend matching a default backend has custom config values, those should override the default config values.
    This includes deep merging of nested config objects at any nesting level.

    Args:
        custom_backends: User-provided backend configurations from config file
        default_backends: Default backend configurations based on runtime

    Returns:
        Merged dictionary of backend configurations
    """
    merged_backends: dict[BackendName, BackendConfig] = {}

    # Add default backends first
    for backend_name, backend_config in default_backends.items():
        merged_backends[backend_name] = backend_config

    # Override with custom backends
    for backend_name, custom_config in custom_backends.items():
        # If the custom backend matches a default, merge the configs
        if backend_name in merged_backends:
            default_config = merged_backends[backend_name]

            # Get the full default config data
            default_data = default_config.model_dump()

            # Get custom values (only what was explicitly set)
            custom_data = custom_config.model_dump(exclude_unset=True)

            # Deep merge the dictionaries
            merged_data = _deep_merge_dicts(default_data, custom_data)

            # Reconstruct the config object from merged data
            merged_config = type(default_config)(**merged_data)
            merged_backends[backend_name] = merged_config
        # Otherwise, just add the custom backend
        else:
            merged_backends[backend_name] = custom_config

    # When platform.runtime is NONE (docker/k8s unavailable), the default
    # deployments_plugin backend is the only viable option. Force-enable it and
    # disable any other enabled custom backends so the registry sees exactly one.
    if get_platform_config().runtime == Runtime.NONE:
        fallback_default = default_backends.get("deployments_plugin")
        if fallback_default is not None and fallback_default.enabled:
            current = merged_backends.get("deployments_plugin", fallback_default)
            if not current.enabled:
                logger.warning(
                    "Backend 'deployments_plugin' was disabled in config but the platform runtime is "
                    "not available; force-enabling 'deployments_plugin' since it is the only viable "
                    "backend in this state."
                )
                current = current.model_copy(update={"enabled": True})
            merged_backends["deployments_plugin"] = current

            for name in list(merged_backends):
                cfg = merged_backends[name]
                if name != "deployments_plugin" and cfg.enabled:
                    logger.warning(
                        "Backend '%s' was enabled in config but the platform runtime "
                        "is not available; disabling it and using 'deployments_plugin' instead.",
                        name,
                    )
                    merged_backends[name] = cfg.model_copy(update={"enabled": False})

    return merged_backends


class ControllerConfig(BaseModel):
    """Configuration for the Models Controller service."""

    interval_seconds: int = Field(
        default=5, description="Interval in seconds for the Models Controller to run its control loop"
    )
    backends: dict[BackendName, BackendConfig] = Field(
        default_factory=dict, description="Dict of custom backend configurations for the Models Controller"
    )
    model_deployment_garbage_collection_ttl_seconds: int = Field(
        default=30,
        description="Time-to-live in seconds for DELETED deployments before they are permanently removed from the database",
    )

    error_deployment_ttl_seconds: int = Field(
        default=10800,  # 3 hours
        description="Time-to-live in seconds for ERROR deployments before backend resources are garbage collected",
    )

    # Drift recovery configuration
    drift_recovery_max_attempts: int = Field(
        default=5,
        description="Maximum number of attempts to recover a deployment when backend resources are lost",
    )
    drift_recovery_base_delay_seconds: int = Field(
        default=30,
        description="Base delay in seconds between drift recovery attempts (used for exponential backoff)",
    )
    drift_recovery_max_delay_seconds: int = Field(
        default=300,
        description="Maximum delay in seconds between drift recovery attempts (caps exponential backoff)",
    )

    provider_discovery_timeout_seconds: int = Field(
        default=180,
        ge=1,
        description=(
            "Per-request timeout in seconds for GET /v1/models provider autodiscovery via the inference gateway. "
            "External providers (e.g. NVIDIA Build) can return large model lists and need longer than the SDK default."
        ),
    )
    provider_discovery_max_retries: int = Field(
        default=0,
        ge=0,
        description=(
            "SDK retry count for provider autodiscovery requests. The controller loop already retries on "
            "each cycle; disabling SDK retries avoids multi-minute retry storms on slow upstreams."
        ),
    )

    @field_validator("backends", mode="before")
    @classmethod
    def validate_backends(cls, v: Any) -> dict[str, BackendConfig]:
        """Parse backend configs using the correct model based on the backend name key.

        This is needed because Pydantic's union type doesn't use dict keys for discrimination.
        Without this validator, Pydantic would always use the first union member (DockerBackendConfigModel)
        for all backends, regardless of the key name.
        """
        if not isinstance(v, dict):
            return v

        result: dict[str, BackendConfig] = {}
        for backend_name, config_data in v.items():
            # If already a BackendConfig instance, use it directly
            if isinstance(config_data, DeploymentsPluginBackendConfigModel):
                result[backend_name] = config_data
                continue

            # Get the appropriate config model for this backend name
            config_model = BACKEND_CONFIG_MODELS.get(backend_name)
            if config_model is None:
                raise ValueError(f"Unknown backend: {backend_name}. Available: {list(BACKEND_CONFIG_MODELS.keys())}")

            # Parse the config data using the correct model
            if isinstance(config_data, dict):
                result[backend_name] = config_model(**config_data)
            else:
                result[backend_name] = config_model.model_validate(config_data)

        return result


def default_trusted_hf_models() -> list[str]:
    """Returns the default trusted model IDs in huggingface."""
    return [
        r"nvidia/.*",
    ]


def default_trusted_ngc_models() -> list[str]:
    """Returns the default trusted model IDs in NGC."""
    return [
        r"nvidia/.*",
    ]


class TrustRemoteCodeConfig(BaseModel):
    hf_allow_list: list[str] = Field(
        default_factory=default_trusted_hf_models,
        description="List of repo IDs or regex patterns trusted for model loading from HF (direct match or fullmatch).",
    )

    ngc_allow_list: list[str] = Field(
        default_factory=default_trusted_ngc_models,
        description="List of org/team/name strings or regex patterns for trusted for model loading from NGC (direct match or fullmatch).",
    )
    enabled: bool = Field(default=True, description="Whether to allow trust_remote_code anywhere in the platform")


class ToolCallPluginConfig(BaseModel):
    enabled: bool = Field(
        default=False,
        description=(
            "Whether to allow custom tool-call parser plugins. "
            "When disabled (default), any tool_call_plugin value supplied via API "
            "or fileset metadata is rejected or stripped. Enable with caution — "
            "plugins execute arbitrary Python code inside the inference container."
        ),
    )


class ModelsConfig(create_service_config_class("models")):  # type: ignore
    """
    Consolidated configuration for the Models service.
    Contains both API and Controller configurations.
    """

    huggingface_model_puller: str = Field(
        default_factory=lambda: get_qualified_image("nhx-api"),
        description="Image used to pull model weights. Its entrypoint is overridden to the "
        "Hugging Face CLI ('hf download ...'), so any image with the 'hf' CLI on PATH works. "
        "Defaults to the platform's nhx-api image (registry/tag from platform config); override "
        "to use a different puller image.",
    )
    controller: ControllerConfig = Field(
        default_factory=ControllerConfig, description="Controller service configuration"
    )
    parallelism: ParallelismConfig = Field(
        default_factory=ParallelismConfig,
        description="Parallelism estimation heuristics (model size thresholds, TP/PP/DP/CP/EP costs, balance bonuses)",
    )
    trust_remote_code: TrustRemoteCodeConfig = Field(
        default_factory=TrustRemoteCodeConfig,
        description="Configuration for trust_remote_code in the models service",
    )
    tool_call_plugin: ToolCallPluginConfig = Field(
        default_factory=ToolCallPluginConfig,
        description="Configuration for tool_call_plugin in the models service",
    )


# Module-level singleton instances
config = get_service_config(ModelsConfig)
backends = merge_backends(
    config.controller.backends,
    get_default_backends_for_runtime(get_platform_config().runtime),
)
