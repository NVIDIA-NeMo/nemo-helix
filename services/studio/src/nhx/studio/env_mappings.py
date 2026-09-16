# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment variable mappings for Studio UI runtime injection.

This module defines the mapping between STUDIO_UI_* markers in the built
Vite bundle and their corresponding global_settings paths.

To add a new environment variable:
1. Add the marker to `.env.fastapi` in web/packages/studio/env/
2. Add the mapping entry below with the marker name and config path

The config_path uses dot notation to traverse nested config objects:
- "studio.platform_base_url" -> global_settings.studio.platform_base_url
"""

from dataclasses import dataclass


@dataclass
class EnvMapping:
    """Mapping from a STUDIO_UI_* marker to a global_settings path."""

    marker: str
    """The marker string to replace (e.g., 'STUDIO_UI_VITE_PLATFORM_BASE_URL')"""

    config_path: str
    """Dot-notation path to the value in global_settings (e.g., 'studio.platform_base_url')"""

    default: str = ""
    """Default value if the config path cannot be resolved"""


# Define all environment variable mappings here
# Each entry maps a STUDIO_UI_* marker to a global_settings attribute
ENV_MAPPINGS: list[EnvMapping] = [
    EnvMapping(
        marker="STUDIO_UI_VITE_PLATFORM_BASE_URL",
        config_path="studio.platform_base_url",
    ),
    EnvMapping(marker="STUDIO_UI_VITE_APP_ENV", config_path="studio.app_env", default="production"),
    EnvMapping(marker="STUDIO_UI_VITE_AUTH_AUTHORITY", config_path="auth.oidc.issuer"),
    EnvMapping(marker="STUDIO_UI_VITE_AUTH_CLIENT_ID", config_path="auth.oidc.client_id"),
    EnvMapping(marker="STUDIO_UI_VITE_AUTH_SCOPES", config_path="auth.oidc.default_scopes"),
    EnvMapping(marker="STUDIO_UI_VITE_AUTH_SCOPE_PREFIX", config_path="auth.oidc.scope_prefix"),
    EnvMapping(marker="STUDIO_UI_VITE_DATA_STORE_MICROSERVICE_URL", config_path="studio.data_store_url"),
    EnvMapping(
        marker="STUDIO_UI_VITE_NIM_PROXY_MICROSERVICE_INTERNAL_URL", config_path="studio.nim_proxy_internal_url"
    ),
    EnvMapping(marker="STUDIO_UI_VITE_NIM_PROXY_MICROSERVICE_URL", config_path="studio.nim_proxy_url"),
    EnvMapping(marker="STUDIO_UI_VITE_OTEL_COLLECTOR_URL", config_path="studio.otel.collector_url"),
    EnvMapping(
        marker="STUDIO_UI_VITE_OTEL_SERVICE_NAME", config_path="studio.otel.service_name", default="nemo-studio-ui"
    ),
    EnvMapping(marker="STUDIO_UI_VITE_SANDBOX_ENABLED", config_path="studio.sandbox_enabled"),
    EnvMapping(marker="STUDIO_UI_VITE_STUDIO_FLYWHEEL_ENABLED", config_path="studio.flywheel_enabled"),
    EnvMapping(marker="STUDIO_UI_VITE_TELEMETRY_ENABLED", config_path="studio.telemetry_enabled", default="false"),
    # Feature Flags (VITE_FF_* prefix)
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_AGENT_OPTIMIZATIONS_ENABLED",
        config_path="studio.feature_flags.agent_optimizations_enabled",
        default="false",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_AGENT_OVERVIEW_ENABLED",
        config_path="studio.feature_flags.agent_overview_enabled",
        default="false",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_AGENTS_ENABLED", config_path="studio.feature_flags.agents_enabled", default="true"
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_ANONYMIZER_ENABLED",
        config_path="studio.feature_flags.anonymizer_enabled",
        default="true",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_BASE_MODELS_ENABLED",
        config_path="studio.feature_flags.base_models_enabled",
        default="true",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_ASSISTANT_STUDIO_ENABLED",
        config_path="studio.feature_flags.assistant_studio_enabled",
        default="false",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_COPILOT_STUDIO_ENABLED",
        config_path="studio.feature_flags.copilot_studio_enabled",
        default="false",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_CUSTOMIZER_ENABLED",
        config_path="studio.feature_flags.customizer_enabled",
        default="true",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_DASHBOARD_ENABLED",
        config_path="studio.feature_flags.dashboard_enabled",
        default="false",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_DATA_DESIGNER_ENABLED",
        config_path="studio.feature_flags.data_designer_enabled",
        default="true",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_DATASETS_ENABLED",
        config_path="studio.feature_flags.datasets_enabled",
        default="true",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_DEPLOYMENTS_ENABLED",
        config_path="studio.feature_flags.deployments_enabled",
        default="false",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_EVALUATOR_BENCHMARKS_ENABLED",
        config_path="studio.feature_flags.evaluator_benchmarks_enabled",
        default="false",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_EVALUATOR_ENABLED",
        config_path="studio.feature_flags.evaluator_enabled",
        default="true",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_EXPERIMENT",
        config_path="studio.feature_flags.experiment",
        default="true",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_FILESET_DETAILS_ENABLED",
        config_path="studio.feature_flags.fileset_details_enabled",
        default="false",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_GUARDRAILS_ENABLED",
        config_path="studio.feature_flags.guardrails_enabled",
        default="false",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_INFERENCE_PROVIDER_ENABLED",
        config_path="studio.feature_flags.inference_provider_enabled",
        default="false",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_INTAKE_ENABLED", config_path="studio.feature_flags.intake_enabled", default="true"
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_JOBS_ENABLED", config_path="studio.feature_flags.jobs_enabled", default="true"
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_MEMBERS_ENABLED", config_path="studio.feature_flags.members_enabled", default="true"
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_MODEL_COMPARE_ENABLED",
        config_path="studio.feature_flags.model_compare_enabled",
        default="false",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_MONITOR_ENABLED",
        config_path="studio.feature_flags.monitor_enabled",
        default="false",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_OPTIMIZER_ENABLED",
        config_path="studio.feature_flags.optimizer_enabled",
        default="true",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_PLUGINS_ENABLED",
        config_path="studio.feature_flags.plugins_enabled",
        default="true",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_SAFE_SYNTHESIZER_ENABLED",
        config_path="studio.feature_flags.safe_synthesizer_enabled",
        default="true",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_SECRETS_ENABLED", config_path="studio.feature_flags.secrets_enabled", default="true"
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_SETTINGS_ENABLED", config_path="studio.feature_flags.settings_enabled", default="true"
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_TOOL_CALLING_ENABLED",
        config_path="studio.feature_flags.tool_calling_enabled",
        default="false",
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_TOUR_ENABLED", config_path="studio.feature_flags.tour_enabled", default="true"
    ),
    EnvMapping(
        marker="STUDIO_UI_VITE_FF_TRACE_GRAPH_ENABLED",
        config_path="studio.feature_flags.trace_graph_enabled",
        default="false",
    ),
]
