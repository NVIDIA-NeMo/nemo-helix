/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { resolveBrowserBaseUrl } from '@nemo/sdk/src/utils/url';
import { featureFlags } from '@studio/constants/featureFlags';

/**
 * Use this function to get environment variables.
 * We use import.meta.env to get the environment variables, but replace at runtime to support dynamic k8s environment variables.
 * @param envVarKey - The key of the environment variable to get.
 * @returns The value of the environment variable, or undefined if the environment variable is not set.
 */
const getEnvVar = (envVarKey: string) => import.meta.env[envVarKey] as string;

// Special keyword env vars
export const IS_PROD = import.meta.env.PROD;
export const BASE_URL = import.meta.env.BASE_URL as string;

// Platform base URL — single endpoint for all microservices
export const PLATFORM_BASE_URL = resolveBrowserBaseUrl(getEnvVar('VITE_PLATFORM_BASE_URL'));

// Vars to indicate whether certain microservices should be turned off, to
// distinguish that logic from code that calls the URL itself
export const AGENT_OPTIMIZATIONS_ENABLED = featureFlags.agentOptimizationsEnabled !== false;
export const AGENT_OVERVIEW_ENABLED = featureFlags.agentOverviewEnabled !== false;
export const AGENTS_ENABLED = featureFlags.agentsEnabled !== false;
export const ANONYMIZER_ENABLED = featureFlags.anonymizerEnabled !== false;
export const BASE_MODELS_ENABLED = featureFlags.baseModelsEnabled !== false;
export const ASSISTANT_STUDIO_ENABLED =
  featureFlags.assistantStudioEnabled !== false || featureFlags.copilotStudioEnabled !== false;
export const CUSTOMIZER_ENABLED = featureFlags.customizerEnabled !== false;
export const DASHBOARD_ENABLED = featureFlags.dashboardEnabled !== false;
export const DATA_DESIGNER_ENABLED = featureFlags.dataDesignerEnabled !== false;
export const DATASETS_ENABLED = featureFlags.datasetsEnabled !== false;
export const DEPLOYMENTS_ENABLED = featureFlags.deploymentsEnabled !== false;
export const EVALUATOR_ENABLED = featureFlags.evaluatorEnabled !== false;
export const EVALUATOR_BENCHMARKS_ENABLED = featureFlags.evaluatorBenchmarksEnabled !== false;
export const EXPERIMENT_ENABLED = featureFlags.experiment !== false;
export const FILESET_DETAILS_ENABLED = featureFlags.filesetDetailsEnabled !== false;
export const INFERENCE_PROVIDER_ENABLED = featureFlags.inferenceProviderEnabled !== false;
export const INTAKE_ENABLED = featureFlags.intakeEnabled !== false;
export const JOBS_ENABLED = featureFlags.jobsEnabled !== false;
export const MEMBERS_ENABLED = featureFlags.membersEnabled !== false;
export const MODEL_COMPARE_ENABLED = featureFlags.modelCompareEnabled !== false;
export const MONITOR_ENABLED = featureFlags.monitorEnabled !== false;
export const OPTIMIZER_ENABLED = featureFlags.optimizerEnabled !== false;
export const PLUGINS_ENABLED = featureFlags.pluginsEnabled !== false;
export const SAFE_SYNTHESIZER_ENABLED = featureFlags.safeSynthesizerEnabled !== false;
export const SECRETS_ENABLED = featureFlags.secretsEnabled !== false;
export const SETTINGS_ENABLED = featureFlags.settingsEnabled !== false;
export const GUARDRAILS_ENABLED = featureFlags.guardrailsEnabled !== false;
export const TOOL_CALLING_ENABLED = featureFlags.toolCallingEnabled !== false;
export const TOUR_ENABLED = featureFlags.tourEnabled !== false;

// Vars used by OpenTelemetry
export const TELEMETRY_ENABLED = getEnvVar('VITE_TELEMETRY_ENABLED')?.toLowerCase() === 'true';
const normalizedBaseUrl = BASE_URL.replace(/\/+$/, '');
export const OTEL_PROXY_URL = `${normalizedBaseUrl}/telemetry`;
export const OTEL_SERVICE_NAME = getEnvVar('VITE_OTEL_SERVICE_NAME');

export const isLocalDevelopmentEnv = getEnvVar('VITE_IS_LOC_ENV')?.toLowerCase() === 'true';

// Vars used by the oidc provider
export const AUTH_CLIENT_ID = getEnvVar('VITE_AUTH_CLIENT_ID');
export const AUTH_AUTHORITY = getEnvVar('VITE_AUTH_AUTHORITY');
export const AUTH_SCOPES = getEnvVar('VITE_AUTH_SCOPES');
export const AUTH_SCOPE_PREFIX = getEnvVar('VITE_AUTH_SCOPE_PREFIX');

export const VERSION_SHA = getEnvVar('VITE_VERSION_SHA');
