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
export const AGENT_CONTAINER_DEPLOYMENTS_ENABLED =
  featureFlags.agentContainerDeploymentsEnabled !== false;
export const AGENT_OPTIMIZATIONS_ENABLED = featureFlags.agentOptimizationsEnabled !== false;
export const AGENT_OPTIMIZATION_FORM_ENABLED = featureFlags.agentOptimizationFormEnabled !== false;
export const AGENT_OVERVIEW_ENABLED = featureFlags.agentOverviewEnabled !== false;
export const ANONYMIZER_ENABLED = featureFlags.anonymizerEnabled !== false;
export const ASSISTANT_STUDIO_ENABLED = featureFlags.assistantStudioEnabled !== false;
export const CUSTOMIZER_ENABLED = featureFlags.customizerEnabled !== false;
export const DASHBOARD_SANDBOX_ENABLED = featureFlags.dashboardSandboxEnabled !== false;
// The /dashboard route is reachable if either variant behind it is enabled — kept as one derived
// constant so the two flags can't drift out of sync across the route table, the side-nav
// link, and the default-landing redirect (each of which needs this exact condition).
export const DASHBOARD_ROUTE_ENABLED = ASSISTANT_STUDIO_ENABLED || DASHBOARD_SANDBOX_ENABLED;
export const EVALUATOR_ENABLED = featureFlags.evaluatorEnabled !== false;
export const EVALUATOR_BENCHMARKS_ENABLED = featureFlags.evaluatorBenchmarksEnabled !== false;
export const EXPERIMENT_ENABLED = featureFlags.experiment !== false;
export const FILESET_DETAILS_ENABLED = featureFlags.filesetDetailsEnabled !== false;
export const INFERENCE_PROVIDER_ENABLED = featureFlags.inferenceProviderEnabled !== false;
export const INTAKE_ENABLED = featureFlags.intakeEnabled !== false;
export const MEMBERS_ENABLED = featureFlags.membersEnabled !== false;
export const MODEL_COMPARE_ENABLED = featureFlags.modelCompareEnabled !== false;
export const MODEL_EVALUATION_FORM_ENABLED = featureFlags.modelEvaluationFormEnabled !== false;
export const MONITOR_ENABLED = featureFlags.monitorEnabled !== false;
export const OPTIMIZER_ENABLED = featureFlags.optimizerEnabled !== false;
export const GUARDRAILS_ENABLED = featureFlags.guardrailsEnabled !== false;

// Vars used by OpenTelemetry
export const TELEMETRY_ENABLED = getEnvVar('VITE_TELEMETRY_ENABLED')?.toLowerCase() === 'true';
const normalizedBaseUrl = BASE_URL.replace(/\/+$/, '');
export const OTEL_PROXY_URL = `${normalizedBaseUrl}/telemetry`;
export const OTEL_SERVICE_NAME = getEnvVar('VITE_OTEL_SERVICE_NAME');

export const isLocalDevelopmentEnv = getEnvVar('VITE_IS_LOC_ENV')?.toLowerCase() === 'true';

// Vars used by the oidc provider
export const AUTH_CLIENT_ID = getEnvVar('VITE_AUTH_CLIENT_ID');
export const AUTH_AUTHORITY = getEnvVar('VITE_AUTH_AUTHORITY');
export const AUTH_BEARER_TOKEN_SOURCE = getEnvVar('VITE_AUTH_BEARER_TOKEN_SOURCE');
export const AUTH_SCOPES = getEnvVar('VITE_AUTH_SCOPES');
export const AUTH_SCOPE_PREFIX = getEnvVar('VITE_AUTH_SCOPE_PREFIX');

export const VERSION_SHA = getEnvVar('VITE_VERSION_SHA');
