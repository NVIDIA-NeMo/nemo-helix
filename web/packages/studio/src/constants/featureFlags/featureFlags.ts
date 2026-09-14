/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { booleanFlag, FlagDescriptor, previewFlag } from '@studio/constants/featureFlags/utils';
import { z } from 'zod';

// ============================================================================
// FEATURE FLAGS CONFIGURATION
// ============================================================================
//
// This file defines feature flags for the application. Flags are read from
// environment variables at runtime and parsed into type-safe values.
//
// During development, edit `.env.dev.local` and the dev server will restart.
//
// ## Adding a new flag:
//
// 1. Add the env var to your .env file:
//    VITE_FF_MY_NEW_FLAG=true
//
// 2. Add the flag definition to `flagDefinitions` below:
//    myNewFlag: booleanFlag('VITE_FF_MY_NEW_FLAG', false),
//
// 3. Add the flag to the `service/studio/src/nmp/studio/env_mappings.py`. Example:
//    EnvMapping(marker="STUDIO_UI_VITE_FF_MY_NEW_FLAG", config_path="studio.feature_flags.my_new_flag", default="false"),
//
// 4. Use it in your code:
//    import { featureFlags } from '@studio/constants/featureFlags';
//    if (featureFlags.myNewFlag) { ... }
//
// ## Removing a flag:
//
// 1. Remove all usage from the codebase
// 2. Remove from `flagDefinitions` below
// 3. Remove from .env files
//
// ## Naming conventions:
//
// - Env var: VITE_FF_<SCREAMING_SNAKE_CASE>  (prefix distinguishes from other config)
// - Config key: camelCase
//
// ============================================================================

// --- Flag definitions ---
// Add new flags here. Each flag maps an env var to a typed value.

export const flagDefinitions = {
  agentOptimizationsEnabled: booleanFlag('VITE_FF_AGENT_OPTIMIZATIONS_ENABLED', false),
  agentOverviewEnabled: booleanFlag('VITE_FF_AGENT_OVERVIEW_ENABLED', false),
  agentsEnabled: previewFlag('VITE_FF_AGENTS_ENABLED', true),
  anonymizerEnabled: booleanFlag('VITE_FF_ANONYMIZER_ENABLED', true),
  baseModelsEnabled: previewFlag('VITE_FF_BASE_MODELS_ENABLED', true),
  assistantStudioEnabled: previewFlag('VITE_FF_ASSISTANT_STUDIO_ENABLED', false),
  copilotStudioEnabled: previewFlag('VITE_FF_COPILOT_STUDIO_ENABLED', false),
  customizerEnabled: previewFlag('VITE_FF_CUSTOMIZER_ENABLED', true),
  dashboardEnabled: previewFlag('VITE_FF_DASHBOARD_ENABLED', false),
  dataDesignerEnabled: previewFlag('VITE_FF_DATA_DESIGNER_ENABLED', true),
  datasetsEnabled: previewFlag('VITE_FF_DATASETS_ENABLED', true),
  deploymentsEnabled: previewFlag('VITE_FF_DEPLOYMENTS_ENABLED'),
  evaluatorBenchmarksEnabled: previewFlag('VITE_FF_EVALUATOR_BENCHMARKS_ENABLED', false),
  evaluatorEnabled: previewFlag('VITE_FF_EVALUATOR_ENABLED', true),
  experiment: previewFlag('VITE_FF_EXPERIMENT', true),
  filesetDetailsEnabled: previewFlag('VITE_FF_FILESET_DETAILS_ENABLED'),
  guardrailsEnabled: previewFlag('VITE_FF_GUARDRAILS_ENABLED'),
  inferenceProviderEnabled: previewFlag('VITE_FF_INFERENCE_PROVIDER_ENABLED'),
  intakeEnabled: previewFlag('VITE_FF_INTAKE_ENABLED', true),
  jobsEnabled: previewFlag('VITE_FF_JOBS_ENABLED', true),
  membersEnabled: previewFlag('VITE_FF_MEMBERS_ENABLED'),
  modelCompareEnabled: previewFlag('VITE_FF_MODEL_COMPARE_ENABLED'),
  monitorEnabled: previewFlag('VITE_FF_MONITOR_ENABLED'),
  optimizerEnabled: previewFlag('VITE_FF_OPTIMIZER_ENABLED', true),
  pluginsEnabled: previewFlag('VITE_FF_PLUGINS_ENABLED', true),
  safeSynthesizerEnabled: previewFlag('VITE_FF_SAFE_SYNTHESIZER_ENABLED', true),
  secretsEnabled: previewFlag('VITE_FF_SECRETS_ENABLED', true),
  settingsEnabled: previewFlag('VITE_FF_SETTINGS_ENABLED', true),
  toolCallingEnabled: booleanFlag('VITE_FF_TOOL_CALLING_ENABLED'),
  tourEnabled: booleanFlag('VITE_FF_TOUR_ENABLED', true),
  traceGraphEnabled: booleanFlag('VITE_FF_TRACE_GRAPH_ENABLED'),
} as const;

// --- Types ---

type FlagDefinitions = typeof flagDefinitions;

export type FeatureFlags = {
  [K in keyof FlagDefinitions]: FlagDefinitions[K] extends FlagDescriptor
    ? z.infer<FlagDefinitions[K]['schema']>
    : never;
};
