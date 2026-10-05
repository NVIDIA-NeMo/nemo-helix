// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Whether the agent has anything for Platform to run. An agent registered from traces is created
 * with `config: {}` (it only anchors telemetry from an agent that runs outside Platform), and `{}`
 * is truthy, so a bare `!!config` check lets it through to a deploy that always fails.
 */
export const hasAgentConfig = (config: object | null | undefined): boolean =>
  !!config && Object.keys(config).length > 0;

/** Shown wherever deploying is unavailable because the agent has no config. */
export const NO_CONFIG_DEPLOY_MESSAGE =
  'This agent was registered from traces and runs outside Platform. Upload the agent to deploy it here.';
