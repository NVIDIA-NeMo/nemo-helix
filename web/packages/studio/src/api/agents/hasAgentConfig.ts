// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Whether the agent has anything for Platform to run. An agent can exist without a config (today,
 * "Register from traces" creates one with `config: {}` to anchor telemetry), and `{}` is truthy, so
 * a bare `!!config` check lets it through to a deploy that always fails.
 */
export const hasAgentConfig = (config: object | null | undefined): boolean =>
  !!config && Object.keys(config).length > 0;

/** Shown wherever deploying is unavailable because the agent has no config. */
export const NO_CONFIG_DEPLOY_MESSAGE =
  'This agent does not have a configuration and cannot be deployed here. To deploy it, upload an agent configuration under a new name.';
