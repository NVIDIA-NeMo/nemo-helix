// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import type { RegisterAgentRequestConfig } from '@nemo/sdk/generated/agents/schema/RegisterAgentRequestConfig';
import type { FilesetLocation } from '@studio/api/files/types';
import type { UseMutationOptions } from '@tanstack/react-query';

export interface CreateAgentFromTrialParams {
  workspace: string;
  name: string;
  /** The agent the study optimized, whose spec fileset the new agent needs a copy of. */
  source: FilesetLocation & Pick<Agent, 'description' | 'config_format'>;
  config: RegisterAgentRequestConfig;
  replaceOrphanedFileset?: boolean;
}

export type UseCreateAgentFromTrialOptions = Omit<
  UseMutationOptions<Agent, Error, CreateAgentFromTrialParams>,
  'mutationFn'
>;

/**
 * Mirrors `McpStatusResponse` in `plugins/nemo-agents/src/nemo_agents_plugin/fabric/mcp_status.py`.
 * Both sides are checked against `@studio/mocks/mcpStatusResponse.json`; update all three together.
 */
export const KNOWN_MCP_SERVER_STATES = ['running', 'unresolved', 'failed', 'unsupported'] as const;

export type KnownMcpServerState = (typeof KNOWN_MCP_SERVER_STATES)[number];

/** Probe outcome for one MCP server; newer runtimes may report states this UI predates. */
export type McpServerState = KnownMcpServerState | (string & {});

export interface McpToolStatus {
  name: string;
  description?: string | null;
  /** JSON Schema for the tool's arguments. */
  input_schema?: Record<string, unknown>;
  /** False when the agent's tool policy hides the tool from the model. */
  exposed: boolean;
}

export interface McpServerStatus {
  name: string;
  transport: string;
  /** Command for stdio servers, URL otherwise. */
  target: string;
  args: string[];
  state: McpServerState;
  tools: McpToolStatus[];
  error?: string | null;
  stderr?: string | null;
}

export interface McpStatusResponse {
  servers: McpServerStatus[];
  checked_at: string;
}

/** An MCP server as the agent config declares it, before any deployment probes it. */
export interface DeclaredMcpServer {
  name: string;
  transport: string;
  target: string;
}
