// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { customFetch } from '@nemo/sdk/generated/fetchers/agents';
import type { DeclaredMcpServer, McpStatusResponse } from '@studio/api/agents/types';
import type { AgentConfig } from '@studio/components/dataViews/AgentsDataView';
import { useQuery } from '@tanstack/react-query';

/** Probing spawns every stdio server, so keep results instead of re-probing on each render. */
const MCP_STATUS_STALE_TIME_MS = 60_000;

export const getMcpStatusQueryKey = (workspace: string, deployment: string) =>
  ['agents', workspace, 'deployments', deployment, 'mcp-status'] as const;

/** Served by the deployment's Fabric server through the agents gateway; not in the OpenAPI spec. */
export const fetchMcpStatus = (workspace: string, deployment: string, signal?: AbortSignal) =>
  customFetch<McpStatusResponse>({
    url: `/apis/agents/v2/workspaces/${encodeURIComponent(workspace)}/deployments/${encodeURIComponent(deployment)}/-/v1/mcp/servers`,
    method: 'GET',
    signal,
  });

/** Probes the MCP servers of a running deployment; disabled until one is given. */
export const useMcpStatus = (workspace: string, deployment?: string) =>
  useQuery({
    queryKey: getMcpStatusQueryKey(workspace, deployment ?? ''),
    queryFn: ({ signal }) => fetchMcpStatus(workspace, deployment ?? '', signal),
    enabled: !!workspace && !!deployment,
    staleTime: MCP_STATUS_STALE_TIME_MS,
    refetchOnWindowFocus: false,
    retry: false,
  });

export const getDeclaredMcpServers = (config: AgentConfig | undefined): DeclaredMcpServer[] =>
  Object.entries(config?.mcp?.servers ?? {}).map(([name, server]) => ({
    name,
    transport: server?.transport ?? 'streamable_http',
    target: server?.url ?? '',
  }));
