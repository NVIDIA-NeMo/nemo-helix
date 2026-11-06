// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import type { AgentConfig } from '@nemo/common/src/utils/agents';
import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import {
  Badge,
  Banner,
  CodeSnippet,
  Flex,
  Skeleton,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import type {
  DeclaredMcpServer,
  KnownMcpServerState,
  McpServerState,
  McpServerStatus,
} from '@studio/api/agents/types';
import { getDeclaredMcpServers, useMcpStatus } from '@studio/api/agents/useMcpStatus';
import { ChatInfoSection } from '@studio/routes/agents/AgentDetailRoute/ChatInfoSection';
import { McpToolRow } from '@studio/routes/agents/AgentDetailRoute/McpToolRow';
import { AxiosError } from 'axios';
import type { ComponentProps, FC, ReactNode } from 'react';

interface StateBadge {
  label: string;
  color: ComponentProps<typeof Badge>['color'];
}

const UNKNOWN_STATE_BADGE: StateBadge = { label: 'Unknown', color: 'gray' };

const STATE_BADGES: Partial<Record<string, StateBadge>> = {
  running: { label: 'Running', color: 'green' },
  unresolved: { label: 'Not on PATH', color: 'red' },
  failed: { label: 'Failed to start', color: 'red' },
  unsupported: UNKNOWN_STATE_BADGE,
  not_started: { label: 'Not started', color: 'gray' },
} satisfies Record<KnownMcpServerState | 'not_started', StateBadge>;

/** Long unbroken strings (URLs, paths, stderr) must wrap inside the narrow pane. */
const WRAP_ANYWHERE = '[overflow-wrap:anywhere]';

const pluralize = (count: number, noun: string) => `${count} ${noun}${count === 1 ? '' : 's'}`;

interface ServerGroupProps {
  server: DeclaredMcpServer;
  state: McpServerState | 'not_started';
  /** Known only once the server has been probed. */
  toolCount?: number;
  children?: ReactNode;
}

/** A bordered group whose shaded header names the server and whose body holds its tools. */
const ServerGroup: FC<ServerGroupProps> = ({ server, state, toolCount, children }) => {
  const badge = STATE_BADGES[state] ?? UNKNOWN_STATE_BADGE;

  return (
    <Stack
      role="group"
      aria-label={`${server.name} MCP server`}
      className="overflow-hidden rounded-md border border-base"
    >
      <Stack gap="density-xs" className="bg-surface-sunken px-density-md py-density-sm">
        <Flex align="center" justify="between" gap="density-sm">
          <Flex align="center" gap="density-xs" className="min-w-0">
            <Badge kind="outline" color="gray" size="small" className="shrink-0">
              MCP
            </Badge>
            <Text kind="label/bold/md" className="truncate">
              {server.name}
            </Text>
          </Flex>
          <Badge kind="solid" color={badge.color} size="small" className="shrink-0">
            {badge.label}
          </Badge>
        </Flex>
        <Text kind="body/regular/sm" className="text-secondary">
          {[server.transport, toolCount === undefined ? undefined : pluralize(toolCount, 'tool')]
            .filter(Boolean)
            .join(' · ')}
        </Text>
        {/* The command usually echoes the server name; it only helps diagnose a failed launch, and an
          unresolved one already names it in its error. */}
        {state === 'failed' ? (
          <Text kind="body/regular/sm" className={`font-mono text-secondary ${WRAP_ANYWHERE}`}>
            {server.target}
          </Text>
        ) : null}
      </Stack>
      {children ? (
        <Stack gap="density-sm" className="border-t border-base px-density-md py-density-sm">
          {children}
        </Stack>
      ) : null}
    </Stack>
  );
};

const ServerSection: FC<{ server: McpServerStatus }> = ({ server }) => {
  const hasBody = Boolean(server.error || server.stderr || server.tools.length);

  return (
    <ServerGroup
      server={server}
      state={server.state}
      toolCount={server.state === 'running' ? server.tools.length : undefined}
    >
      {hasBody ? (
        <>
          {server.error ? (
            <Text kind="body/regular/sm" className={`text-feedback-danger ${WRAP_ANYWHERE}`}>
              {server.error}
            </Text>
          ) : null}
          {server.stderr ? (
            <CodeSnippet value={server.stderr} language="text" copyButton="inline" rows={8} />
          ) : null}
          {server.tools.length > 0 ? (
            <Stack role="list" aria-label={`${server.name} tools`}>
              {server.tools.map((tool) => (
                <McpToolRow key={tool.name} tool={tool} />
              ))}
            </Stack>
          ) : null}
        </>
      ) : null}
    </ServerGroup>
  );
};

const isNotFound = (error: Error) => error instanceof AxiosError && error.response?.status === 404;

const ProbeError: FC<{ error: Error }> = ({ error }) => (
  <Banner kind="inline" status={isNotFound(error) ? 'warning' : 'error'}>
    <Stack gap="density-xs" className="min-w-0">
      <Text kind="body/regular/sm">
        {isNotFound(error)
          ? "This deployment's runtime doesn't report MCP status. Redeploy it to pick up the latest runtime."
          : 'Could not check MCP servers.'}
      </Text>
      {isNotFound(error) ? null : (
        <Text kind="body/regular/sm" className={`text-secondary ${WRAP_ANYWHERE}`}>
          {getErrorMessage(error)}
        </Text>
      )}
    </Stack>
  </Banner>
);

const Muted: FC<{ children: ReactNode }> = ({ children }) => (
  <Text kind="body/regular/sm" className="text-secondary">
    {children}
  </Text>
);

interface AgentToolsPanelProps {
  workspace: string;
  config?: AgentConfig;
  /** The header's selection; tools are only probed when it is running. */
  deployment?: AgentDeployment;
}

/** Chat tab section listing each declared MCP server, its launch state, and the tools it serves. */
export const AgentToolsPanel: FC<AgentToolsPanelProps> = ({ workspace, config, deployment }) => {
  const runningDeployment = deployment?.status === 'running' ? deployment.name : undefined;
  const { data, error, isLoading } = useMcpStatus(workspace, runningDeployment);
  const declared = getDeclaredMcpServers(config);

  const renderBody = () => {
    if (!runningDeployment) {
      if (declared.length === 0) return <Muted>This agent declares no MCP servers.</Muted>;
      return (
        <>
          <Muted>Servers start when a deployment is running.</Muted>
          {declared.map((server) => (
            <ServerGroup key={server.name} server={server} state="not_started" />
          ))}
        </>
      );
    }
    if (isLoading) return <Skeleton className="h-24 w-full" />;
    if (error) return <ProbeError error={error} />;
    if (!data?.servers.length) return <Muted>This agent declares no MCP servers.</Muted>;
    return (
      <>
        {data.servers.map((server) => (
          <ServerSection key={server.name} server={server} />
        ))}
      </>
    );
  };

  return (
    <ChatInfoSection title="MCP Servers">
      <Stack gap="density-md">{renderBody()}</Stack>
    </ChatInfoSection>
  );
};
