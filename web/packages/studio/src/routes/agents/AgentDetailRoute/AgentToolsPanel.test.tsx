// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema';
import type { McpStatusResponse } from '@studio/api/agents/types';
import { AgentToolsPanel } from '@studio/routes/agents/AgentDetailRoute/AgentToolsPanel';
import { render, screen, within } from '@testing-library/react';
import { AxiosError, type AxiosResponse } from 'axios';

const mocks = vi.hoisted(() => ({ useMcpStatus: vi.fn() }));

vi.mock('@studio/api/agents/useMcpStatus', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/api/agents/useMcpStatus')>()),
  useMcpStatus: mocks.useMcpStatus,
}));

const config = {
  mcp: { servers: { iocs: { transport: 'stdio', url: 'email-security-triage-iocs' } } },
};

const status: McpStatusResponse = {
  checked_at: '2026-10-30T12:00:00Z',
  servers: [
    {
      name: 'iocs',
      transport: 'stdio',
      target: 'email-security-triage-iocs',
      args: [],
      state: 'unresolved',
      tools: [],
      error: "Command 'email-security-triage-iocs' was not found on the runtime PATH.",
    },
    {
      name: 'calc',
      transport: 'stdio',
      target: 'calculator-server',
      args: [],
      state: 'running',
      tools: [
        {
          name: 'add',
          description: 'Add two numbers.',
          input_schema: { type: 'object', properties: { a: { type: 'integer' } }, required: ['a'] },
          exposed: true,
        },
        { name: 'wipe', exposed: false },
      ],
    },
    {
      name: 'broken',
      transport: 'stdio',
      target: 'broken-server',
      args: [],
      state: 'failed',
      tools: [],
      error: 'Connection closed',
      stderr: 'ModuleNotFoundError: No module named foo',
    },
  ],
};

const queryResult = (overrides: Record<string, unknown> = {}) => ({
  data: undefined,
  error: null,
  isLoading: false,
  isFetching: false,
  dataUpdatedAt: 0,
  refetch: vi.fn(),
  ...overrides,
});

const running = { name: 'triage-dep', status: 'running' } as AgentDeployment;

describe('AgentToolsPanel', () => {
  beforeEach(() => mocks.useMcpStatus.mockReset());

  it('lists declared servers as not started without a running deployment', () => {
    mocks.useMcpStatus.mockReturnValue(queryResult());
    render(
      <AgentToolsPanel
        workspace="default"
        config={config}
        deployment={{ name: 'triage-dep', status: 'failed' } as AgentDeployment}
      />
    );

    expect(mocks.useMcpStatus).toHaveBeenCalledWith('default', undefined);
    expect(screen.getByText('iocs')).toBeInTheDocument();
    expect(screen.queryByText('email-security-triage-iocs')).not.toBeInTheDocument();
    expect(screen.getByText('Not started')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Recheck MCP servers' })).toBeDisabled();
  });

  it('shows runtime state, errors, stderr, and tool policy for a running deployment', () => {
    mocks.useMcpStatus.mockReturnValue(queryResult({ data: status, dataUpdatedAt: Date.now() }));
    render(<AgentToolsPanel workspace="default" config={config} deployment={running} />);

    expect(mocks.useMcpStatus).toHaveBeenCalledWith('default', 'triage-dep');
    expect(screen.getByText('Not on PATH')).toBeInTheDocument();
    expect(screen.getByText(/was not found on the runtime PATH/)).toBeInTheDocument();
    expect(screen.getByText('Failed to start')).toBeInTheDocument();
    expect(screen.getByText('ModuleNotFoundError: No module named foo')).toBeInTheDocument();
    expect(screen.getByText('broken-server')).toBeInTheDocument();
    expect(screen.queryByText('calculator-server')).not.toBeInTheDocument();
    expect(screen.getByText('Running')).toBeInTheDocument();

    const calc = within(screen.getByRole('group', { name: 'calc MCP server' }));
    expect(calc.getByText('stdio · 2 tools')).toBeInTheDocument();
    const tools = within(calc.getByRole('list', { name: 'calc tools' }));
    expect(tools.getAllByRole('listitem')).toHaveLength(2);
    expect(tools.getByText('Blocked')).toBeInTheDocument();
    expect(tools.getByText('Add two numbers.')).toBeInTheDocument();
  });

  it('falls back to an unknown badge for states this UI predates', () => {
    const [server] = status.servers;
    const data = { ...status, servers: [{ ...server, state: 'auth_required', error: null }] };
    mocks.useMcpStatus.mockReturnValue(queryResult({ data }));
    render(<AgentToolsPanel workspace="default" config={config} deployment={running} />);

    expect(screen.getByText('Unknown')).toBeInTheDocument();
  });

  it('reports a probe error', () => {
    mocks.useMcpStatus.mockReturnValue(queryResult({ error: new Error('gateway timeout') }));
    render(<AgentToolsPanel workspace="default" config={config} deployment={running} />);

    expect(screen.getByText(/Could not check MCP servers/)).toBeInTheDocument();
  });

  it('asks for a redeploy when the runtime predates the status endpoint', () => {
    const error = new AxiosError('Not Found', '404', undefined, undefined, {
      status: 404,
    } as AxiosResponse);
    mocks.useMcpStatus.mockReturnValue(queryResult({ error }));
    render(<AgentToolsPanel workspace="default" config={config} deployment={running} />);

    expect(screen.getByText(/doesn't report MCP status. Redeploy it/)).toBeInTheDocument();
  });

  it('says when the agent declares no MCP servers', () => {
    mocks.useMcpStatus.mockReturnValue(queryResult());
    render(<AgentToolsPanel workspace="default" config={{}} />);

    expect(screen.getByText('This agent declares no MCP servers.')).toBeInTheDocument();
  });
});
