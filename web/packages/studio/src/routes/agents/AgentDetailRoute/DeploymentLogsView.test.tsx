// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema';
import { withHelixBrowserAuthRequest } from '@nemo/sdk/src/utils/platformRequest';
import { DeploymentLogsView } from '@studio/routes/agents/AgentDetailRoute/DeploymentLogsView';
import { streamSse, type StreamSseOptions } from '@studio/util/sseStream';
import { render, screen, waitFor } from '@testing-library/react';

const mocks = vi.hoisted(() => ({
  loggerWarn: vi.fn(),
  streamSse: vi.fn(),
  useAgentsGetDeploymentLogs: vi.fn(),
  withHelixBrowserAuthRequest: vi.fn(),
}));

vi.mock('@nemo/common/src/components/LogViewer', () => ({
  LogViewer: ({ logStatus }: { logStatus?: string }) => (
    <div data-testid="log-viewer" data-log-status={logStatus ?? 'none'} />
  ),
}));

vi.mock('@nemo/common/src/utils/logger', () => ({
  logger: {
    warn: mocks.loggerWarn,
  },
}));

vi.mock('@nemo/sdk/generated/agents/agent-deployments', () => ({
  getAgentsStreamDeploymentLogsQueryKey: (workspace: string, deploymentName: string) => [
    `/apis/agents/v2/workspaces/${workspace}/deployments/${deploymentName}/logs/stream`,
  ],
  useAgentsGetDeploymentLogs: mocks.useAgentsGetDeploymentLogs,
}));

vi.mock('@nemo/sdk/src/utils/platformRequest', () => ({
  withHelixBrowserAuthRequest: mocks.withHelixBrowserAuthRequest,
}));

vi.mock('@studio/providers/auth/useOidcBearerToken', () => ({
  useOidcBearerToken: () => 'stale-token',
}));

vi.mock('@studio/util/sseStream', () => ({
  streamSse: mocks.streamSse,
}));

const deployment = {
  name: 'calc-dep',
  status: 'running',
} as AgentDeployment;

describe('DeploymentLogsView', () => {
  beforeEach(() => {
    mocks.loggerWarn.mockReset();
    mocks.streamSse.mockReset();
    mocks.streamSse.mockResolvedValue(undefined);
    mocks.useAgentsGetDeploymentLogs.mockReset();
    mocks.useAgentsGetDeploymentLogs.mockReturnValue({
      data: { data: [], next_offset: 7 },
      isLoading: false,
    });
    mocks.withHelixBrowserAuthRequest.mockReset();
    mocks.withHelixBrowserAuthRequest.mockImplementation(async (init?: RequestInit) => {
      const headers = new Headers(init?.headers);
      headers.delete('Authorization');
      headers.set('X-Source', 'NeMo Studio');
      return { ...init, credentials: 'include', headers };
    });
  });

  it('streams deployment logs with server-session credentials and no stale bearer token', async () => {
    render(<DeploymentLogsView workspace="default" deployments={[deployment]} />);

    await waitFor(() => expect(withHelixBrowserAuthRequest).toHaveBeenCalled());
    await waitFor(() => expect(streamSse).toHaveBeenCalled());

    const options = vi.mocked(streamSse).mock.calls[0]?.[1];
    const headers = new Headers(options?.headers);
    expect(options?.credentials).toBe('include');
    expect(headers.get('X-Source')).toBe('NeMo Studio');
    expect(headers.has('Authorization')).toBe(false);
  });

  it('logs an accurate setup failure when authentication request setup fails', async () => {
    mocks.withHelixBrowserAuthRequest.mockRejectedValue(new Error('discovery unavailable'));

    render(<DeploymentLogsView workspace="default" deployments={[deployment]} />);

    await waitFor(() =>
      expect(mocks.loggerWarn).toHaveBeenCalledWith(
        'Unable to start log stream for deployment calc-dep',
        expect.any(Error)
      )
    );
    expect(streamSse).not.toHaveBeenCalled();
    expect(screen.getByTestId('log-viewer')).toHaveAttribute('data-log-status', 'none');
  });

  it('badges the log as live while the stream is open', async () => {
    mocks.streamSse.mockReturnValue(new Promise(() => {}));

    render(<DeploymentLogsView workspace="default" deployments={[deployment]} />);

    await waitFor(() => expect(streamSse).toHaveBeenCalled());
    expect(screen.getByTestId('log-viewer')).toHaveAttribute('data-log-status', 'live');
  });

  it.each(['failed', 'deleting'] as const)(
    'badges the log as ended for a %s deployment even while the stream is open',
    async (status) => {
      mocks.streamSse.mockReturnValue(new Promise(() => {}));

      render(<DeploymentLogsView workspace="default" deployments={[{ ...deployment, status }]} />);

      await waitFor(() => expect(streamSse).toHaveBeenCalled());
      expect(screen.getByTestId('log-viewer')).toHaveAttribute('data-log-status', 'ended');
    }
  );

  it('drops the live badge once the stream gives up, e.g. on a 401', async () => {
    mocks.streamSse.mockImplementation(async (_url: string, options: StreamSseOptions) => {
      options.onError?.(new Error('SSE request failed: 401'));
    });

    render(<DeploymentLogsView workspace="default" deployments={[deployment]} />);

    await waitFor(() =>
      expect(screen.getByTestId('log-viewer')).toHaveAttribute('data-log-status', 'none')
    );
  });
});
