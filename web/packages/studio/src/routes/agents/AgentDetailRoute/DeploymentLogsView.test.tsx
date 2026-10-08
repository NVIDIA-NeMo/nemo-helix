// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema';
import { withHelixBrowserAuthRequest } from '@nemo/sdk/src/utils/platformRequest';
import { DeploymentLogsView } from '@studio/routes/agents/AgentDetailRoute/DeploymentLogsView';
import { streamSse } from '@studio/util/sseStream';
import { render, waitFor } from '@testing-library/react';

const mocks = vi.hoisted(() => ({
  loggerWarn: vi.fn(),
  streamSse: vi.fn(),
  useAgentsGetDeploymentLogs: vi.fn(),
  withHelixBrowserAuthRequest: vi.fn(),
}));

vi.mock('@nemo/common/src/components/LogViewer', () => ({
  LogViewer: () => <div data-testid="log-viewer" />,
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
  });
});
