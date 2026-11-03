// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_POLLING_INTERVAL_LONG, JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { useListEvaluations } from '@nemo/sdk/generated/platform/evaluations';
import { useAgentDeployments } from '@studio/api/agents/useAgentDeployments';
import { useAgentDetails } from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook } from '@testing-library/react';
import type { ReactNode } from 'react';

vi.mock('@nemo/common/src/providers/toast/useToast', () => ({
  useToast: () => ({ error: vi.fn() }),
}));
vi.mock('@nemo/sdk/generated/agents/agents', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/agents/agents')>()),
  useAgentsGetAgent: vi.fn(() => ({})),
}));
vi.mock('@nemo/sdk/generated/agents/agent-deployments', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/agents/agent-deployments')>()),
  useAgentsDeleteDeployment: vi.fn(() => ({})),
}));
vi.mock('@nemo/sdk/generated/platform/evaluations', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/evaluations')>()),
  useListEvaluations: vi.fn(() => ({})),
}));
vi.mock('@studio/api/agents/useAgentDeployments', () => ({
  useAgentDeployments: vi.fn(() => ({ deployments: [] })),
}));
vi.mock('@studio/api/evaluation/evaluator-jobs', () => ({ fetchEvaluatorJobs: vi.fn(() => []) }));

const AGENT = 'my-agent';

const queryClient = new QueryClient();
const wrapper = ({ children }: { children: ReactNode }) => (
  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
);

const render = (opts: { evalsEnabled?: boolean }) =>
  renderHook(() => useAgentDetails({ workspace: 'default', agentName: AGENT, ...opts }), {
    wrapper,
  });

const lastQueryOptions = (hook: typeof useListEvaluations) =>
  (vi.mocked(hook).mock.lastCall?.at(-1) as { query: Record<string, unknown> }).query;

const deploymentInterval = (status: string) => {
  const refetchInterval = vi.mocked(useAgentDeployments).mock.lastCall?.[0]
    .refetchInterval as (query: { state: { data: unknown } }) => number | false;
  return refetchInterval({ state: { data: [{ agent: AGENT, status }] } });
};

describe('useAgentDetails', () => {
  it('skips evaluations when they are disabled', () => {
    render({ evalsEnabled: false });
    expect(lastQueryOptions(useListEvaluations).enabled).toBe(false);
  });

  it('fetches evaluations when they are enabled', () => {
    render({ evalsEnabled: true });
    expect(lastQueryOptions(useListEvaluations).enabled).toBe(true);
  });

  it('polls transitional deployments on the short interval', () => {
    render({});
    expect(deploymentInterval('pending')).toBe(JOB_POLLING_INTERVAL_MS);
  });

  it('polls idle deployments on the long interval', () => {
    render({});
    expect(deploymentInterval('running')).toBe(JOB_POLLING_INTERVAL_LONG);
  });
});
