// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_POLLING_INTERVAL_LONG, JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { useAgentsListDeployments } from '@nemo/sdk/generated/agents/agent-deployments';
import { useListEvaluations } from '@nemo/sdk/generated/platform/evaluations';
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
  useAgentsListDeployments: vi.fn(() => ({})),
  useAgentsDeleteDeployment: vi.fn(() => ({})),
}));
vi.mock('@nemo/sdk/generated/platform/evaluations', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/evaluations')>()),
  useListEvaluations: vi.fn(() => ({})),
}));
vi.mock('@studio/api/evaluation/evaluator-jobs', () => ({ fetchEvaluatorJobs: vi.fn(() => []) }));

const AGENT = 'my-agent';

const wrapper = ({ children }: { children: ReactNode }) => (
  <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
);

const render = (opts: { evalsEnabled?: boolean; pollDeployments?: boolean }) =>
  renderHook(() => useAgentDetails({ workspace: 'default', agentName: AGENT, ...opts }), {
    wrapper,
  });

const lastQueryOptions = (hook: typeof useListEvaluations | typeof useAgentsListDeployments) =>
  (vi.mocked(hook).mock.lastCall?.at(-1) as { query: Record<string, unknown> }).query;

const deploymentInterval = (status: string) => {
  const refetchInterval = lastQueryOptions(useAgentsListDeployments).refetchInterval as (query: {
    state: { data: unknown };
  }) => number | false;
  return refetchInterval({ state: { data: { data: [{ agent: AGENT, status }] } } });
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

  it('stops idle deployment polling when not requested', () => {
    render({ pollDeployments: false });
    expect(deploymentInterval('running')).toBe(false);
    expect(deploymentInterval('pending')).toBe(JOB_POLLING_INTERVAL_MS);
  });

  it('polls idle deployments on the long interval when requested', () => {
    render({ pollDeployments: true });
    expect(deploymentInterval('running')).toBe(JOB_POLLING_INTERVAL_LONG);
  });
});
