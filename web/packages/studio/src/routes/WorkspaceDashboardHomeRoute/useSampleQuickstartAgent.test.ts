// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_POLLING_INTERVAL_LONG } from '@nemo/common/src/constants';
import { useAgentsListDeployments } from '@nemo/sdk/generated/agents/agent-deployments';
import { useAgentsGetAgent } from '@nemo/sdk/generated/agents/agents';
import {
  SAMPLE_AGENT_NAME,
  SAMPLE_WORKSPACE_PREFIX,
  useSampleQuickstartAgent,
} from '@studio/routes/WorkspaceDashboardHomeRoute/useSampleQuickstartAgent';
import { renderHook } from '@testing-library/react';

vi.mock('@nemo/sdk/generated/agents/agents', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/agents/agents')>()),
  useAgentsGetAgent: vi.fn(),
}));
vi.mock('@nemo/sdk/generated/agents/agent-deployments', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/agents/agent-deployments')>()),
  useAgentsListDeployments: vi.fn(),
}));

const SAMPLE_WORKSPACE = `${SAMPLE_WORKSPACE_PREFIX}1a2b3c4d`;
const AGENT = { name: SAMPLE_AGENT_NAME, description: 'Sample email triage agent.' };

const deployment = (suffix: string, status: string) => ({
  name: `${SAMPLE_AGENT_NAME}-${suffix}`,
  agent: SAMPLE_AGENT_NAME,
  status,
});

const mockAgent = (data: unknown, isFetched = true) =>
  vi.mocked(useAgentsGetAgent).mockReturnValue({ data, isFetched } as never);

const mockDeployments = (deployments: unknown[], isFetched = true) =>
  vi.mocked(useAgentsListDeployments).mockReturnValue({
    data: isFetched ? { data: deployments } : undefined,
    isFetched,
  } as never);

const sampleStateFromHook = () =>
  renderHook(() => useSampleQuickstartAgent(SAMPLE_WORKSPACE, true)).result.current;

const sampleAgentFromHook = () => {
  const sample = sampleStateFromHook();
  return sample.state === 'ready' ? sample.agent : undefined;
};

/** The `query` options the hook handed to a mocked SDK hook on its last render. */
const lastQueryOptions = (hook: typeof useAgentsGetAgent | typeof useAgentsListDeployments) => {
  const options = vi.mocked(hook).mock.lastCall?.at(-1) as { query: Record<string, unknown> };
  return options.query;
};

describe('useSampleQuickstartAgent', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockAgent(AGENT);
    mockDeployments([deployment('9f2a1c00', 'running')]);
  });

  describe('choosing the deployment', () => {
    it('names a running deployment in the chat command over any other', () => {
      mockDeployments([deployment('0000aaaa', 'failed'), deployment('9f2a1c00', 'running')]);

      expect(sampleAgentFromHook()).toMatchObject({
        status: 'running',
        deploymentName: `${SAMPLE_AGENT_NAME}-9f2a1c00`,
      });
    });

    it('names a starting deployment, which will become routable', () => {
      mockDeployments([deployment('0000aaaa', 'failed'), deployment('5b6c7d00', 'starting')]);

      expect(sampleAgentFromHook()).toMatchObject({
        status: 'starting',
        deploymentName: `${SAMPLE_AGENT_NAME}-5b6c7d00`,
      });
    });

    it.each(['failed', 'deleting'])(
      'shows a %s deployment on the badge but keeps it out of the chat command',
      (status) => {
        mockDeployments([deployment('0000aaaa', status)]);

        const agent = sampleAgentFromHook();

        // A copied command naming it would fail as not routable; the placeholder says why.
        expect(agent?.status).toBe(status);
        expect(agent?.deploymentName).toBeUndefined();
      }
    );

    it('labels the badge "No deployments" when the agent has none', () => {
      mockDeployments([{ name: 'other-12345678', agent: 'other-agent', status: 'running' }]);

      expect(sampleAgentFromHook()).toMatchObject({
        statusLabel: 'No deployments',
        deploymentName: undefined,
      });
    });

    it('shows "Unknown", not "No deployments", when the deployments fetch failed', () => {
      // Errored with no prior success: fetched, but there is no response to read.
      vi.mocked(useAgentsListDeployments).mockReturnValue({
        data: undefined,
        isFetched: true,
      } as never);

      expect(sampleAgentFromHook()).toMatchObject({
        status: 'unknown',
        statusLabel: undefined,
        deploymentName: undefined,
      });
    });
  });

  describe('loading', () => {
    it('holds back until the first deployments response', () => {
      mockDeployments([], false);

      expect(sampleStateFromHook()).toEqual({ state: 'loading' });
    });

    it('does not ask for deployments until the agent exists', () => {
      mockAgent(undefined, false);

      expect(sampleStateFromHook()).toEqual({ state: 'loading' });
      expect(lastQueryOptions(useAgentsListDeployments).enabled).toBe(false);
    });

    it('fetches nothing when disabled, and reports the sample unavailable', () => {
      const { result } = renderHook(() => useSampleQuickstartAgent('some-workspace', false));

      expect(result.current).toEqual({ state: 'unavailable' });
      expect(lastQueryOptions(useAgentsGetAgent).enabled).toBe(false);
      expect(lastQueryOptions(useAgentsListDeployments).enabled).toBe(false);
    });
  });

  describe('a missing agent', () => {
    it('is unavailable once the lookup settles without one (404, 403, 5xx alike)', () => {
      mockAgent(undefined, true);

      expect(sampleStateFromHook()).toEqual({ state: 'unavailable' });
    });

    it('is not retried: a 404 is an answer, not a transient failure', () => {
      sampleAgentFromHook();
      const retry = lastQueryOptions(useAgentsGetAgent).retry as (
        failureCount: number,
        error: unknown
      ) => boolean;

      expect(retry(0, { status: 404 })).toBe(false);
      expect(retry(0, { status: 503 })).toBe(true);
      expect(retry(3, { status: 503 })).toBe(false);
    });

    it('is polled for until it appears, so a re-run of `nemo setup` shows the panel', () => {
      sampleAgentFromHook();
      const refetchInterval = lastQueryOptions(useAgentsGetAgent).refetchInterval as (query: {
        state: { data: unknown };
      }) => number | false;

      expect(refetchInterval({ state: { data: undefined } })).toBe(JOB_POLLING_INTERVAL_LONG);
      expect(refetchInterval({ state: { data: AGENT } })).toBe(false);
    });
  });
});
