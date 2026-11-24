// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { agentsCreateExecuteJob, agentsRegisterAgent } from '@nemo/sdk/generated/agents/agents';
import { agentsListPluginManagedAgentTypesRoute } from '@nemo/sdk/generated/agents/plugin-managed-agent-types';
import type { Agent } from '@nemo/sdk/generated/agents/schema';
import { compassGetAnalysisConfig } from '@nemo/sdk/generated/compass/compass-analysis-configs';
import type { AtifIngestRequest } from '@nemo/sdk/generated/platform/schema';
import {
  agentsFromTrajectories,
  COMPASS_AGENT_TYPE,
  isQualifiedModelRef,
  resolveCompassAgent,
  triggerInsightsRun,
  triggerInsightsRuns,
} from '@studio/api/insightsAnalysis';
import {
  MOCK_COMPASS_AGENT,
  mockAnalysisJob,
  mockCompassAgentType,
} from '@studio/mocks/handlers/insights';
import { AxiosError, AxiosHeaders } from 'axios';

vi.mock('@nemo/sdk/generated/compass/compass-analysis-configs', async (importOriginal) => ({
  ...(await importOriginal<
    typeof import('@nemo/sdk/generated/compass/compass-analysis-configs')
  >()),
  compassGetAnalysisConfig: vi.fn(),
}));

vi.mock('@nemo/sdk/generated/agents/agents', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/agents/agents')>()),
  agentsCreateExecuteJob: vi.fn(),
  agentsRegisterAgent: vi.fn(),
}));

vi.mock('@nemo/sdk/generated/agents/plugin-managed-agent-types', async (importOriginal) => ({
  ...(await importOriginal<
    typeof import('@nemo/sdk/generated/agents/plugin-managed-agent-types')
  >()),
  agentsListPluginManagedAgentTypesRoute: vi.fn(),
}));

const getConfig = vi.mocked(compassGetAnalysisConfig);
const createJob = vi.mocked(agentsCreateExecuteJob);
const registerAgent = vi.mocked(agentsRegisterAgent);
const listTypes = vi.mocked(agentsListPluginManagedAgentTypesRoute);

const contextOf = (index = 0) => createJob.mock.calls[index][1].spec.context;

const config = (overrides: Record<string, unknown> = {}) => ({
  name: 'email-security-triage',
  workspace: 'default',
  agent: 'email-security-triage',
  enabled: true,
  default_model: 'default/nvidia-nemotron-3-nano-30b-a3b',
  fast_model: 'default/nvidia-nemotron-3-nano-30b-a3b',
  id: 'insights-analysis-config-1',
  created_at: '2026-08-14T09:00:00Z',
  created_by: 'user@example.com',
  updated_at: '2026-08-14T09:00:00Z',
  updated_by: 'user@example.com',
  entity_id: 'insights-analysis-config-1',
  parent: 'ws-default',
  db_version: 1,
  ...overrides,
});

const axiosErrorWithStatus = (status: number, detail?: string) =>
  new AxiosError('Request failed', 'ERR_BAD_REQUEST', undefined, undefined, {
    status,
    statusText: '',
    data: detail ? { detail } : {},
    headers: {},
    config: { headers: new AxiosHeaders() },
  });

beforeEach(() => {
  vi.resetAllMocks();
  listTypes.mockResolvedValue([mockCompassAgentType(MOCK_COMPASS_AGENT)]);
  createJob.mockResolvedValue(mockAnalysisJob('email-security-triage', 'created'));
});

describe('agentsFromTrajectories', () => {
  it('deduplicates and preserves first-seen order', () => {
    const trajectories = [
      { agent: { name: 'b' } },
      { agent: { name: 'a' } },
      { agent: { name: 'b' } },
    ] as AtifIngestRequest[];

    expect(agentsFromTrajectories(trajectories)).toEqual(['b', 'a']);
  });

  it('skips trajectories with no agent name', () => {
    const trajectories = [{ agent: {} }, { agent: { name: 'a' } }] as AtifIngestRequest[];

    expect(agentsFromTrajectories(trajectories)).toEqual(['a']);
  });
});

describe('triggerInsightsRun', () => {
  it('runs the Compass agent with the config model pair', async () => {
    getConfig.mockResolvedValue(config());

    const result = await triggerInsightsRun('default', 'email-security-triage');

    expect(result).toEqual({
      agent: 'email-security-triage',
      status: 'started',
      jobName: 'analysis-run-1',
    });
    expect(createJob).toHaveBeenCalledWith('default', {
      spec: {
        agent: 'compass-analyst',
        context: {
          agent: 'email-security-triage',
          default_model: 'default/nvidia-nemotron-3-nano-30b-a3b',
          fast_model: 'default/nvidia-nemotron-3-nano-30b-a3b',
        },
      },
    });
    expect(registerAgent).not.toHaveBeenCalled();
  });

  it('scopes the run to a time and an evaluation', async () => {
    getConfig.mockResolvedValue(config());

    await triggerInsightsRun(
      'default',
      'email-security-triage',
      {},
      { since: '2026-08-13T09:00:00Z', evaluation_id: 'eval-1' }
    );

    expect(contextOf()).toMatchObject({
      since: '2026-08-13T09:00:00Z',
      evaluation_id: 'eval-1',
    });
  });

  it('creates the Compass agent when the workspace has none', async () => {
    getConfig.mockResolvedValue(config());
    listTypes.mockResolvedValue([mockCompassAgentType()]);
    registerAgent.mockResolvedValue({ name: 'compass-analyst' } as Agent);

    const result = await triggerInsightsRun('default', 'email-security-triage');

    expect(result.status).toBe('started');
    expect(registerAgent).toHaveBeenCalledWith('default', {
      name: 'compass-analyst',
      plugin_managed_agent: { type: COMPASS_AGENT_TYPE },
    });
    expect(createJob.mock.calls[0][1].spec.agent).toBe('compass-analyst');
  });

  it('reports an error when the Compass agent type is not installed', async () => {
    getConfig.mockResolvedValue(config());
    listTypes.mockResolvedValue([]);

    const result = await triggerInsightsRun('default', 'email-security-triage');

    expect(result).toMatchObject({ status: 'error' });
    expect(result.message).toContain(COMPASS_AGENT_TYPE);
    expect(createJob).not.toHaveBeenCalled();
  });

  it('uses a config the caller already holds instead of fetching it', async () => {
    const result = await triggerInsightsRun(
      'default',
      'email-security-triage',
      {},
      { storedConfig: config({ fast_model: 'default/held-fast' }) }
    );

    expect(result.status).toBe('started');
    expect(getConfig).not.toHaveBeenCalled();
    expect(contextOf()).toMatchObject({ fast_model: 'default/held-fast' });
  });

  it('reports not-enabled when the agent has no analysis config', async () => {
    getConfig.mockRejectedValue(axiosErrorWithStatus(404));

    const result = await triggerInsightsRun('default', 'recipe-agent');

    expect(result.status).toBe('not-enabled');
    expect(result.message).toContain('nemo compass analysis enable --agent recipe-agent');
    expect(createJob).not.toHaveBeenCalled();
  });

  it('surfaces the reason Compass rejected the run', async () => {
    getConfig.mockResolvedValue(config());
    createJob.mockRejectedValue(
      axiosErrorWithStatus(422, "Model 'default/missing' was not found in workspace 'default'.")
    );

    const result = await triggerInsightsRun('default', 'email-security-triage');

    expect(result).toMatchObject({
      status: 'error',
      message: "Model 'default/missing' was not found in workspace 'default'.",
    });
  });

  it('reports not-enabled when the stored config has no model pair', async () => {
    getConfig.mockResolvedValue(config({ default_model: '', fast_model: '' }));

    const result = await triggerInsightsRun('default', 'email-security-triage');

    expect(result.status).toBe('not-enabled');
    expect(createJob).not.toHaveBeenCalled();
  });

  it('surfaces a non-404 config failure as an error', async () => {
    getConfig.mockRejectedValue(axiosErrorWithStatus(500, 'Failed to get analysis config.'));

    const result = await triggerInsightsRun('default', 'email-security-triage');

    expect(result).toMatchObject({ status: 'error', message: 'Failed to get analysis config.' });
  });

  it('surfaces a job creation failure as an error', async () => {
    getConfig.mockResolvedValue(config());
    createJob.mockRejectedValue(new Error('boom'));

    const result = await triggerInsightsRun('default', 'email-security-triage');

    expect(result).toMatchObject({ status: 'error', message: 'boom' });
  });
});

describe('isQualifiedModelRef', () => {
  it.each([
    ['default/nvidia-nemotron-mini-4b-instruct', true],
    ['nvidia-nemotron-mini-4b-instruct', false],
    ['default/', false],
    ['/model', false],
    ['a/b/c', false],
    ['', false],
  ])('%s -> %s', (ref, expected) => {
    expect(isQualifiedModelRef(ref)).toBe(expected);
  });
});

describe('triggerInsightsRun overrides', () => {
  it('replaces the stored pair with the supplied overrides', async () => {
    getConfig.mockResolvedValue(config());

    await triggerInsightsRun('default', 'email-security-triage', {
      default_model: 'default/override-slow',
      fast_model: 'default/override-fast',
    });

    expect(contextOf()).toMatchObject({
      default_model: 'default/override-slow',
      fast_model: 'default/override-fast',
    });
  });

  it('keeps the stored value for a blank override half', async () => {
    getConfig.mockResolvedValue(config());

    await triggerInsightsRun('default', 'email-security-triage', {
      default_model: '   ',
      fast_model: 'default/override-fast',
    });

    expect(contextOf()).toMatchObject({
      default_model: 'default/nvidia-nemotron-3-nano-30b-a3b',
      fast_model: 'default/override-fast',
    });
  });

  it('lets an override supply a pair the stored config is missing', async () => {
    getConfig.mockResolvedValue(config({ default_model: '', fast_model: '' }));

    const result = await triggerInsightsRun('default', 'email-security-triage', {
      default_model: 'default/override-slow',
      fast_model: 'default/override-fast',
    });

    expect(result.status).toBe('started');
  });

  it('rejects an unqualified ref before creating the job', async () => {
    getConfig.mockResolvedValue(config());

    const result = await triggerInsightsRun('default', 'email-security-triage', {
      default_model: 'nvidia-nemotron-mini-4b-instruct',
    });

    expect(result.status).toBe('error');
    expect(result.message).toContain('workspace/name format');
    expect(createJob).not.toHaveBeenCalled();
  });

  it('rejects an unqualified ref that came from the stored config', async () => {
    getConfig.mockResolvedValue(config({ default_model: 'nvidia-nemotron-mini-4b-instruct' }));

    const result = await triggerInsightsRun('default', 'email-security-triage');

    expect(result.status).toBe('error');
    expect(result.message).toContain('workspace/name format');
    expect(createJob).not.toHaveBeenCalled();
  });
});

describe('triggerInsightsRuns', () => {
  it('applies the overrides to every agent', async () => {
    getConfig.mockImplementation(async (_workspace, agent) => config({ agent }));

    await triggerInsightsRuns('default', ['a', 'b'], { default_model: 'default/override-slow' });

    expect(createJob).toHaveBeenCalledTimes(2);
    for (const index of [0, 1]) {
      expect(contextOf(index)).toMatchObject({ default_model: 'default/override-slow' });
    }
  });

  it('resolves the Compass agent once for the whole batch', async () => {
    getConfig.mockImplementation(async (_workspace, agent) => config({ agent }));

    await triggerInsightsRuns('default', ['a', 'b', 'c']);

    expect(createJob).toHaveBeenCalledTimes(3);
    expect(listTypes).toHaveBeenCalledTimes(1);
  });

  it('returns one result per agent, in order', async () => {
    getConfig.mockImplementation(async (_workspace, agent) => {
      if (agent === 'missing') throw axiosErrorWithStatus(404);
      return config({ agent });
    });

    const results = await triggerInsightsRuns('default', ['a', 'missing', 'b']);

    expect(results.map(({ agent, status }) => [agent, status])).toEqual([
      ['a', 'started'],
      ['missing', 'not-enabled'],
      ['b', 'started'],
    ]);
  });
});

describe('resolveCompassAgent', () => {
  it('uses the agent that won a concurrent create', async () => {
    listTypes
      .mockResolvedValueOnce([mockCompassAgentType()])
      .mockResolvedValueOnce([mockCompassAgentType(MOCK_COMPASS_AGENT)]);
    registerAgent.mockRejectedValue(axiosErrorWithStatus(409));

    await expect(resolveCompassAgent('default')).resolves.toBe('compass-analyst');
  });

  it('explains when an agent that is not a Compass agent has the default name', async () => {
    listTypes.mockResolvedValue([mockCompassAgentType()]);
    registerAgent.mockRejectedValue(axiosErrorWithStatus(409));

    await expect(resolveCompassAgent('default')).rejects.toThrow(
      `nemo agents create ${COMPASS_AGENT_TYPE} --name <name> --workspace default`
    );
  });
});
