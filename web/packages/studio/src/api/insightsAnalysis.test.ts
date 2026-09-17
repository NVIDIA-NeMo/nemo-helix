// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { insightsGetAnalysisConfig } from '@nemo/sdk/generated/insights/insights-analysis-configs';
import { insightsCreateAnalyzeJob } from '@nemo/sdk/generated/insights/insights-analysis-jobs';
import type { AtifIngestRequest } from '@nemo/sdk/generated/platform/schema';
import {
  agentsFromTrajectories,
  isQualifiedModelRef,
  triggerInsightsRun,
  triggerInsightsRuns,
} from '@studio/api/insightsAnalysis';
import { AxiosError, AxiosHeaders } from 'axios';

vi.mock('@nemo/sdk/generated/insights/insights-analysis-configs', async (importOriginal) => ({
  ...(await importOriginal<
    typeof import('@nemo/sdk/generated/insights/insights-analysis-configs')
  >()),
  insightsGetAnalysisConfig: vi.fn(),
}));

vi.mock('@nemo/sdk/generated/insights/insights-analysis-jobs', () => ({
  insightsCreateAnalyzeJob: vi.fn(),
}));

const getConfig = vi.mocked(insightsGetAnalysisConfig);
const createJob = vi.mocked(insightsCreateAnalyzeJob);

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
  it('creates an analyze job with the config model pair', async () => {
    getConfig.mockResolvedValue(config());
    createJob.mockResolvedValue({
      name: 'analyze-job-1',
      spec: { agent: '', default_model: '', fast_model: '' },
    });

    const result = await triggerInsightsRun('default', 'email-security-triage');

    expect(result).toEqual({
      agent: 'email-security-triage',
      status: 'started',
      jobName: 'analyze-job-1',
    });
    expect(createJob).toHaveBeenCalledWith('default', {
      description: expect.stringContaining('email-security-triage'),
      spec: {
        agent: 'email-security-triage',
        default_model: 'default/nvidia-nemotron-3-nano-30b-a3b',
        fast_model: 'default/nvidia-nemotron-3-nano-30b-a3b',
      },
    });
  });

  it('reports not-enabled when the agent has no analysis config', async () => {
    getConfig.mockRejectedValue(axiosErrorWithStatus(404));

    const result = await triggerInsightsRun('default', 'recipe-agent');

    expect(result.status).toBe('not-enabled');
    expect(result.message).toContain('nemo insights analysis enable --agent recipe-agent');
    expect(createJob).not.toHaveBeenCalled();
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
    createJob.mockResolvedValue({
      name: 'analyze-job-1',
      spec: { agent: '', default_model: '', fast_model: '' },
    });

    await triggerInsightsRun('default', 'email-security-triage', {
      default_model: 'default/override-slow',
      fast_model: 'default/override-fast',
    });

    expect(createJob).toHaveBeenCalledWith(
      'default',
      expect.objectContaining({
        spec: expect.objectContaining({
          default_model: 'default/override-slow',
          fast_model: 'default/override-fast',
        }),
      })
    );
  });

  it('keeps the stored value for a blank override half', async () => {
    getConfig.mockResolvedValue(config());
    createJob.mockResolvedValue({
      name: 'analyze-job-1',
      spec: { agent: '', default_model: '', fast_model: '' },
    });

    await triggerInsightsRun('default', 'email-security-triage', {
      default_model: '   ',
      fast_model: 'default/override-fast',
    });

    expect(createJob).toHaveBeenCalledWith(
      'default',
      expect.objectContaining({
        spec: expect.objectContaining({
          default_model: 'default/nvidia-nemotron-3-nano-30b-a3b',
          fast_model: 'default/override-fast',
        }),
      })
    );
  });

  it('lets an override supply a pair the stored config is missing', async () => {
    getConfig.mockResolvedValue(config({ default_model: '', fast_model: '' }));
    createJob.mockResolvedValue({
      name: 'analyze-job-1',
      spec: { agent: '', default_model: '', fast_model: '' },
    });

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
    createJob.mockResolvedValue({
      name: 'analyze-job-1',
      spec: { agent: '', default_model: '', fast_model: '' },
    });

    await triggerInsightsRuns('default', ['a', 'b'], { default_model: 'default/override-slow' });

    expect(createJob).toHaveBeenCalledTimes(2);
    for (const call of createJob.mock.calls) {
      expect(call[1].spec.default_model).toBe('default/override-slow');
    }
  });

  it('returns one result per agent, in order', async () => {
    getConfig.mockImplementation(async (_workspace, agent) => {
      if (agent === 'missing') throw axiosErrorWithStatus(404);
      return config({ agent });
    });
    createJob.mockResolvedValue({
      name: 'analyze-job-1',
      spec: { agent: '', default_model: '', fast_model: '' },
    });

    const results = await triggerInsightsRuns('default', ['a', 'missing', 'b']);

    expect(results.map(({ agent, status }) => [agent, status])).toEqual([
      ['a', 'started'],
      ['missing', 'not-enabled'],
      ['b', 'started'],
    ]);
  });
});
