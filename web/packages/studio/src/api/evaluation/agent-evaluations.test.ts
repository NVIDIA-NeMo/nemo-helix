// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentEvaluateJob } from '@nemo/sdk/generated/evals/schema';
import {
  type AgentEvalBundle,
  type AgentEvalResult,
  agentNameForJob,
  aggregateScoresOf,
  evalConfigName,
  fetchAgentEvalJob,
  fetchAgentEvalJobs,
  fetchAgentEvalResultsForJobs,
  joinBundleByTask,
  parseBundleRef,
} from '@studio/api/evaluation/agent-evaluations';

const customFetchMock = vi.fn();
vi.mock('@nemo/sdk/generated/fetchers/evals', () => ({
  customFetch: (...args: unknown[]) => customFetchMock(...args),
}));

const filesDownloadFileMock = vi.fn();
vi.mock('@nemo/sdk/generated/platform/files', () => ({
  filesDownloadFile: (...args: unknown[]) => filesDownloadFileMock(...args),
}));

beforeEach(() => {
  customFetchMock.mockReset();
  filesDownloadFileMock.mockReset();
});

const baseJob = (overrides: Partial<AgentEvaluateJob> = {}): AgentEvaluateJob =>
  ({
    name: 'eval-1',
    workspace: 'ws-a',
    status: 'completed',
    created_at: '2026-05-05T00:00:00Z',
    updated_at: '2026-05-05T00:01:00Z',
    spec: { target: { kind: 'agent', agent: { name: 'support-bot-mini' } }, tasks: [{}, {}] },
    ...overrides,
  }) as AgentEvaluateJob;

describe('fetchAgentEvalJobs', () => {
  it('walks all pages until a short page ends pagination', async () => {
    const page1 = Array.from({ length: 50 }, (_, i) => baseJob({ name: `j-${i}` }));
    const page2 = Array.from({ length: 50 }, (_, i) => baseJob({ name: `j-${50 + i}` }));
    const page3 = [baseJob({ name: 'j-100' })];
    customFetchMock
      .mockResolvedValueOnce({ data: page1 })
      .mockResolvedValueOnce({ data: page2 })
      .mockResolvedValueOnce({ data: page3 });
    const all = await fetchAgentEvalJobs('ws-a', new AbortController().signal);
    expect(all).toHaveLength(101);
    expect(customFetchMock).toHaveBeenCalledTimes(3);
  });

  it('targets the agent-evaluate endpoint', async () => {
    customFetchMock.mockResolvedValueOnce({ data: [] });
    await fetchAgentEvalJobs('ws-a', new AbortController().signal);
    expect(customFetchMock).toHaveBeenCalledWith(
      expect.objectContaining({
        url: '/apis/evals/v2/workspaces/ws-a/agent-evaluate/jobs',
      })
    );
  });
});

describe('fetchAgentEvalJob', () => {
  it('returns the job when the platform responds with one', async () => {
    customFetchMock.mockResolvedValueOnce(baseJob({ name: 'eval-42' }));
    const job = await fetchAgentEvalJob('ws-a', 'eval-42', new AbortController().signal);
    expect(job?.name).toBe('eval-42');
  });

  it('returns null on 404', async () => {
    customFetchMock.mockRejectedValueOnce({ response: { status: 404 } });
    const job = await fetchAgentEvalJob('ws-a', 'missing', new AbortController().signal);
    expect(job).toBeNull();
  });
});

describe('agentNameForJob', () => {
  it('reads the agent name from the target', () => {
    expect(agentNameForJob(baseJob())).toBe('support-bot-mini');
  });

  it('returns null when no target agent is set', () => {
    expect(agentNameForJob(baseJob({ spec: {} as AgentEvaluateJob['spec'] }))).toBeNull();
  });
});

describe('evalConfigName', () => {
  it('reads the fileset name from spec.labels.eval_config_fileset', () => {
    const job = baseJob({
      spec: {
        target: { kind: 'agent', agent: { name: 'a' } },
        tasks: [{}],
        labels: { eval_config_fileset: 'wise-blue' },
      } as unknown as AgentEvaluateJob['spec'],
    });
    expect(evalConfigName(job)).toBe('wise-blue');
  });

  it('returns null when labels.eval_config_fileset is absent, ignoring any description', () => {
    expect(evalConfigName(baseJob({ description: 'legacy-desc' }))).toBeNull();
  });
});

describe('fetchAgentEvalResultsForJobs', () => {
  it('returns an empty map without fetching when there are no job names', async () => {
    const map = await fetchAgentEvalResultsForJobs('ws-a', [], new AbortController().signal);
    expect(map.size).toBe(0);
    expect(customFetchMock).not.toHaveBeenCalled();
  });

  it('keys results by name for the requested jobs', async () => {
    customFetchMock.mockResolvedValue({
      data: [
        { name: 'eval-1', scores: { scores: [] } },
        { name: 'eval-2', scores: { scores: [] } },
      ],
    });
    const map = await fetchAgentEvalResultsForJobs(
      'ws-a',
      ['eval-1', 'eval-2'],
      new AbortController().signal
    );
    expect(map.get('eval-1')?.name).toBe('eval-1');
    expect(map.get('eval-2')?.name).toBe('eval-2');
    expect(map.size).toBe(2);
  });
});

describe('aggregateScoresOf', () => {
  it('flattens the nested scores shape', () => {
    const result = {
      scores: {
        scores: [{ name: 'llm-judge.accuracy', count: 5, nan_count: 0, score_type: 'range' }],
      },
    } as AgentEvalResult;
    expect(aggregateScoresOf(result)).toHaveLength(1);
    expect(aggregateScoresOf(null)).toEqual([]);
  });
});

describe('parseBundleRef', () => {
  it('splits "workspace/fileset#inner/path"', () => {
    expect(parseBundleRef('default/job-fileset-x#results/attempt-1/agent-eval-results')).toEqual({
      fileset: 'job-fileset-x',
      innerPath: 'results/attempt-1/agent-eval-results',
    });
  });

  it('returns null without a fragment', () => {
    expect(parseBundleRef('default/job-fileset-x')).toBeNull();
  });
});

describe('joinBundleByTask', () => {
  it('joins tasks, trials, and scores by task id', () => {
    const bundle: AgentEvalBundle = {
      tasks: [
        {
          id: 'A',
          intent: 'classify',
          inputs: { instruction: 'email a' },
          reference: { label: 'phishing' },
        },
      ],
      trials: [
        { id: 't1', task_id: 'A', status: 'completed', output: { output_text: 'phishing' } },
      ],
      scores: [
        {
          id: 's1',
          task_id: 'A',
          trial_id: 't1',
          metric_type: 'llm-judge',
          status: 'completed',
          outputs: [{ name: 'accuracy', value: 1 }],
          diagnostics: [],
        },
      ],
    };
    const [row] = joinBundleByTask(bundle);
    expect(row.taskId).toBe('A');
    expect(row.responseText).toBe('phishing');
    expect(row.instruction).toBe('email a');
    expect(row.reference).toEqual({ label: 'phishing' });
    expect(row.scores).toEqual([{ name: 'llm-judge.accuracy', value: 1 }]);
  });

  it('returns [] for a null bundle', () => {
    expect(joinBundleByTask(null)).toEqual([]);
  });

  it("attaches only the selected trial's scores when a task has multiple trials", () => {
    const bundle: AgentEvalBundle = {
      tasks: [{ id: 'A', inputs: { instruction: 'email a' } }],
      // Two trials for task A; the join keeps the last one (t2).
      trials: [
        { id: 't1', task_id: 'A', status: 'completed', output: { output_text: 'benign' } },
        { id: 't2', task_id: 'A', status: 'completed', output: { output_text: 'phishing' } },
      ],
      scores: [
        {
          id: 's1',
          task_id: 'A',
          trial_id: 't1',
          metric_type: 'llm-judge',
          status: 'completed',
          outputs: [{ name: 'accuracy', value: 0 }],
          diagnostics: ['t1-diag'],
        },
        {
          id: 's2',
          task_id: 'A',
          trial_id: 't2',
          metric_type: 'llm-judge',
          status: 'completed',
          outputs: [{ name: 'accuracy', value: 1 }],
          diagnostics: ['t2-diag'],
        },
      ],
    };
    const [row] = joinBundleByTask(bundle);
    // responseText and scores/diagnostics must come from the same trial (t2).
    expect(row.responseText).toBe('phishing');
    expect(row.scores).toEqual([{ name: 'llm-judge.accuracy', value: 1 }]);
    expect(row.diagnostics).toEqual(['t2-diag']);
  });

  it('normalizes serialized NaN score values for display', () => {
    const bundle: AgentEvalBundle = {
      tasks: [{ id: 'A' }],
      trials: [{ id: 't1', task_id: 'A', status: 'completed' }],
      scores: [
        {
          id: 's1',
          task_id: 'A',
          trial_id: 't1',
          metric_type: 'llm-judge',
          status: 'completed',
          outputs: [{ name: 'accuracy', value: 'NaN' }],
        },
      ],
    };

    expect(joinBundleByTask(bundle)[0].scores).toEqual([
      { name: 'llm-judge.accuracy', value: null },
    ]);
  });
});
