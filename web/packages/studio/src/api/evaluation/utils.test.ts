// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getMetricsAsList, targetNameForEvalJob, toEvalJobRow } from '@studio/api/evaluation/utils';
import { makeEvaluatorJob } from '@studio/mocks/evaluation/jobs';

describe('registered runner jobs', () => {
  it.each(['fabric', 'harbor', 'gym'])(
    'joins %s jobs by publication rather than runner name',
    (kind) => {
      const job = makeEvaluatorJob({
        spec: {
          target: { kind, name: 'unrelated-runner', source: { agent: 'my-agent' } },
          publication: { intake: { evaluation_id: 'published-eval' } },
        },
      });
      expect(targetNameForEvalJob(job)).toBe('my-agent');
      expect(toEvalJobRow(job)).toMatchObject({
        agentName: 'my-agent',
        evaluationName: 'published-eval',
        status: 'completed',
      });
    }
  );
});

describe('getMetricsAsList', () => {
  it('returns empty array for undefined tasks', () => {
    expect(getMetricsAsList(undefined)).toEqual([]);
  });

  it('returns empty array for empty tasks', () => {
    expect(getMetricsAsList({})).toEqual([]);
  });

  it('extracts metrics from nested structure', () => {
    const tasks = {
      task1: {
        metrics: {
          accuracy: {
            scores: {
              overall: { value: 0.95 },
            },
          },
        },
      },
    };
    const result = getMetricsAsList(tasks);
    expect(result).toEqual([{ task: 'task1', metric: 'accuracy', key: 'overall', value: '0.95' }]);
  });

  it('truncates values to 5 chars', () => {
    const tasks = {
      t: {
        metrics: {
          m: {
            scores: {
              s: { value: 0.123456789 },
            },
          },
        },
      },
    };
    const result = getMetricsAsList(tasks);
    expect(result[0].value).toBe('0.123');
  });

  it('skips entries with null/undefined values', () => {
    const tasks = {
      t: {
        metrics: {
          m: {
            scores: {
              s1: { value: null },
              s2: { value: undefined },
              s3: { value: 42 },
            },
          },
        },
      },
    };
    const result = getMetricsAsList(tasks);
    expect(result).toHaveLength(1);
    expect(result[0].key).toBe('s3');
  });

  it('skips tasks without metrics', () => {
    const tasks = { t: {} };
    expect(getMetricsAsList(tasks as never)).toEqual([]);
  });

  it('skips metrics without scores', () => {
    const tasks = { t: { metrics: { m: {} } } };
    expect(getMetricsAsList(tasks as never)).toEqual([]);
  });
});
