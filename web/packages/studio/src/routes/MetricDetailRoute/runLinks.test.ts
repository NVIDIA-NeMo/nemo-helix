// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { MetricRun } from '@studio/api/metrics/metricScores';
import { runHref, targetHref } from '@studio/routes/MetricDetailRoute/runLinks';
import {
  getAgentDetailRoute,
  getAgentEvaluationDetailRoute,
  getEvaluationResultDetailsRoute,
  getModelDetailRoute,
} from '@studio/routes/utils';

const run = (overrides: Partial<MetricRun>): MetricRun => ({
  name: 'run',
  kind: 'dataset',
  jobId: 'job',
  createdAt: '2026-11-01T00:00:00Z',
  scores: {},
  ...overrides,
});

describe('runHref', () => {
  it('links each kind to its own results page by job', () => {
    expect(runHref('w', run({}))).toBe(getEvaluationResultDetailsRoute('w', 'job'));
    expect(runHref('w', run({ kind: 'agent', name: 'taskset-run' }))).toBe(
      getAgentEvaluationDetailRoute('w', 'job')
    );
  });
});

describe('targetHref', () => {
  it.each([
    ['agent', 'a', getAgentDetailRoute('w', 'a')],
    ['fabric', 'a', getAgentDetailRoute('w', 'a')],
    ['model', 'w/llama', getModelDetailRoute('w', 'llama')],
    ['openai-runner', 'gpt', undefined],
    [undefined, 'x', undefined],
  ])('maps a %s target', (targetKind, targetName, expected) => {
    expect(targetHref('w', run({ targetKind, targetName }))).toBe(expected);
  });
});
