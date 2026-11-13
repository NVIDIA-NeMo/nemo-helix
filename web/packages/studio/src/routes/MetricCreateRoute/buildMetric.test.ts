// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { buildMetric } from '@studio/routes/MetricCreateRoute/buildMetric';
import {
  METRIC_FORM_DEFAULTS,
  type MetricFormValues,
} from '@studio/routes/MetricCreateRoute/types';

const values = (overrides: Partial<MetricFormValues>): MetricFormValues => ({
  ...METRIC_FORM_DEFAULTS,
  name: 'm',
  ...overrides,
});

describe('buildMetric', () => {
  it('builds a reference metric against the canonical reference field', () => {
    const metric = buildMetric(values({ type: 'exact-match' }), 'default');
    expect(metric.metric_type).toBe('exact-match');
    expect(metric.payload).toMatchObject({
      kind: 'inline',
      metric: { type: 'exact-match', reference: '{{reference}}' },
    });
    expect(metric.outputs.map((output) => output.name)).toEqual(['exact-match']);
  });

  it('builds a judge with a chat prompt and an output per rubric score and its label', () => {
    const metric = buildMetric(
      values({ body: { ...METRIC_FORM_DEFAULTS.body, judgeModel: 'judge' } }),
      'default'
    );
    expect(metric.payload).toMatchObject({
      metric: {
        type: 'llm-judge',
        model: 'default/judge',
        prompt_template: { messages: [{ role: 'user' }] },
      },
    });
    expect(metric.outputs.map((output) => output.name)).toEqual([
      'correctness',
      'correctness.label',
    ]);
  });

  it('uses the typed description over the generated one', () => {
    expect(buildMetric(values({ type: 'f1', description: ' Mine ' }), 'w').metadata).toMatchObject({
      description: 'Mine',
    });
    expect(buildMetric(values({ type: 'f1' }), 'w').metadata?.description).toContain('f1');
  });
});
