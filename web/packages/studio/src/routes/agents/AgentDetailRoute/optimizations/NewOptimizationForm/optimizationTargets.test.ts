// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { optimizationTargets } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/optimizationTargets';
import type { AgentEvaluationRow } from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';

const experiment = (id: string, name: string | null) => ({
  id,
  name,
  description: null,
  isFavorite: false,
  showsEvaluationsOverTime: false,
});

const evaluation = (
  overrides: Partial<AgentEvaluationRow> & Pick<AgentEvaluationRow, 'name' | 'experiments'>
): AgentEvaluationRow =>
  ({
    id: overrides.name,
    workspace: 'default',
    experiment_ids: overrides.experiments.map((e) => e.id),
    dataset_name: 'default/data',
    experiment_group_id: overrides.experiments[0]?.id ?? '',
    ...overrides,
  }) as AgentEvaluationRow;

describe('optimizationTargets', () => {
  it('keeps the most recent evaluation per experiment', () => {
    const targets = optimizationTargets([
      evaluation({
        name: 'older',
        created_at: '2026-08-01T00:00:00Z',
        experiments: [experiment('exp-1', 'quality')],
      }),
      evaluation({
        name: 'newer',
        created_at: '2026-08-20T00:00:00Z',
        experiments: [experiment('exp-1', 'quality')],
      }),
    ]);

    expect(targets).toHaveLength(1);
    expect(targets[0].evaluation.name).toBe('newer');
  });

  it('sorts experiments by recency, so the default pick is the freshest', () => {
    const targets = optimizationTargets([
      evaluation({
        name: 'a',
        created_at: '2026-08-01T00:00:00Z',
        experiments: [experiment('exp-1', 'quality')],
      }),
      evaluation({
        name: 'b',
        created_at: '2026-08-20T00:00:00Z',
        experiments: [experiment('exp-2', 'safety')],
      }),
    ]);

    expect(targets.map((target) => target.experimentId)).toEqual(['exp-2', 'exp-1']);
  });

  it('counts an evaluation toward every experiment it belongs to', () => {
    const targets = optimizationTargets([
      evaluation({
        name: 'shared',
        created_at: '2026-08-20T00:00:00Z',
        experiments: [experiment('exp-1', 'quality'), experiment('exp-2', null)],
      }),
    ]);

    expect(targets.map((target) => target.experimentId).sort()).toEqual(['exp-1', 'exp-2']);
  });

  it('falls back to published score keys when the evaluation names no evaluators', () => {
    const targets = optimizationTargets([
      evaluation({
        name: 'scored',
        created_at: '2026-08-20T00:00:00Z',
        experiments: [experiment('exp-1', 'quality')],
        aggregate_scores: { 'number-check.verdict_match': { mean: 0.9 } },
      }),
    ]);

    expect(targets[0].evaluators).toEqual(['verdict_match']);
  });
});
