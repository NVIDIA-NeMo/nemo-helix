// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  deriveEvaluatorMetrics,
  LATENCY_METRIC,
  TOKENS_METRIC,
} from '@studio/components/charts/ExperimentParetoChart/utils';
import {
  columnExtremes,
  defaultBaselineName,
  describeDelta,
  DURATION_METRIC,
  scoreDelta,
  scoreHeader,
} from '@studio/components/dataViews/ExperimentComparisonDataView/utils';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';

const evaluation = (opts: {
  name: string;
  createdAt?: string;
  latency?: number;
  tokens?: number;
  durationSec?: string;
  scores?: Record<string, number>;
}): EvaluationRow =>
  ({
    name: opts.name,
    id: opts.name,
    created_at: opts.createdAt,
    latency_ms: opts.latency == null ? undefined : { mean: opts.latency },
    tokens: opts.tokens == null ? undefined : { mean: opts.tokens },
    metadata: opts.durationSec == null ? undefined : { eval_duration_sec: opts.durationSec },
    aggregate_scores: opts.scores
      ? Object.fromEntries(Object.entries(opts.scores).map(([name, mean]) => [name, { mean }]))
      : undefined,
  }) as unknown as EvaluationRow;

const baseline = evaluation({
  name: 'baseline',
  createdAt: '2026-10-11T02:00:00',
  latency: 67_000,
  tokens: 27_000,
  durationSec: '582',
  scores: { judge: 55, check: 0.66 },
});
const tuned = evaluation({
  name: 'tuned',
  createdAt: '2026-10-11T02:30:00',
  latency: 71_000,
  tokens: 33_000,
  durationSec: '457',
  scores: { judge: 89, check: 0.95 },
});
const unscored = evaluation({ name: 'unscored', createdAt: '2026-10-11T03:00:00' });

const judge = (() => {
  const metric = deriveEvaluatorMetrics([baseline]).find((m) => m.id === 'evaluators.judge');
  if (!metric) throw new Error('judge metric not found');
  return metric;
})();

describe('defaultBaselineName', () => {
  it('picks the oldest evaluation regardless of row order', () => {
    expect(defaultBaselineName([unscored, tuned, baseline])).toBe('baseline');
  });

  it('prefers a dated evaluation over one without a created time', () => {
    const undated = evaluation({ name: 'undated' });
    expect(defaultBaselineName([undated, tuned])).toBe('tuned');
  });

  it('is undefined for an empty experiment', () => {
    expect(defaultBaselineName([])).toBeUndefined();
  });
});

describe('scoreDelta', () => {
  it('subtracts the baseline score from the row score', () => {
    expect(scoreDelta(judge, tuned, baseline)).toBe(34);
  });

  it('is undefined for the baseline row itself', () => {
    expect(scoreDelta(judge, baseline, baseline)).toBeUndefined();
  });

  it('is undefined when either side lacks the score', () => {
    expect(scoreDelta(judge, unscored, baseline)).toBeUndefined();
    expect(scoreDelta(judge, tuned, unscored)).toBeUndefined();
  });
});

describe('describeDelta', () => {
  it.each([
    { delta: 34, direction: 'max', text: '+34', tone: 'better' },
    { delta: -0.2081, direction: 'max', text: '-0.208', tone: 'worse' },
    { delta: -500, direction: 'min', text: '-500', tone: 'better' },
    { delta: 500, direction: 'min', text: '+500', tone: 'worse' },
    { delta: -0.0001, direction: 'max', text: '0', tone: 'neutral' },
  ] as const)('formats $delta ($direction) as $text, $tone', ({ delta, direction, text, tone }) => {
    expect(describeDelta(delta, direction)).toEqual({ text, tone });
  });
});

describe('columnExtremes', () => {
  const rows = [baseline, tuned, unscored];

  it('takes the highest score as best and the lowest as worst', () => {
    expect(columnExtremes(rows, judge)).toEqual({ best: 89, worst: 55 });
  });

  it.each([
    { metric: LATENCY_METRIC, best: 67_000, worst: 71_000 },
    { metric: TOKENS_METRIC, best: 27_000, worst: 33_000 },
    { metric: DURATION_METRIC, best: 457_000, worst: 582_000 },
  ])('takes the lowest $metric.id as best and the highest as worst', ({ metric, best, worst }) => {
    expect(columnExtremes(rows, metric)).toEqual({ best, worst });
  });

  it('is undefined when every value ties', () => {
    const zeroLatency = [
      evaluation({ name: 'a', latency: 0 }),
      evaluation({ name: 'b', latency: 0 }),
    ];
    expect(columnExtremes(zeroLatency, LATENCY_METRIC)).toBeUndefined();
  });

  it('is undefined with a single value', () => {
    expect(columnExtremes([tuned, unscored], judge)).toBeUndefined();
  });
});

describe('scoreHeader', () => {
  it.each([
    { id: 'evaluators.string-check.string-check', header: 'String check' },
    { id: 'evaluators.llm-judge.behaviour', header: 'Behaviour (LLM judge)' },
    { id: 'evaluators.exact_match', header: 'Exact match' },
    { id: 'evaluators.llm-judge.Tool-Use', header: 'Tool use (LLM judge)' },
  ])('labels $id as $header', ({ id, header }) => {
    expect(scoreHeader(id)).toBe(header);
  });
});
