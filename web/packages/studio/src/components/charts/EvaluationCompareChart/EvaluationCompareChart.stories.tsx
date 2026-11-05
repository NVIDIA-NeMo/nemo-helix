// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EvaluationCompareResponse } from '@nemo/sdk/generated/platform/schema';
import type { Meta, StoryObj } from '@storybook/react';
import { EvaluationCompareChart } from '@studio/components/charts/EvaluationCompareChart';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { http, HttpResponse } from 'msw';

const COMPARE_API = `${PLATFORM_BASE_URL}/apis/intake/v2/workspaces/:workspace/evaluations/compare`;

const TEST_CASE_COUNT = 40;

const evaluation = (
  name: string,
  createdAt: string,
  scores: Record<string, number>,
  cost: number,
  latencyMs: number,
  tokens: number
): EvaluationRow =>
  ({
    id: name,
    name,
    workspace: 'default',
    experiment_ids: ['support-agent'],
    experiment_group_id: 'support-agent',
    dataset_name: 'support-bench',
    created_at: createdAt,
    run_count: TEST_CASE_COUNT,
    test_case_count: TEST_CASE_COUNT,
    aggregate_scores: Object.fromEntries(
      Object.entries(scores).map(([key, mean]) => [key, { mean, count: TEST_CASE_COUNT }])
    ),
    cost_usd: { mean: cost, count: TEST_CASE_COUNT },
    latency_ms: { mean: latencyMs, count: TEST_CASE_COUNT },
    tokens: { mean: tokens, count: TEST_CASE_COUNT },
    metadata: { eval_duration_sec: String(Math.round((latencyMs * TEST_CASE_COUNT) / 4000)) },
  }) as EvaluationRow;

const EVALUATIONS: EvaluationRow[] = [
  evaluation(
    'baseline-llama-70b',
    '2026-10-01T09:00:00Z',
    { 'correctness.score': 0.71, 'helpfulness.score': 0.78, 'tool_use.score': 0.64 },
    0.042,
    5400,
    6100
  ),
  evaluation(
    'prompt-v2',
    '2026-10-04T09:00:00Z',
    { 'correctness.score': 0.79, 'helpfulness.score': 0.83, 'tool_use.score': 0.61 },
    0.046,
    5900,
    6800
  ),
  evaluation(
    'nemotron-routing',
    '2026-10-08T09:00:00Z',
    { 'correctness.score': 0.76, 'helpfulness.score': 0.8, 'tool_use.score': 0.7 },
    0.019,
    3100,
    5200
  ),
];

/** Per-run lean of each evaluator away from the baseline, applied to a deterministic subset of cases. */
const LEAN: Record<string, Record<string, number>> = {
  'prompt-v2': { 'correctness.score': 0.25, 'helpfulness.score': 0.25, 'tool_use.score': -0.25 },
  'nemotron-routing': { 'correctness.score': 0.25, 'tool_use.score': 0.25 },
};

/** Counts and an interval consistent with the per-run lean, so the bars and the summary line agree. */
const compareResponse = (candidate: string, metric: string): EvaluationCompareResponse => {
  const lean = LEAN[candidate]?.[metric] ?? 0;
  const counts =
    lean > 0
      ? { better: 20, same: 10, worse: 10 }
      : lean < 0
        ? { better: 10, same: 10, worse: 20 }
        : { better: 13, same: 14, worse: 13 };
  return {
    workspace: 'default',
    baseline: 'baseline-llama-70b',
    candidate,
    metric: `evaluators.${metric}`,
    higher_is_better: true,
    n_pairs: TEST_CASE_COUNT,
    ...counts,
    mean_paired_difference: lean * 0.1,
    interval_low: lean * 0.1 - 0.05,
    interval_high: lean * 0.1 + 0.05,
    p_candidate_better: 0.5 + lean * 0.4,
  } as EvaluationCompareResponse;
};

const compareHandler = http.get(COMPARE_API, ({ request }) => {
  const url = new URL(request.url);
  const metric = (url.searchParams.get('metric') ?? '').replace(/^evaluators\./, '');
  return HttpResponse.json(compareResponse(url.searchParams.get('candidate') ?? '', metric));
});

const meta: Meta<typeof EvaluationCompareChart> = {
  title: 'Charts/EvaluationCompareChart',
  component: EvaluationCompareChart,
  args: { workspace: 'default', evaluations: EVALUATIONS },
  parameters: { msw: { handlers: [compareHandler] } },
  decorators: [
    (Story) => (
      <div className="max-w-5xl">
        <Story />
      </div>
    ),
  ],
};

export default meta;

type Story = StoryObj<typeof EvaluationCompareChart>;

export const ThreeRuns: Story = {};

export const TwoRuns: Story = {
  args: { evaluations: EVALUATIONS.slice(0, 2) },
};
