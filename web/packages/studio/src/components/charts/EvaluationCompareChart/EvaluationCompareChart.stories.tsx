// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  EvaluationSessionResponsesPage,
  EvaluationSessionResponse,
} from '@nemo/sdk/generated/platform/schema';
import type { Meta, StoryObj } from '@storybook/react';
import { EvaluationCompareChart } from '@studio/components/charts/EvaluationCompareChart';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { http, HttpResponse } from 'msw';

const SESSIONS_API = `${PLATFORM_BASE_URL}/apis/intake/v2/workspaces/:workspace/evaluations/:name/sessions`;

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

const sessionsFor = (evaluationName: string): EvaluationSessionResponse[] => {
  const lean = LEAN[evaluationName];
  return Array.from({ length: TEST_CASE_COUNT }, (_, index) => {
    const score = (evaluator: string, base: number) => {
      if (!lean) return base;
      const noise = [-0.25, 0, 0.25][Math.floor(index / 3) % 3] ?? 0;
      const shift = index % 3 === 0 ? (lean[evaluator] ?? 0) : 0;
      return Math.min(1, Math.max(0, base + noise + shift));
    };
    return {
      workspace: 'default',
      evaluation_name: evaluationName,
      session_id: `${evaluationName}-${index}`,
      test_case_name: `case-${index}`,
      trace_id: `trace-${evaluationName}-${index}`,
      root_span_id: `span-${index}`,
      started_at: '2026-10-01T09:00:00Z',
      status: 'success',
      evaluator_scores: {
        'correctness.score': score('correctness.score', index % 3 === 0 ? 0.5 : 0.75),
        'helpfulness.score': score('helpfulness.score', 0.5),
        'tool_use.score': score('tool_use.score', index % 2 === 0 ? 0.5 : 0.75),
      },
    } as EvaluationSessionResponse;
  });
};

const sessionsHandler = http.get(SESSIONS_API, ({ params }) => {
  const data = sessionsFor(String(params.name));
  const page: EvaluationSessionResponsesPage = {
    data,
    pagination: {
      page: 1,
      page_size: 1000,
      current_page_size: data.length,
      total_pages: 1,
      total_results: data.length,
    },
  } as EvaluationSessionResponsesPage;
  return HttpResponse.json(page);
});

const meta: Meta<typeof EvaluationCompareChart> = {
  title: 'Charts/EvaluationCompareChart',
  component: EvaluationCompareChart,
  args: { workspace: 'default', evaluations: EVALUATIONS, onClose: () => {} },
  parameters: { msw: { handlers: [sessionsHandler] } },
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
