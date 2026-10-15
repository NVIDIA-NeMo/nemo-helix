// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ToastProvider } from '@nemo/common/src/providers/toast/ToastProvider';
import type { EvaluateJob } from '@nemo/sdk/generated/evals/schema';
import type { FilesetOutputsPage, ModelEntitysPage } from '@nemo/sdk/generated/platform/schema';
import type { Meta, StoryObj } from '@storybook/react';
import type { MetricItemWithId } from '@studio/components/dataViews/EvaluationMetricsDataView/types';
import { MetricRunSidePanel } from '@studio/components/sidePanels/MetricRunSidePanel';
import { datasets } from '@studio/mocks/datasets';
import { mixedModelEntitysPage } from '@studio/mocks/entity-store/models';
import { http, HttpResponse } from 'msw';
import { Route, Routes } from 'react-router';

const WORKSPACE = 'default';

const mockLlmJudgeMetric = {
  id: 'metric-llm-judge-1',
  name: 'faithfulness',
  workspace: WORKSPACE,
  type: 'llm-judge',
  description: 'Measures whether the generated answer is faithful to the provided context.',
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-01T00:00:00Z',
} as MetricItemWithId;

const mockBLEUMetric = {
  id: 'metric-bleu-1',
  name: 'bleu-score',
  workspace: WORKSPACE,
  type: 'bleu',
  description: 'BLEU score metric for evaluating text generation quality.',
  created_at: '2025-01-02T00:00:00Z',
  updated_at: '2025-01-02T00:00:00Z',
} as MetricItemWithId;

const mockEvaluateJob: EvaluateJob = {
  id: 'eval-job-1',
  name: 'eval-job-1',
  workspace: WORKSPACE,
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T01:00:00Z',
  status: 'completed',
  spec: {
    metrics: [],
    dataset: 'default/test-dataset',
  },
  custom_fields: {},
};

const commonHandlers = [
  http.get<{ workspace: string }, never, ModelEntitysPage>(
    '/apis/models/v2/workspaces/:workspace/models',
    () => HttpResponse.json(mixedModelEntitysPage)
  ),
  http.get<{ workspace: string }, never, FilesetOutputsPage>(
    '/apis/files/v2/workspaces/:workspace/filesets',
    () => HttpResponse.json(datasets as FilesetOutputsPage)
  ),
  http.post<{ workspace: string }, never, EvaluateJob>(
    '/apis/evals/v2/workspaces/:workspace/evaluate/jobs',
    () => HttpResponse.json(mockEvaluateJob)
  ),
];

const meta = {
  component: MetricRunSidePanel,
  title: 'Side Panels/MetricRunSidePanel',
  decorators: [
    (Story) => (
      <ToastProvider>
        <Routes>
          <Route path="/workspaces/:workspace/*" element={<Story />} />
        </Routes>
      </ToastProvider>
    ),
  ],
  parameters: {
    router: { initialPath: `/workspaces/${WORKSPACE}` },
  },
  args: {
    open: true,
    onOpenChange: () => {},
    workspace: WORKSPACE,
    // Render inline so the panel is visible within the Storybook canvas (not via portal).
    attributes: { SidePanel: {} },
  },
} satisfies Meta<typeof MetricRunSidePanel>;

export default meta;
type Story = StoryObj<typeof meta>;

/** Panel opened with a pre-selected LLM Judge metric. Displays an EvalCard instead of a metric selector. */
export const WithLlmJudgeMetric: Story = {
  args: { metric: mockLlmJudgeMetric },
  parameters: { msw: { handlers: commonHandlers } },
};

/** Panel opened with a pre-selected BLEU metric. Shows the type badge in the EvalCard. */
export const WithBLEUMetric: Story = {
  args: { metric: mockBLEUMetric },
  parameters: { msw: { handlers: commonHandlers } },
};

/** Panel opened with no metric — user must pick one from the searchable dropdown. */
export const WithoutMetric: Story = {
  args: { metric: null },
  parameters: { msw: { handlers: commonHandlers } },
};
