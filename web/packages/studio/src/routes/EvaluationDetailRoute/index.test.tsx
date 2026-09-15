// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getInsightsGetInsightQueryKey } from '@nemo/sdk/generated/insights/insights-insights';
import { getGetEvaluationQueryKey } from '@nemo/sdk/generated/platform/evaluations';
import {
  getGetExperimentQueryKey,
  getListEvaluationSessionsQueryKey,
} from '@nemo/sdk/generated/platform/experiments';
import type { EvaluationResponse, ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { ROUTES } from '@studio/constants/routes';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { EvaluationDetailRoute } from '@studio/routes/EvaluationDetailRoute';
import { renderRoute, screen } from '@studio/tests/util/render';
import { http, HttpResponse } from 'msw';

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_OPTIMIZER_ENABLED', 'true');
});

const WORKSPACE = 'test-workspace';
const GROUP_NAME = 'test-group';
const EVALUATION_NAME = 'test-evaluation';

const evaluation = {
  id: 'evaluation-id',
  name: EVALUATION_NAME,
  workspace: WORKSPACE,
  experiment_group_id: 'group-id',
  dataset_name: 'dataset',
  description: 'Evaluation description',
} satisfies Partial<EvaluationResponse>;

const group = {
  id: 'group-id',
  name: GROUP_NAME,
  workspace: WORKSPACE,
  insight_id: 'insight-id',
  default_sort: '-created_at',
  evaluation_count: 0,
} satisfies Partial<ExperimentResponse>;

describe('EvaluationDetailRoute with Optimizer enabled', () => {
  it('renders the originating insight description instead of relabeling the evaluation description', async () => {
    server.use(
      http.get(mockApiUrl(getGetEvaluationQueryKey, ':workspace', ':name'), () =>
        HttpResponse.json(evaluation)
      ),
      http.get(mockApiUrl(getGetExperimentQueryKey, ':workspace', ':name'), () =>
        HttpResponse.json(group)
      ),
      http.get(mockApiUrl(getListEvaluationSessionsQueryKey, ':workspace', ':name'), () =>
        HttpResponse.json({
          data: [],
          pagination: {
            page: 1,
            page_size: 25,
            current_page_size: 0,
            total_pages: 0,
            total_results: 0,
          },
        })
      ),
      http.get(mockApiUrl(getInsightsGetInsightQueryKey, ':workspace', ':insightId'), () =>
        HttpResponse.json({
          id: 'insight-id',
          name: 'insight',
          title: 'Insight',
          description: 'Actual insight description',
          agent: 'agent',
          status: 'open',
          trace_refs: [],
        })
      )
    );

    renderRoute(<EvaluationDetailRoute />, {
      history: `/workspaces/${WORKSPACE}/experiment/${GROUP_NAME}/${EVALUATION_NAME}`,
      routes: [
        {
          path: ROUTES.workspace.evaluationDetail,
          element: <EvaluationDetailRoute />,
        },
      ],
    });

    expect(await screen.findByText('Actual insight description')).toBeInTheDocument();
    expect(screen.getByText('Insight description')).toBeInTheDocument();
    expect(screen.queryByText('Evaluation description')).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: /originating insight/i })).toHaveAttribute(
      'href',
      `/workspaces/${WORKSPACE}/optimizer/insight-id`
    );
  });
});
