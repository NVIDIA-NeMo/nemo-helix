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
  vi.stubEnv('VITE_FF_OPTIMIZER_ENABLED', 'false');
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

describe('EvaluationDetailRoute with Optimizer disabled', () => {
  it('renders the evaluation description without requesting or linking to the insight', async () => {
    const insightRequest = vi.fn();
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
      http.get(mockApiUrl(getInsightsGetInsightQueryKey, ':workspace', ':insightId'), () => {
        insightRequest();
        return HttpResponse.json({});
      })
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

    expect(await screen.findByText('Evaluation description')).toBeInTheDocument();
    expect(insightRequest).not.toHaveBeenCalled();
    expect(screen.queryByText('Insight description')).not.toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /originating insight/i })).not.toBeInTheDocument();
  });
});
