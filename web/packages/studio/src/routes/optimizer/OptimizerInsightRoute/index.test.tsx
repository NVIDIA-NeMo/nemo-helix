// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import { getInsightsGetInsightQueryKey } from '@nemo/sdk/generated/insights/insights-insights';
import type { Insight } from '@nemo/sdk/generated/insights/schema';
import { getListEvaluationsQueryKey } from '@nemo/sdk/generated/platform/evaluations';
import { getListExperimentsQueryKey } from '@nemo/sdk/generated/platform/experiments';
import type { ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { ROUTES } from '@studio/constants/routes';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { OptimizerInsightRoute } from '@studio/routes/optimizer/OptimizerInsightRoute';
import { getOptimizerInsightRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { useParams } from 'react-router';

const WORKSPACE = 'workspace-a';
const INSIGHT_ID = 'insight-a';
const INSIGHT_URL = mockApiUrl(getInsightsGetInsightQueryKey, ':workspace', ':insightId');
const GROUPS_URL = mockApiUrl(getListExperimentsQueryKey, ':workspace');
const EVALUATIONS_URL = mockApiUrl(getListEvaluationsQueryKey, ':workspace');

const insight: Insight = {
  id: INSIGHT_ID,
  entity_id: INSIGHT_ID,
  parent: `ws-${WORKSPACE}`,
  db_version: 1,
  name: INSIGHT_ID,
  workspace: WORKSPACE,
  title: 'Slow responses',
  description: 'The agent responds too slowly.',
  agent: 'research-agent',
  status: 'open',
  trace_refs: [],
  created_at: '2026-07-19T12:00:00Z',
  created_by: 'user@example.com',
  updated_at: '2026-07-19T12:00:00Z',
  updated_by: 'user@example.com',
};

const pagination = ({
  page = 1,
  pageSize = 10,
  currentPageSize = 0,
  totalResults = 0,
}: {
  page?: number;
  pageSize?: number;
  currentPageSize?: number;
  totalResults?: number;
} = {}) => ({
  page,
  page_size: pageSize,
  current_page_size: currentPageSize,
  total_pages: Math.ceil(totalResults / pageSize),
  total_results: totalResults,
});

const makeGroup = (id: string, overrides: Partial<ExperimentResponse> = {}): ExperimentResponse =>
  ({
    id,
    name: `${id}-name`,
    workspace: WORKSPACE,
    insight_id: insight.id,
    default_sort: '-created_at',
    summary: `${id} summary`,
    description: `${id} description`,
    evaluation_count: 3,
    created_at: '2026-07-19T12:00:00Z',
    updated_at: '2026-07-20T12:00:00Z',
    ...overrides,
  }) as ExperimentResponse;

const ExperimentDestination = () => {
  const { experimentName } = useParams();
  return <div>{`Opened ${experimentName}`}</div>;
};

const renderInsight = (history = getOptimizerInsightRoute(WORKSPACE, INSIGHT_ID)) =>
  renderRoute(undefined, {
    history,
    routes: [
      { path: ROUTES.workspace.optimizerInsight, element: <OptimizerInsightRoute /> },
      {
        path: ROUTES.workspace.experimentDetail,
        element: <ExperimentDestination />,
      },
    ],
  });

describe('OptimizerInsightRoute experiments', () => {
  beforeEach(() => {
    server.use(
      http.get(INSIGHT_URL, () => HttpResponse.json(insight)),
      http.get(GROUPS_URL, () => HttpResponse.json({ data: [], pagination: pagination() }))
    );
  });

  it('renders a compact Experiments list without requesting Evaluations', async () => {
    const group = makeGroup('latency-experiment');
    const evaluationRequest = vi.fn(() => HttpResponse.json({}));
    server.use(
      http.get(GROUPS_URL, ({ request }) => {
        const params = new URL(request.url).searchParams;
        expect(params.get('filter[insight_id]')).toBe(INSIGHT_ID);
        return HttpResponse.json({
          data: [group],
          pagination: pagination({ currentPageSize: 1, totalResults: 1 }),
        });
      }),
      http.get(EVALUATIONS_URL, evaluationRequest)
    );

    renderInsight();

    const row = await screen.findByRole('row', { name: new RegExp(group.name) });
    expect(within(row).getByText(String(group.evaluation_count))).toBeInTheDocument();
    expect(evaluationRequest).not.toHaveBeenCalled();
  });

  it('distinguishes a group-list failure from a successful empty page', async () => {
    server.use(http.get(GROUPS_URL, () => new HttpResponse(null, { status: 500 })));
    const { unmount } = renderInsight();

    expect(await screen.findByText('Failed to load experiments')).toBeInTheDocument();
    expect(
      screen.queryByText(ENTITY_EMPTY_STATES.insightExperiments.subheading)
    ).not.toBeInTheDocument();
    unmount();

    server.use(
      http.get(GROUPS_URL, () => HttpResponse.json({ data: [], pagination: pagination() }))
    );
    renderInsight();

    expect(
      await screen.findByText(ENTITY_EMPTY_STATES.insightExperiments.subheading)
    ).toBeInTheDocument();
    expect(screen.queryByText('Failed to load experiments')).not.toBeInTheDocument();
  });

  it('requests and displays only the selected Experiment page', async () => {
    const user = userEvent.setup();
    const firstPageGroups = Array.from({ length: 10 }, (_unused, index) =>
      makeGroup(`page-one-${index + 1}`)
    );
    const secondPageGroup = makeGroup('page-two-1');
    const requests: Array<{ page: string | null; pageSize: string | null }> = [];

    server.use(
      http.get(GROUPS_URL, ({ request }) => {
        const params = new URL(request.url).searchParams;
        const page = params.get('page');
        requests.push({ page, pageSize: params.get('page_size') });
        return HttpResponse.json({
          data: page === '2' ? [secondPageGroup] : firstPageGroups,
          pagination: pagination({
            page: Number(page ?? 1),
            currentPageSize: page === '2' ? 1 : 10,
            totalResults: 11,
          }),
        });
      })
    );

    renderInsight();

    expect(await screen.findByText(firstPageGroups[0].name)).toBeInTheDocument();
    expect(screen.queryByText(secondPageGroup.name)).not.toBeInTheDocument();

    const nextPageButton = screen
      .getAllByRole('button', { name: /next page/i })
      .find((button) => !button.hasAttribute('disabled'));
    if (!nextPageButton) throw new Error('Enabled Experiments next-page button not found');
    await user.click(nextPageButton);

    expect(await screen.findByText(secondPageGroup.name)).toBeInTheDocument();
    expect(screen.queryByText(firstPageGroups[0].name)).not.toBeInTheDocument();
    await waitFor(() =>
      expect(requests).toEqual([
        { page: '1', pageSize: '10' },
        { page: '2', pageSize: '10' },
      ])
    );

    await user.click(screen.getByText(secondPageGroup.name));
    expect(await screen.findByText(`Opened ${secondPageGroup.name}`)).toBeInTheDocument();
  });

  it('reports status mutation failures through the Studio toast', async () => {
    server.use(http.patch(INSIGHT_URL, () => new HttpResponse(null, { status: 500 })));
    renderInsight();

    await userEvent.click(await screen.findByRole('button', { name: 'Delete' }));

    expect(await screen.findByText('Failed to update insight.')).toBeInTheDocument();
  });
});
