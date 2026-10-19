// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { getInsightsListInsightsQueryKey } from '@nemo/sdk/generated/insights/insights-insights';
import type { InsightListItem } from '@nemo/sdk/generated/insights/schema';
import { ROUTES } from '@studio/constants/routes';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { OptimizerRoute } from '@studio/routes/optimizer/OptimizerRoute';
import { getOptimizerRoute } from '@studio/routes/utils';
import { renderRoute, screen, within } from '@studio/tests/util/render';
import { http, HttpResponse } from 'msw';

const INSIGHTS_URL = mockApiUrl(getInsightsListInsightsQueryKey, ':workspace');
const EXPERIMENTS_URL = '*/apis/intake/v2/workspaces/:workspace/experiments';

const makeInsight = (id: string, title: string): InsightListItem => ({
  id,
  entity_id: id,
  parent: `ws-${DEFAULT_WORKSPACE}`,
  db_version: 1,
  name: id,
  workspace: DEFAULT_WORKSPACE,
  title,
  description: `${title} description`,
  agent: 'research-agent',
  status: 'open',
  trace_refs: ['trace-1'],
  created_at: '2026-07-20T12:00:00Z',
  created_by: 'user@example.com',
  updated_at: '2026-07-20T12:00:00Z',
  updated_by: 'user@example.com',
});

const insightsPage = (data: InsightListItem[]) => ({
  data,
  pagination: {
    page: 1,
    page_size: 50,
    current_page_size: data.length,
    total_pages: 1,
    total_results: data.length,
  },
});

const renderList = () =>
  renderRoute(undefined, {
    history: getOptimizerRoute(DEFAULT_WORKSPACE),
    routes: [{ path: ROUTES.workspace.optimizer, element: <OptimizerRoute /> }],
  });

const findCell = async (insightTitle: string, columnName: string): Promise<HTMLElement> => {
  const row = await screen.findByRole('row', { name: new RegExp(insightTitle) });

  const headers = screen.getAllByRole('columnheader');
  const column = headers.findIndex((header) => header.textContent?.includes(columnName));
  if (column < 0) throw new Error(`${columnName} column not found`);

  return within(row).getAllByRole('cell')[column];
};

describe('OptimizerRoute', () => {
  it('renders server-provided list metadata without per-row requests', async () => {
    const experimentRequest = vi.fn(() => HttpResponse.json({}));
    const insights = [
      {
        ...makeInsight('positive', 'Positive count'),
        experiment_group_count: 7,
        last_seen_at: '2026-07-21T12:00:00Z',
      },
      { ...makeInsight('zero', 'Zero count'), experiment_group_count: 0 },
      makeInsight('null', 'Null count'),
    ];
    server.use(
      http.get(INSIGHTS_URL, () => HttpResponse.json(insightsPage(insights))),
      http.get(EXPERIMENTS_URL, experimentRequest)
    );

    renderList();

    expect(await findCell('Positive count', 'Experiments')).toHaveTextContent('7');
    expect(await findCell('Zero count', 'Experiments')).toHaveTextContent('0');
    expect(await findCell('Null count', 'Experiments')).toHaveTextContent('—');
    expect((await findCell('Positive count', 'Last Seen')).querySelector('time')).toHaveAttribute(
      'datetime',
      '2026-07-21T12:00:00Z'
    );
    expect(await findCell('Zero count', 'Last Seen')).toHaveTextContent('—');
    expect(experimentRequest).not.toHaveBeenCalled();
  });
});
