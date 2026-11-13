// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MetricsDataView } from '@studio/components/dataViews/MetricsDataView';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { getMetricsRoute } from '@studio/routes/utils';
import { XL_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

vi.mock('@studio/api/metrics/useMetricScoreHistory', () => ({
  METRIC_HISTORY_LIMIT: 200,
  useMetricScoreHistory: () => ({ results: [], isPending: false }),
}));

const metric = (name: string, metric_type: string) => ({
  id: name,
  name,
  workspace: 'default',
  metric_type,
  outputs: [],
  secrets: {},
  payload_kind: 'inline',
  payload_digest: '',
  bundle_ref: '',
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
});

describe('MetricsDataView', () => {
  it('toggles a type filter when its tag is clicked', async () => {
    const filters: (string | null)[] = [];
    server.use(
      http.get(`${PLATFORM_BASE_URL}/apis/evals/v2/workspaces/default/metrics`, ({ request }) => {
        filters.push(new URL(request.url).searchParams.get('filter'));
        return HttpResponse.json({
          data: [metric('judge', 'llm-judge'), metric('exact', 'exact-match')],
          pagination: {
            page: 1,
            page_size: 50,
            current_page_size: 2,
            total_pages: 1,
            total_results: 2,
          },
        });
      })
    );
    const user = userEvent.setup();
    renderRoute(<MetricsDataView workspace="default" />, {
      history: getMetricsRoute('default'),
      routes: [
        { path: ROUTES.workspace.metrics, element: <MetricsDataView workspace="default" /> },
      ],
    });

    await user.click(await screen.findByRole('button', { name: 'llm-judge' }));
    await waitFor(() =>
      expect(filters.at(-1)).toBe(
        JSON.stringify({ $and: [{ metric_type: { $in: ['llm-judge'] } }] })
      )
    );

    await user.click(screen.getByRole('button', { name: 'llm-judge' }));
    await waitFor(() => expect(filters.at(-1)).toBeNull());

    await user.type(screen.getByPlaceholderText('Search names and descriptions...'), 'ctx');
    await waitFor(
      () =>
        expect(filters.at(-1)).toBe(
          JSON.stringify({
            $and: [{ $or: [{ name: { $like: 'ctx' } }, { description: { $like: 'ctx' } }] }],
          })
        ),
      { timeout: XL_SELECTOR_TIMEOUT }
    );
  });
});
