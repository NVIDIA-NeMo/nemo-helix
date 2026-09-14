// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getInsightsGetInsightQueryKey } from '@nemo/sdk/generated/insights/insights-insights';
import { getListEvaluationsQueryKey } from '@nemo/sdk/generated/platform/evaluations';
import { getGetExperimentQueryKey } from '@nemo/sdk/generated/platform/experiments';
import type { ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { ROUTES } from '@studio/constants/routes';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { ExperimentDetailRoute } from '@studio/routes/ExperimentDetailRoute';
import { renderRoute, screen } from '@studio/tests/util/render';
import { waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_OPTIMIZER_ENABLED', 'false');
});

const WORKSPACE = 'test-workspace';
const GROUP_NAME = 'test-group';

const group = {
  id: 'group-id',
  name: GROUP_NAME,
  workspace: WORKSPACE,
  description: 'Editable group description',
  summary: 'Generated group summary',
  insight_id: 'insight-id',
  default_sort: '-created_at',
  evaluation_count: 0,
} satisfies Partial<ExperimentResponse>;

const mockGroup = (overrides?: Partial<ExperimentResponse>) => {
  const insightRequest = vi.fn();
  server.use(
    http.get(mockApiUrl(getGetExperimentQueryKey, ':workspace', ':name'), () =>
      HttpResponse.json({ ...group, ...overrides })
    ),
    http.get(mockApiUrl(getListEvaluationsQueryKey, ':workspace'), () =>
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

  renderRoute(<ExperimentDetailRoute />, {
    history: `/workspaces/${WORKSPACE}/experiment/${GROUP_NAME}`,
    routes: [
      {
        path: ROUTES.workspace.experimentDetail,
        element: <ExperimentDetailRoute />,
      },
    ],
  });

  return insightRequest;
};

describe('ExperimentDetailRoute', () => {
  it('renders the group description and summary without requesting Optimizer when disabled', async () => {
    const insightRequest = mockGroup();

    expect(await screen.findByText('Editable group description')).toBeInTheDocument();
    expect(screen.getByText('Generated group summary')).toBeInTheDocument();
    expect(insightRequest).not.toHaveBeenCalled();
    expect(screen.queryByRole('link', { name: /originating insight/i })).not.toBeInTheDocument();
  });

  it('omits the summary panel for groups no producer has summarized', async () => {
    mockGroup({ summary: undefined });

    expect(await screen.findByText('Editable group description')).toBeInTheDocument();
    expect(screen.queryByText('Summary')).not.toBeInTheDocument();
  });

  describe('over-time chart visibility', () => {
    const trendToggle = () => screen.getByRole('button', { name: /over time/i });

    beforeEach(() => {
      window.localStorage.clear();
    });

    it('shows the chart unasked when the experiment is flagged to evaluate over time', async () => {
      mockGroup({ show_evaluations_over_time: true });

      expect(await screen.findByText('Editable group description')).toBeInTheDocument();
      // aria-pressed is the toggle's own view of whether the chart is showing.
      expect(trendToggle()).toHaveAttribute('aria-pressed', 'true');
    });

    it('hides the chart on an unflagged experiment', async () => {
      mockGroup({ show_evaluations_over_time: false });

      expect(await screen.findByText('Editable group description')).toBeInTheDocument();
      expect(trendToggle()).toHaveAttribute('aria-pressed', 'false');
    });

    it("lets a viewer's stored choice override the flag it was made against", async () => {
      window.localStorage.setItem(
        `nemo-studio:experiment-trend:${group.id}`,
        JSON.stringify({ visible: false, flag: true })
      );
      mockGroup({ show_evaluations_over_time: true });

      expect(await screen.findByText('Editable group description')).toBeInTheDocument();
      expect(trendToggle()).toHaveAttribute('aria-pressed', 'false');
    });

    it('deletes a stored choice the flag has moved under, so a round trip cannot revive it', async () => {
      const key = `nemo-studio:experiment-trend:${group.id}`;
      // Chosen while the flag was off; the owner has since turned it on.
      window.localStorage.setItem(key, JSON.stringify({ visible: true, flag: false }));
      mockGroup({ show_evaluations_over_time: true });

      expect(await screen.findByText('Editable group description')).toBeInTheDocument();
      // Merely ignoring it would leave it to apply again the next time the flag reads false.
      await waitFor(() => expect(window.localStorage.getItem(key)).toBeNull());
    });

    it('discards a value left in the older bare-boolean format', async () => {
      const key = `nemo-studio:experiment-trend:${group.id}`;
      window.localStorage.setItem(key, 'false');
      mockGroup({ show_evaluations_over_time: true });

      expect(await screen.findByText('Editable group description')).toBeInTheDocument();
      await waitFor(() => expect(window.localStorage.getItem(key)).toBeNull());
      expect(trendToggle()).toHaveAttribute('aria-pressed', 'true');
    });

    it('retires a stored choice once the flag has moved under it', async () => {
      // The viewer hid the chart while the flag was off; the owner has since turned it on.
      window.localStorage.setItem(
        `nemo-studio:experiment-trend:${group.id}`,
        JSON.stringify({ visible: false, flag: false })
      );
      mockGroup({ show_evaluations_over_time: true });

      expect(await screen.findByText('Editable group description')).toBeInTheDocument();
      expect(trendToggle()).toHaveAttribute('aria-pressed', 'true');
    });
  });
});
