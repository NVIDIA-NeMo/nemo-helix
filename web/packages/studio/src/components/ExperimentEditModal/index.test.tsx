// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getListEvaluationsQueryKey } from '@nemo/sdk/generated/platform/evaluations';
import { getGetExperimentQueryKey } from '@nemo/sdk/generated/platform/experiments';
import type { ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { ExperimentEditModal } from '@studio/components/ExperimentEditModal';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { render, screen } from '@studio/tests/util/render';
import { waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const WORKSPACE = 'test-workspace';

const group: ExperimentResponse = {
  id: 'group-id',
  name: 'opt-group',
  workspace: WORKSPACE,
  description: 'Editable group description',
  summary: 'Optimizer-authored summary',
  insight_id: 'insight-id',
  metadata: { producer: 'optimizer' },
  default_sort: '-created_at',
  evaluation_count: 0,
};

describe('ExperimentEditModal', () => {
  it('resends the fields it does not edit, which a full-replace update would otherwise clear', async () => {
    let body: Record<string, unknown> | undefined;
    server.use(
      http.get(mockApiUrl(getListEvaluationsQueryKey, ':workspace'), () =>
        HttpResponse.json({
          data: [],
          pagination: {
            page: 1,
            page_size: 100,
            current_page_size: 0,
            total_pages: 0,
            total_results: 0,
          },
        })
      ),
      http.put(mockApiUrl(getGetExperimentQueryKey, ':workspace', ':name'), async ({ request }) => {
        body = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(group);
      })
    );

    render(<ExperimentEditModal open onClose={vi.fn()} workspace={WORKSPACE} group={group} />);

    await userEvent.click(await screen.findByRole('button', { name: 'Save' }));

    await waitFor(() => expect(body).toBeDefined());
    expect(body).toMatchObject({
      name: group.name,
      description: group.description,
      default_sort: group.default_sort,
      insight_id: group.insight_id,
      summary: group.summary,
      metadata: group.metadata,
    });
  });

  describe('the chart choice a viewer stored', () => {
    const TREND_KEY = `nemo-studio:experiment-trend:${group.id}`;

    const mockRequests = () => {
      let saved = false;
      server.use(
        http.get(mockApiUrl(getListEvaluationsQueryKey, ':workspace'), () =>
          HttpResponse.json({
            data: [],
            pagination: {
              page: 1,
              page_size: 100,
              current_page_size: 0,
              total_pages: 0,
              total_results: 0,
            },
          })
        ),
        http.put(mockApiUrl(getGetExperimentQueryKey, ':workspace', ':name'), () => {
          saved = true;
          return HttpResponse.json(group);
        })
      );
      return () => saved;
    };

    beforeEach(() => {
      window.localStorage.clear();
    });

    it('drops it when the over-time flag is flipped, so the owner edit is not a no-op', async () => {
      const isSaved = mockRequests();
      // Stamped against the flag's current value, so it would otherwise survive a round trip.
      window.localStorage.setItem(TREND_KEY, JSON.stringify({ visible: false, flag: false }));

      render(<ExperimentEditModal open onClose={vi.fn()} workspace={WORKSPACE} group={group} />);
      await userEvent.click(await screen.findByRole('switch', { name: /evaluate over time/i }));
      await userEvent.click(screen.getByRole('button', { name: 'Save' }));

      await waitFor(() => expect(isSaved()).toBe(true));
      await waitFor(() => expect(window.localStorage.getItem(TREND_KEY)).toBeNull());
    });

    it('keeps it when the over-time flag is left alone', async () => {
      const isSaved = mockRequests();
      const stored = JSON.stringify({ visible: false, flag: false });
      window.localStorage.setItem(TREND_KEY, stored);

      render(<ExperimentEditModal open onClose={vi.fn()} workspace={WORKSPACE} group={group} />);
      // Edit something else; a viewer's chart choice is not this save's business.
      await userEvent.click(await screen.findByRole('switch', { name: /favorite/i }));
      await userEvent.click(screen.getByRole('button', { name: 'Save' }));

      await waitFor(() => expect(isSaved()).toBe(true));
      expect(window.localStorage.getItem(TREND_KEY)).toBe(stored);
    });
  });
});
