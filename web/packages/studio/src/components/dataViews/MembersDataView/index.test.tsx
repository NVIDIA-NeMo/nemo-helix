// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_QUERY_RETRY_COUNT } from '@studio/api/queryClient';
import { MembersDataView } from '@studio/components/dataViews/MembersDataView';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { XL_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { menuFor } from '@studio/tests/util/menuFor';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { MemoryRouter } from 'react-router';

const workspace = 'default';
const MEMBERS_URL = `${PLATFORM_BASE_URL}/apis/entities/v2/workspaces/:workspace/members`;

const renderComponent = ({ onRemoveMember = vi.fn() } = {}) =>
  render(
    <MemoryRouter>
      {/* Retry without the exponential backoff, so the retry assertions below stay fast. */}
      <TestProviders
        options={{ queryClientConfig: { defaultOptions: { queries: { retryDelay: 0 } } } }}
      >
        <MembersDataView
          workspace={workspace}
          onAddMember={vi.fn()}
          onEditMember={vi.fn()}
          onRemoveMember={onRemoveMember}
        />
      </TestProviders>
    </MemoryRouter>
  );

/** Serves a 403 and reports how many times the list was requested. */
const mockForbiddenList = () => {
  let requestCount = 0;
  server.use(
    http.get(MEMBERS_URL, () => {
      requestCount += 1;
      return HttpResponse.json({ detail: 'Forbidden' }, { status: 403 });
    })
  );
  return () => requestCount;
};

describe('MembersDataView', () => {
  it('shows Edit Role and Remove, with Remove styled as danger', async () => {
    const user = userEvent.setup();
    const onRemoveMember = vi.fn();
    server.use(
      http.get(MEMBERS_URL, () =>
        HttpResponse.json({
          data: [{ principal: 'alice@example.com', roles: ['Editor'] }],
          pagination: {
            page: 1,
            page_size: 50,
            current_page_size: 1,
            total_pages: 1,
            total_results: 1,
          },
        })
      )
    );
    renderComponent({ onRemoveMember });

    await screen.findByText('alice@example.com', undefined, { timeout: XL_SELECTOR_TIMEOUT });
    const trigger = screen.getByRole('button', { name: 'Open quick actions menu' });
    await user.click(trigger);

    const menu = menuFor(trigger);
    const items = within(menu).getAllByTestId('quick-actions-menu-item');
    expect(items.map((item) => item.textContent)).toEqual(['Edit Role', 'Remove']);
    expect(items.at(-1)).toHaveClass('nv-menu-item--danger');

    await user.click(within(menu).getByRole('menuitem', { name: 'Remove' }));
    expect(onRemoveMember).toHaveBeenCalledWith(
      expect.objectContaining({ principal: 'alice@example.com' })
    );
  });

  describe('Forbidden', () => {
    it('explains the missing permission instead of showing a generic error', async () => {
      mockForbiddenList();

      renderComponent();

      expect(
        await screen.findByText("You don't have permission to view members", undefined, {
          timeout: XL_SELECTOR_TIMEOUT,
        })
      ).toBeInTheDocument();
      expect(screen.getByText(/Contact your workspace administrator/)).toBeInTheDocument();
    });

    it('does not offer a refresh call to action', async () => {
      mockForbiddenList();

      renderComponent();

      await screen.findByText("You don't have permission to view members", undefined, {
        timeout: XL_SELECTOR_TIMEOUT,
      });

      expect(screen.queryByRole('button', { name: 'Refresh Page' })).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: 'Go Back' })).not.toBeInTheDocument();
    });

    it('does not retry the request', async () => {
      const getRequestCount = mockForbiddenList();

      renderComponent();

      await screen.findByText("You don't have permission to view members", undefined, {
        timeout: XL_SELECTOR_TIMEOUT,
      });

      // Give a retry a chance to land before asserting it never happened.
      await waitFor(() => expect(getRequestCount()).toBe(1));
      expect(getRequestCount()).toBe(1);
    });
  });

  describe('Other failures', () => {
    it('still retries a server error', async () => {
      let requestCount = 0;
      server.use(
        http.get(MEMBERS_URL, () => {
          requestCount += 1;
          return HttpResponse.json({ detail: 'Boom' }, { status: 500 });
        })
      );

      renderComponent();

      // The full budget, so dropping back to a single retry fails here.
      await waitFor(() => expect(requestCount).toBe(DEFAULT_QUERY_RETRY_COUNT + 1), {
        timeout: XL_SELECTOR_TIMEOUT,
      });
      expect(
        screen.queryByText("You don't have permission to view members")
      ).not.toBeInTheDocument();
    });
  });
});
