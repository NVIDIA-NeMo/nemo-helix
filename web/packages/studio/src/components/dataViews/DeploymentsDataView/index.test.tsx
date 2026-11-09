// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeploymentsDataView } from '@studio/components/dataViews/DeploymentsDataView';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { renderRoute } from '@studio/tests/util/render';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

vi.mock('use-debounce', () => ({
  useDebounce: (value: unknown) => [value, { cancel: () => {}, flush: () => {} }],
}));

const DEPLOYMENTS_URL = `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/deployments`;

const emptyDeploymentsPage = {
  data: [],
  pagination: {
    page: 1,
    page_size: 50,
    current_page_size: 0,
    total_pages: 0,
    total_results: 0,
  },
};

const captureRequests = () => {
  const requestUrls: string[] = [];
  server.use(
    http.get(DEPLOYMENTS_URL, ({ request }) => {
      requestUrls.push(request.url);
      return HttpResponse.json(emptyDeploymentsPage);
    })
  );
  return requestUrls;
};

const renderDataView = (initialEntry: string) =>
  renderRoute(
    <DeploymentsDataView
      workspace="default"
      onDeploymentRowClick={() => {}}
      onRequestDeleteDeployment={() => {}}
    />,
    {
      history: initialEntry,
      routes: [
        {
          path: '/workspaces/:workspace/deployments',
          element: (
            <DeploymentsDataView
              workspace="default"
              onDeploymentRowClick={() => {}}
              onRequestDeleteDeployment={() => {}}
            />
          ),
        },
      ],
    }
  );

describe('DeploymentsDataView', () => {
  afterEach(() => {
    server.resetHandlers();
  });

  it.each([
    { status: 'READY', disabled: false },
    { status: 'DELETING', disabled: true },
    { status: 'DELETED', disabled: true },
  ])(
    'shows only Delete, styled as danger, for a $status deployment',
    async ({ status, disabled }) => {
      const user = userEvent.setup();
      server.use(
        http.get(DEPLOYMENTS_URL, () =>
          HttpResponse.json({
            ...emptyDeploymentsPage,
            data: [{ name: 'my-deployment', workspace: 'default', status }],
            pagination: {
              ...emptyDeploymentsPage.pagination,
              current_page_size: 1,
              total_results: 1,
            },
          })
        )
      );
      renderDataView('/workspaces/default/deployments');

      await screen.findByText('my-deployment');
      await user.click(screen.getByRole('button', { name: 'Open quick actions menu' }));

      const items = await screen.findAllByTestId('quick-actions-menu-item');
      expect(items.map((item) => item.textContent)).toEqual(['Delete']);
      expect(items[0]).toHaveClass('nv-menu-item--danger');
      expect(items[0]).toHaveProperty('disabled', disabled);
    }
  );

  it('sends no filter[ or search params on initial render', async () => {
    const requestUrls = captureRequests();
    renderDataView('/workspaces/default/deployments');

    await waitFor(() => expect(requestUrls.length).toBeGreaterThan(0));

    const keys = Array.from(new URL(requestUrls.at(-1)!).searchParams.keys());
    expect(keys.some((k) => k.startsWith('filter['))).toBe(false);
    expect(keys.some((k) => k === 'search' || k.startsWith('search['))).toBe(false);
  });

  it('sends filter[name][$like] when name search is active', async () => {
    const requestUrls = captureRequests();
    renderDataView('/workspaces/default/deployments?s=my-deployment');

    await waitFor(() => {
      expect(requestUrls.some((u) => new URL(u).searchParams.has('filter[name][$like]'))).toBe(
        true
      );
    });

    const url = new URL(requestUrls.at(-1)!);
    expect(url.searchParams.get('filter[name][$like]')).toBe('my-deployment');
    expect(url.searchParams.has('filter[name]')).toBe(false);
  });

  it('never emits a top-level search or search[ key', async () => {
    const requestUrls = captureRequests();
    renderDataView('/workspaces/default/deployments?s=foo');

    await waitFor(() => expect(requestUrls.length).toBeGreaterThan(0));

    for (const raw of requestUrls) {
      const params = new URL(raw).searchParams;
      expect(params.has('search')).toBe(false);
      expect(Array.from(params.keys()).some((k) => k.startsWith('search['))).toBe(false);
    }
  });
});
