// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import { VirtualModelsDataView } from '@studio/components/dataViews/VirtualModelsDataView';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { useLocation } from 'react-router';

vi.mock('use-debounce', () => ({
  useDebounce: (value: unknown) => [value, { cancel: () => {}, flush: () => {} }],
}));

const VMS_URL = `${PLATFORM_BASE_URL}/apis/inference-gateway/v2/workspaces/:workspace/virtual-models`;

const page = (data: unknown[]) => ({
  data,
  pagination: {
    page: 1,
    page_size: 50,
    current_page_size: data.length,
    total_pages: data.length ? 1 : 0,
    total_results: data.length,
  },
});

const sampleVm = {
  id: 'default/my-vm',
  name: 'my-vm',
  workspace: 'default',
  default_model_entity: 'default/gpt-4o',
  autoprovisioned: false,
  request_middleware: [{ name: 'nemo-switchyard', config_type: 'translate' }],
  response_middleware: [{ name: 'nemo-switchyard', config_type: 'random_routing' }],
  post_response_middleware: [],
  created_at: '2026-07-01T00:00:00Z',
  created_by: null,
  updated_at: '2026-07-01T00:00:00Z',
  updated_by: null,
  entity_id: 'default/my-vm',
  parent: '',
};

const LocationProbe = () => {
  const location = useLocation();
  return <output data-testid="location">{location.pathname}</output>;
};

const renderDataViewAt = (entry: string) =>
  renderRoute(<VirtualModelsDataView workspace="default" />, {
    history: entry,
    routes: [
      {
        path: '/workspaces/:workspace/virtual-models',
        element: (
          <>
            <VirtualModelsDataView workspace="default" />
            <LocationProbe />
          </>
        ),
      },
      // Detail pages are exercised in VirtualModelDetailRoute's own suite; here they only need to
      // exist so navigation away from the list resolves to a route.
      {
        path: '/workspaces/:workspace/virtual-models/:virtualModelName/details',
        element: <LocationProbe />,
      },
      {
        path: '/workspaces/:workspace/virtual-models/:virtualModelName/chat',
        element: <LocationProbe />,
      },
    ],
  });

const renderDataView = () => renderDataViewAt('/workspaces/default/virtual-models');

describe('VirtualModelsDataView', () => {
  afterEach(() => {
    server.resetHandlers();
  });

  it('renders virtual models with a middleware call count', async () => {
    server.use(http.get(VMS_URL, () => HttpResponse.json(page([sampleVm]))));
    renderDataView();

    expect(await screen.findByText('my-vm')).toBeInTheDocument();
    expect(screen.getByText('default/gpt-4o')).toBeInTheDocument();
    expect(screen.getByText('2 calls')).toBeInTheDocument();
  });

  it('requests only non-autoprovisioned virtual models', async () => {
    const urls: string[] = [];
    server.use(
      http.get(VMS_URL, ({ request }) => {
        urls.push(request.url);
        return HttpResponse.json(page([sampleVm]));
      })
    );
    renderDataView();

    await waitFor(() => expect(urls.length).toBeGreaterThan(0));
    const params = new URL(urls.at(-1)!).searchParams;
    expect(params.get('exclude_autoprovisioned')).toBe('true');
  });

  it('sends filter[name][$like] when a name search is active', async () => {
    const urls: string[] = [];
    server.use(
      http.get(VMS_URL, ({ request }) => {
        urls.push(request.url);
        return HttpResponse.json(page([sampleVm]));
      })
    );
    renderDataViewAt('/workspaces/default/virtual-models?s=my-vm');

    await waitFor(() =>
      expect(urls.some((u) => new URL(u).searchParams.has('filter[name][$like]'))).toBe(true)
    );
    expect(new URL(urls.at(-1)!).searchParams.get('filter[name][$like]')).toBe('my-vm');
  });

  it('shows an empty state when there are no virtual models', async () => {
    server.use(http.get(VMS_URL, () => HttpResponse.json(page([]))));
    renderDataView();

    expect(await screen.findByText(ENTITY_EMPTY_STATES.virtualModels.heading)).toBeInTheDocument();
  });

  it('deletes a virtual model through the row action', async () => {
    const user = userEvent.setup();
    let deleted = false;
    server.use(
      http.get(VMS_URL, () => HttpResponse.json(page(deleted ? [] : [sampleVm]))),
      http.delete(`${VMS_URL}/:name`, () => {
        deleted = true;
        return new HttpResponse(null, { status: 204 });
      })
    );
    renderDataView();

    await screen.findByText('my-vm');

    await user.click(screen.getByRole('button', { name: 'Open quick actions menu' }));
    const items = await screen.findAllByTestId('quick-actions-menu-item');
    expect(items.map((item) => item.textContent)).toEqual(['View', 'Chat', 'Delete']);
    expect(items.at(-1)).toHaveClass('nv-menu-item--danger');
    await user.click(await screen.findByText('Delete'));
    expect(
      await screen.findByRole('heading', { name: "Delete virtual model 'my-vm'?" })
    ).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Delete Virtual Model' }));

    await waitFor(() => expect(deleted).toBe(true));
  });

  it('navigates to the chat page through the row action', async () => {
    const user = userEvent.setup();
    server.use(http.get(VMS_URL, () => HttpResponse.json(page([sampleVm]))));
    renderDataView();

    await screen.findByText('my-vm');

    await user.click(screen.getByRole('button', { name: 'Open quick actions menu' }));
    await user.click(await screen.findByText('Chat'));

    await waitFor(() =>
      expect(screen.getByTestId('location')).toHaveTextContent(
        '/workspaces/default/virtual-models/my-vm/chat'
      )
    );
  });

  it('navigates to the details page when a row is clicked', async () => {
    const user = userEvent.setup();
    server.use(http.get(VMS_URL, () => HttpResponse.json(page([sampleVm]))));
    renderDataView();

    await user.click(await screen.findByText('my-vm'));

    await waitFor(() =>
      expect(screen.getByTestId('location')).toHaveTextContent(
        '/workspaces/default/virtual-models/my-vm/details'
      )
    );
  });
});
