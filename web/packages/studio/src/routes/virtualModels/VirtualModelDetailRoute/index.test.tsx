// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { VirtualModelChatTab } from '@studio/routes/virtualModels/VirtualModelChatTab';
import { VirtualModelDetailRoute } from '@studio/routes/virtualModels/VirtualModelDetailRoute';
import { VirtualModelDetailsTab } from '@studio/routes/virtualModels/VirtualModelDetailsTab';
import { renderRoute, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse, delay } from 'msw';
import { Navigate } from 'react-router';

const VM_DETAIL_URL = `${PLATFORM_BASE_URL}/apis/inference-gateway/v2/workspaces/:workspace/virtual-models/:name`;

const sampleVm = {
  id: 'default/my-vm',
  name: 'my-vm',
  workspace: 'default',
  default_model_entity: 'default/gpt-4o',
  autoprovisioned: true,
  override_proxy: 'example-plugin.my-proxy',
  models: [{ model: 'default/gpt-4o', backend_format: null }],
  request_middleware: [
    { name: 'nemo-switchyard', config_type: 'translate', config: { target_format: 'anthropic' } },
  ],
  response_middleware: [],
  post_response_middleware: [],
  created_at: '2026-07-01T00:00:00Z',
  created_by: null,
  updated_at: '2026-07-01T00:00:00Z',
  updated_by: null,
  entity_id: 'default/my-vm',
  parent: '',
  db_version: 1,
};

const renderDetailAt = (entry: string) =>
  renderRoute(<VirtualModelDetailRoute />, {
    history: entry,
    routes: [
      {
        path: '/workspaces/:workspace/virtual-models/:virtualModelName',
        element: <VirtualModelDetailRoute />,
        errorElement: <RouteErrorPanel />,
        children: [
          { index: true, element: <Navigate to="details" replace /> },
          {
            path: '/workspaces/:workspace/virtual-models/:virtualModelName/details',
            element: <VirtualModelDetailsTab />,
          },
          {
            path: '/workspaces/:workspace/virtual-models/:virtualModelName/chat',
            element: <VirtualModelChatTab />,
          },
        ],
      },
    ],
  });

describe('VirtualModelDetailRoute', () => {
  afterEach(() => {
    server.resetHandlers();
  });

  it('renders the heading and both tabs from the path before the model loads', async () => {
    server.use(
      http.get(VM_DETAIL_URL, async () => {
        await delay('infinite');
        return HttpResponse.json(sampleVm);
      })
    );
    renderDetailAt('/workspaces/default/virtual-models/my-vm/details');

    expect(await screen.findByText('my-vm')).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Details' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Chat' })).toBeInTheDocument();
    expect(screen.getByTestId('spinner')).toBeInTheDocument();
  });

  it('renders core fields and middleware config read-only', async () => {
    server.use(http.get(VM_DETAIL_URL, () => HttpResponse.json(sampleVm)));
    renderDetailAt('/workspaces/default/virtual-models/my-vm/details');

    expect(await screen.findByText('example-plugin.my-proxy')).toBeInTheDocument();
    expect(screen.getAllByText('default/gpt-4o').length).toBeGreaterThan(0);
    expect(screen.getByText('nemo-switchyard')).toBeInTheDocument();
    expect(screen.getByText('translate')).toBeInTheDocument();
    expect(screen.getByText(/target_format/)).toBeInTheDocument();
  });

  it('shows "None" for empty middleware pipelines', async () => {
    server.use(http.get(VM_DETAIL_URL, () => HttpResponse.json(sampleVm)));
    renderDetailAt('/workspaces/default/virtual-models/my-vm/details');

    expect(await screen.findByText('Post-response')).toBeInTheDocument();
    expect(screen.getAllByText('None').length).toBeGreaterThan(0);
  });

  it('defers a missing virtual model to the route error panel', async () => {
    server.use(http.get(VM_DETAIL_URL, () => new HttpResponse(null, { status: 404 })));
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
    renderDetailAt('/workspaces/default/virtual-models/missing-vm/details');

    expect(await screen.findByText('Not Found')).toBeInTheDocument();
    consoleError.mockRestore();
  });

  it('renders the chat tab without requesting the virtual model', async () => {
    const user = userEvent.setup();
    let detailRequests = 0;
    server.use(
      http.get(VM_DETAIL_URL, () => {
        detailRequests += 1;
        return HttpResponse.json(sampleVm);
      })
    );
    renderDetailAt('/workspaces/default/virtual-models/my-vm/chat');

    expect(await screen.findByRole('textbox', { name: 'Task prompt' })).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Message my-vm')).toBeInTheDocument();
    expect(detailRequests).toBe(0);

    await user.click(screen.getByRole('button', { name: 'Inference parameters' }));
    expect(screen.getByRole('spinbutton', { name: 'Max tokens value' })).toHaveValue(4096);
  });

  it('redirects the bare detail path to the details tab', async () => {
    server.use(http.get(VM_DETAIL_URL, () => HttpResponse.json(sampleVm)));
    renderDetailAt('/workspaces/default/virtual-models/my-vm');

    expect(await screen.findByText('example-plugin.my-proxy')).toBeInTheDocument();
  });
});
