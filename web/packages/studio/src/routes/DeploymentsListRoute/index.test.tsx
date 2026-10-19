// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { DeploymentsListRoute } from '@studio/routes/DeploymentsListRoute';
import { renderRoute } from '@studio/tests/util/render';
import { act, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const DEPLOYMENT_NAME = 'codellama-70b';
const DETAILS_PATH = `/workspaces/default/deployments/${DEPLOYMENT_NAME}/details`;

const deployment = {
  name: DEPLOYMENT_NAME,
  namespace: 'default',
  status: 'ready',
  config: 'codellama-70b-config',
  config_version: 1,
  created_at: '2026-01-01T00:00:00Z',
};

/**
 * Renders the deployments route deep-linked to the details side panel.
 *
 * Uses the `routes` option deliberately: it builds a data router
 * (`createMemoryRouter` + the `react-router/dom` `RouterProvider`), which is the
 * only path that honors `flushSync`. The declarative `MemoryRouter` fallback in
 * `renderRoute` drops the option entirely.
 */
const renderDetailsPanel = () =>
  renderRoute(undefined, {
    history: DETAILS_PATH,
    routes: [
      { path: ROUTES.workspace.deployments, element: <DeploymentsListRoute /> },
      { path: ROUTES.workspace.deploymentsDeployment, element: <DeploymentsListRoute /> },
    ],
  });

describe('DeploymentsListRoute details side panel', () => {
  beforeEach(() => {
    server.use(
      http.get(
        `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/deployments/${DEPLOYMENT_NAME}`,
        () => HttpResponse.json(deployment)
      ),
      http.get(
        `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/deployment-configs/:config/versions/:version`,
        () => HttpResponse.json({ name: deployment.config, description: 'Test config' })
      )
    );
  });

  afterEach(() => {
    server.resetHandlers();
  });

  it('closes exactly once when dismissed', async () => {
    const user = userEvent.setup();
    renderDetailsPanel();

    const dialog = await screen.findByRole('dialog');
    await waitFor(() => expect(dialog).toHaveAttribute('data-state', 'open'));

    // Count re-shows rather than watching `data-state`: the global test setup stubs
    // `MutationObserver` to a no-op, so attribute-level observation is unavailable.
    // `showModal` is the direct signal — `useDialog`'s layout effect calls it when it
    // observes a stale `open === true` against an already-closed <dialog>, which is
    // exactly the spurious re-open between the two closes.
    const showModalSpy = vi.spyOn(HTMLDialogElement.prototype, 'showModal');

    await user.click(screen.getByRole('button', { name: /close side panel/i }));

    // KUI's animate-out is a real 200ms timeout and the close finalizes after it, so
    // the resulting state updates land outside React's event loop. Advance real time
    // inside `act` so those updates — and any deferred transition that would re-open
    // the panel — are flushed and attributed correctly.
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 500));
    });

    await waitFor(() => expect(dialog).toHaveAttribute('data-state', 'closed'));

    // Dismissing must never re-show the dialog. Bug: 1 (the spurious re-open between
    // the two close animations). Fixed: 0.
    expect(showModalSpy).not.toHaveBeenCalled();
    expect(dialog).toHaveAttribute('data-state', 'closed');

    showModalSpy.mockRestore();
  });
});
