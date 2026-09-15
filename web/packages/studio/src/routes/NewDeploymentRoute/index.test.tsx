// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { DeploymentsListRoute } from '@studio/routes/DeploymentsListRoute';
import { NewDeploymentRoute } from '@studio/routes/NewDeploymentRoute';
import { renderRoute } from '@studio/tests/util/render';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const renderPage = (history: string) =>
  renderRoute(undefined, {
    history,
    routes: [
      { path: ROUTES.workspace.deployments, element: <DeploymentsListRoute /> },
      { path: ROUTES.workspace.deploymentsNew, element: <NewDeploymentRoute /> },
    ],
  });

describe('NewDeploymentRoute', () => {
  it('renders the wizard as a page', async () => {
    renderPage('/workspaces/default/deployments/~new');
    expect(await screen.findByRole('button', { name: 'Deploy' })).toBeInTheDocument();
    // A page, not a panel over the list: the deployments list is not rendered
    // behind it. (`role="dialog"` is unusable as a signal here — KUI's closed
    // info popovers carry it too.)
    expect(screen.queryByPlaceholderText('Search Deployments...')).not.toBeInTheDocument();
  });

  it('prefills the Workspace source from `?model=`', async () => {
    renderPage('/workspaces/default/deployments/~new?model=default%2Fmy-model');
    expect(await screen.findByRole('radio', { name: /model/i })).toBeChecked();
  });

  it('returns to the deployments list on Cancel', async () => {
    const user = userEvent.setup();
    renderPage('/workspaces/default/deployments/~new');
    await user.click(await screen.findByRole('button', { name: 'Cancel' }));
    expect(await screen.findByPlaceholderText('Search Deployments...')).toBeInTheDocument();
  });
});
