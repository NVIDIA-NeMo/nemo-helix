// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { DeploymentsListRoute } from '@studio/routes/DeploymentsListRoute';
import { NewDeploymentRoute } from '@studio/routes/NewDeploymentRoute';
import { renderRoute } from '@studio/tests/util/render';
import { screen, within } from '@testing-library/react';
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

  /**
   * Regression guard matching the fine-tune wizard's: `CreateSecretModal` renders its own
   * `<form>`, and a form nested inside this page's `<form>` never receives its submit event
   * (whatwg/dom#756), so React never runs its `onSubmit` and submitting it navigated the
   * page instead of creating a secret. jsdom bubbles the nested submit and so cannot
   * reproduce the navigation — assert the structure that caused it.
   */
  it('renders the create-secret modal outside the wizard form', async () => {
    const user = userEvent.setup();
    renderPage('/workspaces/default/deployments/~new');

    await user.click(await screen.findByRole('radio', { name: 'HuggingFace' }));
    await user.click(await screen.findByRole('combobox', { name: /HuggingFace Secret/i }));
    await user.click(await screen.findByRole('menuitem', { name: 'New Secret' }));

    expect(await screen.findByText('Create Secret')).toBeInTheDocument();
    // Testing Library has no query for DOM structure, and structure is the whole point here.
    // eslint-disable-next-line testing-library/no-node-access
    expect(document.querySelector('form form')).toBeNull();
  });

  it('selects the newly created secret without reopening the dropdown', async () => {
    const user = userEvent.setup();
    renderPage('/workspaces/default/deployments/~new');

    await user.click(await screen.findByRole('radio', { name: 'HuggingFace' }));
    await user.click(await screen.findByRole('combobox', { name: /HuggingFace Secret/i }));
    await user.click(await screen.findByRole('menuitem', { name: 'New Secret' }));

    // The page has its own "Name" field, so scope the modal's inputs to the dialog.
    const dialog = within(await screen.findByRole('dialog', { name: 'Create Secret' }));
    await user.type(await dialog.findByRole('textbox', { name: 'Name' }), 'hf-token');
    // Masked input — rendered as a password field, so it has no `textbox` role.
    // Masked input — a password field, so no `textbox` role; the label also matches the
    // field's help text, hence `selector`.
    await user.type(await dialog.findByLabelText('Value', { selector: 'input' }), 'hf_value');
    await user.click(await dialog.findByRole('button', { name: 'Create' }));

    // The name comes from the created secret the API returns, not from what was typed —
    // the mock names every created secret `test-secret`.
    expect(await screen.findByRole('combobox', { name: /HuggingFace Secret/i })).toHaveTextContent(
      'test-secret'
    );
  });

  it('returns to the deployments list on Cancel', async () => {
    const user = userEvent.setup();
    renderPage('/workspaces/default/deployments/~new');
    await user.click(await screen.findByRole('button', { name: 'Cancel' }));
    expect(await screen.findByPlaceholderText('Search Deployments...')).toBeInTheDocument();
  });
});
