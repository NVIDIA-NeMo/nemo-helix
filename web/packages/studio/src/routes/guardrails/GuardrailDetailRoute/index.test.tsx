// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { mockGuardrailConfigs } from '@studio/mocks/handlers/guardrails';
import { server } from '@studio/mocks/node';
import { GuardrailChecksTab } from '@studio/routes/guardrails/GuardrailChecksTab';
import { GUARDRAIL_CHECKS_DEFAULT_SUB_TAB } from '@studio/routes/guardrails/GuardrailChecksTab/constants';
import { GuardrailConfigTab } from '@studio/routes/guardrails/GuardrailConfigTab';
import { GuardrailDetailRoute } from '@studio/routes/guardrails/GuardrailDetailRoute';
import { getGuardrailDetailRoute } from '@studio/routes/utils';
import { XL_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { delay, http, HttpResponse } from 'msw';
import { Navigate } from 'react-router';

const WORKSPACE = 'default';

beforeEach(() => {
  localStorage.clear();
});

const routes = [
  {
    path: ROUTES.workspace.guardrailDetail,
    element: <GuardrailDetailRoute />,
    errorElement: <RouteErrorPanel />,
    children: [
      { index: true, element: <Navigate to="config" replace /> },
      { path: ROUTES.workspace.guardrailConfig, element: <GuardrailConfigTab /> },
      {
        path: ROUTES.workspace.guardrailChecks,
        element: <Navigate to={GUARDRAIL_CHECKS_DEFAULT_SUB_TAB} replace />,
      },
      { path: ROUTES.workspace.guardrailChecksSubTab, element: <GuardrailChecksTab /> },
    ],
  },
  {
    path: ROUTES.workspace.guardrails,
    element: <div data-testid="guardrails-list">LIST</div>,
  },
];

const renderDetail = (name: string) =>
  renderRoute(undefined, {
    history: getGuardrailDetailRoute(WORKSPACE, name),
    routes,
  });

describe('GuardrailDetailRoute', () => {
  it('renders the config details from the detail endpoint', async () => {
    renderDetail('pii-filter');

    expect(
      await screen.findByText('pii-filter', undefined, { timeout: XL_SELECTOR_TIMEOUT })
    ).toBeInTheDocument();
    // The index route redirects to `config`; react-router v7 runs that navigation
    // inside startTransition, so the tab content resolves asynchronously.
    expect(
      await screen.findByText('Guardrail Configuration', undefined, {
        timeout: XL_SELECTOR_TIMEOUT,
      })
    ).toBeInTheDocument();
    expect(screen.getByText('Self Checks')).toBeInTheDocument();

    // pii-filter uses rails Studio cannot configure yet; they stay visible in the JSON
    // rather than disappearing because no rail definition claims them. CodeMirror splits
    // each line into syntax-coloured spans, so read the editor's text rather than
    // matching nodes.
    expect(screen.getByText('Configuration JSON')).toBeInTheDocument();
    const json = screen.getByTestId('nv-code-editor-root').textContent ?? '';
    expect(json).toContain('check toxicity');
    expect(json).toContain('text-embedding-ada-002');
  });

  it('hides the draft actions until there are edits', async () => {
    renderDetail('pii-filter');
    await screen.findByText('pii-filter', undefined, { timeout: XL_SELECTOR_TIMEOUT });
    expect(screen.queryByRole('button', { name: 'Save Guardrail' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Reset' })).not.toBeInTheDocument();
  });

  it('reveals Reset and Save Guardrail after editing, and Reset discards them', async () => {
    const user = userEvent.setup();
    renderDetail('pii-filter');
    await screen.findByText('pii-filter', undefined, { timeout: XL_SELECTOR_TIMEOUT });

    // Tab content arrives after the index route's redirect settles (see above).
    await user.click(
      await screen.findByRole('switch', { name: 'Self Checks' }, { timeout: XL_SELECTOR_TIMEOUT })
    );

    expect(await screen.findByRole('button', { name: 'Save Guardrail' })).toBeInTheDocument();
    const reset = screen.getByRole('button', { name: 'Reset' });

    await user.click(reset);

    expect(screen.queryByRole('button', { name: 'Save Guardrail' })).not.toBeInTheDocument();
    expect(screen.getByRole('switch', { name: 'Self Checks' })).not.toBeChecked();
  });

  it('suspends to the loading fallback while fetching', async () => {
    server.use(
      http.get(
        `${PLATFORM_BASE_URL}/apis/guardrails/v2/workspaces/:workspace/configs/:name`,
        async () => {
          await delay();
          return HttpResponse.json(mockGuardrailConfigs[0]);
        }
      )
    );
    renderDetail('pii-filter');
    expect(await screen.findByTestId('spinner')).toBeInTheDocument();
  });

  it('renders the route error panel when the config cannot be loaded', async () => {
    // React and the router log the error the route throws to its errorElement.
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
    renderDetail('does-not-exist');
    expect(
      await screen.findByTestId('error-panel', undefined, {
        timeout: XL_SELECTOR_TIMEOUT,
      })
    ).toBeInTheDocument();
    expect(screen.getByText('Not Found')).toBeInTheDocument();
    consoleError.mockRestore();
  });

  it('saves the draft and clears the draft actions', async () => {
    const user = userEvent.setup();
    renderDetail('pii-filter');
    await screen.findByText('pii-filter', undefined, { timeout: XL_SELECTOR_TIMEOUT });

    // Tab content arrives after the index route's redirect settles (see above).
    await user.click(
      await screen.findByRole('switch', { name: 'Self Checks' }, { timeout: XL_SELECTOR_TIMEOUT })
    );
    await user.click(await screen.findByRole('button', { name: 'Save Guardrail' }));

    await waitFor(() =>
      expect(screen.queryByRole('button', { name: 'Save Guardrail' })).not.toBeInTheDocument()
    );
  });
});
